"""
distilbert_model.py
-------------------
DistilBERT embedding with three operational modes for the experiment:

  distilbert_frozen  — pretrained weights frozen, [CLS] projected to 128-dim,
                       only the projection + MLP head are trained.
  distilbert_ft      — end-to-end fine-tuning of DistilBERT + projection + MLP.
  distilbert_mlm_ft  — domain-adaptive MLM pretraining first, then fine-tune.

The custom token vocabulary (AMT_HIGH, SEG_MICRO, TIME_LATE_NIGHT, …) is
treated as natural language: DistilBERT's WordPiece tokenizer splits these
tokens into subwords. [CLS] and [SEP] map to DistilBERT's own special tokens.

For the frozen variant, embeddings are pre-computed once (efficient).
For the FT variants, the full forward pass runs each training batch.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ─── Tokenised Dataset ───────────────────────────────────────────────────────

class TokenizedDataset(Dataset):
    """
    On-the-fly HuggingFace tokenisation of account token strings.

    Tokenisation happens in __getitem__ — avoids holding 356k × 512-token
    tensors in memory simultaneously.
    """

    def __init__(
        self,
        token_strings: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.token_strings = token_strings
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.token_strings[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def _collate_pad(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate to batch-maximum length (avoids fixed max_length padding waste)."""
    input_ids = [item["input_ids"] for item in batch]
    masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    max_len = max(t.size(0) for t in input_ids)
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, msk) in enumerate(zip(input_ids, masks)):
        n = ids.size(0)
        padded_ids[i, :n] = ids
        padded_masks[i, :n] = msk

    return {"input_ids": padded_ids, "attention_mask": padded_masks, "label": labels}


# ─── Frozen-variant embedder ─────────────────────────────────────────────────

class DistilBERTEmbedder:
    """
    Pre-compute projected [CLS] embeddings for the distilbert_frozen variant.

    DistilBERT weights are never updated. A randomly-initialised projection
    Linear(768 → cls_projection_dim) is also frozen — the MLP downstream is the
    only thing that trains. This tests what the off-the-shelf representation
    already knows about fraud-related language.

    Parameters
    ----------
    model_name        : HuggingFace model identifier or local path
    cls_projection_dim: output embedding dimension (128)
    device            : torch device for inference
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        cls_projection_dim: int = 128,
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.cls_projection_dim = cls_projection_dim
        self.embedding_dim = cls_projection_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info("Loading DistilBERT tokeniser + model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._bert = AutoModel.from_pretrained(model_name).to(self.device)
        bert_dim = getattr(self._bert.config, "dim", None) or self._bert.config.hidden_size
        self._projection = nn.Linear(bert_dim, cls_projection_dim).to(self.device)

        # Freeze everything — projection is random but frozen for fair comparison
        for p in self._bert.parameters():
            p.requires_grad = False
        for p in self._projection.parameters():
            p.requires_grad = False

    def precompute_cls_vectors(
        self,
        token_strings: List[str],
        batch_size: int = 128,
        checkpoint_path: Optional["str | Path"] = None,
    ) -> np.ndarray:
        """
        Run all sequences through DistilBERT and return raw [CLS] hidden states.

        Returns np.ndarray of shape (n, 768), dtype float32.
        These are seed-independent (frozen pretrained weights) and can be
        cached on disk and reused across seeds — only apply_projection() differs.

        Parameters
        ----------
        checkpoint_path : optional directory for mid-run checkpointing.
            When provided, each batch is flushed to a memmap file on disk and a
            progress.json tracks how many samples are done.  If the process is
            killed and restarted with the same path, inference resumes from the
            last completed batch — no work is lost.
        """
        import json as _json

        n = len(token_strings)
        bert_dim = getattr(self._bert.config, "dim", None) or self._bert.config.hidden_size
        self._bert.eval()

        # ── checkpoint / memmap setup ────────────────────────────────────────
        n_done = 0
        memmap_arr: Optional[np.ndarray] = None

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            ckpt.mkdir(parents=True, exist_ok=True)
            arr_file  = ckpt / "cls_partial.dat"
            prog_file = ckpt / "progress.json"

            if prog_file.exists():
                n_done = _json.loads(prog_file.read_text()).get("n_done", 0)
                if n_done >= n:
                    logger.info("Checkpoint complete (%d / %d) — loading from memmap", n_done, n)
                    memmap_arr = np.memmap(arr_file, dtype=np.float32, mode="r", shape=(n, bert_dim))
                    return np.array(memmap_arr)
                logger.info("Resuming CLS precompute from %d / %d", n_done, n)

            mode = "r+" if arr_file.exists() and n_done > 0 else "w+"
            memmap_arr = np.memmap(arr_file, dtype=np.float32, mode=mode, shape=(n, bert_dim))
        else:
            all_cls: List[np.ndarray] = []

        # ── inference loop ───────────────────────────────────────────────────
        for start in range(n_done, n, batch_size):
            batch_texts = token_strings[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self._bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_hidden = outputs.last_hidden_state[:, 0, :]

            batch_cls = cls_hidden.cpu().numpy().astype(np.float32)
            end = start + len(batch_texts)

            if memmap_arr is not None:
                memmap_arr[start:end] = batch_cls
                memmap_arr.flush()
            else:
                all_cls.append(batch_cls)

            if (start // batch_size + 1) % 25 == 0 or end >= n:
                logger.info("  DistilBERT cls precompute: %d / %d", end, n)
                if checkpoint_path is not None:
                    Path(checkpoint_path, "progress.json").write_text(
                        _json.dumps({"n_done": end})
                    )

        if memmap_arr is not None:
            return np.array(memmap_arr)   # copy memmap → regular array before closing
        return np.concatenate(all_cls, axis=0)

    def apply_projection(self, cls_vectors: np.ndarray) -> np.ndarray:
        """
        Apply the frozen Linear(768 → cls_projection_dim) to pre-computed CLS vectors.

        This is a cheap CPU matrix multiply (~0.1s for 356k samples) and is
        seed-specific (random init), so it runs once per seed on the cached vectors.
        """
        self._projection.eval()
        tensor = torch.from_numpy(cls_vectors).to(self.device)
        with torch.inference_mode():
            projected = self._projection(tensor)
        return projected.cpu().numpy().astype(np.float32)

    def precompute_embeddings(
        self,
        token_strings: List[str],
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Run all sequences through DistilBERT and return projected [CLS] vectors.

        Returns
        -------
        np.ndarray of shape (n, cls_projection_dim), dtype float32
        """
        cls_vectors = self.precompute_cls_vectors(token_strings, batch_size=batch_size)
        return self.apply_projection(cls_vectors)

    def save(self, path: "str | Path") -> None:
        """Save BERT + projection state so embeddings are reproducible."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._bert.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(
            {
                "bert_state": self._bert.state_dict(),
                "projection_state": self._projection.state_dict(),
                "cls_projection_dim": self.cls_projection_dim,
                "max_length": self.max_length,
            },
            path / "embedder_states.pt",
        )
        logger.info("DistilBERTEmbedder saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path", device: Optional[torch.device] = None) -> "DistilBERTEmbedder":
        path = Path(path)
        ckpt = torch.load(path / "embedder_states.pt", map_location="cpu")
        obj = cls(
            model_name=str(path),
            cls_projection_dim=ckpt["cls_projection_dim"],
            device=device,
            max_length=ckpt.get("max_length", 128),
        )
        obj._bert.load_state_dict(ckpt["bert_state"])
        obj._projection.load_state_dict(ckpt["projection_state"])
        return obj


# ─── End-to-end classifier (FT and MLM+FT variants) ──────────────────────────

class DistilBERTClassifier(nn.Module):
    """
    End-to-end DistilBERT + projection + MLP head for fine-tuning variants.

    Parameters
    ----------
    model_name        : pretrained model name or local path (e.g. after MLM pretraining)
    cls_projection_dim: DistilBERT hidden → this dim via Linear+ReLU
    hidden_dims       : MLP hidden layer sizes after projection
    dropout           : dropout rate in MLP
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        cls_projection_dim: int = 128,
        hidden_dims: List[int] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = getattr(self.bert.config, "dim", None) or self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(bert_dim, cls_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        dims = [cls_projection_dim] + list(hidden_dims)
        mlp_layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        mlp_layers.append(nn.Linear(dims[-1], 1))
        self.mlp_head = nn.Sequential(*mlp_layers)

        self.config_params = {
            "model_name": model_name,
            "cls_projection_dim": cls_projection_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return raw logit of shape (batch,)."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_hidden)
        return self.mlp_head(projected).squeeze(-1)

    def save(self, path: "str | Path") -> None:
        """Save BERT backbone (save_pretrained) + projection/MLP head state_dict."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.bert.save_pretrained(path / "bert_backbone")
        torch.save(
            {
                "projection_state": self.projection.state_dict(),
                "mlp_head_state": self.mlp_head.state_dict(),
                "config_params": self.config_params,
            },
            path / "heads.pt",
        )
        logger.info("DistilBERTClassifier saved → %s", path)

    @classmethod
    def load(
        cls, path: "str | Path", device: Optional[torch.device] = None
    ) -> "DistilBERTClassifier":
        path = Path(path)
        ckpt = torch.load(path / "heads.pt", map_location="cpu")
        cfg = ckpt["config_params"]
        model = cls(
            model_name=str(path / "bert_backbone"),
            cls_projection_dim=cfg["cls_projection_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )
        model.projection.load_state_dict(ckpt["projection_state"])
        model.mlp_head.load_state_dict(ckpt["mlp_head_state"])
        if device:
            model.to(device)
        return model


# ─── MLM domain-adaptive pretraining ─────────────────────────────────────────

def pretrain_mlm(
    train_token_strings: List[str],
    model_name: str,
    save_path: "str | Path",
    mlm_epochs: int = 5,
    mlm_lr: float = 5e-5,
    mlm_batch_size: int = 32,
    mlm_probability: float = 0.15,
    max_length: int = 512,
    seed: int = 42,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional["str | Path"] = None,
) -> str:
    """
    Domain-adaptive MLM pretraining on the fraud-detection token corpus.

    Adapts DistilBERT's weights to the custom token vocabulary
    (AMT_HIGH, SEG_MICRO, TIME_LATE_NIGHT, …) before fine-tuning.

    Returns
    -------
    str : path to the saved pretrained model directory (for subsequent fine-tuning)
    """
    from transformers import DataCollatorForLanguageModeling

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for MLM pretraining: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    class _MLMDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tok = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

    dataset = _MLMDataset(train_token_strings, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=mlm_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=mlm_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    import json as _json
    start_epoch = 1
    _mlm_ckpt_dir = None
    if checkpoint_path is not None:
        _mlm_ckpt_dir = Path(checkpoint_path)
        _mlm_ckpt_dir.mkdir(parents=True, exist_ok=True)
        _prog = _mlm_ckpt_dir / "progress.json"
        _ckpt = _mlm_ckpt_dir / "checkpoint.pt"
        if _prog.exists() and _ckpt.exists():
            _epoch_done = _json.loads(_prog.read_text()).get("epoch_done", 0)
            if _epoch_done >= mlm_epochs and save_path.exists():
                logger.info("MLM pretraining already complete (%d/%d) — skipping", _epoch_done, mlm_epochs)
                return str(save_path)
            if _epoch_done > 0:
                logger.info("Resuming MLM pretraining from epoch %d/%d", _epoch_done + 1, mlm_epochs)
                _saved = torch.load(_ckpt, map_location=device)
                model.load_state_dict(_saved["model_state"])
                optimizer.load_state_dict(_saved["optimizer_state"])
                if "scaler_state" in _saved:
                    scaler.load_state_dict(_saved["scaler_state"])
                start_epoch = _epoch_done + 1

    model.train()

    use_amp = device.type == "cuda"
    for epoch in range(start_epoch, mlm_epochs + 1):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += outputs.loss.item()
        avg_loss = total_loss / len(loader)
        logger.info("MLM epoch %d/%d  loss=%.4f", epoch, mlm_epochs, avg_loss)
        if _mlm_ckpt_dir is not None:
            torch.save(
                {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scaler_state": scaler.state_dict()},
                _mlm_ckpt_dir / "checkpoint.pt",
            )
            (_mlm_ckpt_dir / "progress.json").write_text(_json.dumps({"epoch_done": epoch}))
            logger.info("MLM checkpoint saved: epoch %d/%d", epoch, mlm_epochs)

    # Save the backbone (not the MLM head) for subsequent fine-tuning.
    # DistilBERT wraps it as model.distilbert, BERT as model.bert.
    backbone = getattr(model, "distilbert", None) or model.bert
    backbone.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("MLM pretrained backbone saved → %s", save_path)
    return str(save_path)


# ─── End-to-end DistilBERT training loop ─────────────────────────────────────

def train_distilbert(
    train_token_strings: List[str],
    train_labels: np.ndarray,
    val_token_strings: List[str],
    val_labels: np.ndarray,
    model_name: str,
    cfg: dict,
    seed: int,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional["str | Path"] = None,
) -> Tuple["DistilBERTClassifier", Dict]:
    """
    End-to-end fine-tuning for distilbert_ft and distilbert_mlm_ft variants.

    Uses dual learning rates:
      - DistilBERT backbone: lr_distilbert (2e-5)
      - Projection + MLP head: lr_mlp (1e-3)

    Returns
    -------
    best_model : DistilBERTClassifier with best validation weights
    history    : dict with train_loss, val_pr_auc, best_epoch
    """
    from sklearn.metrics import average_precision_score

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_c = cfg.get("models", {})
    tc = models_c.get("training", {})
    dc = models_c.get("distilbert", {})
    mlp_c = models_c.get("downstream_mlp", {})

    batch_size = tc.get("batch_size", 64)
    max_epochs = tc.get("max_epochs_distilbert", 10)
    patience = tc.get("early_stopping_patience", 5)
    lr_bert = tc.get("lr_distilbert", 2e-5)
    lr_mlp = tc.get("lr_mlp", 1e-3)
    cls_projection_dim = dc.get("cls_projection_dim", 128)
    hidden_dims = mlp_c.get("hidden_dims", [256, 128])
    dropout = mlp_c.get("dropout", 0.3)
    max_length = dc.get("max_length", 512)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = DistilBERTClassifier(
        model_name=model_name,
        cls_projection_dim=cls_projection_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # Dual learning rates: slow for BERT backbone, fast for new heads
    param_groups = [
        {"params": model.bert.parameters(), "lr": lr_bert},
        {"params": list(model.projection.parameters()) + list(model.mlp_head.parameters()),
         "lr": lr_mlp},
    ]
    optimizer = torch.optim.Adam(param_groups)

    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("DistilBERT pos_weight=%.2f", pos_weight.item())

    train_dataset = TokenizedDataset(train_token_strings, train_labels, tokenizer, max_length)
    val_dataset = TokenizedDataset(val_token_strings, val_labels, tokenizer, max_length)

    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: _collate_pad(b, pad_id), persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
        collate_fn=lambda b: _collate_pad(b, pad_id), persistent_workers=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    best_pr_auc = -1.0
    best_state: Optional[Dict] = None
    best_epoch = 0
    no_improve = 0
    train_losses: List[float] = []
    val_pr_aucs: List[float] = []

    import json as _json
    start_epoch = 1
    _ft_ckpt_dir = None
    if checkpoint_path is not None:
        _ft_ckpt_dir = Path(checkpoint_path)
        _ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        _prog = _ft_ckpt_dir / "progress.json"
        _ckpt = _ft_ckpt_dir / "checkpoint.pt"
        if _prog.exists() and _ckpt.exists():
            _epoch_done = _json.loads(_prog.read_text()).get("epoch_done", 0)
            if _epoch_done > 0:
                logger.info("Resuming FT training from epoch %d/%d", _epoch_done + 1, max_epochs)
                _saved = torch.load(_ckpt, map_location=device)
                model.load_state_dict(_saved["model_state"])
                optimizer.load_state_dict(_saved["optimizer_state"])
                if "scaler_state" in _saved:
                    scaler.load_state_dict(_saved["scaler_state"])
                best_pr_auc = _saved["best_pr_auc"]
                best_epoch = _saved["best_epoch"]
                best_state = _saved["best_state"]
                no_improve = _saved["no_improve"]
                train_losses = _saved["train_losses"]
                val_pr_aucs = _saved["val_pr_aucs"]
                start_epoch = _epoch_done + 1

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * len(labels)
        epoch_loss /= len(train_labels)
        train_losses.append(epoch_loss)

        model.eval()
        all_scores: List[np.ndarray] = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for batch in val_loader:
                iids = batch["input_ids"].to(device)
                amsk = batch["attention_mask"].to(device)
                scores = torch.sigmoid(model(iids, amsk)).cpu().numpy()
                all_scores.append(scores)
        val_scores = np.concatenate(all_scores)
        val_pr_auc = float(average_precision_score(val_labels, val_scores))
        val_pr_aucs.append(val_pr_auc)

        logger.info(
            "BERT epoch %2d/%d  loss=%.4f  val_pr_auc=%.4f",
            epoch, max_epochs, epoch_loss, val_pr_auc,
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if _ft_ckpt_dir is not None:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_pr_auc": best_pr_auc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "no_improve": no_improve,
                "train_losses": train_losses,
                "val_pr_aucs": val_pr_aucs,
            }, _ft_ckpt_dir / "checkpoint.pt")
            (_ft_ckpt_dir / "progress.json").write_text(_json.dumps({"epoch_done": epoch}))
            logger.info("FT checkpoint saved: epoch %d/%d", epoch, max_epochs)

        if no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (best val_pr_auc=%.4f at epoch %d)",
                epoch, best_pr_auc, best_epoch,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_loss": train_losses,
        "val_pr_auc": val_pr_aucs,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_pr_auc,
    }
    return model, history
