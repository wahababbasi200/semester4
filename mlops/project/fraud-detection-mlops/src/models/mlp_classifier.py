"""
mlp_classifier.py
-----------------
PyTorch MLP classifier that operates on pre-computed embedding vectors.

Used as the **common downstream classifier** for all six embedding variants,
ensuring a fair comparison: differences in final performance reflect embedding
quality, not classifier differences.

Architecture:
    input → [Linear → ReLU → Dropout] × n_hidden_layers → Linear(1) → logit
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


# ─── Dataset ─────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Dataset wrapping pre-computed numpy embedding arrays and binary labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        # Convert once; __getitem__ just indexes — avoids double memory
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─── Model ───────────────────────────────────────────────────────────────────

class FraudMLP(nn.Module):
    """
    Feed-forward MLP for binary fraud classification on embedding vectors.

    Parameters
    ----------
    input_dim   : dimension of the input embedding
    hidden_dims : sizes of hidden layers, e.g. [256, 128]
    dropout     : dropout probability after each hidden activation
    activation  : "relu" | "gelu" | "tanh"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (256, 128),
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "activation": activation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit of shape (batch,). Use sigmoid for probabilities."""
        return self.net(x).squeeze(-1)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    cfg: dict,
    seed: int,
    device: Optional[torch.device] = None,
) -> Tuple["FraudMLP", Dict]:
    """
    Train a FraudMLP with early stopping on validation PR-AUC.

    Parameters
    ----------
    X_train / X_val : pre-computed embedding matrices, shape (n, input_dim)
    y_train / y_val : binary label arrays, shape (n,)
    input_dim       : must match embedding dim
    cfg             : full CFG dict from experiment_config.yaml
    seed            : random seed for reproducibility
    device          : torch device; auto-detected if None

    Returns
    -------
    best_model : FraudMLP loaded with the best checkpoint weights
    history    : dict with keys train_loss, val_pr_auc, best_epoch
    """
    from sklearn.metrics import average_precision_score

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Hyper-parameters from config ──────────────────────────────────────────
    mc = cfg.get("models", {})
    tc = mc.get("training", {})
    mlp_c = mc.get("downstream_mlp", {})
    hidden_dims = mlp_c.get("hidden_dims", [256, 128])
    dropout = mlp_c.get("dropout", 0.3)
    activation = mlp_c.get("activation", "relu")
    batch_size = tc.get("batch_size", 64)
    max_epochs = tc.get("max_epochs_mlp", 50)
    patience = tc.get("early_stopping_patience", 5)
    lr = tc.get("lr_mlp", 1e-3)

    # ── Class imbalance weight ─────────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    logger.info("pos_weight=%.2f (n_pos=%d, n_neg=%d)", pos_weight.item(), n_pos, n_neg)

    # ── Model, loss, optimiser ─────────────────────────────────────────────────
    model = FraudMLP(input_dim, hidden_dims, dropout, activation).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        EmbeddingDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_pr_auc = -1.0
    best_state: Optional[Dict] = None
    best_epoch = 0
    no_improve = 0
    train_losses: List[float] = []
    val_pr_aucs: List[float] = []

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
        epoch_loss /= len(y_train)
        train_losses.append(epoch_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(X_val_t)).cpu().numpy()
        val_pr_auc = float(average_precision_score(y_val, val_scores))
        val_pr_aucs.append(val_pr_auc)

        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_pr_auc=%.4f", epoch, max_epochs, epoch_loss, val_pr_auc
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
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


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_mlp(model: "FraudMLP", path: "str | Path") -> None:
    """Save model config + state_dict so it can be reconstructed without CFG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config}, path)
    logger.info("FraudMLP saved → %s", path)


def load_mlp(path: "str | Path", device: Optional[torch.device] = None) -> "FraudMLP":
    """Reconstruct a FraudMLP from a checkpoint file."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = FraudMLP(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model
