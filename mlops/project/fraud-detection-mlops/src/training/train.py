"""
train.py
--------
Phase 4 training orchestration.

Runs 6 embedding variants × 3 seeds = 18 experiments, logging everything to
MLflow. All hyper-parameters come from experiment_config.yaml via the CFG
singleton.

Usage:
    python -m src.training.train                          # all 18 runs
    python -m src.training.train --variants tfidf word2vec --seeds 42
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CFG
from src.utils.mlflow_utils import (
    log_artifact_file,
    log_artifacts_dir,
    log_metrics,
    log_params,
    setup_mlflow,
    start_run,
)
from src.models.mlp_classifier import FraudMLP, save_mlp, train_mlp
from src.models.tfidf_model import TFIDFEmbedder
from src.models.word2vec_model import Word2VecEmbedder
from src.models.fasttext_model import FastTextEmbedder
from src.models.doc2vec_model import Doc2VecEmbedder
from src.models.distilbert_model import (
    DistilBERTClassifier,
    DistilBERTEmbedder,
    pretrain_mlm,
    train_distilbert,
)
from src.training.evaluate import (
    evaluate_distilbert,
    evaluate_embeddings,
    format_results_table,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PREPROCESSING_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing"
ARTIFACTS_PATH = PROJECT_ROOT / "model_artifacts"
BERT_CLS_CACHE = PROJECT_ROOT / "data" / "processed" / "bert_cls_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_VARIANTS = [
    "tfidf",
    "word2vec",
    "fasttext",
    "doc2vec",
    "distilbert_frozen",
    "distilbert_ft",
    "distilbert_mlm_ft",
]
ALL_SEEDS = [42, 123, 456]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load tokenised parquets produced by Phase 2–3.

    Returns
    -------
    train_df, val_df, test_df — each with columns:
        token_string        (str, space-separated)
        is_fraud_receiver   (int, 0/1)
        segment             (str)
    """
    logger.info("Loading tokenised splits from %s", PREPROCESSING_PATH)
    train_df = pd.read_parquet(PREPROCESSING_PATH / "train_tokenized.parquet")
    val_df = pd.read_parquet(PREPROCESSING_PATH / "val_tokenized.parquet")
    test_df = pd.read_parquet(PREPROCESSING_PATH / "test_tokenized.parquet")
    logger.info(
        "Loaded: train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _subsample_train(train_df: pd.DataFrame, ratio: int, seed: int = 42) -> pd.DataFrame:
    """
    Subsample training data while preserving ALL fraud cases.

    Keeps every fraud row and randomly samples `ratio` × n_fraud non-fraud rows.
    Val and test splits are never touched — evaluation remains unbiased.

    Parameters
    ----------
    ratio : int
        How many non-fraud rows to keep per fraud row (e.g. 15 → 15:1 neg:pos).
    seed  : int
        Random seed for reproducible sampling.
    """
    fraud = train_df[train_df["is_fraud_receiver"] == 1]
    non_fraud = train_df[train_df["is_fraud_receiver"] == 0]
    n_keep = min(len(non_fraud), len(fraud) * ratio)
    non_fraud_sampled = non_fraud.sample(n=n_keep, random_state=seed)
    result = pd.concat([fraud, non_fraud_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Subsampled train: %d → %d  (fraud=%d kept, non-fraud=%d/%d, ratio=1:%d)",
        len(train_df), len(result), len(fraud), n_keep, len(non_fraud), ratio,
    )
    return result


def _get_arrays(
    df: pd.DataFrame,
) -> Tuple[List[str], List[List[str]], np.ndarray]:
    """Extract token_strings, token_sequences, and labels from a split DataFrame."""
    token_strings = df["token_string"].tolist()
    token_sequences = [s.split() for s in token_strings]
    labels = df["is_fraud_receiver"].values.astype(np.int32)
    return token_strings, token_sequences, labels


# ─── BERT CLS vector cache (shared across seeds for distilbert_frozen) ────────

def _load_or_compute_cls(
    token_strings: List[str],
    split: str,
    embedder: "DistilBERTEmbedder",
    model_name: str,
    max_length: int,
) -> np.ndarray:
    """
    Load cached 768-dim BERT CLS vectors from disk, or compute + cache them.

    Because the DistilBERT backbone is frozen pretrained weights, its output is
    identical for all seeds. Caching avoids running the slow BERT forward pass
    once per seed — only the cheap apply_projection() call is seed-specific.
    """
    import json

    BERT_CLS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = BERT_CLS_CACHE / f"{split}_cls768.npy"
    meta_file  = BERT_CLS_CACHE / "cache_meta.json"

    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if (
            meta.get("model_name") == model_name
            and meta.get("max_length") == max_length
            and meta.get(f"n_{split}") == len(token_strings)
        ):
            logger.info("Loading cached BERT CLS vectors: %s (%d samples)", split, len(token_strings))
            return np.load(cache_file)
        logger.info("Cache mismatch for %s split — recomputing", split)

    logger.info("Computing BERT CLS vectors: %s split (%d samples)…", split, len(token_strings))
    checkpoint_dir = BERT_CLS_CACHE / f"{split}_ckpt"
    cls_vecs = embedder.precompute_cls_vectors(token_strings, checkpoint_path=checkpoint_dir)
    np.save(cache_file, cls_vecs)

    meta: dict = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    meta.update({"model_name": model_name, "max_length": max_length, f"n_{split}": len(token_strings)})
    meta_file.write_text(json.dumps(meta, indent=2))
    logger.info("Cached BERT CLS vectors → %s", cache_file)
    return cls_vecs


# ─── Artifact helpers ─────────────────────────────────────────────────────────

def _artifact_dir(variant: str, seed: int) -> Path:
    d = ARTIFACTS_PATH / variant / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── Statistical embedder + MLP path ─────────────────────────────────────────
#  Used by: tfidf, word2vec, fasttext, distilbert_frozen

def _run_statistical(
    variant: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Dict:
    """
    Shared training path for all variants that pre-compute embeddings.
    Returns a metrics dict with val_ and test_ prefixed keys.
    """
    np.random.seed(seed)

    train_strings, train_seqs, y_train = _get_arrays(train_df)
    val_strings, val_seqs, y_val = _get_arrays(val_df)
    test_strings, test_seqs, y_test = _get_arrays(test_df)

    adir = _artifact_dir(variant, seed)

    # ── Fit embedder ──────────────────────────────────────────────────────────
    mc = CFG.get("models", {})
    dc = mc.get("distilbert", {})
    if variant == "tfidf":
        tc = mc.get("tfidf", {})
        embedder = TFIDFEmbedder(
            max_features=tc.get("max_features", 5000),
            ngram_range=tuple(tc.get("ngram_range", [1, 3])),
        )
        X_train = embedder.fit_transform(train_strings)
        X_val = embedder.transform(val_strings)
        X_test = embedder.transform(test_strings)
        embedder.save(adir / "embedder.pkl")

    elif variant == "word2vec":
        wc = mc.get("word2vec", {})
        embedder = Word2VecEmbedder(
            vector_size=wc.get("vector_size", 128),
            window=wc.get("window", 5),
            min_count=wc.get("min_count", 1),
            epochs=wc.get("epochs", 20),
            sg=wc.get("sg", 1),
            seed=seed,
        )
        X_train = embedder.fit_transform(train_seqs)
        X_val = embedder.transform(val_seqs)
        X_test = embedder.transform(test_seqs)
        embedder.save(adir / "embedder.model")

    elif variant == "fasttext":
        fc = mc.get("fasttext", {})
        embedder = FastTextEmbedder(
            vector_size=fc.get("vector_size", 128),
            window=fc.get("window", 5),
            min_count=fc.get("min_count", 1),
            epochs=fc.get("epochs", 20),
            seed=seed,
        )
        X_train = embedder.fit_transform(train_seqs)
        X_val = embedder.transform(val_seqs)
        X_test = embedder.transform(test_seqs)
        embedder.save(adir / "embedder.model")

    elif variant == "doc2vec":
        dc2 = mc.get("doc2vec", {})
        embedder = Doc2VecEmbedder(
            vector_size=dc2.get("vector_size", 128),
            window=dc2.get("window", 5),
            min_count=dc2.get("min_count", 1),
            epochs=dc2.get("epochs", 20),
            seed=seed,
        )
        X_train = embedder.fit_transform(train_seqs)   # uses stored doc vectors
        X_val = embedder.transform(val_seqs)           # infers for unseen docs
        X_test = embedder.transform(test_seqs)
        embedder.save(adir / "embedder.model")

    elif variant == "distilbert_frozen":
        model_name = dc.get("model_name", "distilbert-base-uncased")
        cls_dim = dc.get("cls_projection_dim", 128)
        max_length = dc.get("max_length", 128)
        embedder = DistilBERTEmbedder(
            model_name=model_name,
            cls_projection_dim=cls_dim,
            device=device,
            max_length=max_length,
        )
        # BERT backbone is frozen pretrained — CLS vectors are identical for all seeds.
        # Cache to disk so only the first seed pays the inference cost; subsequent
        # seeds skip straight to the cheap apply_projection() linear transform.
        cls_train = _load_or_compute_cls(train_strings, "train", embedder, model_name, max_length)
        cls_val   = _load_or_compute_cls(val_strings,   "val",   embedder, model_name, max_length)
        cls_test  = _load_or_compute_cls(test_strings,  "test",  embedder, model_name, max_length)
        logger.info("Applying seed-%d projection (Linear 768→%d)…", seed, cls_dim)
        X_train = embedder.apply_projection(cls_train)
        X_val   = embedder.apply_projection(cls_val)
        X_test  = embedder.apply_projection(cls_test)
        embedder.save(adir / "embedder")
    else:
        raise ValueError(f"Unknown statistical variant: {variant}")

    input_dim = X_train.shape[1]
    logger.info("Embedding dim: %d", input_dim)

    # ── Train MLP ─────────────────────────────────────────────────────────────
    mlp, history = train_mlp(X_train, y_train, X_val, y_val, input_dim, CFG, seed, device)
    save_mlp(mlp, adir / "mlp.pt")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_metrics = evaluate_embeddings(X_val, y_val, mlp, device, split_name="val")
    opt_threshold = val_metrics["val_threshold"]
    test_metrics = evaluate_embeddings(
        X_test, y_test, mlp, device, threshold=opt_threshold, split_name="test"
    )

    all_metrics = {**val_metrics, **test_metrics}
    all_metrics["best_epoch"] = history["best_epoch"]
    all_metrics["best_val_pr_auc"] = history["best_val_pr_auc"]
    return all_metrics


# ─── End-to-end DistilBERT path ───────────────────────────────────────────────
#  Used by: distilbert_ft, distilbert_mlm_ft

def _run_distilbert_e2e(
    variant: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
) -> Dict:
    """End-to-end fine-tuning path. Returns val_ and test_ prefixed metrics."""
    from transformers import AutoTokenizer

    train_strings, _, y_train = _get_arrays(train_df)
    val_strings, _, y_val = _get_arrays(val_df)
    test_strings, _, y_test = _get_arrays(test_df)

    config_key = "bert" if variant.startswith("bert_") else "distilbert"
    dc = CFG.get("models", {}).get(config_key, {})
    base_model_name = dc.get("model_name", "distilbert-base-uncased")
    adir = _artifact_dir(variant, seed)

    model_name = base_model_name
    if variant in ("distilbert_mlm_ft", "bert_mlm_ft"):
        mlm_save = adir / "pretrained_mlm"
        logger.info("=== MLM pretraining (seed=%d) ===", seed)
        model_name = pretrain_mlm(
            train_token_strings=train_strings,
            model_name=base_model_name,
            save_path=mlm_save,
            mlm_epochs=dc.get("mlm_epochs", 5),
            mlm_lr=dc.get("mlm_lr", 5e-5),
            mlm_batch_size=dc.get("mlm_batch_size", 32),
            mlm_probability=dc.get("mlm_probability", 0.15),
            max_length=dc.get("max_length", 512),
            seed=seed,
            device=device,
            checkpoint_path=adir / "mlm_ckpt",
        )

    logger.info("=== Fine-tuning %s (seed=%d) ===", variant, seed)
    model, history = train_distilbert(
        train_token_strings=train_strings,
        train_labels=y_train,
        val_token_strings=val_strings,
        val_labels=y_val,
        model_name=model_name,
        cfg=CFG,
        seed=seed,
        device=device,
        checkpoint_path=adir / "ft_ckpt",
    )
    model.save(adir / "classifier")

    # Evaluate using a DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from src.models.distilbert_model import TokenizedDataset, _collate_pad

    max_length = dc.get("max_length", 512)
    pad_id = tokenizer.pad_token_id or 0
    batch_size = CFG.get("training", {}).get("batch_size", 64)

    def _make_loader(texts, labels):
        ds = TokenizedDataset(texts, labels, tokenizer, max_length)
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size * 2, shuffle=False, num_workers=0,
            collate_fn=lambda b: _collate_pad(b, pad_id),
        )

    val_loader = _make_loader(val_strings, y_val)
    test_loader = _make_loader(test_strings, y_test)

    val_metrics = evaluate_distilbert(model, val_loader, device, split_name="val")
    opt_threshold = val_metrics["val_threshold"]
    test_metrics = evaluate_distilbert(
        model, test_loader, device, threshold=opt_threshold, split_name="test"
    )

    all_metrics = {**val_metrics, **test_metrics}
    all_metrics["best_epoch"] = history["best_epoch"]
    all_metrics["best_val_pr_auc"] = history["best_val_pr_auc"]
    return all_metrics


# ─── Single experiment run ────────────────────────────────────────────────────

def run_experiment(
    variant: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run one (variant, seed) experiment and return the metrics dict.

    Handles MLflow logging, artifact saving, and error isolation.
    Skips the run entirely if results.json already exists in the artifact dir
    (written only after a fully successful run), so restarts never redo
    completed experiments.
    """
    import json

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{variant}_seed{seed}"
    adir = _artifact_dir(variant, seed)
    results_file = adir / "results.json"

    # ── Skip if already completed ─────────────────────────────────────────────
    if results_file.exists():
        metrics = json.loads(results_file.read_text())
        logger.info(
            "SKIP   %s  (results.json exists)  val_pr_auc=%.4f  test_pr_auc=%.4f",
            run_name,
            metrics.get("val_pr_auc", float("nan")),
            metrics.get("test_pr_auc", float("nan")),
        )
        return metrics

    logger.info("=" * 60)
    logger.info("START  %s", run_name)
    logger.info("=" * 60)

    # Build params dict for MLflow
    mc = CFG.get("models", {})
    variant_key = variant.split("_")[0]  # e.g. "distilbert" for all distilbert variants
    params = {
        "variant": variant,
        "seed": seed,
        "device": str(device),
        **{f"model.{k}": v for k, v in mc.get(variant_key, {}).items()},
        **{f"mlp.{k}": v for k, v in mc.get("downstream_mlp", {}).items()},
        **{f"training.{k}": v for k, v in mc.get("training", {}).items()},
    }

    with start_run(run_name, tags={"variant": variant, "seed": str(seed)}):
        log_params(params)

        if variant in ("distilbert_ft", "distilbert_mlm_ft", "bert_ft", "bert_mlm_ft"):
            metrics = _run_distilbert_e2e(variant, seed, train_df, val_df, test_df, device)
        else:
            metrics = _run_statistical(variant, seed, train_df, val_df, test_df, device)

        log_metrics(metrics)
        log_artifacts_dir(adir)

    # Write completion marker — only reached on full success
    results_file.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "DONE   %s  val_pr_auc=%.4f  test_pr_auc=%.4f",
        run_name,
        metrics.get("val_pr_auc", float("nan")),
        metrics.get("test_pr_auc", float("nan")),
    )
    return metrics


# ─── Main entrypoint ──────────────────────────────────────────────────────────

def main(
    variants: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> None:
    variants = variants or ALL_VARIANTS
    seeds = seeds or ALL_SEEDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    mlflow_cfg = CFG.get("mlflow", {})
    setup_mlflow(
        tracking_uri=mlflow_cfg.get("tracking_uri", "file:./mlruns"),
        experiment_name=mlflow_cfg.get("experiment_name", "fraud-detection-embeddings"),
    )

    train_df, val_df, test_df = load_data()

    tc = CFG.get("models", {}).get("training", {})
    subsample_ratio = tc.get("subsample_nonfraud_ratio", None)
    if subsample_ratio is not None:
        train_df = _subsample_train(train_df, ratio=int(subsample_ratio), seed=42)

    all_results: Dict[str, Dict] = {}
    failed: List[str] = []

    for variant in variants:
        for seed in seeds:
            run_name = f"{variant}_seed{seed}"
            try:
                metrics = run_experiment(variant, seed, train_df, val_df, test_df, device)
                all_results[run_name] = metrics
            except Exception as exc:
                logger.error("FAILED %s: %s", run_name, exc, exc_info=True)
                failed.append(run_name)

    # Summary
    logger.info("\n%s\n%s\n%s", "=" * 70, format_results_table(all_results), "=" * 70)
    if failed:
        logger.warning("Failed runs: %s", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: train all embedding variants")
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    main(args.variants, args.seeds)
