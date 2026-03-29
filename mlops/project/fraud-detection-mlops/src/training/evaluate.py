"""
evaluate.py
-----------
All evaluation logic for the fraud detection experiments.

Primary metric: PR-AUC (average precision score) — chosen because the dataset
has extreme class imbalance (~0.95% fraud rate in training).

All functions accept numpy arrays; they are intentionally framework-agnostic.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ─── Core metric computation ──────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    y_true  : binary ground-truth labels, shape (n,)
    y_score : predicted probabilities for the positive class, shape (n,)
    threshold : decision threshold; if None, the optimal F1 threshold is found.

    Returns
    -------
    dict with keys:
        pr_auc, roc_auc, f1, precision, recall, threshold,
        tp, fp, fn, tn, n_pos, n_neg, prevalence
    """
    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / len(y_true),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Find the decision threshold that maximises F1 on the given data.

    Uses the precision-recall curve from sklearn so all candidate thresholds
    come from the model's actual score distribution — no grid search needed.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns n+1 points; the last has no threshold
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


# ─── Split-level evaluation helpers ──────────────────────────────────────────

def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mlp,                          # FraudMLP (torch module)
    device,                       # torch.device
    batch_size: int = 512,
    threshold: Optional[float] = None,
    split_name: str = "eval",
) -> Dict[str, float]:
    """
    Run the MLP over pre-computed embeddings in batches and compute metrics.

    Returns a flat dict with keys prefixed by split_name (e.g. "val_pr_auc").
    """
    import torch

    mlp.eval()
    all_scores: List[np.ndarray] = []

    X = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size].to(device)
            logits = mlp(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)

    y_score = np.concatenate(all_scores)
    metrics = compute_metrics(labels, y_score, threshold=threshold)

    logger.info(
        "[%s] PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f  thr=%.4f",
        split_name,
        metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"], metrics["threshold"],
    )
    return {f"{split_name}_{k}": v for k, v in metrics.items()}


def evaluate_distilbert(
    model,             # DistilBERTClassifier (torch module)
    dataloader,        # torch DataLoader yielding {"input_ids", "attention_mask", "label"}
    device,
    threshold: Optional[float] = None,
    split_name: str = "eval",
) -> Dict[str, float]:
    """
    Run the end-to-end DistilBERT model over a DataLoader and compute metrics.
    """
    import torch

    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(batch["label"].numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels).astype(int)
    metrics = compute_metrics(y_true, y_score, threshold=threshold)

    logger.info(
        "[%s] PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f  thr=%.4f",
        split_name,
        metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"], metrics["threshold"],
    )
    return {f"{split_name}_{k}": v for k, v in metrics.items()}


# ─── Summary table ────────────────────────────────────────────────────────────

def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Pretty-print a comparison table.

    Parameters
    ----------
    results : {"tfidf_seed42": {"val_pr_auc": ..., "test_pr_auc": ...}, ...}
    """
    header = f"{'Run':<30} {'val_pr_auc':>10} {'test_pr_auc':>11} {'test_roc_auc':>12} {'test_f1':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for run_name, m in sorted(results.items()):
        rows.append(
            f"{run_name:<30} "
            f"{m.get('val_pr_auc', float('nan')):>10.4f} "
            f"{m.get('test_pr_auc', float('nan')):>11.4f} "
            f"{m.get('test_roc_auc', float('nan')):>12.4f} "
            f"{m.get('test_f1', float('nan')):>8.4f}"
        )
    return "\n".join(rows)
