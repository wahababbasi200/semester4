"""
mlflow_utils.py
---------------
Central MLflow interaction layer for Phase 4 experiments.
All other files import from here; nothing else touches the MLflow API directly.

Experiment: 6 embedding variants × 3 seeds = 18 runs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """
    Configure MLflow tracking URI and experiment.
    Falls back to a local file store if the server is unreachable.

    Returns
    -------
    experiment_id : str
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        eid = exp.experiment_id if exp else "0"
        logger.info(
            "MLflow configured: tracking_uri=%s  experiment=%s (id=%s)",
            tracking_uri, experiment_name, eid,
        )
        return eid
    except Exception as exc:
        logger.warning("MLflow server unreachable (%s). Using local file store.", exc)
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else "0"


def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Return an MLflow ActiveRun context manager with standard phase-4 tags.

    Usage:
        with start_run("tfidf_seed42", {"variant": "tfidf", "seed": "42"}):
            log_params(...)
            log_metrics(...)
    """
    default_tags = {"phase": "4"}
    if tags:
        default_tags.update(tags)
    return mlflow.start_run(run_name=run_name, tags=default_tags)


def log_params(params: Dict[str, Any]) -> None:
    """
    Flatten and log a (possibly nested) params dict.

    Nested keys become dot-separated: {"mlp": {"lr": 1e-3}} → {"mlp.lr": "0.001"}
    Values are converted to str and truncated to 500 chars (MLflow limit).
    """
    if mlflow.active_run() is None:
        logger.warning("log_params called outside an active run — skipped.")
        return
    flat = _flatten(params)
    flat_str = {k: str(v)[:500] for k, v in flat.items()}
    try:
        mlflow.log_params(flat_str)
    except Exception as exc:
        logger.warning("mlflow.log_params failed: %s", exc)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a flat metrics dict. Non-numeric values are silently skipped."""
    if mlflow.active_run() is None:
        logger.warning("log_metrics called outside an active run — skipped.")
        return
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    try:
        mlflow.log_metrics(numeric, step=step)
    except Exception as exc:
        logger.warning("mlflow.log_metrics failed: %s", exc)


def log_artifacts_dir(local_dir: "str | Path") -> None:
    """Log all files under local_dir as MLflow artifacts."""
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_artifacts(str(local_dir))
    except Exception as exc:
        logger.warning("mlflow.log_artifacts failed: %s", exc)


def log_artifact_file(path: "str | Path", artifact_path: Optional[str] = None) -> None:
    """Log a single file as an MLflow artifact."""
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:
        logger.warning("mlflow.log_artifact failed: %s", exc)


def get_run_id() -> Optional[str]:
    """Return the active run ID, or None if not inside a run."""
    run = mlflow.active_run()
    return run.info.run_id if run else None


# ─── Internal ────────────────────────────────────────────────────────────────

def _flatten(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dict with dot-notation keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        else:
            out[key] = v
    return out
