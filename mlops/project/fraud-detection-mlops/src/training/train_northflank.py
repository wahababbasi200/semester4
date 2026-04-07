"""
train_northflank.py
-------------------
Northflank-compatible training entrypoint.

Runs experiments sequentially on a single GPU container — no Ray required.
Designed to run as a Northflank Job with GPU resources.

Usage:
    # ── Local (same as train.py) ──────────────────────────────────────
    python -m src.training.train_northflank --variants distilbert_ft bert_ft --seeds 42 123 456

    # ── Northflank Job ────────────────────────────────────────────────
    #   Build:  docker build -f Dockerfile.train -t fraud-train .
    #   Push:   docker tag fraud-train <registry>/fraud-train && docker push <registry>/fraud-train
    #   Run:    Create a Northflank Job with GPU, set the image, and override CMD:
    #           python -m src.training.train_northflank --variants distilbert_ft bert_ft --seeds 42 123 456

    # ── Parallel on Northflank (multiple jobs) ────────────────────────
    #   Submit one job per variant to run on separate GPU containers:
    #   Job 1: python -m src.training.train_northflank --variants distilbert_ft --seeds 42 123 456
    #   Job 2: python -m src.training.train_northflank --variants distilbert_mlm_ft --seeds 42 123 456
    #   Job 3: python -m src.training.train_northflank --variants bert_ft --seeds 42 123 456
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    "bert_ft",
    "bert_mlm_ft",
]
ALL_SEEDS = [42, 123, 456]


def main(
    variants: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> None:
    """
    Run training experiments sequentially on a single GPU.

    This is the Northflank-compatible entrypoint. Unlike train_ray.py which
    distributes experiments across a Ray cluster, this runs them one at a time
    on whatever GPU is available in the container.

    For parallelism on Northflank, submit multiple jobs — one per variant —
    each running on its own GPU container.
    """
    import torch

    from src.utils.config import CFG
    from src.utils.mlflow_utils import setup_mlflow
    from src.training.train import (
        load_data,
        run_experiment,
        _subsample_train,
        format_results_table,
    )

    variants = variants or ALL_VARIANTS
    seeds = seeds or ALL_SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("NORTHFLANK TRAINING")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)
    logger.info("Variants: %s", variants)
    logger.info("Seeds: %s", seeds)
    logger.info("Total experiments: %d", len(variants) * len(seeds))
    logger.info("=" * 70)

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow_cfg = CFG.get("mlflow", {})
    setup_mlflow(
        tracking_uri=mlflow_cfg.get("tracking_uri", "file:./mlruns"),
        experiment_name=mlflow_cfg.get("experiment_name", "fraud-detection-embeddings"),
    )

    # ── Load and subsample data ─────────────────────────────────────────────
    train_df, val_df, test_df = load_data()

    tc = CFG.get("models", {}).get("training", {})
    subsample_ratio = tc.get("subsample_nonfraud_ratio", None)
    if subsample_ratio is not None:
        train_df = _subsample_train(train_df, ratio=int(subsample_ratio), seed=42)

    # ── Run experiments sequentially ────────────────────────────────────────
    all_results: Dict[str, Dict] = {}
    failed: List[str] = []
    start_time = time.time()

    total = len(variants) * len(seeds)
    done = 0

    for variant in variants:
        for seed in seeds:
            done += 1
            run_name = f"{variant}_seed{seed}"
            logger.info("[%d/%d] Starting %s", done, total, run_name)
            exp_start = time.time()

            try:
                metrics = run_experiment(variant, seed, train_df, val_df, test_df, device)
                all_results[run_name] = metrics
                elapsed = time.time() - exp_start
                logger.info(
                    "[%d/%d] DONE  %s  (%.1f min)  test_pr_auc=%.4f",
                    done, total, run_name, elapsed / 60,
                    metrics.get("test_pr_auc", float("nan")),
                )
            except Exception as exc:
                elapsed = time.time() - exp_start
                logger.error(
                    "[%d/%d] FAILED  %s  (%.1f min): %s",
                    done, total, run_name, elapsed / 60, exc,
                    exc_info=True,
                )
                failed.append(run_name)

    # ── Summary ─────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY  (total time: %.1f min)", total_time / 60)
    logger.info("=" * 70)
    for name, result in sorted(all_results.items()):
        logger.info(
            "  %-30s  val_pr_auc=%.4f  test_pr_auc=%.4f",
            name,
            result.get("val_pr_auc", float("nan")),
            result.get("test_pr_auc", float("nan")),
        )
    if failed:
        logger.warning("Failed runs: %s", failed)
        sys.exit(1)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Northflank-compatible training (sequential, single GPU)"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=ALL_VARIANTS, default=None,
        help="Which embedding variants to train (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Which random seeds to use (default: 42, 123, 456)",
    )
    args = parser.parse_args()
    main(variants=args.variants, seeds=args.seeds)
