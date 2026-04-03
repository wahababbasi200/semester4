"""
train_ray.py
------------
Ray-accelerated training orchestration.

Usage:
    # ── Local (CPU-only) ──────────────────────────────────────────
    python -m src.training.train_ray
    python -m src.training.train_ray --max-concurrent 2

    # ── Anyscale hosted cloud (recommended) ──────────────────────────────
    anyscale job submit -f anyscale-job.yaml               # submit as managed job
    anyscale job logs <job-id>                              # view logs
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import ray

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
    # "tfidf",
    # "word2vec",
    # "fasttext",
    # "doc2vec",
    # "distilbert_frozen",

    # ── Fine-tune only ────────────────────────
    "distilbert_ft",

    # ── MLM pretrain + fine-tune────────────────────────────────
    "distilbert_mlm_ft",

    # ── BERT-base (heavier, 110M params) ──────────────────────
    "bert_ft",
    "bert_mlm_ft",
]
ALL_SEEDS = [42, 123, 456]


# Variants that are lightweight
STATISTICAL_VARIANTS = {"tfidf", "word2vec", "fasttext", "doc2vec"}
# Variants that are CPU/GPU-heavy
HEAVY_VARIANTS = {"distilbert_frozen", "distilbert_ft", "distilbert_mlm_ft",
                   "bert_ft", "bert_mlm_ft"}


# ─── Ray remote task ────────────────────────────────────────────────────────

@ray.remote
def run_experiment_remote(
    variant: str,
    seed: int,
    train_parquet_path: str,
    val_parquet_path: str,
    test_parquet_path: str,
) -> Dict:
    """
    Run a single (variant, seed) experiment inside a Ray worker.

    Data is passed by path (not value) to avoid serialising large DataFrames
    through Ray's object store. Each worker loads its own copy.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch

    # Re-add project root inside the worker process
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.utils.config import CFG
    from src.utils.mlflow_utils import (
        log_artifacts_dir,
        log_metrics,
        log_params,
        setup_mlflow,
        start_run,
    )
    from src.training.train import (
        _artifact_dir,
        _run_distilbert_e2e,
        _run_statistical,
        _subsample_train,
    )

    run_name = f"{variant}_seed{seed}"

    # ── Skip if already completed ────────────────────────────────────────────
    adir = _artifact_dir(variant, seed)
    results_file = adir / "results.json"
    if results_file.exists():
        metrics = json.loads(results_file.read_text())
        logger.info("SKIP  %s (results.json exists)", run_name)
        return {"run_name": run_name, "status": "skipped", **metrics}

    # ── Load data inside the worker ──────────────────────────────────────────
    train_df = pd.read_parquet(train_parquet_path)
    val_df = pd.read_parquet(val_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    tc = CFG.get("models", {}).get("training", {})
    subsample_ratio = tc.get("subsample_nonfraud_ratio", None)
    if subsample_ratio is not None:
        train_df = _subsample_train(train_df, ratio=int(subsample_ratio), seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MLflow setup (each worker needs its own connection) ──────────────────
    mlflow_cfg = CFG.get("mlflow", {})
    setup_mlflow(
        tracking_uri=mlflow_cfg.get("tracking_uri", "file:./mlruns"),
        experiment_name=mlflow_cfg.get("experiment_name", "fraud-detection-embeddings"),
    )

    # ── Build params dict for MLflow ─────────────────────────────────────────
    mc = CFG.get("models", {})
    variant_key = variant.split("_")[0]
    params = {
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "distributed": "ray",
        **{f"model.{k}": v for k, v in mc.get(variant_key, {}).items()},
        **{f"mlp.{k}": v for k, v in mc.get("downstream_mlp", {}).items()},
        **{f"training.{k}": v for k, v in mc.get("training", {}).items()},
    }

    logger.info("START  %s  (device=%s)", run_name, device)

    with start_run(run_name, tags={"variant": variant, "seed": str(seed), "runner": "ray"}):
        log_params(params)

        if variant in ("distilbert_ft", "distilbert_mlm_ft", "bert_ft", "bert_mlm_ft"):
            metrics = _run_distilbert_e2e(variant, seed, train_df, val_df, test_df, device)
        else:
            metrics = _run_statistical(variant, seed, train_df, val_df, test_df, device)

        log_metrics(metrics)
        log_artifacts_dir(adir)

    # Write completion marker
    results_file.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "DONE  %s  val_pr_auc=%.4f  test_pr_auc=%.4f",
        run_name,
        metrics.get("val_pr_auc", float("nan")),
        metrics.get("test_pr_auc", float("nan")),
    )
    return {"run_name": run_name, "status": "completed", **metrics}


# ─── Main ────────────────────────────────────────────────────────────────────

def main(
    variants: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    ray_address: Optional[str] = None,
    max_concurrent: Optional[int] = None,
) -> None:
    """
    Launch all experiments as parallel Ray tasks.

    On a CPU-only machine, experiments are scheduled with smart resource
    allocation:
      - Statistical variants (tfidf, word2vec, etc.): 1 CPU each → many run
        in parallel
      - DistilBERT variants: more CPUs reserved → fewer run at once but each
        gets more compute

    Parameters
    ----------
    ray_address     : Ray cluster address (None = start local cluster)
    max_concurrent  : Cap on simultaneous experiments (None = auto)
    """
    variants = variants or ALL_VARIANTS
    seeds = seeds or ALL_SEEDS

    # ── Initialise Ray ───────────────────────────────────────────────────────
    # Anyscale jobs auto-set RAY_ADDRESS; detect and use it
    if ray_address:
        logger.info("Connecting to Ray cluster at %s", ray_address)
        ray.init(address=ray_address)
    elif os.environ.get("RAY_ADDRESS"):
        logger.info("Detected Anyscale/Ray environment (RAY_ADDRESS=%s)",
                     os.environ["RAY_ADDRESS"])
        ray.init(address="auto")
    else:
        logger.info("Starting local Ray cluster (CPU-only)")
        ray.init()

    total_cpus = int(ray.cluster_resources().get("CPU", os.cpu_count() or 4))
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    logger.info("Ray cluster resources: %d CPUs, %d GPUs", total_cpus, total_gpus)

    # ── Data paths (workers load independently) ──────────────────────────────
    preprocessing_path = PROJECT_ROOT / "data" / "processed" / "preprocessing"
    train_path = str(preprocessing_path / "train_tokenized.parquet")
    val_path = str(preprocessing_path / "val_tokenized.parquet")
    test_path = str(preprocessing_path / "test_tokenized.parquet")

    # ── Smart scheduling: lightweight tasks first, heavy tasks after ─────────
    # Sort so statistical variants run first (fast, parallel), then DistilBERT
    # variants run after (slow, need more CPU each).
    tasks = []
    for variant in variants:
        for seed in seeds:
            tasks.append((variant, seed))

    tasks.sort(key=lambda t: (t[0] in HEAVY_VARIANTS, t[0], t[1]))

    # ── Resource allocation per task type ────────────────────────────────────
    # Statistical: 1 CPU each (many run in parallel)
    # DistilBERT:  4 CPUs each (DataLoader workers) + 1 GPU — the GPU does
    #              the heavy lifting, so we don't need to hog CPUs.
    #              On small local clusters (≤16 CPUs), fall back to half.
    cpus_statistical = 1
    cpus_heavy = 4 if total_cpus > 16 else max(total_cpus // 2, 2)

    # If user set max_concurrent, give each task fewer CPUs accordingly
    if max_concurrent:
        cpus_statistical = max(total_cpus // max_concurrent, 1)
        cpus_heavy = max(total_cpus // max_concurrent, 1)

    logger.info(
        "Scheduling: statistical variants → %d CPU each, "
        "DistilBERT variants → %d CPUs each",
        cpus_statistical, cpus_heavy,
    )

    # ── Submit all experiments ────────────────────────────────────────────────
    futures = []
    task_names = []

    for variant, seed in tasks:
        run_name = f"{variant}_seed{seed}"
        task_names.append(run_name)

        if variant in HEAVY_VARIANTS:
            num_cpus = cpus_heavy
            # Always request 1 GPU for heavy variants on cloud clusters.
            # On Anyscale, GPU workers auto-scale when tasks request GPUs.
            # Only fall back to 0 GPUs on local (non-cloud) runs.
            is_cloud = bool(ray_address or os.environ.get("RAY_ADDRESS"))
            num_gpus = 1 if (total_gpus > 0 or is_cloud) else 0
        else:
            num_cpus = cpus_statistical
            num_gpus = 0

        future = run_experiment_remote.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        ).remote(variant, seed, train_path, val_path, test_path)

        futures.append(future)
        logger.info(
            "Submitted: %-25s  (num_cpus=%d, num_gpus=%d)",
            run_name, num_cpus, num_gpus,
        )

    logger.info("All %d experiments submitted to Ray", len(futures))

    # ── Collect results as they complete ─────────────────────────────────────
    all_results = {}
    failed = []
    remaining = list(zip(futures, task_names))

    while remaining:
        ready_refs = [f for f, _ in remaining]
        done, _ = ray.wait(ready_refs, num_returns=1)

        for done_ref in done:
            idx = next(i for i, (f, _) in enumerate(remaining) if f == done_ref)
            _, name = remaining.pop(idx)

            try:
                result = ray.get(done_ref)
                all_results[name] = result
                status = result.get("status", "completed")
                pr_auc = result.get("test_pr_auc", float("nan"))
                logger.info(
                    "FINISHED  %-30s  status=%-9s  test_pr_auc=%.4f",
                    name, status, pr_auc,
                )
            except Exception as exc:
                logger.error("FAILED  %s: %s", name, exc)
                failed.append(name)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
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
    logger.info("=" * 70)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ray-accelerated training for all embedding variants"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=ALL_VARIANTS, default=None,
        help="Which embedding variants to train (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Which random seeds to use (default: 42, 123, 456)",
    )
    parser.add_argument(
        "--ray-address", type=str, default=None,
        help="Ray cluster address (e.g. ray://<ip>:10001). "
             "Omit to start a local cluster on your laptop.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max experiments to run simultaneously (default: auto based on CPU count)",
    )
    args = parser.parse_args()

    main(
        variants=args.variants,
        seeds=args.seeds,
        ray_address=args.ray_address,
        max_concurrent=args.max_concurrent,
    )
