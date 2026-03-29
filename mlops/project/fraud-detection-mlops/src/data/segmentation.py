"""
segmentation.py
---------------
Phase 3: Account behavioral segmentation (SEG_MICRO / SEG_REGULAR / SEG_HEAVY)
based on a composite activity score derived from three account-level features.

Design rationale:
  - EDA showed most fraud receivers are single-use accounts (txn_count=1,
    active_duration=0, small total_volume) → they land in SEG_MICRO.
  - Segmentation is fit on training accounts only to prevent leakage.
  - Thresholds are saved as JSON so the serving container can reproduce
    segment assignment without any training data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEGMENT_NAMES = ["SEG_MICRO", "SEG_REGULAR", "SEG_HEAVY"]
_ACTIVITY_FEATURES = ["total_volume", "txn_count", "active_duration"]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _minmax_normalize(
    series: pd.Series,
    mn: float,
    mx: float,
) -> pd.Series:
    """Min-max scale series to [0, 1] using provided min/max (from training set)."""
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return ((series - mn) / (mx - mn)).clip(0.0, 1.0)




def fit_segmentation(
    train_accounts: pd.DataFrame,
    quantile_boundaries: List[float] = (0.33, 0.75),
) -> Dict:
    """
    Compute segmentation score boundaries from the training set.

    The composite activity score is the equally-weighted sum of min-max-
    normalized total_volume, txn_count, and active_duration (range [0, 3]).
    Two quantile thresholds partition this score into three segments:
      - SEG_MICRO   : score ≤ lower_threshold
      - SEG_REGULAR : lower_threshold < score ≤ upper_threshold
      - SEG_HEAVY   : score > upper_threshold

    Parameters
    ----------
    train_accounts : pd.DataFrame
        Training receiver accounts from temporal_split().
        Must contain: total_volume, txn_count, active_duration.
    quantile_boundaries : list of 2 floats
        Quantile positions for the SEG_MICRO/SEG_REGULAR and
        SEG_REGULAR/SEG_HEAVY boundaries.

    Returns
    -------
    dict with keys:
      - score_thresholds  : [lower, upper] boundary values
      - feature_stats     : per-feature min/max from training set
      - segment_names     : ["SEG_MICRO", "SEG_REGULAR", "SEG_HEAVY"]
      - quantile_boundaries: the input quantile_boundaries
    """
    assert len(quantile_boundaries) == 2, "Need exactly 2 quantile boundaries"
    assert quantile_boundaries[0] < quantile_boundaries[1], "Must be strictly increasing"

    # Compute and store training-set normalization statistics
    feature_stats = {}
    for feat in _ACTIVITY_FEATURES:
        feature_stats[feat] = {
            "min": float(train_accounts[feat].min()),
            "max": float(train_accounts[feat].max()),
        }

    # Composite score on training set
    score = sum(
        _minmax_normalize(
            train_accounts[feat],
            feature_stats[feat]["min"],
            feature_stats[feat]["max"],
        )
        for feat in _ACTIVITY_FEATURES
    )

    lower = float(np.quantile(score, quantile_boundaries[0]))
    upper = float(np.quantile(score, quantile_boundaries[1]))

    # Log training distribution
    n_micro   = int((score <= lower).sum())
    n_heavy   = int((score > upper).sum())
    n_regular = len(score) - n_micro - n_heavy
    logger.info(
        "Segmentation thresholds: SEG_MICRO ≤ %.4f < SEG_REGULAR ≤ %.4f < SEG_HEAVY",
        lower, upper,
    )
    logger.info(
        "Training distribution: SEG_MICRO=%d (%.1f%%), "
        "SEG_REGULAR=%d (%.1f%%), SEG_HEAVY=%d (%.1f%%)",
        n_micro,   100.0 * n_micro   / len(score),
        n_regular, 100.0 * n_regular / len(score),
        n_heavy,   100.0 * n_heavy   / len(score),
    )

    return {
        "score_thresholds":    [lower, upper],
        "feature_stats":       feature_stats,
        "segment_names":       _SEGMENT_NAMES,
        "quantile_boundaries": list(quantile_boundaries),
    }


def assign_segments(
    accounts_df: pd.DataFrame,
    thresholds: Dict,
) -> pd.Series:
    """
    Assign segment labels to accounts using pre-fitted thresholds.

    Normalization uses the training-set min/max stored in thresholds,
    so val/test accounts are scaled consistently (no data leakage).
    Out-of-range values are clipped to [0, 1].

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Receiver accounts (any split).
    thresholds : dict
        Output of fit_segmentation().

    Returns
    -------
    pd.Series of segment strings indexed like accounts_df.
    """
    fstats = thresholds["feature_stats"]
    score = sum(
        _minmax_normalize(
            accounts_df[feat],
            fstats[feat]["min"],
            fstats[feat]["max"],
        )
        for feat in _ACTIVITY_FEATURES
    )

    lower, upper = thresholds["score_thresholds"]
    segment = pd.Series("SEG_REGULAR", index=accounts_df.index, dtype=object)
    segment[score <= lower] = "SEG_MICRO"
    segment[score > upper]  = "SEG_HEAVY"

    logger.info("Segment distribution: %s", segment.value_counts().to_dict())
    return segment


def save_segmentation_config(thresholds: Dict, path: str) -> None:
    """
    Persist segmentation thresholds to JSON.

    This file is baked into the Docker image so the serving container
    can assign segments to new accounts without any training data.

    Parameters
    ----------
    thresholds : dict
        Output of fit_segmentation().
    path : str
        Output path for the JSON file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Segmentation config saved to %s", out_path)


def load_segmentation_config(path: str) -> Dict:
    """Load a previously saved segmentation config."""
    with open(path) as f:
        return json.load(f)
