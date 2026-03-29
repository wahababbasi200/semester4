"""
preprocess.py
-------------
Phase 2: Account-level aggregation, receiver dataset construction,
temporal train/val/test split.

Key design decisions from EDA:
- Receiver-side labeling: nameDest accounts are labeled fraudulent if ANY
  incoming transaction had isFraud=1
- Temporal split by last_step: prevents data leakage from future behaviour
- Most fraud receivers are single-use (1 transaction, 0 active_duration)
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_receiver_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into receiver account-level features.

    Each row represents one unique nameDest with:
      - total_volume    : sum of all incoming transaction amounts
      - txn_count       : number of incoming transactions
      - first_step      : earliest transaction step
      - last_step       : latest transaction step (used for temporal split)
      - active_duration : last_step - first_step (0 for single-use accounts)
      - is_fraud_receiver: 1 if ANY incoming transaction was isFraud=1

    EDA insight: 99.5% of fraud receiver accounts receive exactly 1 fraudulent
    transaction, so active_duration is mostly 0 for fraud accounts.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered PaySim data (TRANSFER + CASH_OUT only), from Phase 1.

    Returns
    -------
    pd.DataFrame indexed by account_id (nameDest).
    """
    logger.info("Building receiver account dataset from %d transactions...", len(df))

    agg = df.groupby("nameDest").agg(
        total_volume=("amount", "sum"),
        txn_count=("amount", "count"),
        first_step=("step", "min"),
        last_step=("step", "max"),
        is_fraud_receiver=("isFraud", "max"),  # 1 if ANY incoming tx was fraud
    )
    agg["active_duration"] = agg["last_step"] - agg["first_step"]
    agg.index.name = "account_id"

    n_fraud = int(agg["is_fraud_receiver"].sum())
    logger.info(
        "Receiver dataset: %d accounts, %d fraud receivers (%.4f%%)",
        len(agg), n_fraud, 100.0 * n_fraud / len(agg),
    )
    return agg


def temporal_split(
    accounts_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split receiver accounts into train / val / test by their last_step.

    Accounts are sorted by last_step ascending. The first train_ratio
    fraction forms training, the next val_ratio validation, and the
    remainder test. This preserves temporal order and prevents leakage
    of future account behaviour into earlier splits.

    EDA insight: Fraud rate shifts over time because fraud volume is roughly
    constant while legitimate volume is cyclical, so random splitting would
    cause distribution mismatch between splits.

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Output of build_receiver_dataset().
    train_ratio, val_ratio : float
        Fractional sizes; test = 1 - train_ratio - val_ratio.

    Returns
    -------
    train, val, test : three pd.DataFrames with the same columns.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    sorted_accts = accounts_df.sort_values("last_step").reset_index()

    n = len(sorted_accts)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted_accts.iloc[:n_train].set_index("account_id")
    val   = sorted_accts.iloc[n_train : n_train + n_val].set_index("account_id")
    test  = sorted_accts.iloc[n_train + n_val :].set_index("account_id")

    train_boundary = int(sorted_accts.iloc[n_train - 1]["last_step"])
    val_boundary   = int(sorted_accts.iloc[n_train + n_val - 1]["last_step"])

    logger.info(
        "Temporal split — train: %d accounts (last_step ≤ %d), "
        "val: %d accounts (last_step ≤ %d), test: %d accounts",
        len(train), train_boundary, len(val), val_boundary, len(test),
    )
    logger.info(
        "Fraud receivers per split — train: %d, val: %d, test: %d",
        int(train["is_fraud_receiver"].sum()),
        int(val["is_fraud_receiver"].sum()),
        int(test["is_fraud_receiver"].sum()),
    )
    return train, val, test


def assign_split_labels(
    filtered_df: pd.DataFrame,
    train_accounts: pd.DataFrame,
    val_accounts: pd.DataFrame,
    test_accounts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tag each transaction in filtered_df with its split label.

    The split label is inherited from the nameDest account's split
    assignment (train / val / test). Transactions whose nameDest does
    not appear in any split are dropped with a warning.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Full filtered transaction dataset from Phase 1.
    train_accounts, val_accounts, test_accounts : pd.DataFrame
        Output of temporal_split().

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    split_map = pd.concat([
        pd.Series("train", index=train_accounts.index),
        pd.Series("val",   index=val_accounts.index),
        pd.Series("test",  index=test_accounts.index),
    ])
    split_map.index.name = "nameDest"

    df = filtered_df.copy()
    df["split"] = df["nameDest"].map(split_map)

    n_missing = int(df["split"].isna().sum())
    if n_missing > 0:
        logger.warning(
            "%d transactions have nameDest not in any split — dropping.", n_missing
        )
        df = df.dropna(subset=["split"])

    for s in ("train", "val", "test"):
        mask = df["split"] == s
        n = int(mask.sum())
        n_f = int(df.loc[mask, "isFraud"].sum())
        logger.info(
            "Split '%s': %d transactions, %d fraud (%.4f%%)", s, n, n_f, 100.0 * n_f / n
        )
    return df


def save_split_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    path: str,
) -> None:
    """
    Save account-level split statistics to JSON.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Account-level split DataFrames from temporal_split().
    path : str
        Output path for the JSON file.
    """
    stats = {
        "split_boundaries": {
            "train_last_step_max": int(train["last_step"].max()),
            "val_last_step_max":   int(val["last_step"].max()),
            "test_last_step_max":  int(test["last_step"].max()),
        },
        "account_counts": {
            "train": int(len(train)),
            "val":   int(len(val)),
            "test":  int(len(test)),
            "total": int(len(train) + len(val) + len(test)),
        },
        "fraud_receiver_counts": {
            "train": int(train["is_fraud_receiver"].sum()),
            "val":   int(val["is_fraud_receiver"].sum()),
            "test":  int(test["is_fraud_receiver"].sum()),
        },
        "fraud_receiver_rates": {
            "train": float(train["is_fraud_receiver"].mean()),
            "val":   float(val["is_fraud_receiver"].mean()),
            "test":  float(test["is_fraud_receiver"].mean()),
        },
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Split stats saved to %s", out_path)
