"""
load_data.py
------------
PaySim dataset loading, schema validation, and basic quality checks.
Single entry point for all downstream modules: load_paysim().
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────

PAYSIM_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "nameDest",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

PAYSIM_DTYPES = {
    "step":            "int64",
    "type":            "object",
    "amount":          "float64",
    "nameOrig":        "object",
    "nameDest":        "object",
    "oldbalanceOrg":   "float64",
    "newbalanceOrig":  "float64",
    "oldbalanceDest":  "float64",
    "newbalanceDest":  "float64",
    "isFraud":         "int64",
    "isFlaggedFraud":  "int64",
}

FRAUD_TRANSACTION_TYPES = {"TRANSFER", "CASH_OUT"}
ALL_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


# ── Public API ──────────────────────────────────────────────────────────────────

def load_paysim(
    path: str | Path,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the PaySim CSV dataset.

    Parameters
    ----------
    path : str or Path
        Path to the PaySim CSV file.
    validate : bool
        If True, run schema validation and quality checks.
    verbose : bool
        If True, print a summary report after loading.

    Returns
    -------
    pd.DataFrame
        Loaded and optionally validated PaySim dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at: {path}\n"
            "Download from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1"
        )

    logger.info(f"Loading PaySim dataset from: {path}")
    df = pd.read_csv(path, dtype=PAYSIM_DTYPES, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    if validate:
        _validate_schema(df)
        _check_data_quality(df)

    if verbose:
        print_dataset_summary(df)

    return df


def filter_fraud_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Filter to only TRANSFER and CASH_OUT transactions (the only types with fraud).

    Returns
    -------
    filtered_df : pd.DataFrame
    stats : dict  — counts before/after, fraud rate before/after
    """
    before_rows = len(df)
    before_fraud = df["isFraud"].sum()
    before_accounts = df["nameDest"].nunique()

    filtered = df[df["type"].isin(FRAUD_TRANSACTION_TYPES)].copy()
    filtered = filtered.reset_index(drop=True)

    after_rows = len(filtered)
    after_fraud = filtered["isFraud"].sum()
    after_accounts = filtered["nameDest"].nunique()

    removed_rows = before_rows - after_rows
    removed_pct = 100 * removed_rows / before_rows

    stats = {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": removed_rows,
        "removed_pct": round(removed_pct, 2),
        "before_fraud_count": int(before_fraud),
        "after_fraud_count": int(after_fraud),
        "before_fraud_rate_pct": round(100 * before_fraud / before_rows, 4),
        "after_fraud_rate_pct": round(100 * after_fraud / after_rows, 4),
        "before_unique_dest_accounts": int(before_accounts),
        "after_unique_dest_accounts": int(after_accounts),
    }

    logger.info(
        f"Filtered to {FRAUD_TRANSACTION_TYPES}: "
        f"{after_rows:,} rows ({removed_pct:.1f}% removed). "
        f"Fraud rate: {stats['after_fraud_rate_pct']:.4f}%"
    )
    return filtered, stats


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame) -> None:
    """Check that all expected columns are present with correct types."""
    missing = set(PAYSIM_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    extra = set(df.columns) - set(PAYSIM_COLUMNS)
    if extra:
        logger.warning(f"Unexpected extra columns (will ignore): {extra}")

    for col, expected_dtype in PAYSIM_DTYPES.items():
        actual = str(df[col].dtype)
        if not actual.startswith(expected_dtype.rstrip("0123456789")):
            logger.warning(f"Column '{col}': expected {expected_dtype}, got {actual}")

    logger.info("Schema validation passed.")


def _check_data_quality(df: pd.DataFrame) -> None:
    """Run basic quality checks and log findings."""
    # Null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found.")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")

    # Transaction types
    observed_types = set(df["type"].unique())
    unexpected_types = observed_types - ALL_TRANSACTION_TYPES
    if unexpected_types:
        logger.warning(f"Unexpected transaction types: {unexpected_types}")

    # Fraud only in expected types
    fraud_types = set(df[df["isFraud"] == 1]["type"].unique())
    if fraud_types != FRAUD_TRANSACTION_TYPES:
        logger.warning(
            f"Fraud found in unexpected types: {fraud_types}. "
            f"Expected only: {FRAUD_TRANSACTION_TYPES}"
        )

    # isFraud binary check
    invalid_fraud_vals = set(df["isFraud"].unique()) - {0, 1}
    if invalid_fraud_vals:
        raise ValueError(f"isFraud contains non-binary values: {invalid_fraud_vals}")

    # Amount sanity
    neg_amounts = (df["amount"] < 0).sum()
    if neg_amounts > 0:
        logger.warning(f"Found {neg_amounts} negative transaction amounts.")

    logger.info("Data quality checks complete.")


# ── Summary reporting ──────────────────────────────────────────────────────────

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a comprehensive human-readable summary of the dataset."""
    n = len(df)
    n_fraud = int(df["isFraud"].sum())
    fraud_rate = 100 * n_fraud / n

    print("\n" + "=" * 60)
    print("PAYSIM DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows:          {n:>12,}")
    print(f"Total columns:       {df.shape[1]:>12}")
    print(f"Memory usage:        {df.memory_usage(deep=True).sum() / 1e6:>11.1f} MB")
    print()
    print(f"Fraud transactions:  {n_fraud:>12,}  ({fraud_rate:.4f}%)")
    print(f"Legit transactions:  {n - n_fraud:>12,}  ({100 - fraud_rate:.4f}%)")
    print()
    print("Transaction type distribution:")
    type_counts = df["type"].value_counts()
    for t, cnt in type_counts.items():
        fraud_in_type = int(df[df["type"] == t]["isFraud"].sum())
        print(f"  {t:<12} {cnt:>9,}  ({100*cnt/n:.2f}%)  fraud: {fraud_in_type:,}")
    print()
    print(f"Unique sender accounts (nameOrig):   {df['nameOrig'].nunique():>10,}")
    print(f"Unique receiver accounts (nameDest): {df['nameDest'].nunique():>10,}")
    print(f"Step range: {df['step'].min()} – {df['step'].max()} ({df['step'].nunique()} unique steps)")
    print()
    print("Amount statistics:")
    amt_stats = df["amount"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    for stat, val in amt_stats.items():
        print(f"  {stat:<8} {val:>14,.2f}")
    print()
    # Null summary
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print("Null values:")
        for col, cnt in nulls[nulls > 0].items():
            print(f"  {col}: {cnt:,}")
    else:
        print("Null values: None")
    print("=" * 60 + "\n")


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Return a structured data quality report as a dict (for saving/logging).
    """
    n = len(df)
    n_fraud = int(df["isFraud"].sum())

    type_stats = {}
    for t in df["type"].unique():
        mask = df["type"] == t
        type_stats[t] = {
            "count": int(mask.sum()),
            "pct": round(100 * mask.sum() / n, 4),
            "fraud_count": int(df.loc[mask, "isFraud"].sum()),
        }

    return {
        "total_rows": n,
        "total_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "fraud_count": n_fraud,
        "fraud_rate_pct": round(100 * n_fraud / n, 6),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts": df.isnull().sum().to_dict(),
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
        "unique_steps": int(df["step"].nunique()),
        "unique_orig_accounts": int(df["nameOrig"].nunique()),
        "unique_dest_accounts": int(df["nameDest"].nunique()),
        "transaction_type_stats": type_stats,
        "amount_stats": {
            k: round(v, 4)
            for k, v in df["amount"].describe(
                percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
            ).to_dict().items()
        },
    }
