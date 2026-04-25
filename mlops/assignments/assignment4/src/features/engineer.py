"""
Feature engineering for IEEE CIS Fraud Detection.
Applied AFTER basic preprocessing (before model training).
Adds domain-informed aggregation features.
"""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Input df should still have raw columns."""
    df = df.copy()

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_cents"] = (df["TransactionAmt"] * 100 % 100).astype(int)
        # Bins: small (<10), medium (10-100), large (100-1000), huge (>1000)
        df["TransactionAmt_bin"] = pd.cut(
            df["TransactionAmt"],
            bins=[0, 10, 100, 1000, np.inf],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(float)

    if "TransactionDT" in df.columns:
        # Seconds since epoch offset → extract time-of-day and day-of-week signals
        START_DT = 1_580_000_000  # approximate reference (arbitrary, consistent)
        seconds = df["TransactionDT"] - START_DT
        df["hour_of_day"] = (seconds // 3600) % 24
        df["day_of_week"] = (seconds // 86400) % 7
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(np.int8)

    # card1 + addr1 combined aggregation signal
    if "card1" in df.columns and "addr1" in df.columns:
        df["card1_addr1"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        # Will be target-encoded in preprocess step; here just create the combo column

    # D-column deltas (timedelta features) — ratio of D1/D2 as signal
    if "D1" in df.columns and "D2" in df.columns:
        df["D1_D2_ratio"] = (df["D1"] / (df["D2"] + 1)).fillna(0)

    # C-column sum (count features sum as activity proxy)
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    if c_cols:
        df["C_sum"] = df[c_cols].sum(axis=1)

    return df


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """Return list of feature columns (exclude target + time + ID cols)."""
    exclude = set(exclude or [])
    exclude.update(["isFraud", "TransactionID", "TransactionDT"])
    return [c for c in df.columns if c not in exclude]
