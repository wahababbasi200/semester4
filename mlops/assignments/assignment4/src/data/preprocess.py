"""
IEEE CIS preprocessing pipeline:
  1. Drop ID column
  2. Missing-indicator features for heavy-NaN columns (>30%)
  3. Binary M-columns: T→1, F→0, NaN→-1
  4. Target encoding for high-cardinality categoricals
  5. Label encoding remaining string columns
  6. Median imputation for remaining numerics
  7. Train/val/test split preserving time order (TransactionDT)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

M_COLS = [f"M{i}" for i in range(1, 10)]
HIGH_CARD_COLS = ["card1", "card2", "card5", "addr1", "P_emaildomain",
                  "R_emaildomain", "DeviceInfo"]
MISSING_IND_THRESHOLD = 0.30
FRAUD_COL = "isFraud"
TIME_COL = "TransactionDT"
DROP_COLS = ["TransactionID"]


def _target_encode(train: pd.DataFrame, cols: list, target: str,
                   smoothing: float = 20.0) -> dict:
    """Return mapping dict {col: {value: encoded_mean}}."""
    global_mean = train[target].mean()
    encoders = {}
    for col in cols:
        if col not in train.columns:
            continue
        stats = train.groupby(col)[target].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
                   (stats["count"] + smoothing)
        encoders[col] = smoothed.to_dict()
    return encoders


def preprocess(df: pd.DataFrame, encoders: dict = None,
               medians: dict = None, fit: bool = True):
    """
    If fit=True: derive encoders + medians from df (training mode).
    If fit=False: apply provided encoders + medians (inference mode).
    Returns (X, y, encoders, medians).
    """
    df = df.copy()

    # Drop non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    if TIME_COL in df.columns:
        drop.append(TIME_COL)
    df = df.drop(columns=drop)

    y = df.pop(FRAUD_COL) if FRAUD_COL in df.columns else None

    # Missing indicators — determined from training split, reused in inference
    if fit:
        missing_rates = df.isnull().mean()
        heavy_null_cols = [
            col for col in missing_rates[missing_rates > MISSING_IND_THRESHOLD].index
            if pd.api.types.is_numeric_dtype(df[col])
        ]
    else:
        heavy_null_cols = (medians or {}).get("__missing_cols__", [])
    for col in heavy_null_cols:
        df[f"{col}_missing"] = df[col].isnull().astype(np.int8)

    # M-columns: T→1, F→0, NaN→-1
    for col in M_COLS:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)

    # Target encoding for high-cardinality cols
    if fit:
        encoders = _target_encode(
            pd.concat([df, y.rename(FRAUD_COL)], axis=1),
            HIGH_CARD_COLS, FRAUD_COL
        )
    global_fraud_mean = y.mean() if y is not None else 0.05
    for col, mapping in (encoders or {}).items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(global_fraud_mean)

    # Label-encode remaining string / object columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Numeric median imputation
    if fit:
        medians = df.select_dtypes(include=[np.number]).median().to_dict()
        medians["__missing_cols__"] = heavy_null_cols
    for col, val in (medians or {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Fill any residual NaN with 0
    df = df.fillna(0)

    return df, y, encoders, medians


def time_aware_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological split — no shuffling — respects TransactionDT order."""
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - val_frac - test_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def fit_and_save(df: pd.DataFrame, out_dir: str):
    """Full preprocessing pipeline: fit + save artifacts."""
    os.makedirs(out_dir, exist_ok=True)

    # Chronological split before any fitting (prevent data leakage)
    train_raw, val_raw, test_raw = time_aware_split(df)

    X_train, y_train, encoders, medians = preprocess(train_raw, fit=True)
    X_val, y_val, _, _ = preprocess(val_raw, encoders=encoders, medians=medians, fit=False)
    X_test, y_test, _, _ = preprocess(test_raw, encoders=encoders, medians=medians, fit=False)

    joblib.dump({"encoders": encoders, "medians": medians}, os.path.join(out_dir, "preprocessor.pkl"))

    train_df = X_train.copy(); train_df[FRAUD_COL] = y_train.values
    val_df = X_val.copy();   val_df[FRAUD_COL] = y_val.values
    test_df = X_test.copy(); test_df[FRAUD_COL] = y_test.values

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    print(f"[preprocess] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} features={X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, X_test, y_test
