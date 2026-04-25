"""Tests for the preprocessing pipeline."""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocess import preprocess, time_aware_split


def _make_sample(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID":  np.arange(n),
        "isFraud":        (rng.random(n) < 0.15).astype(int),
        "TransactionDT":  np.sort(rng.integers(1, 10_000_000, n)),
        "TransactionAmt": rng.exponential(100, n),
        "ProductCD":      rng.choice(["W", "H", "C"], n),
        "card1":          rng.integers(1000, 18000, n).astype(float),
        "card2":          rng.integers(100, 600, n).astype(float),
        "card5":          rng.integers(100, 600, n).astype(float),
        "addr1":          rng.integers(100, 500, n).astype(float),
        "P_emaildomain":  rng.choice(["gmail.com", "yahoo.com", "outlook.com"], n),
        "R_emaildomain":  rng.choice(["gmail.com", "yahoo.com", None], n),
        "DeviceInfo":     rng.choice(["Chrome", "Safari", None], n),
        "M1":             rng.choice(["T", "F", None], n),
        "M2":             rng.choice(["T", "F", None], n),
        "V1":             np.where(rng.random(n) < 0.4, np.nan, rng.random(n)),
        "V2":             np.where(rng.random(n) < 0.6, np.nan, rng.random(n)),
    })


def test_preprocess_no_nulls_after_transform():
    df = _make_sample()
    X, y, enc, med = preprocess(df, fit=True)
    assert X.isnull().sum().sum() == 0, "There should be no nulls after preprocessing"


def test_preprocess_returns_correct_shapes():
    df = _make_sample(n=400)
    X, y, _, _ = preprocess(df, fit=True)
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert X.shape[1] > 10   # should have added missing-indicator features


def test_preprocess_m_columns_encoded():
    df = _make_sample()
    X, y, _, _ = preprocess(df, fit=True)
    for col in ["M1", "M2"]:
        if col in X.columns:
            assert X[col].isin([-1, 0, 1]).all(), f"{col} should be -1/0/1"


def test_preprocess_target_encoding_applied():
    df = _make_sample()
    X, y, enc, _ = preprocess(df, fit=True)
    # card1 should be numeric after target encoding
    if "card1" in X.columns:
        assert pd.api.types.is_numeric_dtype(X["card1"]), "card1 should be numeric after encoding"


def test_time_aware_split_preserves_order():
    df = _make_sample(n=300)
    train, val, test = time_aware_split(df)
    total = len(train) + len(val) + len(test)
    assert total == len(df), "Split should cover all rows"
    # val TransactionDT should all be >= train TransactionDT max (time-ordered)
    if "TransactionDT" in df.columns:
        assert train["TransactionDT"].max() <= val["TransactionDT"].min()


def test_inference_mode_uses_train_encoders():
    df = _make_sample(n=400)
    train = df.iloc[:300]
    test  = df.iloc[300:]
    X_train, y_train, enc, med = preprocess(train, fit=True)
    X_test,  y_test,  _,   _   = preprocess(test, encoders=enc, medians=med, fit=False)
    # Same feature count
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.isnull().sum().sum() == 0
