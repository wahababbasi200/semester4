"""Tests for data schema validation (runs in CI)."""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.validate import validate


def _make_df(rows=2000, fraud_rate=0.1):
    """Create a minimal IEEE CIS-like DataFrame for testing."""
    rng = np.random.default_rng(0)
    n = rows
    return pd.DataFrame({
        "TransactionID":  np.arange(n),
        "isFraud":        (rng.random(n) < fraud_rate).astype(int),
        "TransactionDT":  rng.integers(1, 1_000_000, n),
        "TransactionAmt": rng.exponential(100, n),
        "ProductCD":      rng.choice(["W", "H", "C", "S", "R"], n),
        "card1":          rng.integers(1000, 18000, n).astype(float),
        "M1":             rng.choice(["T", "F", None], n),
        "V1":             np.where(rng.random(n) < 0.3, np.nan, rng.random(n)),
    })


def test_validate_passes_on_valid_data(tmp_path):
    df = _make_df()
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path)
    assert validate(path) is True


def test_validate_fails_on_missing_required_column(tmp_path):
    df = _make_df()
    df = df.drop(columns=["isFraud"])
    path = str(tmp_path / "bad.parquet")
    df.to_parquet(path)
    assert validate(path) is False


def test_validate_fails_on_too_few_rows(tmp_path):
    df = _make_df(rows=50)
    path = str(tmp_path / "tiny.parquet")
    df.to_parquet(path)
    assert validate(path) is False


def test_validate_fails_on_all_null_column(tmp_path):
    df = _make_df()
    df["V1"] = np.nan   # 100% null — should fail
    path = str(tmp_path / "allnull.parquet")
    df.to_parquet(path)
    assert validate(path) is False
