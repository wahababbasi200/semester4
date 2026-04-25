"""
Data validation checks:
  - Schema (required columns present)
  - Missing value thresholds
  - Row count sanity
  - Target distribution
Exits non-zero on failure (CI can catch it).
"""
import sys
import os
import pandas as pd
import argparse


REQUIRED_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
MAX_MISSING_RATIO = 0.99  # flag columns with >99% missing
MIN_ROWS = 1_000
MAX_FRAUD_RATE = 0.60  # sanity upper bound


def validate(path: str) -> bool:
    print(f"[validate] Reading {path}")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    errors = []

    # Schema
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Row count
    if len(df) < MIN_ROWS:
        errors.append(f"Too few rows: {len(df)} < {MIN_ROWS}")

    # Target column
    if "isFraud" in df.columns:
        fraud_rate = df["isFraud"].mean()
        if fraud_rate > MAX_FRAUD_RATE:
            errors.append(f"Fraud rate suspiciously high: {fraud_rate:.2%}")
        print(f"  fraud rate:  {fraud_rate:.2%}")

    # Missing value check
    high_missing = (df.isnull().mean() > MAX_MISSING_RATIO)
    if high_missing.any():
        cols = high_missing[high_missing].index.tolist()
        errors.append(f"Columns with >{MAX_MISSING_RATIO:.0%} missing: {cols[:5]} ...")

    # Missing value report (non-fatal, informational)
    overall_missing = df.isnull().mean().mean()
    print(f"  rows:        {len(df):,}")
    print(f"  columns:     {df.shape[1]}")
    print(f"  avg missing: {overall_missing:.2%}")

    if errors:
        print("\n[validate] FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    print("[validate] PASSED")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to parquet or CSV file to validate")
    args = parser.parse_args()
    ok = validate(args.path)
    sys.exit(0 if ok else 1)
