"""Load the IEEE CIS working sample from disk."""
import os
import pandas as pd


def load_sample(path: str = None) -> pd.DataFrame:
    if path is None:
        from src.config import SAMPLE_FILE
        path = SAMPLE_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample file not found: {path}\n"
            "Run:  python scripts/build_sample.py --tx ... --id ... --out ..."
        )
    df = pd.read_parquet(path)
    print(f"[ingest] Loaded {len(df):,} rows | fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean():.2%})")
    return df
