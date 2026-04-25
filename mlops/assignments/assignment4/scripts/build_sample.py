"""
Build working sample: ALL fraud rows + 50K non-fraud rows from IEEE CIS dataset.
Run this on your local machine or EC2 BEFORE submitting the pipeline.

Usage:
  python scripts/build_sample.py \
    --tx   /path/to/train_transaction.csv \
    --id   /path/to/train_identity.csv \
    --out  E:/Study/semester4/mlops/assignments/assignment4/data/sample_ieee_70k.parquet

On EC2 example:
  python scripts/build_sample.py \
    --tx /mnt/ml-data/assignment4/raw/train_transaction.csv \
    --id /mnt/ml-data/assignment4/raw/train_identity.csv \
    --out /mnt/ml-data/assignment4/data/sample_ieee_70k.parquet
"""
import argparse
import os
import pandas as pd
import numpy as np


def build_sample(tx_path: str, id_path: str, out_path: str,
                 nonfraud_n: int = 50_000, seed: int = 42):
    print(f"[1/4] Reading train_transaction.csv  ({tx_path})")
    tx = pd.read_csv(tx_path)
    print(f"      shape: {tx.shape}  |  fraud rate: {tx['isFraud'].mean():.4%}")

    print(f"[2/4] Reading train_identity.csv  ({id_path})")
    identity = pd.read_csv(id_path)
    print(f"      shape: {identity.shape}")

    print("[3/4] Left-joining on TransactionID (keeping all transactions)")
    df = tx.merge(identity, on="TransactionID", how="left")
    print(f"      merged shape: {df.shape}")

    # Sort by time to preserve temporal ordering (critical for drift simulation)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0]

    rng = np.random.default_rng(seed)
    non_fraud_idx = rng.choice(len(non_fraud), size=min(nonfraud_n, len(non_fraud)), replace=False)
    non_fraud_sample = non_fraud.iloc[non_fraud_idx]

    sample = pd.concat([fraud, non_fraud_sample], ignore_index=True)
    # Re-sort by time after concat
    sample = sample.sort_values("TransactionDT").reset_index(drop=True)

    print(f"[4/4] Sample:  {len(sample):,} rows  |  "
          f"fraud: {sample['isFraud'].sum():,} ({sample['isFraud'].mean():.2%})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sample.to_parquet(out_path, index=False)
    print(f"      Saved → {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", required=True, help="Path to train_transaction.csv")
    parser.add_argument("--id", required=True, help="Path to train_identity.csv")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_ieee_70k.parquet"), help="Output parquet path")
    parser.add_argument("--nonfraud-n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_sample(args.tx, args.id, args.out, args.nonfraud_n, args.seed)
