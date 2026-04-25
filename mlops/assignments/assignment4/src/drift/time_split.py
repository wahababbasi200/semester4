"""
Task 7: Time-based drift simulation.
Split IEEE CIS sample on TransactionDT quartiles:
  Q1+Q2 → training
  Q3     → validation / early-drift detection
  Q4     → "future" distribution (drift introduced here)
"""
import pandas as pd
import numpy as np


def quartile_split(df: pd.DataFrame, time_col: str = "TransactionDT"):
    """Return (q12_train, q3_val, q4_future) DataFrames in time order."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n // 4
    q2_end = n // 2
    q3_end = 3 * n // 4

    q12 = df.iloc[:q2_end]
    q3  = df.iloc[q2_end:q3_end]
    q4  = df.iloc[q3_end:]

    print(f"[drift] Q1+Q2: {len(q12):,}  Q3: {len(q3):,}  Q4: {len(q4):,}")
    print(f"  Q12 fraud: {q12['isFraud'].mean():.2%} | "
          f"Q3: {q3['isFraud'].mean():.2%} | "
          f"Q4: {q4['isFraud'].mean():.2%}")
    return q12, q3, q4


def inject_new_fraud_pattern(df: pd.DataFrame,
                             amt_percentile: float = 95,
                             product_code: str = "W",
                             fraud_injection_rate: float = 0.25) -> pd.DataFrame:
    """
    Inject a synthetic new fraud pattern in Q4 data:
    Transactions with ProductCD == product_code AND high TransactionAmt
    are flipped to fraud at the given rate. Simulates attackers exploiting
    a new vector that the Q1+Q2 model hasn't seen.
    """
    df = df.copy()
    amt_thresh = df["TransactionAmt"].quantile(amt_percentile / 100)
    mask = (df["ProductCD"] == product_code) & (df["TransactionAmt"] >= amt_thresh)
    rng = np.random.default_rng(99)
    flip_idx = df[mask].index[rng.random(mask.sum()) < fraud_injection_rate]
    df.loc[flip_idx, "isFraud"] = 1
    n_injected = len(flip_idx)
    print(f"[drift] Injected {n_injected} new-pattern fraud cases "
          f"(ProductCD={product_code}, amt>={amt_thresh:.0f})")
    return df
