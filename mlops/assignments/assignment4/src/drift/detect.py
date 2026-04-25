"""
Data drift detection using:
  - PSI  (Population Stability Index) per feature
  - KS   (Kolmogorov-Smirnov) test per feature
  - Feature importance comparison between Q12 and Q4 models
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import os


def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI for one feature. <0.1 = stable, 0.1-0.2 = minor, >0.2 = major drift."""
    eps = 1e-6
    combined = np.concatenate([expected, actual])
    _, edges = np.histogram(combined, bins=bins)
    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)

    exp_pct = (exp_hist / len(expected) + eps)
    act_pct = (act_hist / len(actual) + eps)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_drift(train_df: pd.DataFrame, future_df: pd.DataFrame,
                  top_n: int = 20, out_dir: str = "outputs/drift") -> pd.DataFrame:
    """Compute PSI and KS for top_n numeric features."""
    os.makedirs(out_dir, exist_ok=True)

    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["isFraud", "TransactionDT", "TransactionID"]
    num_cols = [c for c in num_cols if c not in exclude][:top_n]

    rows = []
    for col in num_cols:
        if col not in future_df.columns:
            continue
        exp = train_df[col].dropna().values
        act = future_df[col].dropna().values
        if len(exp) < 5 or len(act) < 5:
            continue
        psi = _psi_single(exp, act)
        ks_stat, ks_p = ks_2samp(exp, act)
        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 6),
            "drift_level": "major" if psi > 0.2 else ("minor" if psi > 0.1 else "stable"),
        })

    df_drift = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df_drift.to_csv(os.path.join(out_dir, "drift_metrics.csv"), index=False)

    print(f"\n[drift] Top drifted features (PSI):")
    print(df_drift[["feature", "psi", "ks_stat", "drift_level"]].head(10).to_string(index=False))

    # PSI bar plot
    top = df_drift.head(15)
    colors = ["red" if p > 0.2 else ("orange" if p > 0.1 else "green") for p in top["psi"]]
    plt.figure(figsize=(10, 5))
    plt.bar(top["feature"], top["psi"], color=colors)
    plt.axhline(0.1, ls="--", color="orange", label="Minor drift (0.1)")
    plt.axhline(0.2, ls="--", color="red",    label="Major drift (0.2)")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Feature"); plt.ylabel("PSI")
    plt.title("Feature Distribution Drift (PSI): Q1+Q2 → Q4")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psi_bar.png"), dpi=150)
    plt.close()

    return df_drift


def compare_feature_importance(fi_train: dict, fi_future: dict,
                               top_n: int = 10, out_dir: str = "outputs/drift"):
    """Plot feature importance shift between two XGB models."""
    os.makedirs(out_dir, exist_ok=True)
    all_feats = list(set(list(fi_train.keys()) + list(fi_future.keys())))
    df_fi = pd.DataFrame({
        "feature":  all_feats,
        "q12_imp":  [fi_train.get(f, 0) for f in all_feats],
        "q4_imp":   [fi_future.get(f, 0) for f in all_feats],
    })
    df_fi["delta"] = df_fi["q4_imp"] - df_fi["q12_imp"]
    df_fi = df_fi.sort_values("delta", key=abs, ascending=False).head(top_n)

    x = np.arange(len(df_fi))
    w = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, df_fi["q12_imp"], w, label="Q1+Q2 model")
    plt.bar(x + w/2, df_fi["q4_imp"],  w, label="Q4 model")
    plt.xticks(x, df_fi["feature"], rotation=45, ha="right")
    plt.ylabel("Feature Importance (gain)")
    plt.title("Feature Importance Shift: Q1+Q2 vs Q4")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_shift.png"), dpi=150)
    plt.close()

    df_fi.to_csv(os.path.join(out_dir, "importance_shift.csv"), index=False)
    print(f"[drift] Saved importance shift → {out_dir}/importance_shift.png")
    return df_fi
