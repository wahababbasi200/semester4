"""
Local eval script — replicates full pipeline preprocessing + feature engineering,
then scores the deployed model against the test split (last 15% chronologically).
Usage: python eval_local.py [--variant xgb_class_weight]
"""
import argparse, os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder

M_COLS         = [f"M{i}" for i in range(1, 10)]
HIGH_CARD      = ["card1","card2","card5","addr1","P_emaildomain",
                  "R_emaildomain","DeviceInfo"]
FRAUD_COL      = "isFraud"
DROP_COLS      = ["TransactionID","TransactionDT"]
MISSING_THRESH = 0.30


def apply_preprocess(df_raw, encoders, medians):
    """Mirrors preprocess_op.apply_transforms (test split — no fit)."""
    d = df_raw.copy()
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns])
    y = d.pop(FRAUD_COL)

    miss_rates = d.isnull().mean()
    inds = {
        f"{col}_missing": d[col].isnull().astype(np.int8)
        for col in miss_rates[miss_rates > MISSING_THRESH].index
        if pd.api.types.is_numeric_dtype(d[col])
    }
    if inds:
        d = pd.concat([d, pd.DataFrame(inds, index=d.index)], axis=1)

    for col in M_COLS:
        if col in d.columns:
            d[col] = d[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)

    gm = y.mean()
    for col, mapping in encoders.items():
        if col in d.columns:
            d[col] = d[col].map(mapping).fillna(gm)

    for col in d.select_dtypes(include=["object","category","string"]).columns:
        d[col] = LabelEncoder().fit_transform(d[col].astype(str))

    for col, val in medians.items():
        if col in d.columns:
            d[col] = d[col].fillna(val)
    d = d.fillna(0)
    return d, y


def apply_feateng(d):
    """Mirrors feateng_op.add_features."""
    if "TransactionAmt" in d.columns:
        d["TransactionAmt_log"]   = np.log1p(d["TransactionAmt"])
        d["TransactionAmt_cents"] = (d["TransactionAmt"] * 100 % 100).astype(int)
        d["TransactionAmt_bin"]   = pd.cut(
            d["TransactionAmt"], bins=[0, 10, 100, 1000, np.inf],
            labels=[0, 1, 2, 3], include_lowest=True
        ).astype(float)
    c_cols = [c for c in d.columns if c.startswith("C") and c[1:].isdigit()]
    if c_cols:
        d["C_sum"] = d[c_cols].sum(axis=1)
    if "D1" in d.columns and "D2" in d.columns:
        d["D1_D2_ratio"] = (d["D1"] / d["D2"].replace(0, np.nan)).fillna(0)
    return d


def main(variant):
    print(f"\n=== Evaluating: {variant} ===")

    payload  = joblib.load("models/best_model.pkl")
    prep     = joblib.load("models/preprocessor.pkl")
    encoders = prep["encoders"]
    medians  = prep["medians"]

    df       = pd.read_parquet("data/processed/sample_ieee_70k.parquet")
    df       = df.sort_values("TransactionDT").reset_index(drop=True)
    test_raw = df.iloc[int(len(df) * 0.85):]

    X, y  = apply_preprocess(test_raw, encoders, medians)
    X     = apply_feateng(X)
    X_arr = X.values.astype(np.float32)

    if isinstance(payload, dict) and "selector" in payload:
        X_arr = payload["selector"].transform(X_arr)
        model = payload["model"]
    else:
        model = payload

    proba = model.predict_proba(X_arr)[:, 1]
    y_hat = (proba >= 0.5).astype(int)

    auc    = round(roc_auc_score(y, proba), 4)
    prauc  = round(average_precision_score(y, proba), 4)
    report = classification_report(y, y_hat, target_names=["Legit","Fraud"],
                                   output_dict=True)
    prec   = round(report["Fraud"]["precision"], 4)
    rec    = round(report["Fraud"]["recall"], 4)
    f1     = round(report["Fraud"]["f1-score"], 4)

    print(f"[eval] AUC-ROC={auc}  PR-AUC={prauc}  "
          f"Precision={prec}  Recall={rec}  F1={f1}")
    print(classification_report(y, y_hat, target_names=["Legit","Fraud"]))

    out_dir = f"outputs/{variant}"
    os.makedirs(out_dir, exist_ok=True)

    ConfusionMatrixDisplay(confusion_matrix(y, y_hat),
        display_labels=["Legit","Fraud"]).plot(colorbar=False)
    plt.title(f"{variant} — Confusion Matrix"); plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150); plt.close()

    p, r, _ = precision_recall_curve(y, proba)
    plt.figure(); plt.plot(r, p, color="steelblue", label=f"PR-AUC={prauc}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.title(f"{variant} — PR Curve"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve.png", dpi=150); plt.close()

    print(f"Saved → {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="xgb_class_weight")
    main(ap.parse_args().variant)
