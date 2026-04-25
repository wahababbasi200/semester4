"""
Compare two imbalance-handling strategies on XGBoost:
  1. class_weight  — scale_pos_weight = neg_count / pos_count
  2. SMOTE         — synthetic minority oversampling on training fold only

Outputs a comparison table + confusion matrices.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             average_precision_score)
import xgboost as xgb
import os


def _eval(model, X_test, y_test, name: str, threshold: float = 0.5):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    return {
        "strategy": name,
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"],    4),
        "f1":        round(report["1"]["f1-score"],  4),
        "auc_roc":   round(auc, 4),
        "pr_auc":    round(pr_auc, 4),
    }, proba


def compare_strategies(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray,  y_test: np.ndarray,
                       out_dir: str = "outputs") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    print(f"[imbalance] neg={neg:,}  pos={pos:,}  scale_pos_weight={spw:.2f}")

    results = []
    probas = {}

    # ── Strategy 1: class_weight (scale_pos_weight) ───────────────────────────
    print("[imbalance] Training with class_weight strategy...")
    m1 = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, tree_method="hist",
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    m1.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    r1, p1 = _eval(m1, X_test, y_test, "class_weight")
    results.append(r1); probas["class_weight"] = p1

    # ── Strategy 2: SMOTE ─────────────────────────────────────────────────────
    print("[imbalance] Training with SMOTE strategy...")
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_sm):,} samples (pos={y_sm.sum():,})")

    m2 = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1
    )
    m2.fit(X_sm, y_sm, eval_set=[(X_test, y_test)], verbose=False)
    r2, p2 = _eval(m2, X_test, y_test, "SMOTE")
    results.append(r2); probas["SMOTE"] = p2

    df_results = pd.DataFrame(results)
    print("\n=== Imbalance Strategy Comparison ===")
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(out_dir, "imbalance_comparison.csv"), index=False)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (strat, proba) in zip(axes, probas.items()):
        y_pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Strategy: {strat}")
    plt.suptitle("Imbalance Handling — Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imbalance_confusion_matrices.png"), dpi=150)
    plt.close()
    print(f"[imbalance] Saved → {out_dir}/imbalance_comparison.csv + .png")

    return df_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.ingest import load_sample
    from src.data.preprocess import preprocess, time_aware_split

    df = load_sample()
    train_raw, _, test_raw = time_aware_split(df)
    X_train, y_train, encoders, medians = preprocess(train_raw, fit=True)
    X_test, y_test, _, _ = preprocess(test_raw, encoders=encoders, medians=medians, fit=False)

    compare_strategies(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        out_dir="outputs/imbalance"
    )
