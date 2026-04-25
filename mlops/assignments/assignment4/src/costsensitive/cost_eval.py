"""
Cost-Sensitive Learning — Task 4.

Business impact model:
  - True Positive  (caught fraud):   +$200 (investigation cost offset)
  - False Negative (missed fraud):   -$500 (full fraud loss)
  - False Positive (false alarm):    -$50  (analyst review cost)
  - True Negative  (correct block):  $0

Compare: standard XGB vs cost-sensitive XGB.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             classification_report, precision_recall_curve,
                             average_precision_score)
import matplotlib.pyplot as plt
import joblib
import os


FN_COST = 500
FP_COST = 50
TP_GAIN = 200


def business_utility(y_true: np.ndarray, proba: np.ndarray,
                     threshold: float = 0.5) -> dict:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, cm[0, 1], cm[1, 1])

    utility = TP_GAIN * tp - FN_COST * fn - FP_COST * fp
    total_fraud_loss = FN_COST * (fn + tp)       # total fraud in test set
    avoided_loss = TP_GAIN * tp                  # caught by model
    false_alarm_cost = FP_COST * fp

    return {
        "threshold": round(threshold, 3),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "utility_$": int(utility),
        "fraud_loss_$": int(FN_COST * fn),
        "false_alarm_cost_$": int(false_alarm_cost),
        "avoided_loss_$": int(avoided_loss),
        "total_potential_fraud_$": int(total_fraud_loss),
    }


def optimal_threshold(y_true: np.ndarray, proba: np.ndarray,
                      thresholds: np.ndarray = None) -> float:
    """Find threshold that maximises business utility."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.1, 0.9, 50)
    best_t, best_u = 0.5, -np.inf
    for t in thresholds:
        u = business_utility(y_true, proba, t)["utility_$"]
        if u > best_u:
            best_u, best_t = u, t
    return best_t


def compare_standard_vs_costsensitive(X_train, y_train, X_test, y_test,
                                      out_dir: str = "outputs/costsensitive"):
    os.makedirs(out_dir, exist_ok=True)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    # ── Standard XGB ────────────────────────────────────────────────────────
    print("[cost] Training standard XGB...")
    std_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=1.0, tree_method="hist",
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    std_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    std_proba = std_model.predict_proba(X_test)[:, 1]
    std_threshold = 0.5
    std_results = business_utility(y_test, std_proba, std_threshold)
    std_results["model"] = "standard_xgb"
    std_results["auc_roc"] = round(roc_auc_score(y_test, std_proba), 4)
    r = classification_report(y_test, (std_proba >= std_threshold).astype(int), output_dict=True)
    std_results["recall"] = round(r["1"]["recall"], 4)
    std_results["precision"] = round(r["1"]["precision"], 4)

    # ── Cost-Sensitive XGB ──────────────────────────────────────────────────
    print("[cost] Training cost-sensitive XGB (higher penalty on FN)...")
    cs_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=max(neg / pos, 10.0),  # strong emphasis on recall
        tree_method="hist", eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    cs_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    cs_proba = cs_model.predict_proba(X_test)[:, 1]
    cs_threshold = optimal_threshold(y_test, cs_proba)
    print(f"  Optimal cost threshold: {cs_threshold:.3f}")
    cs_results = business_utility(y_test, cs_proba, cs_threshold)
    cs_results["model"] = "cost_sensitive_xgb"
    cs_results["auc_roc"] = round(roc_auc_score(y_test, cs_proba), 4)
    r2 = classification_report(y_test, (cs_proba >= cs_threshold).astype(int), output_dict=True)
    cs_results["recall"] = round(r2["1"]["recall"], 4)
    cs_results["precision"] = round(r2["1"]["precision"], 4)

    # ── Comparison table ─────────────────────────────────────────────────────
    df = pd.DataFrame([std_results, cs_results])
    print("\n=== Cost-Sensitive Learning Comparison ===")
    print(df[["model", "auc_roc", "recall", "precision",
              "TP", "FP", "FN", "utility_$",
              "fraud_loss_$", "false_alarm_cost_$"]].to_string(index=False))
    df.to_csv(os.path.join(out_dir, "cost_comparison.csv"), index=False)

    # ── Utility vs threshold curve ───────────────────────────────────────────
    thresholds = np.linspace(0.05, 0.95, 80)
    std_utils = [business_utility(y_test, std_proba, t)["utility_$"] for t in thresholds]
    cs_utils  = [business_utility(y_test, cs_proba,  t)["utility_$"] for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, std_utils, label="Standard XGB", color="steelblue")
    plt.plot(thresholds, cs_utils,  label="Cost-Sensitive XGB", color="darkorange")
    plt.axvline(std_threshold, ls="--", color="steelblue", alpha=0.5)
    plt.axvline(cs_threshold,  ls="--", color="darkorange", alpha=0.5)
    plt.xlabel("Decision Threshold"); plt.ylabel("Business Utility ($)")
    plt.title("Business Utility vs Decision Threshold")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "utility_vs_threshold.png"), dpi=150)
    plt.close()

    joblib.dump(cs_model, os.path.join(out_dir, "cost_sensitive_xgb.pkl"))
    print(f"[cost] Saved → {out_dir}/")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.ingest import load_sample
    from src.data.preprocess import preprocess, time_aware_split

    df = load_sample()
    train_raw, _, test_raw = time_aware_split(df)
    X_train, y_train, encoders, medians = preprocess(train_raw, fit=True)
    X_test, y_test, _, _ = preprocess(test_raw, encoders=encoders, medians=medians, fit=False)

    compare_standard_vs_costsensitive(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        out_dir="outputs/costsensitive"
    )
