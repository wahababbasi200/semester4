"""
Hybrid model: Random Forest + Feature Selection.
Step 1: Train RF to get feature importances.
Step 2: SelectFromModel prunes to top features.
Step 3: Retrain RF on pruned feature set.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import os


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       n_estimators: int = 300,
                       threshold: str = "median",
                       random_state: int = 42,
                       out_dir: str = "outputs/rf_fs") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    cw = {0: 1, 1: int(neg / pos)}
    print(f"[rf_fs] class_weight={cw}  n_estimators={n_estimators}")

    # Stage 1: full RF for feature importance
    print("[rf_fs] Stage 1: full RF...")
    rf_full = RandomForestClassifier(
        n_estimators=100, class_weight=cw,
        random_state=random_state, n_jobs=-1, max_depth=10
    )
    rf_full.fit(X_train, y_train)

    # Stage 2: feature selection
    selector = SelectFromModel(rf_full, threshold=threshold, prefit=True)
    X_train_sel = selector.transform(X_train)
    X_test_sel  = selector.transform(X_test)
    n_selected = X_train_sel.shape[1]
    print(f"[rf_fs] Stage 2: selected {n_selected}/{X_train.shape[1]} features")

    # Stage 3: retrain on selected features
    print("[rf_fs] Stage 3: retrain on selected features...")
    rf_final = RandomForestClassifier(
        n_estimators=n_estimators, class_weight=cw,
        random_state=random_state, n_jobs=-1, max_depth=12
    )
    rf_final.fit(X_train_sel, y_train)

    proba = rf_final.predict_proba(X_test_sel)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    metrics = {
        "model": "rf_fs",
        "n_features_selected": n_selected,
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"],    4),
        "f1":        round(report["1"]["f1-score"],  4),
        "auc_roc":   round(auc_roc, 4),
        "pr_auc":    round(pr_auc, 4),
    }
    print(f"[rf_fs] AUC-ROC={auc_roc:.4f}  Recall={metrics['recall']:.4f}  PR-AUC={pr_auc:.4f}")

    joblib.dump({"selector": selector, "model": rf_final}, os.path.join(out_dir, "rf_fs_model.pkl"))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"RF+FS ({n_selected} features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("RF+FS Precision-Recall Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close()

    return metrics
