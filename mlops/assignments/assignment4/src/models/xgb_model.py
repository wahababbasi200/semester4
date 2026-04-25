"""XGBoost fraud detection model — standard and cost-sensitive variants."""
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import joblib
import os


def build_xgb(scale_pos_weight: float = 1.0,
              n_estimators: int = 400,
              max_depth: int = 6,
              learning_rate: float = 0.05,
              random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       cost_sensitive: bool = False,
                       threshold: float = 0.5,
                       out_dir: str = "outputs/xgb") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = (neg / pos) if cost_sensitive else 1.0
    if cost_sensitive:
        spw = max(spw, 10.0)  # extra weight for cost-sensitive regime
    print(f"[xgb] scale_pos_weight={spw:.2f}  cost_sensitive={cost_sensitive}")

    model = build_xgb(scale_pos_weight=spw)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    metrics = {
        "model": "xgb",
        "cost_sensitive": cost_sensitive,
        "threshold": threshold,
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"],    4),
        "f1":        round(report["1"]["f1-score"],  4),
        "auc_roc":   round(auc_roc, 4),
        "pr_auc":    round(pr_auc, 4),
    }
    print(f"[xgb] AUC-ROC={auc_roc:.4f}  Recall={metrics['recall']:.4f}  PR-AUC={pr_auc:.4f}")

    # Save model
    model_path = os.path.join(out_dir, "xgb_model.pkl")
    joblib.dump(model, model_path)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"XGBoost ({'Cost-Sensitive' if cost_sensitive else 'Standard'})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("XGBoost Precision-Recall Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close()

    return metrics
