"""LightGBM fraud detection model."""
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import os


def build_lgbm(is_unbalance: bool = True,
               n_estimators: int = 500,
               num_leaves: int = 64,
               learning_rate: float = 0.05,
               random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        is_unbalance=is_unbalance,
        objective="binary",
        metric="auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       threshold: float = 0.5,
                       out_dir: str = "outputs/lgbm") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("[lgbm] Training LightGBM...")

    model = build_lgbm()
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=callbacks)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_roc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    metrics = {
        "model": "lgbm",
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"],    4),
        "f1":        round(report["1"]["f1-score"],  4),
        "auc_roc":   round(auc_roc, 4),
        "pr_auc":    round(pr_auc, 4),
    }
    print(f"[lgbm] AUC-ROC={auc_roc:.4f}  Recall={metrics['recall']:.4f}  PR-AUC={pr_auc:.4f}")

    joblib.dump(model, os.path.join(out_dir, "lgbm_model.pkl"))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("LightGBM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("LightGBM Precision-Recall Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close()

    return metrics
