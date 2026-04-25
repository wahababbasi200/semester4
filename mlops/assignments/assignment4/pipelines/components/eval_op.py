"""KFP Component 6 — Model Evaluation: full metrics + confusion matrix."""
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image='python:3.9',
    packages_to_install=[
        'pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4',
        'scikit-learn==1.3.2', 'xgboost==2.0.3', 'lightgbm==4.3.0',
        'joblib==1.3.2', 'matplotlib==3.8.2'
    ]
)
def evaluate_model(
    test_data: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics],
    plots_artifact: Output[Dataset],   # stores confusion matrix + PR curve PNGs
    threshold: float = 0.5,
) -> float:   # returns AUC-ROC so pipeline can branch on it
    """Evaluate trained model: Precision/Recall/F1/AUC-ROC/Confusion Matrix."""
    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        confusion_matrix, ConfusionMatrixDisplay,
        average_precision_score, precision_recall_curve,
    )

    FRAUD_COL = "isFraud"

    src = test_data.path
    df = pd.read_parquet(src)
    X_test = df.drop(columns=[FRAUD_COL]).values.astype(np.float32)
    y_test = df[FRAUD_COL].values

    model_path = model_artifact.path
    payload = joblib.load(model_path)

    # Handle RF-FS hybrid (dict with selector + model)
    if isinstance(payload, dict) and "selector" in payload:
        X_test = payload["selector"].transform(X_test)
        model = payload["model"]
    else:
        model = payload

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc_roc  = float(roc_auc_score(y_test, proba))
    pr_auc   = float(average_precision_score(y_test, proba))
    precision = float(report["1"]["precision"])
    recall    = float(report["1"]["recall"])
    f1        = float(report["1"]["f1-score"])
    fpr = float((y_pred[y_test == 0] == 1).sum() / max((y_test == 0).sum(), 1))

    eval_metrics.log_metric("auc_roc",   round(auc_roc, 4))
    eval_metrics.log_metric("pr_auc",    round(pr_auc, 4))
    eval_metrics.log_metric("precision", round(precision, 4))
    eval_metrics.log_metric("recall",    round(recall, 4))
    eval_metrics.log_metric("f1_score",  round(f1, 4))
    eval_metrics.log_metric("false_positive_rate", round(fpr, 4))
    eval_metrics.log_metric("threshold", threshold)

    print(f"[eval] AUC-ROC={auc_roc:.4f}  Recall={recall:.4f}  Precision={precision:.4f}  "
          f"F1={f1:.4f}  PR-AUC={pr_auc:.4f}")

    # Save plots
    plots_dir = plots_artifact.path
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, colorbar=False)
    model_name = model_artifact.metadata.get("model_type", "model")
    ax.set_title(f"{model_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # PR curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(rec_vals, prec_vals, label=f"PR-AUC={pr_auc:.3f}", color="steelblue")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{model_name.upper()} Precision-Recall Curve")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pr_curve.png"), dpi=150)
    plt.close()

    plots_artifact.metadata["auc_roc"] = auc_roc   # used by downstream deploy condition
    return auc_roc
