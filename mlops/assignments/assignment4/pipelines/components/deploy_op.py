"""KFP Component 7 — Conditional Deployment: copy model to serving location."""
from kfp.dsl import component, Input, Model, Dataset


@component(
    base_image='python:3.9',
    packages_to_install=['joblib==1.3.2']
)
def deploy_model(
    model_artifact: Input[Model],
    preprocessor_artifact: Input[Dataset],
    auc_roc: float,
    deploy_threshold: float = 0.85,
    deploy_path: str = "/data/models",    # PVC-mounted serving path
    model_version: str = "v1",
):
    """
    Copy the trained model and preprocessor to the serving path when the
    evaluation metric passes the deployment threshold.

    This keeps deployment gating inside a single executor component because the
    Kubeflow runtime used for this assignment cannot resolve artifact inputs
    passed through nested conditional DAG branches.
    """
    import shutil
    import os
    import json

    if auc_roc < deploy_threshold:
        print(f"[deploy] SKIPPED - AUC-ROC {auc_roc:.4f} < threshold {deploy_threshold:.2f}")
        return

    os.makedirs(deploy_path, exist_ok=True)

    model_src = model_artifact.path
    prep_src  = preprocessor_artifact.path

    model_dst = os.path.join(deploy_path, "best_model.pkl")
    prep_dst  = os.path.join(deploy_path, "preprocessor.pkl")

    shutil.copy(model_src, model_dst)
    shutil.copy(prep_src,  prep_dst)

    # Write version manifest
    meta = {
        "version":          model_version,
        "model_type":       model_artifact.metadata.get("model_type", "unknown"),
        "imbalance":        model_artifact.metadata.get("imbalance_strategy", "unknown"),
        "cost_sensitive":   model_artifact.metadata.get("cost_sensitive", "false"),
    }
    with open(os.path.join(deploy_path, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[deploy] PASSED threshold {deploy_threshold:.2f} with AUC-ROC {auc_roc:.4f}")
    print(f"[deploy] Model deployed to {deploy_path}")
    print(f"  model:        {model_dst}")
    print(f"  preprocessor: {prep_dst}")
    print(f"  meta:         {meta}")
    print("[deploy] FastAPI service will pick up new model on next request (lazy reload).")
