"""KFP Component 7 — Conditional Deployment: copy model to serving location."""
from kfp.dsl import component, Input, Output, Model, Dataset


@component(
    base_image='python:3.9',
    packages_to_install=['joblib==1.3.2']
)
def deploy_model(
    model_artifact: Input[Model],
    preprocessor_artifact: Input[Dataset],
    deploy_path: str = "/data/models",    # PVC-mounted serving path
    model_version: str = "v1",
):
    """
    Copy the trained model and preprocessor to the serving path.
    In production this would trigger a Kubernetes rolling-update;
    here we write to the shared PVC so the FastAPI pod picks it up.
    """
    import shutil
    import os
    import json
    import joblib

    os.makedirs(deploy_path, exist_ok=True)

    model_src = model_artifact.metadata.get("file", model_artifact.path + ".pkl")
    prep_src  = preprocessor_artifact.metadata.get("file", preprocessor_artifact.path + ".pkl")

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

    print(f"[deploy] Model deployed to {deploy_path}")
    print(f"  model:        {model_dst}")
    print(f"  preprocessor: {prep_dst}")
    print(f"  meta:         {meta}")
    print("[deploy] FastAPI service will pick up new model on next request (lazy reload).")
