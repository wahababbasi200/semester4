"""
Assignment 4 — IEEE CIS Fraud Detection Pipeline (KFP v2)

7-step DAG:
  1. ingest_data
  2. validate_data
  3. preprocess_data
  4. engineer_features
  5. train_model
  6. evaluate_model
  7. deploy_model  ← runtime-gated inside the component

Features:
  - Deployment gate based on AUC-ROC threshold
  - Retry with exponential back-off on train/eval/deploy
  - PVC mount via kfp-kubernetes (hostPath PV at /mnt/ml-data/assignment4)
  - Caching fully disabled at component level (A3 lesson)
"""
import sys
import os
import re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp import compiler
from kfp.dsl import pipeline

from pipelines.components.ingest_op   import ingest_data
from pipelines.components.validate_op import validate_data
from pipelines.components.preprocess_op import preprocess_data
from pipelines.components.feateng_op  import engineer_features
from pipelines.components.train_op    import train_model
from pipelines.components.eval_op     import evaluate_model
from pipelines.components.deploy_op   import deploy_model

try:
    from kfp import kubernetes as kfp_k8s
    KFP_K8S_AVAILABLE = True
except ImportError:
    KFP_K8S_AVAILABLE = False
    print("WARNING: kfp-kubernetes not installed — PVC mounting disabled.")


PVC_NAME     = "fraud-a4-pvc"
PVC_MOUNT    = "/data"
DEPLOY_THRESHOLD = 0.85


def _disable_cache(task):
    """Disable KFP task-level caching (client-flag alone is insufficient — A3 lesson)."""
    task.set_caching_options(enable_caching=False)
    return task


def _mount_pvc(task):
    """Mount the assignment PVC if kfp-kubernetes is available."""
    if KFP_K8S_AVAILABLE:
        kfp_k8s.mount_pvc(task, pvc_name=PVC_NAME, mount_path=PVC_MOUNT)
    return task


def patch_compiled_yaml_for_cluster(yaml_path: str) -> int:
    """Normalize generated PVC config for the Kubeflow runtime used in this assignment."""
    path = Path(yaml_path)
    text = path.read_text(encoding="utf-8")
    replacements = 0

    # Flatten the KFP-generated nested pvcNameParameter runtime value into the
    # scalar form this Kubeflow deployment expects.
    text, flattened = re.subn(
        r"(?m)^(?P<indent>\s*)pvcNameParameter:\n"
        r"(?P=indent)  runtimeValue:\n"
        r"(?P=indent)    constant: (?P<name>[^\n]+)\n?",
        r"\g<indent>constant: \g<name>\n",
        text,
    )
    replacements += flattened

    # Older/re-patched YAML can contain a duplicate nested constant block under a
    # pvcMount item that already has "- constant: <pvc>".
    text, deduped = re.subn(
        r"(?m)^(?P<prefix>\s*-\s*constant:\s*(?P<name>[^\n]+)\n"
        r"\s*mountPath:\s*[^\n]+\n)"
        r"\s*constant:\n"
        r"\s*runtimeValue:\n"
        r"\s*constant:\s*(?P=name)\n?",
        r"\g<prefix>",
        text,
    )
    replacements += deduped

    if replacements == 0:
        return 0
    path.write_text(text, encoding="utf-8")
    return replacements


@pipeline(
    name="fraud-detection-pipeline",
    description=(
        "IEEE CIS Fraud Detection: 7-step MLOps pipeline with "
        "AUC-gated deployment and retry mechanisms."
    ),
)
def fraud_detection_pipeline(
    # ── Data params ──────────────────────────────────────────────────────────
    data_path: str = "/data/processed/sample_ieee_70k.parquet",
    deploy_path: str = "/data/models",
    model_version: str = "v1",
    # ── Training params ───────────────────────────────────────────────────────
    model_type: str = "xgb",                   # xgb | lgbm | rf_fs
    imbalance_strategy: str = "class_weight",  # class_weight | smote
    cost_sensitive: bool = False,
    n_estimators: int = 400,
    random_state: int = 42,
    # ── Evaluation params ─────────────────────────────────────────────────────
    eval_threshold: float = 0.5,
    deploy_auc_threshold: float = DEPLOY_THRESHOLD,
):
    # Keep task names aligned with the compiler-generated IDs. On this Kubeflow
    # runtime, custom display names can break downstream artifact resolution
    # because producerTask references use IDs like "ingest-data".
    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    ingest_task = _disable_cache(
        _mount_pvc(ingest_data(data_path=data_path))
    )

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    validate_task = _disable_cache(
        _mount_pvc(validate_data(
            input_dataset=ingest_task.outputs["output_dataset"],
        ))
    )
    validate_task.set_retry(num_retries=2, backoff_duration="30s", backoff_factor=2.0)

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    preprocess_task = _disable_cache(
        _mount_pvc(preprocess_data(
            validated_dataset=validate_task.outputs["validated_dataset"],
        ))
    )
    # Keep requests small enough to schedule on a single-node Minikube cluster.
    preprocess_task.set_cpu_request("250m").set_memory_request("1G")
    preprocess_task.set_cpu_limit("1").set_memory_limit("2G")

    # ── Step 4: Feature Engineering ───────────────────────────────────────────
    feateng_task = _disable_cache(
        _mount_pvc(engineer_features(
            train_data=preprocess_task.outputs["train_data"],
            val_data=preprocess_task.outputs["val_data"],
            test_data=preprocess_task.outputs["test_data"],
        ))
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    train_task = _disable_cache(
        _mount_pvc(train_model(
            train_data=feateng_task.outputs["train_engineered"],
            val_data=feateng_task.outputs["val_engineered"],
            model_type=model_type,
            imbalance_strategy=imbalance_strategy,
            cost_sensitive=cost_sensitive,
            n_estimators=n_estimators,
            random_state=random_state,
        ))
    )
    train_task.set_retry(num_retries=2, backoff_duration="60s", backoff_factor=2.0)
    train_task.set_cpu_request("500m").set_memory_request("2G")
    train_task.set_cpu_limit("2").set_memory_limit("4G")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    eval_task = _disable_cache(
        _mount_pvc(evaluate_model(
            test_data=feateng_task.outputs["test_engineered"],
            model_artifact=train_task.outputs["model_artifact"],
            threshold=eval_threshold,
        ))
    )
    eval_task.set_retry(num_retries=1, backoff_duration="30s", backoff_factor=2.0)

    # ── Step 7: Deployment Gate ───────────────────────────────────────────────
    # Keep deployment in the root DAG. This Kubeflow runtime cannot resolve
    # artifact inputs across nested conditional DAG branches.
    # eval_task has multiple outputs (eval_metrics, plots_artifact, + return float),
    # so KFP v2 names the function return value "Output".
    auc_output = eval_task.outputs["Output"]
    deploy_task = _disable_cache(
        _mount_pvc(deploy_model(
            model_artifact=train_task.outputs["model_artifact"],
            preprocessor_artifact=preprocess_task.outputs["preprocessor_artifact"],
            auc_roc=auc_output,
            deploy_threshold=deploy_auc_threshold,
            deploy_path=deploy_path,
            model_version=model_version,
        ))
    )
    deploy_task.set_retry(num_retries=2, backoff_duration="30s", backoff_factor=2.0)


if __name__ == "__main__":
    os.makedirs("pipelines/compiled", exist_ok=True)
    out = "pipelines/compiled/fraud_pipeline.yaml"
    compiler.Compiler().compile(fraud_detection_pipeline, out)
    patched = patch_compiled_yaml_for_cluster(out)
    print(f"Pipeline compiled -> {out}")
    if patched:
        print(f"Patched {patched} pvcNameParameter entries to constant for cluster compatibility.")
