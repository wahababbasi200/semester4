"""
Submit fraud detection pipeline to Kubeflow on EC2 Minikube.

Prerequisites on EC2:
  kubectl -n kubeflow port-forward --address 0.0.0.0 svc/ml-pipeline-ui 8080:80 &

Usage examples:
  # XGBoost run (default)
  python scripts/submit_pipeline.py

  # LightGBM run
  python scripts/submit_pipeline.py --model lgbm

  # RF hybrid run
  python scripts/submit_pipeline.py --model rf_fs

  # SMOTE imbalance strategy comparison
  python scripts/submit_pipeline.py --model xgb --imbalance smote
  python scripts/submit_pipeline.py --model xgb --imbalance class_weight

  # Cost-sensitive run
  python scripts/submit_pipeline.py --model xgb --cost-sensitive true
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kfp
from pipelines.fraud_pipeline import fraud_detection_pipeline
import kfp.compiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--model", default="xgb", choices=["xgb", "lgbm", "rf_fs"])
    parser.add_argument("--imbalance", default="class_weight", choices=["class_weight", "smote"])
    parser.add_argument("--cost-sensitive", default="false", choices=["true", "false"])
    parser.add_argument("--experiment", default="fraud-detection-a4")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    os.makedirs("pipelines/compiled", exist_ok=True)
    yaml_path = "pipelines/compiled/fraud_pipeline.yaml"

    print(f"Compiling pipeline → {yaml_path}")
    kfp.compiler.Compiler().compile(fraud_detection_pipeline, yaml_path)
    print("  Done.")

    if args.compile_only:
        return

    client = kfp.Client(host=args.host)

    # Ensure experiment exists
    try:
        exp = client.get_experiment(experiment_name=args.experiment)
    except Exception:
        exp = client.create_experiment(args.experiment)

    run_name = f"fraud-{args.model}-{args.imbalance}-costsens{args.cost_sensitive}"
    print(f"Submitting run: {run_name}")

    run = client.run_pipeline(
        experiment_id=exp.experiment_id,
        job_name=run_name,
        pipeline_package_path=yaml_path,
        params={
            "model_type": args.model,
            "imbalance_strategy": args.imbalance,
            "cost_sensitive": args.cost_sensitive == "true",
        },
        enable_caching=False,
    )
    print(f"Run created: {run.run_id}")
    print(f"Monitor at:  {args.host}/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    main()
