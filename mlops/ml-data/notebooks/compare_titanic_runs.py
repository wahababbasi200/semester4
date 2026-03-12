import os

import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "titanic-survival-airflow"
OUTPUT_FILE = "/workspace/notebooks/titanic_run_comparison.csv"

mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print(f"Experiment '{EXPERIMENT_NAME}' was not found.")
    raise SystemExit(1)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)

if runs.empty:
    print(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
    raise SystemExit(1)

columns = [
    "run_id",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1_score",
    "params.model_type",
    "tags.model_type",
    "params.n_estimators",
    "params.max_depth",
    "params.min_samples_split",
    "params.C",
    "tags.model_status",
]

available_columns = [column for column in columns if column in runs.columns]
comparison = runs[available_columns].copy()
comparison.to_csv(OUTPUT_FILE, index=False)

print(comparison.to_string(index=False))

best_run = comparison.iloc[0]
print()
print(f"Best run_id: {best_run['run_id']}")
print(f"Best accuracy: {best_run['metrics.accuracy']}")
print(f"Comparison file saved to: {OUTPUT_FILE}")
