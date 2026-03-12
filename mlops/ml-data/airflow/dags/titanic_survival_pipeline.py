from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, get_current_context
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "titanic-mlops"
REGISTERED_MODEL_NAME = "TitanicSurvivalModel"
DATA_DIR = Path("/opt/airflow/data")
RAW_DATASET_PATH = DATA_DIR / "Titanic-Dataset.csv"
OUTPUT_ROOT = DATA_DIR / "output"
RETRY_MARKER_ROOT = DATA_DIR / "retry_markers"


def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _current_context() -> dict[str, Any]:
    return get_current_context()


def _run_output_dir() -> Path:
    context = _current_context()
    run_dir = OUTPUT_ROOT / _safe_name(context["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _dag_conf() -> dict[str, Any]:
    context = _current_context()
    dag_run = context.get("dag_run")
    if dag_run is None or dag_run.conf is None:
        return {}
    return dict(dag_run.conf)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _pipeline_config() -> dict[str, Any]:
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 2,
        "C": 1.0,
        "solver": "liblinear",
        "test_size": 0.2,
        "random_state": 42,
        "accuracy_threshold": 0.80,
        "force_retry_demo": True,
    }
    config.update(_dag_conf())
    config["model_type"] = str(config["model_type"]).strip().lower()
    config["n_estimators"] = int(config["n_estimators"])
    config["min_samples_split"] = int(config["min_samples_split"])
    config["C"] = float(config["C"])
    config["test_size"] = float(config["test_size"])
    config["random_state"] = int(config["random_state"])
    config["accuracy_threshold"] = float(config["accuracy_threshold"])
    config["force_retry_demo"] = _as_bool(config["force_retry_demo"])

    max_depth = config.get("max_depth")
    if max_depth in (None, "", "None", "null"):
        config["max_depth"] = None
    else:
        config["max_depth"] = int(max_depth)

    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataset() -> str:
    if not RAW_DATASET_PATH.exists():
        raise AirflowException(f"Dataset not found at {RAW_DATASET_PATH}")

    run_dir = _run_output_dir()
    df = pd.read_csv(RAW_DATASET_PATH)
    missing_counts = df.isna().sum().to_dict()

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values count: {missing_counts}")
    logging.info("Dataset shape: %s", df.shape)
    logging.info("Missing values count: %s", missing_counts)

    _write_json(
        run_dir / "ingestion_summary.json",
        {
            "dataset_path": str(RAW_DATASET_PATH),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": missing_counts,
        },
    )
    return str(RAW_DATASET_PATH)


def validate_dataset() -> dict[str, Any]:
    context = _current_context()
    config = _pipeline_config()
    run_dir = _run_output_dir()
    RETRY_MARKER_ROOT.mkdir(parents=True, exist_ok=True)

    marker_path = RETRY_MARKER_ROOT / f"{_safe_name(context['run_id'])}.marker"
    if config["force_retry_demo"] and not marker_path.exists():
        marker_path.write_text("fail-once", encoding="utf-8")
        raise AirflowException("Intentional validation failure to demonstrate Airflow retry behavior.")

    dataset_path = context["ti"].xcom_pull(task_ids="load_dataset")
    df = pd.read_csv(dataset_path)

    age_missing_pct = round(float(df["Age"].isna().mean() * 100), 2)
    embarked_missing_pct = round(float(df["Embarked"].isna().mean() * 100), 2)

    print(f"Age missing percentage: {age_missing_pct}%")
    print(f"Embarked missing percentage: {embarked_missing_pct}%")
    logging.info("Age missing percentage: %.2f%%", age_missing_pct)
    logging.info("Embarked missing percentage: %.2f%%", embarked_missing_pct)

    if age_missing_pct > 30 or embarked_missing_pct > 30:
        raise AirflowException(
            f"Validation failed because missing data exceeded 30%. "
            f"Age={age_missing_pct}%, Embarked={embarked_missing_pct}%"
        )

    if marker_path.exists():
        marker_path.unlink()

    result = {
        "age_missing_pct": age_missing_pct,
        "embarked_missing_pct": embarked_missing_pct,
        "validated": True,
    }
    _write_json(run_dir / "validation_summary.json", result)
    return result


def handle_missing_values() -> str:
    context = _current_context()
    run_dir = _run_output_dir()
    dataset_path = context["ti"].xcom_pull(task_ids="load_dataset")

    df = pd.read_csv(dataset_path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    output_path = run_dir / "missing_handled.csv"
    df.to_csv(output_path, index=False)
    logging.info("Saved missing-value output to %s", output_path)
    return str(output_path)


def engineer_features() -> str:
    context = _current_context()
    run_dir = _run_output_dir()
    dataset_path = context["ti"].xcom_pull(task_ids="load_dataset")

    df = pd.read_csv(dataset_path)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    output_path = run_dir / "engineered_features.csv"
    df.to_csv(output_path, index=False)
    logging.info("Saved engineered-feature output to %s", output_path)
    return str(output_path)


def merge_parallel_outputs() -> str:
    context = _current_context()
    run_dir = _run_output_dir()

    missing_path = context["ti"].xcom_pull(task_ids="handle_missing_values")
    engineered_path = context["ti"].xcom_pull(task_ids="engineer_features")

    missing_df = pd.read_csv(missing_path)
    engineered_df = pd.read_csv(engineered_path)

    missing_df["FamilySize"] = engineered_df["FamilySize"]
    missing_df["IsAlone"] = engineered_df["IsAlone"]

    output_path = run_dir / "prepared_dataset.csv"
    missing_df.to_csv(output_path, index=False)
    logging.info("Merged parallel outputs into %s", output_path)
    return str(output_path)


def encode_features() -> str:
    context = _current_context()
    run_dir = _run_output_dir()
    prepared_path = context["ti"].xcom_pull(task_ids="merge_parallel_outputs")

    df = pd.read_csv(prepared_path)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

    output_path = run_dir / "encoded_dataset.csv"
    df.to_csv(output_path, index=False)
    logging.info("Saved encoded dataset to %s", output_path)
    return str(output_path)


def _build_model(config: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    model_type = config["model_type"]

    if model_type == "logistic_regression":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config["C"],
                        solver=str(config["solver"]),
                        max_iter=1000,
                        random_state=config["random_state"],
                    ),
                ),
            ]
        )
        params = {
            "C": config["C"],
            "solver": str(config["solver"]),
            "max_iter": 1000,
            "random_state": config["random_state"],
        }
        return model, params, "LogisticRegression"

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=config["random_state"],
        )
        params = {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "random_state": config["random_state"],
        }
        return model, params, "RandomForestClassifier"

    raise AirflowException(
        "Unsupported model_type. Use 'random_forest' or 'logistic_regression'."
    )


def train_model() -> dict[str, Any]:
    _configure_mlflow()
    context = _current_context()
    config = _pipeline_config()
    run_dir = _run_output_dir()
    encoded_path = context["ti"].xcom_pull(task_ids="encode_features")

    df = pd.read_csv(encoded_path)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["Survived"] = y_train.values
    test_df = X_test.copy()
    test_df["Survived"] = y_test.values

    train_split_path = run_dir / "train_split.csv"
    test_split_path = run_dir / "test_split.csv"
    train_df.to_csv(train_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)

    mlflow.set_experiment(EXPERIMENT_NAME)
    model, model_params, model_name = _build_model(config)

    with mlflow.start_run(run_name=_safe_name(context["run_id"])) as run:
        model.fit(X_train, y_train)

        mlflow.set_tag("airflow_dag_id", context["dag"].dag_id)
        mlflow.set_tag("airflow_run_id", context["run_id"])
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset_rows", int(df.shape[0]))
        mlflow.log_param("dataset_columns", int(df.shape[1]))
        mlflow.log_param("train_rows", int(X_train.shape[0]))
        mlflow.log_param("test_rows", int(X_test.shape[0]))
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_params(model_params)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_dict(
            {
                "model_type": model_name,
                "model_params": model_params,
                "encoded_dataset_path": encoded_path,
            },
            "training_summary.json",
        )

        result = {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "model_name": model_name,
            "train_split_path": str(train_split_path),
            "test_split_path": str(test_split_path),
            "dataset_rows": int(df.shape[0]),
            "dataset_columns": int(df.shape[1]),
        }

    _write_json(run_dir / "training_result.json", result)
    return result


def evaluate_model() -> dict[str, Any]:
    _configure_mlflow()
    context = _current_context()
    run_dir = _run_output_dir()
    train_result = context["ti"].xcom_pull(task_ids="train_model")

    test_df = pd.read_csv(train_result["test_split_path"])
    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

    model = mlflow.sklearn.load_model(train_result["model_uri"])
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }

    with mlflow.start_run(run_id=train_result["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metrics, "evaluation_metrics.json")

    logging.info("Evaluation metrics: %s", metrics)
    _write_json(run_dir / "evaluation_metrics.json", metrics)
    return metrics


def branch_on_accuracy() -> str:
    context = _current_context()
    config = _pipeline_config()
    evaluation = context["ti"].xcom_pull(task_ids="evaluate_model")
    accuracy = float(evaluation["accuracy"])

    if accuracy >= config["accuracy_threshold"]:
        return "register_model"
    return "reject_model"


def register_model() -> dict[str, Any]:
    _configure_mlflow()
    context = _current_context()
    run_dir = _run_output_dir()
    train_result = context["ti"].xcom_pull(task_ids="train_model")
    evaluation = context["ti"].xcom_pull(task_ids="evaluate_model")

    client = MlflowClient()
    registered_model = mlflow.register_model(
        model_uri=train_result["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered_model.version,
        "source",
        "airflow",
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered_model.version,
        "accuracy",
        str(evaluation["accuracy"]),
    )

    with mlflow.start_run(run_id=train_result["run_id"]):
        mlflow.set_tag("model_status", "registered")
        mlflow.set_tag("registered_model_name", REGISTERED_MODEL_NAME)
        mlflow.set_tag("registered_model_version", registered_model.version)

    result = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": registered_model.version,
        "accuracy": evaluation["accuracy"],
    }
    _write_json(run_dir / "registration_result.json", result)
    return result


def reject_model() -> dict[str, Any]:
    _configure_mlflow()
    context = _current_context()
    config = _pipeline_config()
    run_dir = _run_output_dir()
    train_result = context["ti"].xcom_pull(task_ids="train_model")
    evaluation = context["ti"].xcom_pull(task_ids="evaluate_model")
    accuracy = float(evaluation["accuracy"])
    threshold = float(config["accuracy_threshold"])

    reason = f"Model rejected because accuracy {accuracy:.4f} is below {threshold:.2f}."

    with mlflow.start_run(run_id=train_result["run_id"]):
        mlflow.set_tag("model_status", "rejected")
        mlflow.set_tag("rejection_reason", reason)

    result = {"accuracy": accuracy, "rejection_reason": reason}
    _write_json(run_dir / "rejection_result.json", result)
    logging.info(reason)
    return result


default_args = {
    "owner": "abdul",
}


with DAG(
    dag_id="titanic_survival_pipeline",
    description="Titanic survival pipeline with Airflow orchestration and MLflow tracking.",
    start_date=datetime(2026, 3, 9),
    schedule=None,
    catchup=False,
    default_args=default_args,
    max_active_runs=3,
    doc_md="""
    ## Titanic Survival Pipeline

    Dependency flow:

    `load_dataset -> validate_dataset -> [handle_missing_values, engineer_features]`

    `-> merge_parallel_outputs -> encode_features -> train_model -> evaluate_model`

    `-> branch_on_accuracy -> register_model / reject_model -> finish`

    The DAG is acyclic because every edge moves forward to a later step and no task points back to an upstream task.
    """,
) as dag:
    start = EmptyOperator(task_id="start")

    load_dataset_task = PythonOperator(
        task_id="load_dataset",
        python_callable=load_dataset,
    )

    validate_dataset_task = PythonOperator(
        task_id="validate_dataset",
        python_callable=validate_dataset,
        retries=1,
        retry_delay=timedelta(seconds=10),
    )

    handle_missing_values_task = PythonOperator(
        task_id="handle_missing_values",
        python_callable=handle_missing_values,
    )

    engineer_features_task = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )

    merge_parallel_outputs_task = PythonOperator(
        task_id="merge_parallel_outputs",
        python_callable=merge_parallel_outputs,
    )

    encode_features_task = PythonOperator(
        task_id="encode_features",
        python_callable=encode_features,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=branch_on_accuracy,
    )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    reject_model_task = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    finish = EmptyOperator(
        task_id="finish",
        trigger_rule="none_failed_min_one_success",
    )

    start >> load_dataset_task >> validate_dataset_task
    validate_dataset_task >> handle_missing_values_task
    validate_dataset_task >> engineer_features_task
    handle_missing_values_task >> merge_parallel_outputs_task
    engineer_features_task >> merge_parallel_outputs_task
    merge_parallel_outputs_task >> encode_features_task >> train_model_task
    train_model_task >> evaluate_model_task >> branch_task
    branch_task >> register_model_task >> finish
    branch_task >> reject_model_task >> finish
