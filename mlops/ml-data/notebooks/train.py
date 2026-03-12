import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

MODELS_DIR = "/workspace/models"
REGISTERED_MODEL = "IrisClassifier"


def get_next_semantic_version(client):
    """Read existing semantic_version tags and return the next patch increment."""
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        sem_versions = [
            v.tags["semantic_version"]
            for v in versions
            if "semantic_version" in v.tags
        ]
        if not sem_versions:
            return "0.0.1"

        def parse(v):
            return tuple(int(x) for x in v.split("."))

        major, minor, patch = max(sem_versions, key=parse).split(".")
        return f"{major}.{minor}.{int(patch) + 1}"
    except Exception:
        return "0.0.1"


def train():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("iris-classification")
    client = MlflowClient()

    sem_version = get_next_semantic_version(client)

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = [
        {"n_estimators": 50,  "max_depth": None, "min_samples_split": 2, "random_state": 42},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42},
        {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 2, "random_state": 42},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_split": 5, "random_state": 42},
    ]

    best_result = None
    results = []

    for params in param_grid:
        with mlflow.start_run() as run:
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))

            mlflow.log_params(params)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.set_tag("semantic_version", sem_version)

            result = {
                "params": params,
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "run_id": run.info.run_id,
                "model": model,
            }
            results.append(result)

            print(f"  params={params} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

            if best_result is None or test_acc > best_result["test_accuracy"]:
                best_result = result

    # Register only the best model
    best_model = best_result["model"]
    with mlflow.start_run(run_id=best_result["run_id"]):
        model_info = mlflow.sklearn.log_model(
            best_model, "model", registered_model_name=REGISTERED_MODEL
        )
    mlflow_version = model_info.registered_model_version

    # Tag and alias the best registered version
    client.set_model_version_tag(REGISTERED_MODEL, mlflow_version, "semantic_version", sem_version)
    alias = f"v{sem_version.replace('.', '-')}"
    client.set_registered_model_alias(REGISTERED_MODEL, alias, mlflow_version)

    # Save versioned local copy and log as artifact
    versioned_path = os.path.join(MODELS_DIR, f"model_v{sem_version}.pkl")
    latest_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(best_model, versioned_path)
    joblib.dump(best_model, latest_path)
    with mlflow.start_run(run_id=best_result["run_id"]):
        mlflow.log_artifact(versioned_path, artifact_path="model_pkl")
        mlflow.log_artifact(latest_path, artifact_path="model_pkl")

    return {
        "best_params": best_result["params"],
        "train_accuracy": best_result["train_accuracy"],
        "test_accuracy": best_result["test_accuracy"],
        "semantic_version": sem_version,
        "mlflow_version": mlflow_version,
        "run_id": best_result["run_id"],
        "total_runs": len(results),
    }


if __name__ == "__main__":
    result = train()
    print(result)
