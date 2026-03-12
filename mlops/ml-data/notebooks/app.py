from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
import redis
import os
import json

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL = "IrisClassifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

redis_host = os.environ.get("REDIS_HOST", "redis")
cache = redis.Redis(host=redis_host, port=6379)

model = None
model_source = None


def get_latest_version_info():
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for '{REGISTERED_MODEL}'")
    return max(versions, key=lambda v: int(v.version))


def load_model(version=None):
    """
    Load model through MLflow using semantic version (e.g. '0.0.3') or 'latest'.

    Fallback chain:
      1. MLflow alias URI     models:/IrisClassifier@v0.0.3
      2. MLflow registry URI  models:/IrisClassifier/<mlflow_int_version>
      3. MLflow artifact source path from version metadata
      4. MLflow download_artifacts → joblib load of logged .pkl
    """
    global model, model_source

    # Resolve version info
    if version:
        # Find the MLflow version whose semantic_version tag matches
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        v_info = next(
            (v for v in versions if v.tags.get("semantic_version") == version),
            None
        )
        if v_info is None:
            raise RuntimeError(f"Semantic version '{version}' not found in MLflow registry.")
        alias_uri = f"models:/{REGISTERED_MODEL}@v{version.replace('.', '-')}"
        mlflow_ver = v_info.version
        sem_ver = version
    else:
        v_info = get_latest_version_info()
        mlflow_ver = v_info.version
        sem_ver = v_info.tags.get("semantic_version", f"mlflow-v{mlflow_ver}")
        alias_uri = f"models:/{REGISTERED_MODEL}@v{sem_ver.replace('.', '-')}"

    # --- Step 1: Alias URI ---
    try:
        model = mlflow.sklearn.load_model(alias_uri)
        model_source = f"mlflow alias ({sem_ver})"
        return
    except Exception:
        pass

    # --- Step 2: Registry integer version URI ---
    try:
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/{mlflow_ver}")
        model_source = f"mlflow registry v{mlflow_ver} ({sem_ver})"
        return
    except Exception:
        pass

    # --- Step 3: Source path from version metadata ---
    try:
        model = mlflow.sklearn.load_model(v_info.source)
        model_source = f"mlflow artifact source ({sem_ver})"
        return
    except Exception:
        pass

    # --- Step 4: Download logged .pkl artifact ---
    try:
        pkl_name = f"model_v{sem_ver}.pkl"
        local_path = client.download_artifacts(v_info.run_id, f"model_pkl/{pkl_name}")
        model = joblib.load(local_path)
        model_source = f"mlflow pkl artifact ({sem_ver})"
        return
    except Exception as e:
        raise RuntimeError(f"All MLflow load strategies failed for version '{sem_ver}': {e}")


@app.route("/")
def home():
    return f"ML Model is running! Loaded from: {model_source or 'not loaded'}"


@app.route("/train", methods=["POST"])
def train_route():
    import sys
    sys.path.insert(0, "/workspace/notebooks")
    from train import train
    result = train()
    load_model()
    cache.flushdb()
    result["model_source"] = model_source
    return jsonify({"status": "trained", **result})


@app.route("/models", methods=["GET"])
def list_models():
    """List all registered versions with semantic versions from MLflow."""
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        return jsonify([
            {
                "semantic_version": v.tags.get("semantic_version", "unknown"),
                "mlflow_version": v.version,
                "run_id": v.run_id,
                "stage": v.current_stage,
                "status": v.status,
                "aliases": v.aliases,
                "source": v.source,
                "created_at": v.creation_timestamp,
            }
            for v in sorted(versions, key=lambda v: int(v.version))
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    version = data.get("version")   # semantic version e.g. "0.0.1"

    if version:
        try:
            load_model(version=version)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    if model is None:
        return jsonify({"error": "No model loaded. POST to /train first."}), 400

    key = json.dumps({"features": features, "version": version or "latest"})

    if cache.exists(key):
        return jsonify({"prediction": int(cache.get(key)), "cached": True, "model_source": model_source})

    features_np = np.array(features).reshape(1, -1)
    prediction = model.predict(features_np)[0]
    cache.set(key, int(prediction))

    return jsonify({"prediction": int(prediction), "cached": False, "model_source": model_source})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
