"""Load trained model + preprocessor from disk."""
import os
import joblib
import numpy as np
import pandas as pd

_model = None
_preprocessor = None

MODEL_PATH = os.environ.get("MODEL_PATH", "/data/models/best_model.pkl")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "/data/models/preprocessor.pkl")


def load_artifacts():
    global _model, _preprocessor
    if _model is None:
        print(f"[loader] Loading model from {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    if _preprocessor is None:
        print(f"[loader] Loading preprocessor from {PREPROCESSOR_PATH}")
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    return _model, _preprocessor


def predict(raw_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Run preprocessing + inference. Returns probabilities and labels."""
    from src.data.preprocess import preprocess
    model, prep = load_artifacts()

    X, _, _, _ = preprocess(raw_df, encoders=prep["encoders"],
                             medians=prep["medians"], fit=False)

    if hasattr(model, "selector"):  # RF-FS hybrid
        X_arr = model["selector"].transform(X.values)
        proba = model["model"].predict_proba(X_arr)[:, 1]
    else:
        proba = model.predict_proba(X.values)[:, 1]

    labels = ["fraud" if p >= threshold else "legit" for p in proba]
    return {"probabilities": proba.tolist(), "labels": labels}
