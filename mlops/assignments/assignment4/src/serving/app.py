"""
FastAPI inference service for IEEE CIS Fraud Detection.

Endpoints:
  POST /predict           — run fraud inference
  GET  /health            — health check
  GET  /metrics           — Prometheus metrics

Environment variables:
  MODEL_PATH        — path to .pkl model file (default /data/models/best_model.pkl)
  PREPROCESSOR_PATH — path to preprocessor .pkl
  INFERENCE_THRESHOLD — fraud probability threshold (default 0.5)
"""
import time
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.serving.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT,
    PREDICTION_COUNT, PREDICTION_CONFIDENCE, MISSING_VALUE_RATE,
    MODEL_RECALL, MODEL_PRECISION, REGISTRY,
)
from src.serving.model_loader import load_artifacts, predict

THRESHOLD = float(os.environ.get("INFERENCE_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    # Seed gauges with initial values so dashboards don't show "no data"
    MODEL_RECALL.set(0.0)
    MODEL_PRECISION.set(0.0)
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str = "W"
    card1: float = 0.0
    card2: float = 0.0
    card5: float = 0.0
    addr1: float = 0.0
    P_emaildomain: str = "gmail.com"
    R_emaildomain: str = "gmail.com"
    DeviceInfo: str = "unknown"
    TransactionDT: int = 0

    class Config:
        extra = "allow"   # accept any additional IEEE CIS columns


class PredictRequest(BaseModel):
    transactions: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    count: int


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest):
    try:
        df = pd.DataFrame(req.transactions)

        # Track missing value rate
        missing_rate = df.isnull().mean().mean()
        MISSING_VALUE_RATE.set(float(missing_rate))

        result = predict(df, threshold=THRESHOLD)

        predictions = []
        for i, (prob, label) in enumerate(
                zip(result["probabilities"], result["labels"])):
            PREDICTION_COUNT.labels(label=label).inc()
            PREDICTION_CONFIDENCE.observe(float(prob))
            predictions.append({"fraud_probability": round(prob, 4),
                                 "label": label,
                                 "index": i})

        return PredictResponse(predictions=predictions, count=len(predictions))

    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/update-metrics")
async def update_model_metrics(recall: float, precision: float, auc: float = 0.0):
    """Called by shadow eval CronJob to update model performance gauges."""
    MODEL_RECALL.set(recall)
    MODEL_PRECISION.set(precision)
    return {"status": "updated", "recall": recall, "precision": precision}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=False)
