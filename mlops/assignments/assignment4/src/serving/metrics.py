"""Prometheus metrics collectors for the fraud inference API."""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry()

# System-level
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)
ERROR_COUNT = Counter(
    "fraud_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"],
    registry=REGISTRY,
)

# Model-level
PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Total predictions by label",
    ["label"],       # "fraud" or "legit"
    registry=REGISTRY,
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Model confidence (fraud probability)",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    registry=REGISTRY,
)
MODEL_RECALL = Gauge(
    "fraud_model_recall",
    "Current model recall on labeled holdout (updated by shadow eval)",
    registry=REGISTRY,
)
MODEL_PRECISION = Gauge(
    "fraud_model_precision",
    "Current model precision on labeled holdout",
    registry=REGISTRY,
)
MODEL_AUC = Gauge(
    "fraud_model_auc_roc",
    "Current model AUC-ROC on labeled holdout",
    registry=REGISTRY,
)

# Data-level
FEATURE_PSI = Gauge(
    "fraud_feature_psi",
    "PSI drift score per feature (updated by shadow eval)",
    ["feature"],
    registry=REGISTRY,
)
MISSING_VALUE_RATE = Gauge(
    "fraud_input_missing_rate",
    "Fraction of NaN values in last N requests",
    registry=REGISTRY,
)
