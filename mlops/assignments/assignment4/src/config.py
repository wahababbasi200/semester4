"""Central config — override DATA_ROOT and MODEL_ROOT with env vars for EC2."""
import os

# On Windows dev:  E:/Study/semester4/mlops/assignments/assignment4
# On EC2:          /mnt/ml-data/assignment4
BASE_DIR = os.environ.get("A4_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SAMPLE_FILE = os.path.join(DATA_DIR, "processed", "sample_ieee_70k.parquet")

# Inside Kubeflow pods the PVC is mounted here
# PV hostPath=/mnt/ml-data/assignment4 → pod /data
# sample is at /mnt/ml-data/assignment4/processed/sample_ieee_70k.parquet
POD_DATA_DIR = os.environ.get("POD_DATA_DIR", "/data")
POD_SAMPLE_FILE = os.path.join(POD_DATA_DIR, "processed", "sample_ieee_70k.parquet")

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_COL = "isFraud"
TIME_COL = "TransactionDT"

# Cost-sensitive params  ($500 per missed fraud, $50 per false alarm)
FN_COST = 500
FP_COST = 50
TP_GAIN = 200

# High-cardinality columns to target-encode
HIGH_CARD_COLS = ["card1", "card2", "card5", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]

# Binary M-columns (T/F/NaN)
M_COLS = [f"M{i}" for i in range(1, 10)]

# Deployment threshold
DEPLOY_AUC_THRESHOLD = 0.85
RECALL_ALERT_THRESHOLD = 0.70
PSI_ALERT_THRESHOLD = 0.20
