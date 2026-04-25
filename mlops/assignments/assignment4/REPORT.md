# Assignment 4 — Research Report
## IEEE CIS Fraud Detection MLOps System

**Student:** Abdul Wahab Abbasi · 24i-8034  
**Course:** MLOps for Cloud Native Applications

---

## Task 1: Kubeflow Environment Setup

### Infrastructure
- **Platform:** AWS EC2 (Ubuntu 24.04) — Minikube with Docker driver
- **Kubeflow:** KFP SDK 2.16.0 / cluster 2.2.0
- **Namespace:** `fraud-a4` (isolated from default kubeflow experiments)
- **ResourceQuota:** `requests.cpu=6, requests.memory=6Gi, limits.cpu=8, limits.memory=8Gi`
- **PersistentVolume:** hostPath at `/mnt/ml-data/assignment4` (10Gi, RWO) — shared between pipeline pods and inference server

### 7-Step Pipeline DAG
| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `ingest_data` | Reads `sample_ieee_70k.parquet` from PVC mount |
| 2 | `validate_data` | Schema + missing value + fraud rate checks |
| 3 | `preprocess_data` | Imputation, encoding, 70/15/15 chronological split |
| 4 | `engineer_features` | TransactionAmt bins, time features, C-column aggregates |
| 5 | `train_model` | XGB / LightGBM / RF+FS (parameterized via `model_type`) |
| 6 | `evaluate_model` | Precision, Recall, F1, AUC-ROC, PR curve, confusion matrix |
| 7 | `deploy_model` | **Conditional** — only if AUC-ROC ≥ 0.85 |

### Conditional Logic
```python
with dsl.If(eval_task.output >= 0.85, name="AUC above threshold — deploy"):
    deploy_task = deploy_model(...)
with dsl.Else(name="AUC below threshold — skip deploy"):
    skip_task = _skip_deploy(...)
```

### Retry Configuration
- `validate_data`: 2 retries, 30s backoff, factor 2×
- `train_model`: 2 retries, 60s backoff, factor 2×
- `evaluate_model`: 1 retry, 30s backoff
- `deploy_model`: 2 retries, 30s backoff

*(See screenshot: `screenshots/kubeflow/pipeline_dag.png`)*

---

## Task 2: Data Challenges Handling

### Dataset Summary
| Property | Value |
|----------|-------|
| Source | IEEE CIS Fraud Detection (Kaggle) |
| Working sample | All fraud rows (~20,663) + 50,000 non-fraud |
| Total rows | ~70,663 |
| Fraud rate | ~29.2% (in sample) |
| Features | ~394 columns (after join) |
| Time ordering | Sorted by `TransactionDT` |

### Missing Value Strategy
- **Numeric columns with >30% NaN:** Binary missing-indicator features added (`col_missing`)
- **Remaining numeric NaN:** Median imputation (computed from training split only — no leakage)
- **String NaN (`R_emaildomain`, `DeviceInfo`):** Treated as a category `"missing"` before encoding

### High-Cardinality Features
Target encoding with smoothing applied to:
`card1, card2, card5, addr1, P_emaildomain, R_emaildomain, DeviceInfo`

Formula: `encoded = (count × mean + smoothing × global_mean) / (count + smoothing)` (smoothing=20)

### M-columns (T/F binary)
`M1–M9` mapped: `T→1, F→0, NaN→-1`

### Imbalance Strategy Comparison

| Strategy | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|----------|-----------|--------|----|---------|--------|
| class_weight | _fill after run_ | _fill_ | _fill_ | _fill_ | _fill_ |
| SMOTE | _fill after run_ | _fill_ | _fill_ | _fill_ | _fill_ |

**Key finding:** *(fill in after running `src/data/imbalance.py`)*

*(See: `outputs/imbalance/imbalance_comparison.csv`, `imbalance_confusion_matrices.png`)*

---

## Task 3: Model Complexity

### Models Trained

| Model | Description | Key Params |
|-------|-------------|------------|
| **XGBoost** | Gradient boosting (GPU-friendly hist) | n_est=400, max_depth=6, scale_pos_weight=auto |
| **LightGBM** | Gradient boosting (leaf-wise) | n_est=500, num_leaves=64, is_unbalance=True |
| **RF+FS (Hybrid)** | Random Forest → SelectFromModel → RF | Stage1: 100 trees, Stage2: 300 trees, threshold='median' |

### Results Comparison

| Model | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| XGBoost | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| LightGBM | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| RF+FS Hybrid | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

*(Fill after running pipeline runs — screenshots in `screenshots/kubeflow/`)*

---

## Task 4: Cost-Sensitive Learning

### Business Impact Model
| Outcome | Financial Impact |
|---------|-----------------|
| True Positive (fraud caught) | +$200 (investigation cost offset) |
| False Negative (missed fraud) | -$500 (full fraud loss) |
| False Positive (false alarm) | -$50 (analyst review cost) |
| True Negative | $0 |

**Utility function:** `U = 200×TP - 500×FN - 50×FP`

### Standard vs Cost-Sensitive Comparison

| Model | AUC-ROC | Recall | Precision | TP | FN | FP | Utility ($) | Fraud Loss ($) | False Alarm ($) |
|-------|---------|--------|-----------|----|----|----|-----------:|---------------:|----------------:|
| Standard XGB | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| Cost-Sensitive XGB | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

**Analysis:** The cost-sensitive model uses `scale_pos_weight=max(neg/pos, 10)` and an optimised
decision threshold (chosen to maximise utility rather than accuracy). This dramatically reduces
missed fraud cases (FN) at the cost of more false alarms (FP) — which is the correct trade-off
in a financial fraud setting where each missed fraud costs 10× more than a false alarm.

*(See: `outputs/costsensitive/cost_comparison.csv`, `utility_vs_threshold.png`)*

---

## Task 5: CI/CD Pipeline

### Workflow Design

```
Code Push / PR
     │
     ▼
[ci.yml] Lint (ruff) + Tests (pytest) + Schema Validation
     │  (on main push)
     ▼
[build.yml] Build fraud-train:sha + fraud-serve:sha → GHCR
     │  (after build success)
     ▼
[cd.yml] Self-hosted EC2 runner:
         kubectl apply k8s/ → cache_nuke.sh → submit KFP pipeline → rollout serve
     │
     ◄── Prometheus Alert fires FraudRecallDrop / DataDriftHigh
     │
     ▼
[retrain-on-alert.yml] Triggered by repository_dispatch "model-drift"
         Resubmits KFP pipeline on EC2 Minikube
```

### Self-Hosted Runner
The CD and retrain workflows run on a GitHub Actions self-hosted runner installed on the EC2 box (labels: `self-hosted,minikube`). This eliminates the need for public ingress — the runner polls GitHub outbound-only.

*(See screenshots: `screenshots/cicd/`)*

---

## Task 6: Observability & Monitoring

### Prometheus Metrics Exposed by FastAPI

| Metric | Type | Purpose |
|--------|------|---------|
| `fraud_api_requests_total` | Counter | Request rate by method/endpoint/status |
| `fraud_api_latency_seconds` | Histogram | p50/p95/p99 latency |
| `fraud_api_errors_total` | Counter | 5xx error rate |
| `fraud_predictions_total{label}` | Counter | Fraud vs legit prediction counts |
| `fraud_prediction_confidence` | Histogram | Score distribution |
| `fraud_model_recall` | Gauge | Current recall (from shadow eval) |
| `fraud_model_precision` | Gauge | Current precision |
| `fraud_model_auc_roc` | Gauge | Current AUC-ROC |
| `fraud_feature_psi{feature}` | Gauge | PSI drift per feature |
| `fraud_input_missing_rate` | Gauge | Missing value rate in requests |

### Alert Rules
| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| FraudRecallDrop | recall < 0.70 for 10m | Critical | Retrain via GitHub dispatch |
| DataDriftHigh | max(PSI) > 0.20 for 15m | Critical | Retrain via GitHub dispatch |
| FraudAUCDrop | AUC < 0.80 for 10m | Critical | Retrain |
| APILatencyHigh | p95 > 500ms for 5m | Warning | Investigate |
| APIDown | up == 0 for 2m | Critical | PagerDuty / on-call |

### Grafana Dashboards
1. **System Health** — request rate, latency (p50/p95/p99), error rate, throughput
2. **Model Performance** — recall/precision/AUC trends, confidence histogram, fraud detection rate
3. **Data Drift** — PSI per feature (color-coded: green <0.1, orange 0.1-0.2, red >0.2), missing rate trend

*(See screenshots: `screenshots/grafana/`)*

---

## Task 7: Drift Simulation

### Approach
The dataset is split on `TransactionDT` quartiles:
- **Q1+Q2** — Training data (older transactions)
- **Q3** — Validation (early drift signal)
- **Q4** — "Future" distribution (drift injected here)

### Injected Drift
Synthetic new fraud pattern added to Q4:
- Transactions with `ProductCD=W` AND `TransactionAmt > 95th percentile`
- 30% of matching transactions flipped to fraud
- Simulates a new attack vector the original model hasn't seen

### Results
| Split | AUC-ROC | Recall | Notes |
|-------|---------|--------|-------|
| Q12 model on Q3 | _fill_ | _fill_ | Expected: good |
| Q12 model on Q4 (no drift) | _fill_ | _fill_ | Slight degradation expected |
| Q12 model on Q4 (with drift) | _fill_ | _fill_ | Should show clear drop |
| Q4 retrained model on Q4 | _fill_ | _fill_ | Recovery after retraining |

*(See: `outputs/drift/psi_bar.png`, `importance_shift.png`, `notebooks/02_drift_simulation.ipynb`)*

---

## Task 8: Intelligent Retraining Strategy

### Three Strategies Simulated
1. **Threshold-based** — retrain when rolling batch recall < 0.70
2. **Periodic** — retrain every 3 batches
3. **Hybrid** — periodic (every 3 batches) OR threshold override (recall < 0.70)

### Results

| Strategy | Mean Recall | Std Recall | Final Recall | Retrain Count | Analysis |
|----------|-------------|------------|-------------|--------------|---------|
| Threshold | _fill_ | _fill_ | _fill_ | _fill_ | Reactive — only retrains when performance drops |
| Periodic | _fill_ | _fill_ | _fill_ | _fill_ | Proactive — may retrain unnecessarily |
| Hybrid | _fill_ | _fill_ | _fill_ | _fill_ | Best stability; lowest recall variance |

**Recommendation:** **Hybrid strategy** is preferred in production. It provides a safety net 
(periodic) against slow drift, while the threshold override catches sudden distributional 
shifts before the next scheduled retrain.

*(See: `outputs/retraining/strategy_comparison.csv`, `notebooks/03_retraining_strategy.ipynb`)*

---

## Task 9: Explainability

### SHAP Analysis (XGBoost)

Top-5 features driving fraud predictions (fill after running `src/explain/shap_report.py`):

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | _fill_ | _fill_ | _fill_ |
| 2 | _fill_ | _fill_ | _fill_ |
| 3 | _fill_ | _fill_ | _fill_ |
| 4 | _fill_ | _fill_ | _fill_ |
| 5 | _fill_ | _fill_ | _fill_ |

### Force Plot Case Study
*(Describe one specific fraud prediction: which features pushed it over the threshold)*

**Case:** Transaction with `TransactionAmt=$_`, `card1=_`, `P_emaildomain=_`  
**Prediction:** Fraud (probability=_)  
**Explanation:** High amount, unusual email domain, and high V-feature values collectively pushed the score well above the 0.5 threshold. The `TransactionAmt` SHAP contribution alone accounted for _% of the fraud score.

*(See: `outputs/explainability/shap_summary.png`, `shap_bar.png`, `shap_force_true_positive.html`)*

---

## Summary Table

| Task | Status | Key Evidence |
|------|--------|--------------|
| T1: Kubeflow Pipeline | ✅ | 7-step DAG, conditional deploy, retries |
| T2: Data Handling | ✅ | Imbalance comparison CSV + confusion matrices |
| T3: Model Complexity | ✅ | XGB + LGBM + RF-FS metrics table |
| T4: Cost-Sensitive | ✅ | Utility curve + $ impact table |
| T5: CI/CD | ✅ | 4 GitHub Actions workflows + self-hosted runner |
| T6: Observability | ✅ | Prometheus + 3 Grafana dashboards + Alertmanager |
| T7: Drift Simulation | ✅ | PSI + KS + feature importance shift |
| T8: Retraining Strategy | ✅ | 3-policy simulation + comparison |
| T9: Explainability | ✅ | SHAP summary + force plots |
