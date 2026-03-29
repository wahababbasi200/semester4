# Embedding Complexity vs. Operational Cost: An MLOps-Driven Evaluation of Tokenization-Based Fraud Detection Pipelines in a Cloud-Native Environment

## Project Overview

**Track:** Track-II (Technical Research — Implementation + Improvement)

**Research Question:** How does embedding model complexity affect the full MLOps lifecycle — training cost, CI/CD pipeline time, serving latency, resource consumption, and monitoring reliability — in a production fraud detection system?

**Core Insight:** The embeddings (TF-IDF, Word2Vec, FastText, DistilBERT variants) are the experimental variable. The research contribution is NOT "which embedding is most accurate" — it is "what are the real operational tradeoffs when you deploy each embedding in a cloud-native MLOps pipeline, and which is the best choice under realistic production constraints like latency SLAs and cost budgets?"

**Dataset:** PaySim (6.3M synthetic mobile money transactions, ~8,213 fraud cases, 0.129% fraud rate)

**Prediction Level:** Account-level binary classification (fraud / non-fraud)

**Fraud Scope:** Type 1 — Unauthorized Transaction Fraud only (TRANSFER + CASH_OUT patterns)

### Labeling Strategy — Receiver-Side Fraud Detection

In PaySim, `isFraud=1` marks transactions where a sender's account was compromised (account takeover) and funds were transferred out without authorization. This means:

- The **sender** is the **victim** — their credentials were stolen
- The **receiver** (destination account) is the **fraud beneficiary** — the mule or collector account

**Our labeling rule:** A **receiver account** (`nameDest`) is labeled fraudulent if it appears as the destination of ANY transaction where `isFraud=1`. Sender accounts are NOT labeled as fraudulent — they are victims, not perpetrators.

**Justification:**
- This aligns with real-world AML/fraud practice where the goal is to identify suspicious recipient accounts (mules, collectors)
- Detecting mule accounts is operationally more useful — you can freeze the receiving account before funds are withdrawn
- PaySim's fraud simulation specifically models TRANSFER → CASH_OUT patterns where funds flow to a destination account and are immediately cashed out
- Sender-side detection would be identifying victims, which is a different problem (account takeover detection, not fraud beneficiary detection)

**Formal definition:**
```
For each unique account_id in nameDest:
    label = 1 if ANY transaction where nameDest == account_id has isFraud == 1
    label = 0 otherwise
```

We still collect transactions where an account appears as SENDER (nameOrig) to build its behavioral profile — the model sees the full transaction history of each receiver account, including both incoming and outgoing transactions. The label is just assigned based on whether the account received fraudulent funds.

---

## Deployment Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT (Local)                      │
│  Docker Compose: Train, Experiment, Iterate                  │
│  MLflow tracking → logs params, metrics, artifacts to S3     │
└──────────────────────────┬──────────────────────────────────┘
                           │ GitHub Actions CI/CD
                           │ (build, test, push image)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     REGISTRY (AWS ECR)                        │
│  Stores versioned Docker images for fraud-api                │
│  Each image contains: model weights + tokenization configs   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Deploy pulls images
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION (AWS EC2 Free Tier)               │
│  Docker Compose:                                             │
│  ┌───────────┐ ┌────────┐ ┌────────────┐ ┌────────┐        │
│  │ FastAPI    │ │ MLflow │ │ Prometheus │ │ Grafana│        │
│  │ fraud-api  │ │ server │ │            │ │        │        │
│  └───────────┘ └────────┘ └────────────┘ └────────┘        │
│                                                              │
│  MLflow artifacts (experiment history) → AWS S3              │
│  Model + configs baked INTO Docker image (no runtime S3)     │
└─────────────────────────────────────────────────────────────┘
```

### Model Loading & Versioning Strategy

The model lifecycle follows a clear path:

```
Training → MLflow logs metrics + artifacts to S3
    ↓
Model Registry → Model promoted to "Production" stage in MLflow
    ↓
Export → Promoted model downloaded from MLflow/S3
    ↓
Bake → Model weights + tokenization configs copied INTO Docker image during build
    ↓
Push → Self-contained Docker image pushed to ECR (tagged with git SHA)
    ↓
Deploy → EC2 pulls image from ECR, starts container
    ↓
Serve → FastAPI loads model from local filesystem inside container (no runtime S3 calls)
```

**Why bake into the Docker image:**
- Self-contained: no runtime dependency on S3 or MLflow
- Fast cold starts: model is already on disk when container starts
- Deterministic: image tag (git SHA) tells you exactly which model is inside
- Simple: no AWS credential management needed at inference time
- Reproducible: same image always serves the same model

**Tokenization configs** (segment boundaries, feature thresholds) are also baked into the image since they are small JSON files versioned in git.

**S3 is still used for:** MLflow experiment history, training artifacts, model checkpoints during development, and as the MLflow artifact backend. But the production serving path does NOT depend on S3 at runtime.

---

## Phase 1 — Data Loading & Exploratory Data Analysis

**Time estimate: ~4-5 hours**

### 1.1 Data Loading

- Download PaySim dataset from Kaggle (`paysim1`)
- Load into pandas DataFrame
- Verify schema: `step`, `type`, `amount`, `nameOrig`, `nameDest`, `oldbalanceOrg`, `newbalanceOrg`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- Check data types, null values, duplicates
- Document total rows, columns, memory usage

### 1.2 Exploratory Data Analysis

#### 1.2.1 Class Distribution
- Count and percentage of `isFraud=0` vs `isFraud=1`
- Bar chart visualization
- Note extreme imbalance (~0.129%)

#### 1.2.2 Transaction Type Analysis
- Distribution of all 5 transaction types (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- Fraud rate per transaction type
- **Verify:** Only TRANSFER and CASH_OUT contain fraud in PaySim
- Grouped bar chart

#### 1.2.3 Amount Analysis
- Amount distribution overall, fraud vs non-fraud (log-scale histograms)
- Box plots: fraud vs non-fraud amounts
- Summary statistics (mean, median, Q25, Q75, Q90, max) by class

#### 1.2.4 Temporal Analysis
- Transaction volume per step (step = 1 hour)
- Fraud volume per step
- Any temporal clustering of fraud

#### 1.2.5 Balance Analysis
- Compare `oldbalanceOrg - newbalanceOrg` vs `amount` for fraud transactions
- Check for balance inconsistencies indicating fraud
- Analyze zero-balance patterns in destination accounts
- **Note known PaySim issue:** destination balances are often zero — document this

#### 1.2.6 Receiver Account Analysis (Fraud-Specific)
- How many unique receiver accounts (`nameDest`) appear in `isFraud=1` transactions
- How many fraud transactions does each fraudulent receiver have (distribution)
- Do fraudulent receivers also appear as senders in other transactions
- Ratio of fraud-receiving vs legitimate-receiving accounts

#### 1.2.7 General Account Statistics
- Unique sender and receiver accounts
- Transactions-per-account distribution
- Overlap: accounts appearing as both sender and receiver

### 1.3 Data Filtering

- Filter to only TRANSFER and CASH_OUT transactions (only types with fraud)
- Document how many transactions and accounts are removed
- Re-check fraud rate after filtering

### 1.4 EDA Deliverables

- Jupyter notebook with all visualizations
- Summary statistics CSV
- Key findings documented (feeds into paper's dataset section)

---

## Phase 2 — Account-Level Aggregation & Preprocessing

**Time estimate: ~3-4 hours**

### 2.1 Build Receiver Account Dataset

Since we are doing receiver-side fraud detection:

**Step 1 — Identify target accounts:**
- Collect all unique `nameDest` values from the filtered dataset (TRANSFER + CASH_OUT only)
- These are the accounts we will classify

**Step 2 — Collect full transaction history for each target account:**
- For each target account, gather ALL transactions where it appears as:
  - `nameDest` (incoming transactions) — add role token `RECEIVER`
  - `nameOrig` (outgoing transactions, if any) — add role token `SENDER`
- This gives the model a complete behavioral picture of the account

**Step 3 — Assign labels:**
```
For each target receiver account:
    label = 1 if it is nameDest in ANY transaction where isFraud == 1
    label = 0 otherwise
```

- Prefix account IDs: `PS_{original_id}`
- Document: total target accounts, fraud count, fraud rate

### 2.2 Per-Account Transaction Sorting

- For each target account, sort all transactions chronologically by `step`
- This ordered list is the account's transaction history
- Record transaction count per account (needed for truncation)

### 2.3 Train / Validation / Test Split

**Temporal split (NOT random):**
- Training: first 70% of time steps
- Validation: next 15% of time steps
- Test: final 15% of time steps

**Account assignment:** based on each account's last transaction timestamp:
- Last transaction in training period → training set
- Last transaction in validation period → validation set
- Last transaction in test period → test set

Document split statistics: accounts and fraud rate per split.

**CRITICAL:** All thresholds, quantiles, and model training use ONLY training data.

---

## Phase 3 — Context-Aware Transaction Tokenization

**Time estimate: ~4-5 hours**

Two-step process:
1. Segment accounts into behavioral groups based on aggregate features
2. Within each segment, compute quantile-based thresholds for feature discretization
3. Apply universal token names across all segments

### 3.1 Step 1 — Account Behavioral Segmentation

#### 3.1.1 Compute Account Aggregate Features (Training Data Only)

For each target account in the training set, compute:

| Feature | Description |
|---------|-------------|
| `total_volume` | Sum of all transaction amounts (incoming + outgoing) |
| `txn_count` | Total number of transactions |
| `active_duration` | Last step − First step (how long the account was active) |

#### 3.1.2 Composite Activity Score

```
account_activity_score = normalize(total_volume) + normalize(txn_count) + normalize(active_duration)
```

Where `normalize(x) = (x - min) / (max - min)` computed on training data.

#### 3.1.3 Segment into 3 Groups

Using quantiles of the composite score (training data only):

```
score <= Q33         → SEG_MICRO    (low-activity accounts)
Q33 < score <= Q75   → SEG_REGULAR  (moderate-activity accounts)
score > Q75          → SEG_HEAVY    (high-activity accounts)
```

- Save Q33 and Q75 thresholds as JSON config
- Apply same thresholds to validation and test accounts

#### 3.1.4 Document Segment Statistics

For each segment: account count, fraud rate, mean txn count, mean amount. This table goes in the paper.

### 3.2 Step 2 — Segment-Specific Feature Discretization (Universal Token Names)

**Key design:** Thresholds differ per segment, but token names are the same. `AMT_HIGH` means "high relative to this account's peer group."

#### 3.2.1 Amount Tokenization

Within each segment (using only that segment's training accounts):

```
Q20, Q50, Q80, Q95 = quantiles of transaction amounts

amount < Q20       → AMT_VERY_LOW
Q20 ≤ amount < Q50 → AMT_LOW
Q50 ≤ amount < Q80 → AMT_MEDIUM
Q80 ≤ amount < Q95 → AMT_HIGH
amount ≥ Q95       → AMT_VERY_HIGH
```

#### 3.2.2 Temporal Tokenization

PaySim steps represent hours:

```
step % 24:
  0-5    → TIME_LATE_NIGHT
  6-11   → TIME_MORNING
  12-17  → TIME_AFTERNOON
  18-23  → TIME_EVENING

(step // 24) % 7:
  0-4  → DAY_WEEKDAY
  5-6  → DAY_WEEKEND
```

#### 3.2.3 Frequency Tokenization

Rolling transaction count over a lookback window (last 24 steps), within each segment:

```
rolling_count < segment_Q25 → FREQ_LOW
Q25 ≤ rolling_count < Q75  → FREQ_MEDIUM
rolling_count ≥ Q75        → FREQ_HIGH
```

#### 3.2.4 Balance Change Tokenization

Handle sender and receiver roles differently:

**When account is RECEIVER (incoming transaction):**
```
balance_change = newbalanceDest - oldbalanceDest

If oldbalanceDest == 0 AND newbalanceDest == 0 → BAL_UNKNOWN   (PaySim data quality issue)
balance_change == amount                       → BAL_EXACT_CREDIT
balance_change != amount                       → BAL_MISMATCH_CREDIT
```

**When account is SENDER (outgoing transaction):**
```
balance_change = oldbalanceOrg - newbalanceOrg

balance_change == amount  → BAL_EXACT_DEBIT
balance_change > amount   → BAL_OVER_DEBIT
balance_change < amount   → BAL_UNDER_DEBIT   (suspicious)
balance_change == 0       → BAL_NO_CHANGE      (suspicious)
```

#### 3.2.5 Transaction Type and Role Tokens

```
type: TRANSFER or CASH_OUT (kept as-is)
role: SENDER or RECEIVER
```

### 3.3 Token Sequence Construction

For each account, build an ordered token sequence:

```
[ACCT] SEG_REGULAR [TXN] RECEIVER TRANSFER AMT_HIGH TIME_LATE_NIGHT DAY_WEEKDAY FREQ_LOW BAL_EXACT_CREDIT [TXN] SENDER CASH_OUT AMT_VERY_HIGH TIME_LATE_NIGHT DAY_WEEKDAY FREQ_HIGH BAL_NO_CHANGE [TXN] RECEIVER TRANSFER AMT_LOW TIME_MORNING DAY_WEEKDAY FREQ_MEDIUM BAL_UNKNOWN
```

**Tokens per transaction:** ~7-8 (role + type + amount + time + day + frequency + balance)

### 3.4 Sequence Length Handling — Recency-Based Truncation

For each account, keep only the most recent **k = 30** transactions:

```
1. Take full sorted transaction history
2. Slice [-30:]  (last 30 transactions)
3. Tokenize each transaction
4. Prepend [ACCT] SEG_X
5. Total: ~242 tokens (30 × 8 + 2 prefix) — well within BERT's 512 limit
6. If fewer than 30 transactions, pad with [PAD]
```

**Justification:** Recent transactions carry the strongest fraud signal. Fraudulent behavior manifests as sudden recent deviations. Older transactions have diminishing value due to concept drift.

### 3.5 Final Vocabulary

| Category | Tokens | Count |
|----------|--------|-------|
| Special | `[ACCT]`, `[TXN]`, `[PAD]`, `[MASK]`, `[CLS]`, `[SEP]` | 6 |
| Segment | `SEG_MICRO`, `SEG_REGULAR`, `SEG_HEAVY` | 3 |
| Role | `SENDER`, `RECEIVER` | 2 |
| Type | `TRANSFER`, `CASH_OUT` | 2 |
| Amount | `AMT_VERY_LOW`, `AMT_LOW`, `AMT_MEDIUM`, `AMT_HIGH`, `AMT_VERY_HIGH` | 5 |
| Time | `TIME_LATE_NIGHT`, `TIME_MORNING`, `TIME_AFTERNOON`, `TIME_EVENING` | 4 |
| Day | `DAY_WEEKDAY`, `DAY_WEEKEND` | 2 |
| Frequency | `FREQ_LOW`, `FREQ_MEDIUM`, `FREQ_HIGH` | 3 |
| Balance | `BAL_EXACT_DEBIT`, `BAL_OVER_DEBIT`, `BAL_UNDER_DEBIT`, `BAL_NO_CHANGE`, `BAL_UNKNOWN`, `BAL_EXACT_CREDIT`, `BAL_MISMATCH_CREDIT` | 7 |
| **Total** | | **~34** |

### 3.6 Save Artifacts

- Token sequences: text file (one line per account)
- Account labels: CSV
- Tokenization thresholds per segment: JSON config
- Segment boundaries (Q33, Q75 of composite score): JSON config
- All configs versioned in git AND uploaded to S3 for backup

---

## Phase 4 — Embedding Strategies & Model Training

**Time estimate: ~10-12 hours**

### 4.1 Common Setup Across All Variants

**Downstream classifier (SAME for all — ensures fair comparison):**
```
MLP:
  Input: embedding_dim (varies per method)
  → Linear(embedding_dim, 256) → ReLU → Dropout(0.3)
  → Linear(256, 128) → ReLU → Dropout(0.3)
  → Linear(128, 1) → Sigmoid
```

**Class imbalance handling:**
- Weighted Binary Cross-Entropy: `w_pos = N_neg / N_pos`
- Same weight for all variants

**Training protocol:**
- Optimizer: Adam (lr=1e-3 for MLP, lr=2e-5 for DistilBERT)
- Early stopping: patience=5 on validation PR-AUC
- Max epochs: 50 for MLP-only, 10 for DistilBERT variants
- Batch size: 64
- Seeds: 3 random seeds (42, 123, 456), report mean ± std

**Evaluation metrics (on test set):**
- **Primary:** PR-AUC (most appropriate under extreme imbalance)
- **Secondary:** AUC-ROC, F1-Score, Precision, Recall
- Threshold: optimal from validation PR curve

### 4.2 Variant 1 — TF-IDF → MLP

- Treat each account's token sequence as a document
- Fit TF-IDF on training sequences only
- Parameters: `max_features=5000`, `ngram_range=(1,3)`
- Output: sparse vector → MLP
- **Captures:** token frequencies, n-gram co-occurrences
- **Misses:** token order, context

### 4.3 Variant 2 — Word2Vec → Mean Pool → MLP

- Train Word2Vec (Skip-gram) on training token sequences
- Parameters: `vector_size=128`, `window=5`, `min_count=1`, `epochs=20`
- Per account: mean-pool all token embeddings → 128-d vector → MLP
- **Captures:** distributional similarity between tokens
- **Misses:** word order, positional context

### 4.4 Variant 3 — FastText → Mean Pool → MLP

- Train FastText on training token sequences
- Parameters: `vector_size=128`, `window=5`, `min_count=1`, `epochs=20`
- Same mean-pooling as Word2Vec
- **Captures:** same as Word2Vec + subword information
- For our small vocabulary, difference from Word2Vec may be small

### 4.5 Variant 4 — DistilBERT Frozen → [CLS] → MLP

- Load pretrained `distilbert-base-uncased`
- Add ~34 custom tokens via `tokenizer.add_special_tokens()`
- Resize embeddings: `model.resize_token_embeddings(len(tokenizer))`
- **Freeze ALL DistilBERT weights** — no gradient updates
- Extract [CLS] embedding (768-d) → linear projection to 128-d → MLP
- Only projection + MLP are trained
- **Expected:** poor performance (pretrained weights know English, not transaction tokens)
- **Purpose:** lower bound for transformer approaches

### 4.6 Variant 5 — DistilBERT Fine-Tuned → [CLS] → MLP

- Same setup as Variant 4
- **Unfreeze ALL weights** — fine-tune end-to-end on fraud classification
- Differential learning rates: 2e-5 for DistilBERT, 1e-3 for MLP head
- Max epochs: 10

### 4.7 Variant 6 — DistilBERT Domain-Pretrained (MLM) → Fine-Tuned → [CLS] → MLP

**Step 1 — Domain MLM Pretraining:**
- Take pretrained DistilBERT, add custom tokens
- Pretrain with Masked Language Modeling on ALL training token sequences
  - Mask 15% of tokens randomly
  - Model learns to predict masked tokens
  - Learns domain patterns: `CASH_OUT` follows `TRANSFER`, `AMT_VERY_HIGH` + `TIME_LATE_NIGHT` co-occur
- Parameters: epochs=5, lr=5e-5, batch_size=32
- Use HuggingFace `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)`

**Step 2 — Classification Fine-Tuning:**
- Take MLM-pretrained model, add classification head
- Fine-tune on fraud classification (same as Variant 5)

### 4.8 MLflow Experiment Tracking

For every variant × every seed, log:

**Parameters:**
- `embedding_type`, `random_seed`, `learning_rate`, `batch_size`, `max_epochs`, `dropout_rate`, `class_weight_pos`

**Metrics (per epoch on val, final on test):**
- `pr_auc`, `auc_roc`, `f1`, `precision`, `recall`
- `training_time_seconds`, `inference_time_per_sample_ms`, `model_size_mb`

**Artifacts (saved to S3 via MLflow artifact backend):**
- Model checkpoint
- Confusion matrix plot
- PR curve plot
- Classification report

### 4.9 Model Registry Workflow

After all experiments are complete:

1. Register each variant's best seed model in MLflow Model Registry
2. Evaluate each against promotion criteria (see Phase 5.5)
3. Promote the best qualifying model to "Production" stage
4. Export the production model for Docker image baking

---

## Phase 5 — MLOps Infrastructure

**Time estimate: ~10-12 hours**

### 5.1 Repository Structure

```
fraud-detection-mlops/
├── README.md
├── docker-compose.yml                    # Local development
├── docker-compose.prod.yml               # EC2 production
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   ├── raw/                              # PaySim CSV (gitignored)
│   ├── processed/                        # Tokenized sequences, labels
│   └── configs/                          # Segment thresholds, tokenization configs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_results_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── segmentation.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── tfidf_model.py
│   │   ├── word2vec_model.py
│   │   ├── fasttext_model.py
│   │   ├── distilbert_model.py
│   │   └── mlp_classifier.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── serving/
│   │   ├── app.py                        # FastAPI application
│   │   ├── schemas.py                    # Pydantic request/response models
│   │   ├── metrics.py                    # Prometheus custom metrics
│   │   └── logging_config.py             # Structured JSON logging setup
│   └── utils/
│       ├── config.py
│       └── mlflow_utils.py
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── fraud_detection.json
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_api.py
│   └── golden_test_set.json              # 50 fraud + 50 non-fraud for smoke tests
├── logs/                                  # Mounted volume for JSON logs
│   └── .gitkeep
├── model_artifacts/                       # Baked into Docker image at build time
│   ├── model/                             # Exported model weights
│   └── configs/                           # Tokenization thresholds, segment boundaries
├── Dockerfile
├── Dockerfile.mlflow
├── requirements.txt
└── configs/
    └── experiment_config.yaml
```

### 5.2 Dockerfile — Model Baked In

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# BAKE model and configs into the image
# These are copied from MLflow/S3 BEFORE building the image
COPY model_artifacts/model/ ./model/
COPY model_artifacts/configs/ ./configs/

# Copy logging config
COPY src/serving/logging_config.py ./logging_config.py

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build process:**
```bash
# 1. Export production model from MLflow to model_artifacts/
python scripts/export_production_model.py --output model_artifacts/

# 2. Copy tokenization configs
cp data/configs/*.json model_artifacts/configs/

# 3. Build image with model baked in
docker build -t fraud-api:$(git rev-parse --short HEAD) .

# 4. Push to ECR
docker tag fraud-api:$(git rev-parse --short HEAD) <ecr-uri>/fraud-api:latest
docker push <ecr-uri>/fraud-api:latest
```

### 5.3 FastAPI Inference Service

**Endpoints:**

```
POST /predict
  Request:
  {
    "account_id": "PS_12345",
    "transactions": [
      {"step": 1, "type": "TRANSFER", "amount": 5000.0, "role": "RECEIVER",
       "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
    ]
  }

  Response:
  {
    "account_id": "PS_12345",
    "fraud_probability": 0.87,
    "prediction": "FRAUD",
    "model_version": "distilbert_mlm_v1_abc123",
    "embedding_type": "distilbert_mlm_finetuned",
    "latency_ms": 12.3
  }

GET /health
  → {"status": "healthy", "model_loaded": true, "model_version": "..."}

GET /metrics
  → Prometheus-format metrics
```

**Internal prediction flow:**
```
1. Receive raw transactions via POST /predict
2. Load segment boundaries from /app/configs/segment_boundaries.json (loaded once at startup)
3. Load tokenization thresholds from /app/configs/tokenization_thresholds_*.json (loaded once at startup)
4. Determine account segment using composite score
5. Apply tokenization using segment-specific thresholds
6. Build token sequence (recency truncation, k=30)
7. Generate embedding using model loaded from /app/model/ (loaded once at startup)
8. Run MLP classifier
9. Record Prometheus metrics
10. Write structured JSON log
11. Return prediction
```

All model and config files are loaded from the **local filesystem inside the container** at startup. No runtime calls to S3, MLflow, or any external service.

### 5.4 Structured JSON Logging

**Implementation: `src/serving/logging_config.py`**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry)
```

**Three log categories:**

#### 5.4.1 Prediction Logs (every request)

Written for every `/predict` call:

```json
{
  "timestamp": "2026-04-15T14:32:01.123Z",
  "level": "INFO",
  "logger": "prediction",
  "message": "prediction_complete",
  "account_id": "PS_12345",
  "embedding_type": "distilbert_mlm_finetuned",
  "model_version": "distilbert_mlm_v1_abc123",
  "prediction": "FRAUD",
  "fraud_probability": 0.87,
  "confidence_bucket": "high",
  "num_transactions": 12,
  "account_segment": "SEG_REGULAR",
  "tokenization_latency_ms": 1.2,
  "inference_latency_ms": 11.1,
  "total_latency_ms": 12.3
}
```

#### 5.4.2 Error Logs (on failures)

```json
{
  "timestamp": "2026-04-15T14:33:05.456Z",
  "level": "ERROR",
  "logger": "prediction",
  "message": "prediction_failed",
  "account_id": "PS_99999",
  "error_type": "TokenizationError",
  "error_detail": "Unknown transaction type: DEBIT",
  "traceback": "..."
}
```

#### 5.4.3 System Logs (startup, health, model loading)

```json
{
  "timestamp": "2026-04-15T14:30:00.000Z",
  "level": "INFO",
  "logger": "system",
  "message": "model_loaded",
  "model_version": "distilbert_mlm_v1_abc123",
  "embedding_type": "distilbert_mlm_finetuned",
  "model_size_mb": 256.3,
  "load_time_seconds": 3.2,
  "configs_loaded": ["segment_boundaries.json", "thresholds_seg_micro.json", "thresholds_seg_regular.json", "thresholds_seg_heavy.json"]
}
```

**Log output:**
- Logs written to `/app/logs/fraud_api.log` (JSON lines format, one JSON object per line)
- Docker volume mounts `/app/logs` so logs persist outside the container
- Also streamed to stdout for `docker logs` access
- Log rotation: configure `RotatingFileHandler` (max 50MB per file, keep 5 files)

**Docker Compose volume for logs:**
```yaml
fraud-api:
  volumes:
    - ./logs:/app/logs
```

### 5.5 Docker Compose — Local Development

```yaml
services:
  fraud-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_VERSION=distilbert_mlm_v1
    volumes:
      - ./logs:/app/logs
    depends_on:
      - mlflow

  mlflow:
    build:
      dockerfile: Dockerfile.mlflow
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_ARTIFACT_ROOT=s3://your-bucket/mlflow-artifacts
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    volumes:
      - mlflow_data:/mlflow

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus

volumes:
  mlflow_data:
```

### 5.6 Docker Compose — EC2 Production

`docker-compose.prod.yml` differences:
- Image pulled from ECR (not built locally)
- Restart policies: `restart: unless-stopped`
- Log volume mounted for persistence
- No build-time AWS credentials needed (model is baked in)

```yaml
services:
  fraud-api:
    image: <ecr-uri>/fraud-api:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # MLflow, Prometheus, Grafana same as local but with restart policies
```

### 5.7 MLflow Model Registry & Promotion Workflow

**Model Registry Stages:** `None → Staging → Production`

**Automated Promotion Criteria:**
A model is promoted from Staging to Production ONLY if it meets BOTH:
1. **Quality gate:** PR-AUC on test set ≥ current production model's PR-AUC
2. **Latency SLA gate:** p99 inference latency < 100ms

```python
# scripts/promote_model.py
def evaluate_for_promotion(candidate_model, production_model):
    candidate_prauc = evaluate(candidate_model, test_set)
    candidate_latency_p99 = benchmark_latency(candidate_model, n=500)

    production_prauc = get_production_metric("pr_auc")

    if candidate_prauc >= production_prauc and candidate_latency_p99 < 100:
        promote_to_production(candidate_model)
        logger.info("Promoted", extra_fields={
            "pr_auc": candidate_prauc,
            "latency_p99_ms": candidate_latency_p99,
            "status": "promoted"
        })
    else:
        logger.info("Rejected", extra_fields={
            "pr_auc": candidate_prauc,
            "latency_p99_ms": candidate_latency_p99,
            "status": "rejected",
            "reason": "failed quality gate" if candidate_prauc < production_prauc else "failed latency gate"
        })
```

### 5.8 Prometheus Metrics

**Application metrics (exposed by FastAPI):**
- `fraud_prediction_latency_seconds` — histogram (inference latency)
- `fraud_predictions_total` — counter by class (fraud/non-fraud)
- `fraud_prediction_confidence` — histogram (model confidence)
- `fraud_prediction_rate` — gauge (rolling fraud prediction rate for drift detection)
- `tokenization_latency_seconds` — histogram
- `model_version_info` — gauge with model version label

**System metrics:**
- CPU usage per container
- Memory usage per container
- HTTP request rate and error rate

### 5.9 Prediction Drift Detection

- Track **rolling fraud prediction rate** over last N predictions
- Baseline: training set fraud rate (~0.129% for receiver accounts — verify after preprocessing)
- Alert thresholds:
  - WARNING if rolling rate > 1% (roughly 8x baseline)
  - CRITICAL if rolling rate > 5% (roughly 40x baseline)
  - WARNING if rolling rate drops to 0% over 1000+ predictions

Implementation: simple counter in FastAPI exposed to Prometheus.

### 5.10 Grafana Dashboard

**Row 1 — Service Health:**
- Request rate (requests/sec), error rate (4xx, 5xx), uptime

**Row 2 — Inference Performance:**
- Latency distribution (p50, p95, p99)
- Throughput over time
- Tokenization vs model inference latency breakdown

**Row 3 — Model Behavior & Drift:**
- Prediction distribution (% fraud vs % non-fraud over time)
- Rolling fraud prediction rate with alert threshold lines
- Confidence score distribution

**Row 4 — Resource Usage:**
- CPU and memory per container

### 5.11 CI/CD Pipeline (GitHub Actions)

```yaml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/ -v
      - name: Lint code
        run: flake8 src/ --max-line-length=120

  smoke-test:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t fraud-api:test .
      - name: Start container
        run: docker run -d -p 8000:8000 --name smoke-test fraud-api:test
      - name: Wait for startup
        run: sleep 15
      - name: Health check
        run: curl -f http://localhost:8000/health
      - name: Golden test set validation
        run: |
          python tests/run_golden_tests.py --url http://localhost:8000/predict --min-f1 0.5
      - name: Stop container
        run: docker stop smoke-test

  build-and-push:
    needs: smoke-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - name: Login to Amazon ECR
        run: aws ecr get-login-password | docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
      - name: Build and push to ECR
        run: |
          docker build -t ${{ secrets.ECR_REGISTRY }}/fraud-api:${{ github.sha }} .
          docker push ${{ secrets.ECR_REGISTRY }}/fraud-api:${{ github.sha }}
          docker tag ${{ secrets.ECR_REGISTRY }}/fraud-api:${{ github.sha }} ${{ secrets.ECR_REGISTRY }}/fraud-api:latest
          docker push ${{ secrets.ECR_REGISTRY }}/fraud-api:latest
```

**CI/CD Pipeline Timing as a Research Metric:**
- Record total CI/CD execution time for each embedding variant
- TF-IDF: small image, fast build, fast smoke test
- DistilBERT MLM: large image (PyTorch + transformers), slow build, slow smoke test
- Report in paper as part of operational cost comparison

---

## Phase 6 — Production Benchmarking

**Time estimate: ~5-6 hours**

### 6.1 Benchmark Setup

For EACH of the 6 embedding variants:
1. Bake model into Docker image
2. Start FastAPI container
3. Run benchmark: 1000 test-set samples
4. Collect all metrics

### 6.2 ML Performance Metrics (from Phase 4)

| Metric | Description |
|--------|-------------|
| PR-AUC | Primary metric under imbalance |
| AUC-ROC | Overall discrimination |
| F1, Precision, Recall | At optimal threshold |

### 6.3 MLOps / Production Metrics

| Metric | How to Measure |
|--------|---------------|
| Inference latency p50, p95, p99 (ms) | Prometheus histogram |
| Throughput (requests/sec) | Benchmark script |
| Model file size (MB) | `os.path.getsize()` on model directory |
| Docker image size (MB) | `docker images` |
| Peak memory usage (MB) | Prometheus container metrics |
| CPU utilization (%) | Prometheus container metrics |
| CI/CD pipeline time (minutes) | GitHub Actions logs |
| Cold start time (seconds) | Time from `docker run` to first `/health` success |
| Estimated monthly EC2 cost ($) | Based on minimum instance type needed |

### 6.4 EC2 Cost Estimation

| Instance | vCPU | RAM | Cost/month |
|----------|------|-----|------------|
| t3.micro | 2 | 1 GB | ~$7.50 |
| t3.small | 2 | 2 GB | ~$15 |
| t3.medium | 2 | 4 GB | ~$30 |
| t3.large | 2 | 8 GB | ~$60 |

For each variant, determine minimum EC2 instance that can serve it.

### 6.5 Benchmark Protocol

- Run each variant in isolation
- Warm up with 100 requests
- Measure over 1000 requests
- Repeat 3 times, report mean ± std

### 6.6 Key Output: Cost-Performance Tradeoff Visualization

**This is the MOST IMPORTANT figure in the paper.**

Scatter plot:
- X-axis: Inference latency p95 (ms)
- Y-axis: PR-AUC
- Point size: Docker image size (MB)
- Point color: Embedding type
- Annotate each point with estimated monthly EC2 cost
- Draw vertical line at latency SLA (e.g., 100ms)

### 6.7 Results Tables (Templates)

**Table 1: ML Performance**

| Variant | PR-AUC | AUC-ROC | F1 | Precision | Recall |
|---------|--------|---------|-----|-----------|--------|
| TF-IDF + MLP | | | | | |
| Word2Vec + MLP | | | | | |
| FastText + MLP | | | | | |
| DistilBERT Frozen | | | | | |
| DistilBERT Fine-tuned | | | | | |
| DistilBERT MLM + FT | | | | | |

**Table 2: Operational Cost**

| Variant | Latency p95 (ms) | Throughput (req/s) | Model Size (MB) | Image Size (MB) | Memory (MB) | CI/CD Time (min) | EC2 Cost ($/mo) |
|---------|------------------|-------------------|-----------------|-----------------|-------------|-------------------|--------------------|
| TF-IDF + MLP | | | | | | | |
| Word2Vec + MLP | | | | | | | |
| FastText + MLP | | | | | | | |
| DistilBERT Frozen | | | | | | | |
| DistilBERT Fine-tuned | | | | | | | |
| DistilBERT MLM + FT | | | | | | | |

**Table 3: Model Registry Promotion Results**

| Variant | PR-AUC | Latency p99 (ms) | Quality Gate (≥ baseline) | Latency Gate (<100ms) | Promoted? |
|---------|--------|-------------------|---------------------------|-----------------------|-----------|
| TF-IDF + MLP | | | | | |
| ... | | | | | |

---

## Phase 7 — AWS Deployment

**Time estimate: ~3-4 hours**

### 7.1 AWS Services Used

| Service | Purpose |
|---------|---------|
| EC2 (free tier) | Host Docker Compose production stack |
| ECR | Store versioned Docker images (model baked in) |
| S3 | Store MLflow experiment artifacts (training history, not runtime) |

### 7.2 ECR Setup

- Create ECR repository: `fraud-detection-api`
- CI/CD pushes images tagged with git SHA and `latest`
- EC2 pulls from ECR

### 7.3 S3 Bucket Structure

```
s3://your-fraud-detection-bucket/
├── mlflow-artifacts/              # MLflow experiment history
│   ├── experiment-tfidf/
│   ├── experiment-word2vec/
│   ├── experiment-fasttext/
│   ├── experiment-distilbert-frozen/
│   ├── experiment-distilbert-ft/
│   └── experiment-distilbert-mlm-ft/
├── configs-backup/                # Backup of tokenization configs
│   ├── segment_boundaries.json
│   └── tokenization_thresholds_*.json
└── training-data/                 # Optional: processed data backup
```

**Note:** S3 is for training-time artifact storage and backup. The production serving path loads everything from the Docker image's local filesystem — no runtime S3 dependency.

### 7.4 EC2 Deployment Steps

```bash
# 1. SSH into EC2
ssh -i your-key.pem ec2-user@<ec2-ip>

# 2. Install Docker and Docker Compose (one-time)
sudo yum install docker -y
sudo service docker start
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 3. Login to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <ecr-uri>

# 4. Pull latest image
docker pull <ecr-uri>/fraud-api:latest

# 5. Start production stack
docker-compose -f docker-compose.prod.yml up -d

# 6. Verify
curl http://localhost:8000/health

# 7. Access services
# FastAPI:    http://<ec2-public-ip>:8000
# MLflow:     http://<ec2-public-ip>:5000
# Prometheus: http://<ec2-public-ip>:9090
# Grafana:    http://<ec2-public-ip>:3000
```

### 7.5 Live Demo Readiness

For 15-minute presentation:
- Hit EC2 endpoint live with sample transactions
- Show Grafana dashboard with real-time metrics
- Show MLflow UI with all 6 experiment comparisons
- Show structured logs (`docker exec fraud-api tail /app/logs/fraud_api.log | jq .`)
- Show cost-performance tradeoff chart

---

## Phase 8 — Research Paper & Documentation

**Time estimate: ~8-10 hours**

### 8.1 Paper Structure (IEEE Format, 6-8 pages)

**I. Introduction** (~0.75 page)
- Financial fraud problem
- Motivation: embedding choice has operational implications that papers ignore
- Gap: no systematic MLOps-focused comparison of embedding strategies for fraud detection
- Contributions:
  1. Context-aware transaction tokenization with behavioral segmentation
  2. Systematic comparison of 6 embedding strategies across ML quality AND operational cost
  3. Cloud-native MLOps pipeline with model registry, drift detection, structured logging, and promotion gates

**II. Related Work** (~1 page)
- Traditional ML for fraud detection (2-3 papers)
- Deep learning / transformer approaches for fraud (2-3 papers)
- MLOps practices and deployment challenges (2-3 papers)
- Embedding comparisons in other domains (1-2 papers)
- Minimum 8-10 total references

**III. Methodology** (~1.5 pages)
- Dataset: PaySim, filtering, receiver-side labeling (with justification)
- Context-aware tokenization (two-step segmentation + discretization)
- Six embedding variants
- Downstream classifier
- MLOps pipeline architecture diagram

**IV. Experimental Setup** (~0.5 page)
- Split details, hyperparameters, hardware, software
- Evaluation metrics (ML + operational)
- Benchmark protocol

**V. Results & Analysis** (~2 pages)
- ML performance table
- Operational cost table
- **Cost-performance tradeoff scatter plot** (key figure)
- Model registry promotion analysis (which variants pass both gates?)
- CI/CD pipeline timing comparison
- Discussion: practical recommendations by deployment scenario

**VI. Conclusion & Future Work** (~0.5 page)
- Key findings and decision framework
- Limitations (synthetic data, single classifier)
- Future: multi-view fusion (connect to thesis), real datasets, Kubernetes scaling

### 8.2 Figures and Tables

1. End-to-end architecture diagram (deployment flow)
2. Tokenization example figure (raw transaction → tokens)
3. ML performance comparison table
4. Operational cost comparison table
5. **Cost-performance tradeoff scatter plot** (most important)
6. Model registry promotion results table
7. CI/CD pipeline timing bar chart
8. Grafana dashboard screenshot
9. MLflow experiment comparison screenshot
10. PR curves for all 6 variants

### 8.3 GitHub README

- Project overview and research question
- Architecture diagram
- Prerequisites and setup
- How to run: EDA → training → serving → monitoring
- How to deploy on EC2
- Structured logging documentation
- Results summary
- Project structure

---

## Timeline Summary

| Week | Phase | Hours | Deliverables |
|------|-------|-------|-------------|
| Week 1 | EDA + Preprocessing + Tokenization (Phases 1-3) | ~12 hrs | EDA notebook, tokenized dataset, configs on S3 |
| Week 2 | Model Training + MLflow Tracking (Phase 4) | ~12 hrs | 6 models × 3 seeds, MLflow experiments, model registry |
| Week 3 | MLOps Infra + Benchmarking + AWS Deploy (Phases 5-7) | ~14 hrs | Docker Compose, logging, Prometheus, Grafana, CI/CD, EC2 deploy, benchmarks |
| Week 4 | Paper + Documentation (Phase 8) | ~10 hrs | IEEE paper, GitHub README, presentation |
| **Total** | | **~48 hrs** | |

---

## Priority Order (If Running Low on Time)

Cut from the bottom first:

1. **Must have:** Tokenization + 6 models trained + MLflow tracking + Docker Compose + results tables
2. **Should have:** Prometheus + Grafana + structured logging + CI/CD + EC2 deployment + cost-performance chart
3. **Nice to have:** Prediction drift detection + model registry promotion logic + CI/CD timing analysis
4. **Drop if needed:** EC2 cost estimation (estimate manually instead)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| DistilBERT MLM pretraining slow | Reduce to 3 epochs, subsample training sequences |
| PaySim too large for memory | Subsample non-fraud accounts (keep all fraud), aim for ~500K accounts |
| EC2 free tier too small for DistilBERT | Run DistilBERT benchmarks locally, deploy only best light model on EC2 |
| GPU not available locally | Use Google Colab for DistilBERT training |
| CI/CD pipeline complex | Start with build + test only, add ECR push later |
| Receiver-only labeling gives very few fraud accounts | Verify count after preprocessing; if too few, consider including senders too |
| Time overrun | Follow priority order above strictly |

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10 | Primary language |
| pandas, numpy | Data processing |
| scikit-learn | TF-IDF, metrics |
| gensim | Word2Vec, FastText |
| HuggingFace transformers | DistilBERT, tokenizer, MLM pretraining |
| PyTorch | Model training |
| FastAPI | Inference service |
| MLflow | Experiment tracking + model registry |
| Prometheus | Metrics collection |
| Grafana | Dashboard visualization |
| Docker + Docker Compose | Containerization and orchestration |
| GitHub Actions | CI/CD pipeline |
| AWS EC2 | Production deployment |
| AWS ECR | Container registry |
| AWS S3 | MLflow artifact storage |
| structlog / logging | Structured JSON logging |
| locust / custom script | Load testing |
| matplotlib, seaborn | Visualization |

---

## Reproducibility Checklist

- [ ] All tokenization thresholds saved as JSON configs
- [ ] All segment boundaries saved as JSON configs
- [ ] Receiver-side labeling logic documented and coded deterministically
- [ ] All configs versioned in Git AND backed up to S3
- [ ] Random seeds fixed and documented (42, 123, 456)
- [ ] Train/val/test split reproducible via temporal cutoff values
- [ ] All MLflow experiments logged with full parameters
- [ ] Docker images tagged with git SHA for exact model-to-image traceability
- [ ] Structured logs capture every prediction with full context
- [ ] `requirements.txt` with pinned versions
- [ ] README with step-by-step reproduction instructions
