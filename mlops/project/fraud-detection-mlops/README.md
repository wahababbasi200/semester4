# Embedding Complexity vs. Operational Cost
## An MLOps-Driven Evaluation of Tokenization-Based Fraud Detection Pipelines

**Research Question:** How does embedding model complexity affect the full MLOps lifecycle — training cost, CI/CD pipeline time, serving latency, resource consumption, and monitoring reliability — in a production fraud detection system?

---

## Architecture

```
Local (Docker Compose) → GitHub Actions CI/CD → AWS ECR → AWS EC2 (Production)
     Training + MLflow         Build, Test, Push       Docker images    Docker Compose Stack
```

## Six Embedding Variants Compared

| Variant | Embedding | Operational Cost |
|---------|-----------|-----------------|
| 1 | TF-IDF → MLP | Lowest |
| 2 | Word2Vec → Mean Pool → MLP | Low |
| 3 | FastText → Mean Pool → MLP | Low |
| 4 | DistilBERT Frozen → [CLS] → MLP | High |
| 5 | DistilBERT Fine-tuned → [CLS] → MLP | High |
| 6 | DistilBERT MLM Pretrained → Fine-tuned → [CLS] → MLP | Highest |

## Dataset

**PaySim** — 6.3M synthetic mobile money transactions, ~8,213 fraud cases (0.129% fraud rate)
Download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place at `data/raw/PS_20174392719_1491204439457_log.csv`

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd fraud-detection-mlops
pip install -r requirements.txt

# 2. Download PaySim dataset (see above)

# 3. Run EDA (Phase 1)
jupyter notebook notebooks/01_eda.ipynb

# 4. Run preprocessing and tokenization (Phases 2-3)
jupyter notebook notebooks/02_preprocessing.ipynb

# 5. Train all 6 variants (Phase 4)
python src/training/train.py --config configs/experiment_config.yaml

# 6. Start local stack (Phase 5)
docker-compose up -d

# 7. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"account_id": "PS_12345", "transactions": [...]}'
```

## Project Structure

```
fraud-detection-mlops/
├── src/data/          # Data loading, preprocessing, segmentation, tokenization
├── src/models/        # 6 embedding model implementations
├── src/training/      # Training loop and evaluation
├── src/serving/       # FastAPI app, Prometheus metrics, structured logging
├── src/utils/         # Config and MLflow utilities
├── notebooks/         # EDA, preprocessing, results analysis
├── monitoring/        # Prometheus and Grafana configurations
├── tests/             # Unit tests and golden test set
├── configs/           # Experiment configuration YAML
├── data/              # Raw data (gitignored), processed outputs, configs
├── model_artifacts/   # Model weights and tokenization configs (baked into Docker)
├── Dockerfile         # Production inference image
├── docker-compose.yml # Local development stack
└── .github/workflows/ # CI/CD pipeline
```

## Reproduction

- Random seeds: 42, 123, 456 (all experiments report mean ± std)
- All tokenization thresholds saved as JSON in `data/configs/`
- Train/val/test split: temporal 70/15/15% by account's last transaction step
- Docker images tagged with git SHA for exact model traceability

## Stack

Python 3.10 · PyTorch · HuggingFace Transformers · gensim · scikit-learn · FastAPI · MLflow · Prometheus · Grafana · Docker · GitHub Actions · AWS (EC2, ECR, S3)
