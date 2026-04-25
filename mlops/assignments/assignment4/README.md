# Assignment 4 — IEEE CIS Fraud Detection MLOps
**Student:** Abdul Wahab Abbasi · 24i-8034  
**Course:** MLOps for Cloud Native Applications  
**Deadline:** 25-April-2026

---

## Folder Structure
```
assignment4/
├── src/               # All Python source modules
│   ├── data/          # ingest, validate, preprocess, imbalance
│   ├── features/      # feature engineering
│   ├── models/        # xgb, lgbm, rf+fs hybrid
│   ├── costsensitive/ # cost-sensitive training & business impact
│   ├── drift/         # time-based drift simulation & PSI/KS detection
│   ├── explain/       # SHAP explainability
│   └── serving/       # FastAPI + Prometheus metrics
├── pipelines/
│   ├── components/    # 7 KFP components (ingest→validate→preprocess→feateng→train→eval→deploy)
│   ├── fraud_pipeline.py  # Main KFP pipeline (dsl.If conditional + retries)
│   └── compiled/      # compiled YAML (auto-generated)
├── k8s/               # Kubernetes manifests (namespace, quota, PVC, inference deployment)
├── observability/     # Prometheus rules, Alertmanager config, Grafana dashboards
├── docker/            # Dockerfile.train + Dockerfile.serve
├── .github/workflows/ # ci.yml, build.yml, cd.yml, retrain-on-alert.yml
├── tests/             # Unit tests (pytest)
├── notebooks/         # Drift simulation (02) + Retraining strategy (03)
├── scripts/           # build_sample.py, submit_pipeline.py, cache_nuke.sh
└── data/              # sample_ieee_70k.parquet (all fraud + 50K non-fraud)
```

---

## Quick Start

### Step 1 — Build the dataset sample (run on Windows or EC2)
```bash
# Point to your raw IEEE CIS files
python scripts/build_sample.py \
  --tx /path/to/train_transaction.csv \
  --id /path/to/train_identity.csv \
  --out data/sample_ieee_70k.parquet
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run tests locally (CI check)
```bash
pytest tests/ -v
```

### Step 4 — Compile the pipeline
```bash
python pipelines/fraud_pipeline.py
# → pipelines/compiled/fraud_pipeline.yaml
```

---

## EC2 / Kubeflow Deployment

### Prerequisites (already done from Assignment 3)
- Minikube running: `minikube status`
- KFP UI accessible: `kubectl -n kubeflow port-forward --address 0.0.0.0 svc/ml-pipeline-ui 8080:80 &`

### Setup PVC + namespace
```bash
# Copy sample data to EC2 mount
scp data/sample_ieee_70k.parquet ec2-user@EC2_IP:/mnt/ml-data/assignment4/data/

# Apply k8s manifests on EC2
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/resourcequota.yaml
kubectl apply -f k8s/pv-pvc.yaml
```

### Flush cache (IMPORTANT — from A3 experience)
```bash
bash scripts/cache_nuke.sh
```

### Submit pipeline runs
```bash
# XGBoost with class_weight (default)
python scripts/submit_pipeline.py --model xgb --imbalance class_weight

# LightGBM run
python scripts/submit_pipeline.py --model lgbm

# RF+FS hybrid
python scripts/submit_pipeline.py --model rf_fs

# SMOTE comparison
python scripts/submit_pipeline.py --model xgb --imbalance smote

# Cost-sensitive
python scripts/submit_pipeline.py --model xgb --cost-sensitive true
```

---

## Observability Setup (on EC2)

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f observability/prometheus-values.yaml

# Apply alert rules
kubectl apply -f observability/alert-rules.yaml

# Deploy inference API
kubectl apply -f k8s/inference-deploy.yaml

# Access Grafana (NodePort 30300)
echo "Grafana: http://$(minikube ip):30300  (admin / fraud-admin-2024)"

# Import dashboards manually:
# Grafana → + → Import → upload JSON from observability/grafana/*.json
```

---

## CI/CD Setup

### GitHub Actions (automatic on push)
- **ci.yml** — runs on every push: lint + tests + schema validation
- **build.yml** — runs on main push: builds + pushes Docker images to GHCR
- **cd.yml** — runs on self-hosted EC2 runner: applies k8s manifests + submits KFP pipeline

### Self-hosted runner (one-time EC2 setup)
```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz
./config.sh --url https://github.com/GITHUB_USER/REPO \
  --token <RUNNER_TOKEN> --labels self-hosted,minikube --unattended
sudo ./svc.sh install && sudo ./svc.sh start
```

### Alertmanager → GitHub retrain trigger
```bash
# Test webhook manually (replace TOKEN and REPO)
curl -X POST \
  -H "Authorization: token $GITHUB_PAT" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/GITHUB_USER/GITHUB_REPO/dispatches \
  -d '{"event_type":"model-drift","client_payload":{"reason":"test"}}'
# → HTTP 204 = success → retrain-on-alert.yml fires
```

---

## Screenshots Checklist
```
screenshots/
├── kubeflow/
│   ├── pipeline_dag.png          # 7-step DAG with conditional branch
│   ├── run_succeeded.png         # All steps green
│   ├── conditional_branch.png    # dsl.If AUC >= 0.85 branch
│   └── retry_logs.png            # Retry mechanism in action
├── cicd/
│   ├── ci_green.png              # CI workflow passing
│   ├── build_images.png          # Docker images in GHCR
│   ├── cd_run.png                # CD deployment workflow
│   └── retrain_triggered.png     # retrain-on-alert.yml fired by Alertmanager
├── grafana/
│   ├── system_health.png
│   ├── model_performance.png
│   └── data_drift.png
├── alerts/
│   └── alert_firing.png          # Prometheus/Grafana alert in FIRING state
└── explainability/
    ├── shap_summary.png
    ├── shap_bar.png
    └── shap_force_true_positive.html
```
