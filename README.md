# ml-data

MLOps project workspace for Semester 4. Contains notebooks, DAGs, and a Docker Compose setup that runs Jupyter, MLflow, Airflow, Redis, and a Flask API together.

## Project structure

```
ml-data/
├── airflow/
│   ├── dags/          # Airflow DAG definitions
│   ├── logs/          # task logs
│   └── plugins/
├── data/              # datasets
├── models/            # saved model files
├── mlruns/            # MLflow artifact store (shared volume)
├── notebooks/         # Jupyter notebooks and Python scripts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Services

| Container | Image | Port | Purpose |
|---|---|---|---|
| `ml_jupyter` | custom (Dockerfile) | 8889 | JupyterLab for experimentation |
| `ml_mlflow` | mlflow v2.15.1 | 5000 | Experiment tracking and model registry |
| `ml_app_container` | custom (Dockerfile) | 5001 | Flask prediction API with Redis caching |
| `redis_container` | redis:alpine | 6379 | Caching layer for the API |
| `airflow_webserver` | custom (airflow/) | 8080 | Airflow UI |
| `airflow_scheduler` | custom (airflow/) | — | DAG task executor |
| `airflow_postgres` | postgres:15 | — | Airflow metadata database |

## Getting started

```bash
# Start MLflow and Postgres first
docker compose up -d mlflow airflow-postgres

# Initialize Airflow (first time only)
docker compose run --rm airflow-init

# Start everything else
docker compose up -d jupyter redis ml-app airflow-webserver airflow-scheduler
```

| UI | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| JupyterLab | http://localhost:8889 |
| Flask API | http://localhost:5001 |
