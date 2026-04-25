#!/bin/bash
# Full KFP cache nuke — run before every clean pipeline test on EC2 Minikube.
# From Assignment 3 CLAUDE.md: the tasks table in mlpipeline DB is the real cache source.
set -e

echo "=== KFP Cache Nuke ==="

MYSQL_POD=$(kubectl -n kubeflow get pod -l app=mysql -o jsonpath='{.items[0].metadata.name}')
echo "MySQL pod: $MYSQL_POD"

echo "[1/4] Flushing mlpipeline.tasks (the real cache index)..."
kubectl -n kubeflow exec $MYSQL_POD -- mysql -u root mlpipeline -e "DELETE FROM tasks;"

echo "[2/4] Flushing MLMD executions..."
kubectl -n kubeflow exec $MYSQL_POD -- mysql -u root metadb -e "
DELETE FROM Event;
DELETE FROM EventPath;
DELETE FROM ExecutionProperty;
DELETE FROM Association;
DELETE FROM Execution;
"

echo "[3/4] Flushing v1 cache-server DB..."
kubectl -n kubeflow exec $MYSQL_POD -- mysql -u root cachedb -e "DELETE FROM execution_caches;"

echo "[4/4] Restarting stateful services..."
kubectl -n kubeflow rollout restart deployment ml-pipeline metadata-grpc-deployment cache-server
kubectl -n kubeflow rollout status deployment ml-pipeline --timeout=120s

echo "=== Cache nuke complete. Next run will spawn fresh executor pods. ==="
