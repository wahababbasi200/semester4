#!/usr/bin/env bash
# submit_anyscale_job.sh
# ----------------------
# Submits the Anyscale training job, temporarily overriding .gitignore
# so that processed parquet data files get uploaded to the cluster.
#
# Usage:
#   bash submit_anyscale_job.sh
#   bash submit_anyscale_job.sh --wait    # wait and stream logs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GITIGNORE=".gitignore"
GITIGNORE_BAK=".gitignore.bak"

# Check that data files exist locally
if [ ! -f "data/processed/preprocessing/train_tokenized.parquet" ]; then
    echo "ERROR: Training data not found at data/processed/preprocessing/"
    echo "Run notebooks/02_preprocessing.ipynb first to generate parquet files."
    exit 1
fi

echo "==> Temporarily modifying .gitignore to include data files..."
cp "$GITIGNORE" "$GITIGNORE_BAK"

# Remove lines that block parquet/processed data upload
sed -i '/^data\/processed\/$/d' "$GITIGNORE"
sed -i '/^\*\.parquet$/d' "$GITIGNORE"

echo "==> Submitting Anyscale job..."
anyscale job submit -f anyscale-job.yaml "$@"
SUBMIT_EXIT=$?

echo "==> Restoring .gitignore..."
mv "$GITIGNORE_BAK" "$GITIGNORE"

if [ $SUBMIT_EXIT -eq 0 ]; then
    echo "==> Job submitted successfully!"
    echo "    Monitor at: https://console.anyscale.com"
    echo "    Or run: anyscale job logs <job-id>"
else
    echo "==> Job submission failed (exit code $SUBMIT_EXIT)"
    exit $SUBMIT_EXIT
fi
