#!/usr/bin/env bash
# submit_anyscale_job.sh
# ----------------------
# Submits the Anyscale training job with only the required data files.
# Creates a lightweight staging directory to avoid uploading unnecessary
# large files (paysim_filtered.parquet = 114MB, etc.).
#
# Usage:
#   bash submit_anyscale_job.sh
#   bash submit_anyscale_job.sh --wait    # wait and stream logs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/processed/preprocessing"

# Check that data files exist locally
if [ ! -f "$DATA_DIR/train_tokenized.parquet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/"
    echo "Run notebooks/02_preprocessing.ipynb first to generate parquet files."
    exit 1
fi

# Create a lightweight staging directory with only needed files
STAGING_DIR=$(mktemp -d)
trap "rm -rf $STAGING_DIR" EXIT

echo "==> Creating staging directory with only required files..."

# Copy source code and configs (small files)
cp -r src/ "$STAGING_DIR/src/"
cp -r configs/ "$STAGING_DIR/configs/"
mkdir -p "$STAGING_DIR/data/configs/"
cp -r data/configs/* "$STAGING_DIR/data/configs/"
cp anyscale-job.yaml "$STAGING_DIR/"
cp requirements.txt "$STAGING_DIR/" 2>/dev/null || true

# Copy ONLY the 3 tokenized parquets needed for training (~40MB total)
mkdir -p "$STAGING_DIR/$DATA_DIR"
cp "$DATA_DIR/train_tokenized.parquet" "$STAGING_DIR/$DATA_DIR/"
cp "$DATA_DIR/val_tokenized.parquet" "$STAGING_DIR/$DATA_DIR/"
cp "$DATA_DIR/test_tokenized.parquet" "$STAGING_DIR/$DATA_DIR/"

echo "    Staging size: $(du -sh "$STAGING_DIR" | cut -f1)"

echo "==> Submitting Anyscale job from staging directory..."
cd "$STAGING_DIR"
anyscale job submit -f anyscale-job.yaml "$@"
SUBMIT_EXIT=$?

if [ $SUBMIT_EXIT -eq 0 ]; then
    echo "==> Job submitted successfully!"
    echo "    Monitor at: https://console.anyscale.com"
    echo "    Or run: anyscale job logs <job-id>"
else
    echo "==> Job submission failed (exit code $SUBMIT_EXIT)"
    exit $SUBMIT_EXIT
fi
