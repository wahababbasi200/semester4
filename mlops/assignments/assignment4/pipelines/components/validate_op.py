"""KFP Component 2 — Data Validation: schema + missing value + row count checks."""
from kfp.dsl import component, Input, Output, Dataset, Metrics


@component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4']
)
def validate_data(
    input_dataset: Input[Dataset],
    validated_dataset: Output[Dataset],
    validation_metrics: Output[Metrics],
    min_rows: int = 1000,
    max_missing_ratio: float = 0.99,
):
    """Validate schema, row count, and missing values. Fail fast on errors."""
    import pandas as pd
    import shutil
    import os

    src = input_dataset.metadata.get("file", input_dataset.path + ".parquet")
    print(f"[validate] Reading from {src}")
    df = pd.read_parquet(src)

    errors = []
    REQUIRED = ["isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}")

    high_null = df.columns[df.isnull().mean() > max_missing_ratio].tolist()
    if high_null:
        errors.append(f"Columns >{max_missing_ratio:.0%} null: {high_null[:5]}")

    fraud_rate = float(df["isFraud"].mean())
    overall_missing = float(df.isnull().mean().mean())

    validation_metrics.log_metric("rows", len(df))
    validation_metrics.log_metric("columns", df.shape[1])
    validation_metrics.log_metric("fraud_rate", round(fraud_rate, 4))
    validation_metrics.log_metric("avg_missing_rate", round(overall_missing, 4))
    validation_metrics.log_metric("validation_passed", 1 if not errors else 0)

    if errors:
        raise ValueError(f"Validation FAILED:\n" + "\n".join(errors))

    print(f"[validate] PASSED — rows={len(df):,} fraud={fraud_rate:.2%} missing={overall_missing:.2%}")

    # Pass through to next component
    out_path = validated_dataset.path + ".parquet"
    shutil.copy(src, out_path)
    validated_dataset.metadata["file"] = out_path
    validated_dataset.metadata["rows"] = len(df)
