"""KFP Component 1 — Data Ingestion: reads sample parquet from PVC mount."""
from kfp.dsl import component, Output, Dataset


@component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4']
)
def ingest_data(
    data_path: str,          # path inside pod (PVC mount), e.g. /data/sample_ieee_70k.parquet
    output_dataset: Output[Dataset],
):
    """Read IEEE CIS sample from PVC-mounted path and pass as KFP artifact."""
    import pandas as pd
    import os

    print(f"[ingest] Reading from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Sample not found at {data_path}. "
            "Ensure the PVC is mounted and build_sample.py was run first."
        )

    df = pd.read_parquet(data_path)
    print(f"[ingest] Shape: {df.shape}  |  fraud rate: {df['isFraud'].mean():.2%}")

    # Write directly to the managed artifact path so downstream tasks can resolve it.
    df.to_parquet(output_dataset.path, index=False)
    output_dataset.metadata["rows"] = len(df)
    output_dataset.metadata["fraud_rate"] = round(float(df["isFraud"].mean()), 4)

    print(f"[ingest] Written to artifact: {output_dataset.path}")
