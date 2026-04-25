"""KFP Component 4 — Feature Engineering: domain-informed feature creation."""
from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4']
)
def engineer_features(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    test_data: Input[Dataset],
    train_engineered: Output[Dataset],
    val_engineered: Output[Dataset],
    test_engineered: Output[Dataset],
):
    """Add TransactionAmt bins, time features, and C-column aggregates."""
    import pandas as pd
    import numpy as np

    FRAUD_COL = "isFraud"

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "TransactionAmt" in d.columns:
            d["TransactionAmt_log"]    = np.log1p(d["TransactionAmt"])
            d["TransactionAmt_cents"]  = (d["TransactionAmt"] * 100 % 100).astype(int)
            d["TransactionAmt_bin"]    = pd.cut(
                d["TransactionAmt"], bins=[0, 10, 100, 1000, np.inf],
                labels=[0, 1, 2, 3], include_lowest=True
            ).astype(float)

        c_cols = [c for c in d.columns if c[0] == "C" and c[1:].isdigit()]
        if c_cols:
            d["C_sum"] = d[c_cols].sum(axis=1)

        if "D1" in d.columns and "D2" in d.columns:
            d["D1_D2_ratio"] = (d["D1"] / (d["D2"].replace(0, np.nan))).fillna(0)

        return d

    for in_art, out_art, name in [
        (train_data, train_engineered, "train"),
        (val_data,   val_engineered,   "val"),
        (test_data,  test_engineered,  "test"),
    ]:
        src = in_art.metadata.get("file", in_art.path + ".parquet")
        df = pd.read_parquet(src)
        df_eng = add_features(df)
        out_path = out_art.path + ".parquet"
        df_eng.to_parquet(out_path, index=False)
        out_art.metadata["file"] = out_path
        out_art.metadata["rows"] = len(df_eng)
        out_art.metadata["fraud_rate"] = df_eng[FRAUD_COL].mean() if FRAUD_COL in df_eng.columns else 0.0
        print(f"[feateng] {name}: {df.shape[1]} → {df_eng.shape[1]} features")
