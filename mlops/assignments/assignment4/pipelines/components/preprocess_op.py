"""KFP Component 3 — Data Preprocessing: imputation, encoding, train/val/test split."""
from kfp.dsl import component, Input, Output, Dataset


@component(
    base_image='python:3.9',
    packages_to_install=[
        'pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4',
        'scikit-learn==1.3.2', 'category_encoders==2.6.3', 'joblib==1.3.2'
    ]
)
def preprocess_data(
    validated_dataset: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    preprocessor_artifact: Output[Dataset],   # saves encoders + medians
):
    """
    Full preprocessing:
      - Missing indicators for >30% null numeric columns
      - M-columns: T→1, F→0, NaN→-1
      - Target encoding (smoothed) for high-cardinality cols
      - Label encoding for remaining strings
      - Median imputation for remaining numerics
      - Chronological 70/15/15 split
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import joblib

    M_COLS = [f"M{i}" for i in range(1, 10)]
    HIGH_CARD = ["card1", "card2", "card5", "addr1", "P_emaildomain",
                 "R_emaildomain", "DeviceInfo"]
    FRAUD_COL = "isFraud"
    DROP_COLS = ["TransactionID", "TransactionDT"]
    MISSING_THRESH = 0.30

    src = validated_dataset.path
    df = pd.read_parquet(src)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Chronological split first (prevent leakage)
    n = len(df)
    val_start  = int(n * 0.70)
    test_start = int(n * 0.85)
    train_raw = df.iloc[:val_start]
    val_raw   = df.iloc[val_start:test_start]
    test_raw  = df.iloc[test_start:]

    def target_encode(train_df, cols, target, smoothing=20.0):
        global_mean = train_df[target].mean()
        enc = {}
        for col in cols:
            if col not in train_df.columns:
                continue
            stats = train_df.groupby(col)[target].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
                       (stats["count"] + smoothing)
            enc[col] = smoothed.to_dict()
        return enc

    def apply_transforms(data, encoders, medians, fit=False, train_for_enc=None):
        d = data.copy()
        drop = [c for c in DROP_COLS if c in d.columns]
        d = d.drop(columns=drop)
        y = d.pop(FRAUD_COL) if FRAUD_COL in d.columns else None

        # Missing indicators
        miss_rates = d.isnull().mean()
        missing_indicators = {
            f"{col}_missing": d[col].isnull().astype(np.int8)
            for col in miss_rates[miss_rates > MISSING_THRESH].index
            if pd.api.types.is_numeric_dtype(d[col])
        }
        if missing_indicators:
            d = pd.concat([d, pd.DataFrame(missing_indicators, index=d.index)], axis=1)

        # M-columns
        for col in M_COLS:
            if col in d.columns:
                d[col] = d[col].map({"T": 1, "F": 0}).fillna(-1).astype(np.int8)

        # Target encoding
        gm = y.mean() if y is not None else 0.05
        for col, mapping in encoders.items():
            if col in d.columns:
                d[col] = d[col].map(mapping).fillna(gm)

        # Label encode remaining strings
        string_like_cols = [
            col for col in d.columns
            if (
                pd.api.types.is_object_dtype(d[col])
                or pd.api.types.is_string_dtype(d[col])
                or pd.api.types.is_categorical_dtype(d[col])
            )
        ]
        for col in string_like_cols:
            le = LabelEncoder()
            d[col] = le.fit_transform(d[col].astype(str))

        # Median imputation
        for col, val in medians.items():
            if col in d.columns:
                d[col] = d[col].fillna(val)
        d = d.fillna(0)
        return d, y

    # Fit encoders on training set
    encoders = target_encode(train_raw, HIGH_CARD, FRAUD_COL)
    # Compute medians after dropping non-feature cols
    tmp = train_raw.drop(columns=[c for c in DROP_COLS if c in train_raw.columns])
    tmp = tmp.drop(columns=[FRAUD_COL], errors="ignore")
    for col in M_COLS:
        if col in tmp.columns:
            tmp[col] = tmp[col].map({"T": 1, "F": 0}).fillna(-1)
    gm = train_raw[FRAUD_COL].mean()
    for col, mapping in encoders.items():
        if col in tmp.columns:
            tmp[col] = tmp[col].map(mapping).fillna(gm)
    medians = tmp.select_dtypes(include=[np.number]).median().to_dict()
    del tmp

    def write_split(split_name, raw_df, out_art):
        X, y = apply_transforms(raw_df, encoders, medians)
        X[FRAUD_COL] = y.to_numpy()
        path = out_art.path
        X.to_parquet(path, index=False)
        out_art.metadata["rows"] = len(X)
        out_art.metadata["fraud_rate"] = round(float(y.mean()), 4)
        print(f"[preprocess] {split_name}: {len(X):,} rows  fraud={y.mean():.2%}  features={X.shape[1] - 1}")

    # Transform and write one split at a time to keep peak memory low.
    write_split("train", train_raw, train_data)
    write_split("val", val_raw, val_data)
    write_split("test", test_raw, test_data)

    # Save preprocessor
    prep_path = preprocessor_artifact.path
    joblib.dump({"encoders": encoders, "medians": medians}, prep_path)
    print(f"[preprocess] Preprocessor saved → {prep_path}")
