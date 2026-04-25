"""KFP Component 5 — Model Training: XGBoost / LightGBM / RF+FS."""
from kfp.dsl import component, Input, Output, Dataset, Model


@component(
    base_image='python:3.9',
    packages_to_install=[
        'pandas==2.1.4', 'pyarrow==15.0.2', 'numpy==1.26.4',
        'scikit-learn==1.3.2', 'xgboost==2.0.3', 'lightgbm==4.3.0',
        'imbalanced-learn==0.12.2', 'joblib==1.3.2'
    ]
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model_artifact: Output[Model],
    model_type: str = "xgb",               # "xgb" | "lgbm" | "rf_fs"
    imbalance_strategy: str = "class_weight",  # "class_weight" | "smote"
    cost_sensitive: bool = False,
    n_estimators: int = 400,
    random_state: int = 42,
):
    """Train fraud detection model with configurable type and imbalance strategy."""
    import pandas as pd
    import numpy as np
    import joblib

    FRAUD_COL = "isFraud"

    def load_split(art):
        src = art.metadata.get("file", art.path + ".parquet")
        d = pd.read_parquet(src)
        X = d.drop(columns=[FRAUD_COL])
        y = d[FRAUD_COL]
        return X.values.astype(np.float32), y.values.astype(np.int32)

    X_train, y_train = load_split(train_data)
    X_val,   y_val   = load_split(val_data)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    print(f"[train] model={model_type}  imbalance={imbalance_strategy}  "
          f"cost_sensitive={cost_sensitive}  neg={neg}  pos={pos}  spw={spw:.2f}")

    # Apply SMOTE before training if requested
    if imbalance_strategy == "smote":
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[train] After SMOTE: {len(X_train):,} samples")
        effective_spw = 1.0   # SMOTE already balanced
    else:
        effective_spw = spw

    if cost_sensitive:
        effective_spw = max(effective_spw, 10.0)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    if model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
            scale_pos_weight=effective_spw, tree_method="hist",
            eval_metric="logloss", random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=100)
        payload = model

    # ── LightGBM ─────────────────────────────────────────────────────────────
    elif model_type == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, num_leaves=64, learning_rate=0.05,
            is_unbalance=(imbalance_strategy == "class_weight"),
            objective="binary", metric="auc",
            random_state=random_state, n_jobs=-1, verbose=-1
        )
        cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cbs)
        payload = model

    # ── Random Forest + Feature Selection (Hybrid) ───────────────────────────
    elif model_type == "rf_fs":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        cw = {0: 1, 1: int(max(neg / pos, 5))}
        rf_full = RandomForestClassifier(
            n_estimators=100, class_weight=cw,
            random_state=random_state, n_jobs=-1, max_depth=10
        )
        rf_full.fit(X_train, y_train)
        selector = SelectFromModel(rf_full, threshold="median", prefit=True)
        X_tr_sel = selector.transform(X_train)
        X_vl_sel = selector.transform(X_val)
        print(f"[train] RF-FS selected {X_tr_sel.shape[1]}/{X_train.shape[1]} features")

        rf_final = RandomForestClassifier(
            n_estimators=n_estimators, class_weight=cw,
            random_state=random_state, n_jobs=-1, max_depth=12
        )
        rf_final.fit(X_tr_sel, y_train)
        payload = {"selector": selector, "model": rf_final}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model_path = model_artifact.path + ".pkl"
    joblib.dump(payload, model_path)
    model_artifact.metadata["file"] = model_path
    model_artifact.metadata["model_type"] = model_type
    model_artifact.metadata["imbalance_strategy"] = imbalance_strategy
    model_artifact.metadata["cost_sensitive"] = str(cost_sensitive)
    print(f"[train] Model saved → {model_path}")
