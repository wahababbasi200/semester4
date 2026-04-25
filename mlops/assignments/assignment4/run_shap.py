"""Task 9 — SHAP explainability runner. Trains XGB once, then generates SHAP outputs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')

import xgboost as xgb
import joblib
from src.data.ingest import load_sample
from src.data.preprocess import preprocess, time_aware_split
from src.explain.shap_report import generate_shap_report

print("[shap-runner] Loading and preprocessing data...")
df = load_sample()
train_raw, _, test_raw = time_aware_split(df)
X_train, y_train, encoders, medians = preprocess(train_raw, fit=True)
X_test,  y_test,  _,        _       = preprocess(test_raw,
                                                  encoders=encoders,
                                                  medians=medians,
                                                  fit=False)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
print(f"[shap-runner] Training XGB (n_est=300)...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=neg/pos, tree_method="hist",
    eval_metric="logloss", random_state=42, n_jobs=-1
)
model.fit(X_train.values, y_train.values, verbose=False)

os.makedirs("outputs/xgb", exist_ok=True)
joblib.dump(model, "outputs/xgb/xgb_model.pkl")
joblib.dump({"encoders": encoders, "medians": medians}, "outputs/xgb/preprocessor.pkl")

print("[shap-runner] Generating SHAP report...")
generate_shap_report(model, X_test, y_test.values,
                     model_type="xgb",
                     out_dir="outputs/explainability",
                     sample_n=2000)
