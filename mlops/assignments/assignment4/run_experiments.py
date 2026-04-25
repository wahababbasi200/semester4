"""
Direct experiment runner — Tasks 2, 3, 4.
Bypasses Kubeflow; runs all model variants locally on EC2.
Usage: python run_experiments.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

from src.data.ingest import load_sample
from src.data.preprocess import preprocess, time_aware_split
from src.data.imbalance import compare_strategies
from src.models.lgbm_model import train_and_evaluate as lgbm_eval
from src.models.hybrid_rf_fs import train_and_evaluate as rf_eval
from src.costsensitive.cost_eval import compare_standard_vs_costsensitive
import pandas as pd

print("=" * 60)
print("Loading and preprocessing data...")
print("=" * 60)

df = load_sample()
train_raw, _, test_raw = time_aware_split(df)
X_train, y_train, encoders, medians = preprocess(train_raw, fit=True)
X_test,  y_test,  _,        _       = preprocess(test_raw,
                                                  encoders=encoders,
                                                  medians=medians,
                                                  fit=False)
X_tr = X_train.values
y_tr = y_train.values
X_te = X_test.values
y_te = y_test.values
print(f"Train: {X_tr.shape}  Test: {X_te.shape}  Fraud rate: {y_te.mean():.2%}\n")

# ── Task 2: Imbalance comparison ─────────────────────────────────────────────
print("=" * 60)
print("TASK 2 — Imbalance Strategy Comparison (class_weight vs SMOTE)")
print("=" * 60)
df_imbalance = compare_strategies(X_tr, y_tr, X_te, y_te,
                                  out_dir="outputs/imbalance")
print()

# ── Task 3: Model complexity ──────────────────────────────────────────────────
print("=" * 60)
print("TASK 3 — LightGBM")
print("=" * 60)
lgbm_metrics = lgbm_eval(X_tr, y_tr, X_te, y_te, out_dir="outputs/lgbm")
print()

print("=" * 60)
print("TASK 3 — RF + Feature Selection Hybrid")
print("=" * 60)
rf_metrics = rf_eval(X_tr, y_tr, X_te, y_te, out_dir="outputs/rf_fs")
print()

# ── Task 4: Cost-sensitive ────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4 — Cost-Sensitive Learning")
print("=" * 60)
df_cost = compare_standard_vs_costsensitive(X_tr, y_tr, X_te, y_te,
                                            out_dir="outputs/costsensitive")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL RESULTS SUMMARY")
print("=" * 60)

rows = []
for _, r in df_imbalance.iterrows():
    rows.append({"model": f"xgb_{r['strategy']}", "precision": r["precision"],
                 "recall": r["recall"], "f1": r["f1"],
                 "auc_roc": r["auc_roc"], "pr_auc": r["pr_auc"]})
rows.append({k: lgbm_metrics[k] for k in ["model","precision","recall","f1","auc_roc","pr_auc"]})
rows.append({k: rf_metrics[k]   for k in ["model","precision","recall","f1","auc_roc","pr_auc"]})

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv("outputs/all_metrics_summary.csv", index=False)
print("\nSaved → outputs/all_metrics_summary.csv")
print("All outputs saved to outputs/")
