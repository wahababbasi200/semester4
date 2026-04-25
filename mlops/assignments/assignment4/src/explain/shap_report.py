"""
Task 9: Model Explainability using SHAP.
Generates:
  - shap_summary.png   — beeswarm summary for top-20 features
  - shap_bar.png       — mean |SHAP| bar chart
  - top_features.csv   — ranked feature importance
  - shap_force_*.html  — 3 individual force plots (one TP, one FP, one FN)
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_shap_report(model, X_test: pd.DataFrame, y_test: np.ndarray,
                         model_type: str = "xgb",
                         out_dir: str = "outputs/explainability",
                         sample_n: int = 2000):
    os.makedirs(out_dir, exist_ok=True)

    # Subsample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(sample_n, len(X_test)), replace=False)
    X_sub = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
    y_sub = y_test[idx]

    print(f"[shap] Computing SHAP values for {len(X_sub):,} samples ({model_type})...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # For LightGBM binary classification, shap_values may be a list
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    feature_names = (X_sub.columns.tolist()
                     if hasattr(X_sub, "columns")
                     else [f"f{i}" for i in range(X_sub.shape[1])])

    # ── Summary beeswarm ────────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(sv, X_sub, feature_names=feature_names,
                      max_display=20, show=False)
    plt.title(f"SHAP Summary — {model_type.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Bar chart (mean |SHAP|) ─────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_feats  = [feature_names[i] for i in top_idx]
    top_values = mean_abs[top_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(top_feats[::-1], top_values[::-1], color="steelblue")
    plt.xlabel("Mean |SHAP| value")
    plt.title(f"Top-20 Feature Importance (SHAP) — {model_type.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=150)
    plt.close()

    # ── Top features CSV ────────────────────────────────────────────────────
    df_imp = pd.DataFrame({"feature": top_feats, "mean_abs_shap": top_values.round(6)})
    df_imp.to_csv(os.path.join(out_dir, "top_features.csv"), index=False)

    # ── Force plots for 3 cases ─────────────────────────────────────────────
    proba = model.predict_proba(X_sub)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    cases = {
        "true_positive":  np.where((y_sub == 1) & (y_pred == 1))[0],
        "false_positive": np.where((y_sub == 0) & (y_pred == 1))[0],
        "false_negative": np.where((y_sub == 1) & (y_pred == 0))[0],
    }

    for case_name, case_idx in cases.items():
        if len(case_idx) == 0:
            print(f"  [shap] No {case_name} cases found, skipping force plot")
            continue
        i = case_idx[0]
        row = X_sub.iloc[[i]] if hasattr(X_sub, "iloc") else X_sub[[i]]
        sv_row = explainer(row)
        html = shap.plots.force(sv_row[0], show=False, matplotlib=False)
        html_path = os.path.join(out_dir, f"shap_force_{case_name}.html")
        shap.save_html(html_path, html)
        print(f"  [shap] Saved {case_name} force plot → {html_path}")

    print(f"[shap] All outputs saved → {out_dir}/")
    return df_imp


if __name__ == "__main__":
    import sys
    import joblib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.data.ingest import load_sample
    from src.data.preprocess import preprocess, time_aware_split

    df = load_sample()
    _, _, test_raw = time_aware_split(df)
    # Assume preprocessor was already saved
    prep = joblib.load("outputs/xgb/preprocessor.pkl") if os.path.exists("outputs/xgb/preprocessor.pkl") else None
    if prep:
        X_test, y_test, _, _ = preprocess(test_raw, encoders=prep["encoders"],
                                          medians=prep["medians"], fit=False)
        model = joblib.load("outputs/xgb/xgb_model.pkl")
        generate_shap_report(model, X_test, y_test.values,
                             model_type="xgb", out_dir="outputs/explainability")
    else:
        print("No preprocessor found. Run train_local.py first.")
