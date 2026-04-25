"""Smoke tests for model training and evaluation."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_arrays(n=400, n_features=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (rng.random(n) < 0.15).astype(np.int32)
    return X, y


def test_xgb_trains_and_returns_probabilities():
    import xgboost as xgb
    X, y = _make_arrays()
    X_tr, y_tr = X[:300], y[:300]
    X_te, _y_te = X[300:], y[300:]

    model = xgb.XGBClassifier(
        n_estimators=20, max_depth=3, tree_method="hist",
        eval_metric="logloss", random_state=42, n_jobs=1
    )
    model.fit(X_tr, y_tr, verbose=False)
    proba = model.predict_proba(X_te)[:, 1]

    assert proba.shape == (len(X_te),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_lgbm_trains_and_returns_probabilities():
    import lightgbm as lgb
    X, y = _make_arrays()
    X_tr, y_tr = X[:300], y[:300]
    X_te, y_te = X[300:], y[300:]

    model = lgb.LGBMClassifier(
        n_estimators=20, num_leaves=16, random_state=42, n_jobs=1, verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.log_evaluation(-1)])
    proba = model.predict_proba(X_te)[:, 1]

    assert proba.shape == (len(X_te),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_rf_fs_selects_features_and_predicts():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel

    X, y = _make_arrays()
    X_tr, y_tr = X[:300], y[:300]
    X_te = X[300:]

    rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    rf.fit(X_tr, y_tr)
    selector = SelectFromModel(rf, threshold="median", prefit=True)
    X_sel_tr = selector.transform(X_tr)
    X_sel_te = selector.transform(X_te)

    assert X_sel_tr.shape[1] < X_tr.shape[1]   # features were reduced

    rf2 = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    rf2.fit(X_sel_tr, y_tr)
    proba = rf2.predict_proba(X_sel_te)[:, 1]

    assert proba.shape == (len(X_te),)


def test_cost_utility_calculation():
    from src.costsensitive.cost_eval import business_utility

    y_true = np.array([0, 0, 1, 1, 1, 0])
    proba  = np.array([0.1, 0.9, 0.9, 0.1, 0.8, 0.2])

    result = business_utility(y_true, proba, threshold=0.5)
    assert "utility_$" in result
    assert "TP" in result and "FN" in result and "FP" in result
    # TP=2, FP=1, FN=1 → utility = 200*2 - 500*1 - 50*1 = 400-500-50 = -150
    assert result["TP"] == 2
    assert result["FN"] == 1


def test_psi_stable_on_same_distribution():
    from src.drift.detect import _psi_single
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000)
    psi = _psi_single(data, data + rng.normal(0, 0.01, 1000))
    assert psi < 0.1, f"PSI should be near 0 for near-identical distributions, got {psi}"


def test_psi_high_on_different_distribution():
    from src.drift.detect import _psi_single
    rng = np.random.default_rng(0)
    train = rng.standard_normal(1000)
    future = rng.standard_normal(1000) * 3 + 5   # very different
    psi = _psi_single(train, future)
    assert psi > 0.2, f"PSI should be high for shifted distribution, got {psi}"
