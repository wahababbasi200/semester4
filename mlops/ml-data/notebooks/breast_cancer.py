from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

mlflow.set_experiment("Breast_Cancer_Experiment")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Logistic Regression versions ---
lr_configs = [
    {"C": 1.0,  "solver": "lbfgs",  "version": "lr-v1"},
    {"C": 0.1,  "solver": "lbfgs",  "version": "lr-v2"},
    {"C": 10.0, "solver": "lbfgs",  "version": "lr-v3"},
]

for cfg in lr_configs:
    with mlflow.start_run(run_name=cfg["version"]):
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("version", cfg["version"])

        lr = LogisticRegression(max_iter=1000, C=cfg["C"], solver=cfg["solver"])
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("solver", cfg["solver"])
        mlflow.log_param("C", cfg["C"])

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred))
        mlflow.log_metric("f1",        f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(lr, "model")

# --- Random Forest versions ---
rf_configs = [
    {"n_estimators": 50,  "max_depth": None, "version": "rf-v1"},
    {"n_estimators": 100, "max_depth": None, "version": "rf-v2"},
    {"n_estimators": 100, "max_depth": 5,    "version": "rf-v3"},
]

for cfg in rf_configs:
    with mlflow.start_run(run_name=cfg["version"]):
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("version", cfg["version"])

        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", cfg["n_estimators"])
        mlflow.log_param("max_depth", cfg["max_depth"])

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred))
        mlflow.log_metric("f1",        f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(rf, "model")
