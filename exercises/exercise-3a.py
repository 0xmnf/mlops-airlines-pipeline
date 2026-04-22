"""Sweep six hyperparameter configurations as separate MLflow runs."""
import warnings
import logging

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from _utils import load_flight_data

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


X_train, X_test, y_train, y_test = load_flight_data()

mlflow.set_experiment("flight-delay-classifier")

candidate_configs = [
    {"n_estimators":   1, "max_depth":    1, "random_state": 42},
    {"n_estimators":   5, "max_depth":    2, "random_state": 42, "max_features": 1},
    {"n_estimators":  25, "max_depth":    5, "random_state": 42},
    {"n_estimators": 100, "max_depth":   10, "random_state": 42},
    {"n_estimators": 300, "max_depth":   20, "random_state": 42},
    {"n_estimators": 500, "max_depth": None, "random_state": 42},
]


# TODO — log each config as its own run, then compare them in the MLflow UI.
#
# for run_num, config in enumerate(candidate_configs, 1):
#     with mlflow.start_run(run_name=f"config-{run_num}"):
#         mlflow.log_params(config)
#
#         model = RandomForestClassifier(**config)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         mlflow.log_metrics({
#             "accuracy":  accuracy_score(y_test, y_pred),
#             "precision": precision_score(y_test, y_pred),
#             "recall":    recall_score(y_test, y_pred),
#             "f1":        f1_score(y_test, y_pred),
#         })
#
#         signature = infer_signature(X_train, model.predict(X_train))
#         mlflow.sklearn.log_model(model, name="model", signature=signature)
#
# print(
#     "Sweep done.\n"
#     "In the MLflow UI, open this experiment and compare the runs.\n"
#     "Which configuration would you recommend for production, and on what evidence?"
# )
