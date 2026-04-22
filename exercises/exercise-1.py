"""Track one training run: log params, metrics, and the model artifact."""
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

# Pick one training configuration. Try others later to see how the metrics move.
forest_config = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
# Alternatives:
#   {"n_estimators":  50, "max_depth":    5, "random_state": 42}
#   {"n_estimators": 200, "max_depth": None, "random_state": 42}
#   {"n_estimators":  10, "max_depth":    3, "random_state": 42}


# TODO — log this run's params, metrics, and fitted model to MLflow.
#
# mlflow.set_experiment("<your-experiment-name>")
#
# with mlflow.start_run() as run:
#     mlflow.log_params(forest_config)
#
#     model = RandomForestClassifier(**forest_config)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     mlflow.log_metrics({
#         "accuracy":  accuracy_score(y_test, y_pred),
#         "precision": precision_score(y_test, y_pred),
#         "recall":    recall_score(y_test, y_pred),
#         "f1":        f1_score(y_test, y_pred),
#     })
#
#     signature = infer_signature(X_train, model.predict(X_train))
#     mlflow.sklearn.log_model(model, name="model", signature=signature)
#
#     print(
#         f"Run {run.info.run_id} logged.\n"
#         "In the MLflow UI, open this experiment and find the run.\n"
#         "Could someone six months from now reproduce this training from what was logged?"
#     )
