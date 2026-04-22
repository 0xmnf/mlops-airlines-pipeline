"""Register the trained model and assign the @production alias."""
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

mlflow.set_experiment("<your-experiment-name>")

forest_config = {"n_estimators": 100, "max_depth": 10, "random_state": 42}


with mlflow.start_run() as run:
    mlflow.log_params(forest_config)

    model = RandomForestClassifier(**forest_config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metrics({
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    })

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, name="model", signature=signature)

    # TODO — register this run's model and point @production at it.
    #
    # model_uri     = f"runs:/{run.info.run_id}/model"
    # registered    = mlflow.register_model(model_uri, "FlightDelayClassifier")
    # client        = mlflow.tracking.MlflowClient()
    # client.set_registered_model_alias(
    #     name="FlightDelayClassifier",
    #     alias="production",
    #     version=registered.version,
    # )
    #
    # print(
    #     f"FlightDelayClassifier v{registered.version} registered.\n"
    #     "In the MLflow UI, find the model entry and note which version holds the @production alias.\n"
    #     "If you wanted to promote a new version next week, how many lines of application code would change?"
    # )
