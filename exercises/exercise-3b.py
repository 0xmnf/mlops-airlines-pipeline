"""Promote the best sweep run to @production only if it beats the champion."""
import warnings
import logging

import mlflow

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


EXPERIMENT_NAME = "flight-delay-classifier"
MODEL_NAME      = "FlightDelayClassifier"

client = mlflow.tracking.MlflowClient()


# TODO — query the best sweep run, compare to the current @production,
# and reassign the alias only if the challenger wins.
#
# experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
# best_run = client.search_runs(
#     experiment_ids=[experiment.experiment_id],
#     order_by=["metrics.f1 DESC"],
#     max_results=1,
# )[0]
# best_candidate_f1 = best_run.data.metrics["f1"]
#
# try:
#     prod_version = client.get_model_version_by_alias(MODEL_NAME, "production")
# except mlflow.exceptions.RestException:
#     raise SystemExit("No @production yet — did you run exercise-2?")
#
# production_f1 = client.get_run(prod_version.run_id).data.metrics["f1"]
#
# if best_candidate_f1 > production_f1:
#     registered = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", MODEL_NAME)
#     client.set_registered_model_alias(
#         name=MODEL_NAME,
#         alias="production",
#         version=registered.version,
#     )
#
# print(
#     "Gate evaluated.\n"
#     "In the MLflow UI, look at the @production alias on FlightDelayClassifier.\n"
#     "Did it move? What evidence made that decision, and should a human still review it?"
# )
