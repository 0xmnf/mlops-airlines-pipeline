"""Load the @production model by alias and predict on one flight."""
import warnings
import logging

import mlflow
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


flight = {
    "departure_delay": 60,
    "distance":      1200,
    "air_time":       180,
    "month":            2,
    "day_of_week":      4,
}


# TODO — load the @production model and predict on the flight above.
#
# model      = mlflow.pyfunc.load_model("models:/FlightDelayClassifier@production")
# prediction = model.predict(pd.DataFrame([flight]))[0]
#
# print(
#     f"Prediction: {'DELAYED' if prediction == 1 else 'ON TIME'}\n"
#     "In the MLflow UI, open the run behind @production — that is the model that just answered.\n"
#     "Notice that this script never names a version. If you promoted a new version tomorrow, what would change here?"
# )
