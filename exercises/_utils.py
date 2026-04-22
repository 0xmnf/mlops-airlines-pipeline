"""Shared Gold-layer data loader for the MLOps exercises (ex-1, 2, 3a)."""
import warnings
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

FEATURES = ["departure_delay", "distance", "air_time", "month", "day_of_week"]


def load_flight_data():
    flights  = pd.read_csv("data/gold/fact_flights_sample.csv")
    airlines = pd.read_csv("data/gold/dim_airline.csv")
    dates    = pd.read_csv("data/gold/dim_date.csv")
    df = flights.merge(airlines, on="airline_id").merge(dates, on="date_id")
    df["is_delayed"] = (df["arrival_delay"] > 15).astype(int)
    X = df[FEATURES].dropna()
    y = df.loc[X.index, "is_delayed"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
