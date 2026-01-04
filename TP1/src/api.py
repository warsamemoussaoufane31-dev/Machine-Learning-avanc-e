from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

artifact_path = "models/first_pass_models.joblib"

try:
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    model = artifact["model"]
except FileNotFoundError:
    pipeline = None
    model = None

app = FastAPI(
    title="California Housing Price Prediction API",
    description="API pour prédire le prix médian des maisons en Californie.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "California Housing ML API", "status": "ok"}


@app.post("/predict")
def predict(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
):
    if pipeline is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model first.")

    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])

    prepared = pipeline.transform(data)
    prediction = model.predict(prepared)
    price = float(prediction[0])

    return {
        "predicted_median_house_value": price
    }
