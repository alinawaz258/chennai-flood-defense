from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .routing_engine import calculate_safe_route

app = FastAPI()

# Allow frontend to communicate with backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "ml_models/saved_models/rf_flood_model.pkl"


@lru_cache(maxsize=1)
def load_flood_model():
    """Load and cache the model globally so each worker reads it only once."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


class SensorData(BaseModel):
    rainfall_mm: float
    soil_moisture: float
    drain_capacity: float


class RouteRequest(BaseModel):
    start: str
    destination: str
    flooded_areas: list[str]


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"status": "Chennai Flood API is running!"}


@app.post("/predict_flood")
async def predict_flood(data: SensorData) -> dict[str, int | str]:
    model = load_flood_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run train_flood_model.py first.",
        )

    # NumPy float32 keeps inference fast and memory-efficient on edge devices.
    features = np.asarray(
        [[data.rainfall_mm, data.soil_moisture, data.drain_capacity]],
        dtype=np.float32,
    )
    prediction = int(model.predict(features)[0])

    return {
        "ward_status": "Flooded" if prediction == 1 else "Safe",
        "risk_level": prediction,
    }


@app.post("/safe_route")
async def get_safe_route(req: RouteRequest) -> dict[str, list[str]]:
    route = calculate_safe_route(req.start, req.destination, req.flooded_areas)
    if not route:
        raise HTTPException(status_code=404, detail="No valid route found.")
    return {"safe_route": route}
