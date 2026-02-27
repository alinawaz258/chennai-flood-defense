from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from typing import List

# Absolute-safe import
try:
    from routing_engine import calculate_safe_route
except ImportError:
    calculate_safe_route = None

app = FastAPI(title="Chennai Flood API")

# CORS (limit this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= LOAD MODEL SAFELY =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ml_models", "saved_models", "rf_flood_model.pkl")

if os.path.exists(model_path):
    flood_model = joblib.load(model_path)
else:
    flood_model = None

# ========= SCHEMAS =========
class SensorData(BaseModel):
    rainfall_mm: float
    soil_moisture: float
    drain_capacity: float

class RouteRequest(BaseModel):
    start: str
    destination: str
    flooded_areas: List[str]

# ========= ROUTES =========
@app.get("/")
def read_root():
    return {"status": "Chennai Flood API is running!"}

@app.post("/predict_flood")
def predict_flood(data: SensorData):
    if flood_model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not trained yet. Run train_flood_model.py first."
        )

    try:
        features = [[data.rainfall_mm, data.soil_moisture, data.drain_capacity]]
        prediction = flood_model.predict(features)[0]

        status = "Flooded" if prediction == 1 else "Safe"

        return {
            "ward_status": status,
            "risk_level": int(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/safe_route")
def get_safe_route(req: RouteRequest):
    if calculate_safe_route is None:
        raise HTTPException(
            status_code=500,
            detail="Routing engine not available."
        )

    try:
        route = calculate_safe_route(
            req.start,
            req.destination,
            req.flooded_areas
        )

        if not route:
            raise HTTPException(
                status_code=404,
                detail="No valid route found."
            )

        return {"safe_route": route}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
