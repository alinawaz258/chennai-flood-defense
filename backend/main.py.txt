from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
from .routing_engine import calculate_safe_route

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI model
model_path = "ml_models/saved_models/rf_flood_model.pkl"
if os.path.exists(model_path):
    flood_model = joblib.load(model_path)
else:
    flood_model = None

# Data schemas for API requests
class SensorData(BaseModel):
    rainfall_mm: float
    soil_moisture: float
    drain_capacity: float

class RouteRequest(BaseModel):
    start: str
    destination: str
    flooded_areas: list[str]

@app.get("/")
def read_root():
    return {"status": "Chennai Flood API is running!"}

@app.post("/predict_flood")
def predict_flood(data: SensorData):
    if not flood_model:
        return {"error": "Model not trained yet. Run train_flood_model.py first."}
    
    features = [[data.rainfall_mm, data.soil_moisture, data.drain_capacity]]
    prediction = flood_model.predict(features)[0]
    
    status = "Flooded" if prediction == 1 else "Safe"
    return {"ward_status": status, "risk_level": int(prediction)}

@app.post("/safe_route")
def get_safe_route(req: RouteRequest):
    route = calculate_safe_route(req.start, req.destination, req.flooded_areas)
    if not route:
        return {"error": "No valid route found."}
    return {"safe_route": route}