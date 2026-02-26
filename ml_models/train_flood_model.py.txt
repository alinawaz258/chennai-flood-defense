from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_OUTPUT_PATH = "ml_models/saved_models/rf_flood_model.pkl"

# 1. Create compact float32/int8 training data for lower memory footprint.
raw_data = {
    "rainfall_mm": [10, 50, 120, 5, 80, 150, 20, 100],
    "soil_moisture": [30, 60, 95, 20, 80, 99, 40, 85],
    "drain_capacity": [80, 50, 10, 90, 30, 5, 75, 20],
    "flooded": [0, 0, 1, 0, 1, 1, 0, 1],
}
df = pd.DataFrame(raw_data).astype(
    {
        "rainfall_mm": np.float32,
        "soil_moisture": np.float32,
        "drain_capacity": np.float32,
        "flooded": np.int8,
    }
)

# 2. Vectorized feature/target extraction.
X = df[["rainfall_mm", "soil_moisture", "drain_capacity"]].to_numpy(dtype=np.float32)
y = df["flooded"].to_numpy(dtype=np.int8)

# 3. Train a smaller RF for edge inference speed and reduced model size.
model = RandomForestClassifier(
    n_estimators=64,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)
print("Model trained successfully.")

# 4. Save compressed model artifact to reduce disk footprint.
os.makedirs("ml_models/saved_models", exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH, compress=3)
print(f"Model saved to {MODEL_OUTPUT_PATH}")
