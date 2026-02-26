# Chennai Flood Defense System 🌊
Built for the AMD Slingshot Hackathon.

## Overview
An AI-powered urban flood management system for Chennai. It uses predictive modeling to forecast waterlogging, dynamic graph routing to redirect traffic away from flooded zones, and simulated edge-IoT sensors for automated water removal.

## Tech Stack
* **Backend:** FastAPI, Python
* **AI/ML:** Scikit-Learn (Random Forest), Pandas
* **Routing:** NetworkX
* **Frontend:** HTML, Vanilla JS

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python ml_models/train_flood_model.py`
4. Start the backend server: `uvicorn backend.main:app --reload`
5. Open `frontend/index.html` in your browser.