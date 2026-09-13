
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel


# ---------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------
app = FastAPI(
    title="Tourism Wellness Prediction API",
    description="Prediction API for the Wellness Tourism Package model.",
    version="1.0.0"
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_REPO_FILE = BASE_DIR / "model_repo.txt"
MODEL_OUTPUT_FILE = BASE_DIR / "model_output.txt"

# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------
MODEL_REPO = (
    MODEL_REPO_FILE.read_text().strip()
    if MODEL_REPO_FILE.exists()
    else os.getenv("HF_MODEL_REPO")
)

MODEL_FILENAME = (
    MODEL_OUTPUT_FILE.read_text().strip()
    if MODEL_OUTPUT_FILE.exists()
    else os.getenv("MODEL_FILENAME")
)

HF_TOKEN = os.getenv("HF_TOKEN_ML")

if not MODEL_REPO:
    raise RuntimeError("Model repository is not configured." )

if not MODEL_FILENAME:
    raise RuntimeError("Model filename is not configured." )

# ---------------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------------
class CustomerInput(BaseModel):
    Age: float
    TypeofContact: str
    CityTier: int
    Occupation: str
    Gender: str
    NumberOfPersonVisiting: int
    PreferredPropertyStar: float
    MaritalStatus: str
    NumberOfTrips: float
    Passport: int
    OwnCar: int
    NumberOfChildrenVisiting: float
    Designation: str
    MonthlyIncome: float
    PitchSatisfactionScore: int
    ProductPitched: str
    NumberOfFollowups: float
    DurationOfPitch: float

INPUT_COLUMNS = list(CustomerInput.model_fields.keys())

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model_path = hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILENAME,token=HF_TOKEN)
model = joblib.load(model_path)

if hasattr(model, "steps"):
    model_type = type(model.steps[-1][1]).__name__
    pipeline_steps = [name for name, _ in model.steps]
else:
    model_type = type(model).__name__
    pipeline_steps = []

# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Wellness Tourism Prediction API",
        "status": "online"
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILENAME,
        "model_type": model_type,
        "input_features": len(INPUT_COLUMNS),
        "pipeline_steps": pipeline_steps
    }

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
@app.post("/predict")
def predict(customer: CustomerInput):

    try:
        data = customer.model_dump()
        input_df = pd.DataFrame( [[data[col] for col in INPUT_COLUMNS]], columns=INPUT_COLUMNS)
        prediction = int(model.predict(input_df)[0] )
        probability = float(  model.predict_proba(input_df)[0][1] )

        return {
            "prediction": prediction,
            "prediction_label": (
                "Likely to Purchase"
                if prediction == 1
                else "Unlikely to Purchase"  ),
            "purchase_probability": round(
                probability,
                4
            )
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
