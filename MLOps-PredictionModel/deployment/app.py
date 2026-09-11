import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from hf_inference import load_model

app = FastAPI(title="Wellness Tourism Purchase Prediction API")

class Customer(BaseModel):
    Age: int
    Duration: int
    MonthlyIncome: float

MODEL_REPO = os.getenv("MODEL_REPO")
MODEL_FILENAME = os.getenv("MODEL_FILENAME")

model = load_model(MODEL_REPO, MODEL_FILENAME)

@app.post("/predict")
def predict_purchase(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    pred = model.predict(df)[0]
    return {
        "prediction": int(pred),
        "message": "Will purchase package" if pred == 1 else "Will not purchase package"
    }
