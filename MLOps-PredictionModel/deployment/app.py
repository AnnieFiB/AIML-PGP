from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from hf_inference import load_model

app = FastAPI(title="Wellness Tourism Purchase Prediction API")

class Customer(BaseModel):
    Age: int
    Duration: int
    MonthlyIncome: float

# These will be set from environment/config in deployment, not hardcoded in code
MODEL_REPO = "Omotayof/wellness-tourism-model"
MODEL_FILENAME = "xgboost.joblib"  # set via CI/CD based on training output

model = load_model(MODEL_REPO, MODEL_FILENAME)

@app.post("/predict")
def predict_purchase(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    pred = model.predict(df)[0]
    return {
        "prediction": int(pred),
        "message": "Will purchase package" if pred == 1 else "Will not purchase package"
    }
