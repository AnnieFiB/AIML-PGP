
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from pathlib import Path

from deployment_files.feature_engineering import create_features

app = Flask(__name__)

# Base directory of app.py
BASE_DIR = Path(__file__).resolve().parent

# Load model
model = joblib.load(
    BASE_DIR / "superkart_sales_forecasting_model_v1_0.joblib"
)

# Load input columns
with open(BASE_DIR / "model_input_columns.json", "r") as f:
    input_columns = json.load(f)


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "SuperKart Sales Forecasting API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        input_df = pd.DataFrame([data])

        input_df = input_df[input_columns]

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_sales": round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860
    )
