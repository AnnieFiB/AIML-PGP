
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from pathlib import Path

app = Flask(__name__)

# Base directory of app.py
BASE_DIR = Path(__file__).resolve().parent

# Load model
model = joblib.load(
    BASE_DIR / "superkart_sales_forecasting_model_v1_0.joblib"
)

# Load raw input columns expected by the API
with open(BASE_DIR / "model_input_columns.json", "r") as f:
    input_columns = json.load(f)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Forecasting API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "input_columns": input_columns
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON payload received"
            }), 400

        input_df = pd.DataFrame([data])

        missing_cols = [
            col for col in input_columns
            if col not in input_df.columns
        ]

        if missing_cols:
            return jsonify({
                "error": "Missing required input columns",
                "missing_columns": missing_cols,
                "expected_columns": input_columns
            }), 400

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
