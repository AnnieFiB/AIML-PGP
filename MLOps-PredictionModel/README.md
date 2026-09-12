# `🌍`Tourism Wellness Package Prediction

An end-to-end **MLOps classification project** that predicts whether a customer is likely to purchase a wellness tourism package.

## Project Objective

Automate customer purchase prediction to support targeted marketing and improve customer acquisition.

## MLOps Workflow

```text
GitHub Push
     ↓
HF Dataset Hub
     ↓
Clean → Split → Preprocess
     ↓
Train → Tune → MLflow
     ↓
Evaluate → Select Best Model
     ↓
HF Model Hub
     ↓
Streamlit Community Cloud
     ↓
Prediction App
```

## Models

Six classification models were evaluated:

`Decision Tree` • `Bagging` • `Random Forest` • `AdaBoost` • `Gradient Boosting` • `XGBoost`

**F1 Score** is used for final model selection.

## Tech Stack

| Component           | Technology                    |
| ------------------- | ----------------------------- |
| Data & Modelling    | Pandas, Scikit-learn, XGBoost |
| Experiment Tracking | MLflow                        |
| Dataset Registry    | Hugging Face                  |
| Model Registry      | Hugging Face                  |
| CI/CD               | GitHub Actions                |
| Deployment          | Streamlit Community Cloud     |

## Project Structure

```text
MLOps-Tourism-Package/
├── src/                 # Data preparation & training
├── notebooks/           # Analysis notebook
├── data/                # Dataset
├── models/              # Model artifacts
├── deployment/          # Streamlit application
├── .github/workflows/   # CI/CD pipeline
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
python -m src.train_model
streamlit run deployment/streamlit_app.py
```

## Output

The deployed application uses **18 customer features** to return:

- Purchase prediction
- Purchase probability
- Model and pipeline information
