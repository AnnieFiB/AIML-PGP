# 🌍 Tourism Wellness Package Prediction

An end-to-end **MLOps classification project** that predicts whether a customer is likely to purchase a wellness tourism package.

## Project Objective

Automate customer purchase prediction to support targeted marketing and improve customer acquisition.

## MLOps Workflow

```text
                         GitHub Push
                              │
                              ▼
┌──────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────────────────┐
│ STAGE 1              │ │ STAGE 2                   │ │ STAGE 3                            │
│ DATASET REGISTRY     │▶│ MODEL DEVELOPMENT         │▶│ DEPLOYMENT                         │
│                      │ │                           │ │                                    │
│ tourism.csv          │ │ Clean → Split → Process   │ │ GitHub Actions                     │
│      ↓               │ │ Train → Tune → MLflow     │ │       │                            │
│ HF Dataset Hub       │ │ Evaluate → Select         │ │   ┌───┴───────────────┐            │
│                      │ │       ↓                   │ │   ▼                   ▼            │
│                      │ │ HF Model Hub              │ │ HF Docker Space   Streamlit Cloud │
│                      │ │                           │ │ Streamlit + API    Prediction App  │
└──────────────────────┘ └───────────────────────────┘ └────────────────────────────────────┘
```

The **Hugging Face Docker Space** combines a Streamlit interface with a FastAPI prediction service, while the existing application is also available through **Streamlit Community Cloud**.

## Models

Six classification models were evaluated:

`Decision Tree` • `Bagging` • `Random Forest` • `AdaBoost` • `Gradient Boosting` • `XGBoost`

**F1 Score** is used for final model selection.

## Tech Stack

| Component           | Technology                                 |
| ------------------- | ------------------------------------------ |
| Data & Modelling    | Pandas, Scikit-learn, XGBoost              |
| Experiment Tracking | MLflow                                     |
| Dataset Registry    | Hugging Face Dataset Hub                   |
| Model Registry      | Hugging Face Model Hub                     |
| API                 | FastAPI                                    |
| User Interface      | Streamlit                                  |
| CI/CD               | GitHub Actions                             |
| Deployment          | HF Docker Space, Streamlit Community Cloud |

## Project Structure

```text
MLOps-Tourism-Package/
├── src/                       # Data preparation & training
├── notebooks/                 # Analysis notebook
├── data/                      # Dataset
├── models/                    # Local model artifacts
│
├── deployment/
│   ├── streamlit_app.py       # Streamlit Community Cloud
│   ├── requirements.txt
│   │
│   └── hf_space/
│       ├── app.py             # FastAPI backend
│       ├── streamlit.py       # Streamlit frontend
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml # CI/CD pipeline
│
├── requirements.txt
└── README.md
```

## Deployment

Two application deployments are maintained:

**Hugging Face Docker Space**

```text
Streamlit UI → FastAPI → HF Model Hub → Prediction
```

**Streamlit Community Cloud**

```text
Streamlit App → HF Model Hub → Prediction
```

The Hugging Face Space is automatically updated by the GitHub Actions deployment stage.

## Run Locally

### Model Training

```bash
pip install -r requirements.txt
python -m src.train_model
```

### Streamlit Community App

```bash
streamlit run deployment/streamlit_app.py
```

### Hugging Face Docker Application

```bash
docker build -t tourism-prediction deployment/hf_space
docker run -p 7860:7860 tourism-prediction
```

## Output

The deployed applications use **18 customer features** to return:

- Purchase prediction
- Purchase probability
- Model and pipeline information
