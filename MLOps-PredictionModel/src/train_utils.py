import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier


from src.hf_utils import *
from src.data_prep import *


# ---------------------------------------------------------
# 1. PREPROCESSING PIPELINE (OneHotEncoder + StandardScaler)
# ---------------------------------------------------------
def build_preprocessor(df, target_col="ProdTaken"):
    print("Building preprocessing pipeline...")

    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"  Numeric features: {numeric_features}")
    print(f"  Categorical features: {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# 2. MODEL DEFINITIONS + PARAMETER GRIDS
# ---------------------------------------------------------

def get_models_and_params():
    models = {
        "decision_tree": DecisionTreeClassifier(),
        "bagging": BaggingClassifier(),
        "random_forest": RandomForestClassifier(),
        "adaboost": AdaBoostClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    param_grids = {
        "decision_tree": {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },

        "bagging": {
            "model__n_estimators": [10, 50, 100],
            "model__max_samples": [0.5, 0.7, 1.0],
            "model__max_features": [0.5, 0.7, 1.0]
        },

        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10]
        },

        "adaboost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.5, 1.0]
        },

        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5]
        },

        "xgboost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0]
        }
    }

    return models, param_grids



# ---------------------------------------------------------
# 3. TRAIN + TUNE + LOG PARAMETERS (MLflow)
# ---------------------------------------------------------
def train_and_tune(model_name, model, params, preprocessor, X_train, y_train, X_test, y_test):
    print(f"\nTraining and tuning model: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    with mlflow.start_run(run_name=model_name):
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"\n  =========== Model: {model_name} ===========")
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        return best_model, f1,acc

# ---------------------------------------------------------
# 4. SUMMARY TABLE OF ALL MODELS
# ---------------------------------------------------------
def summarize_models(results):

    summary = []

    for name, metrics in results.items():
        summary.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "F1 Score": round(metrics["f1"], 4)
        })

    df_summary = pd.DataFrame(summary).sort_values("F1 Score", ascending=False)
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    display(df_summary)

    return df_summary

# ---------------------------------------------------------
# 5. SELECT BEST MODEL
# ---------------------------------------------------------
def select_best_model(results):
    best_name = max(results, key=lambda m: results[m]["f1"])
    best_model = results[best_name]["model"]

    print(f"\nBest model selected: {best_name}")
    print(f"  F1 Score: {results[best_name]['f1']:.4f}")

    return best_name, best_model


# ---------------------------------------------------------
# 6. SAVE + UPLOAD BEST MODEL TO HUGGING FACE MODEL HUB
# ---------------------------------------------------------
def register_model_hf(model, model_name, model_repo, token=None):
    """
    Register the best model in the Hugging Face Model Hub.
    """

    if token is None:
        token = os.getenv("HF_TOKEN_ML")

    print("\n Registering best model in Hugging Face Model Hub...")

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{model_name}.joblib"
    joblib.dump(model, save_path)

    upload_model_files(
        filepaths=[save_path],
        model_repo=model_repo,
        token=token
    )


    print(f"Model uploaded to HF Model Hub: {model_repo}/{model_name}.joblib")


