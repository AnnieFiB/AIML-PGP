
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

from src.hf_utils import upload_to_hf

filepaths= os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------
# 1. SPLIT DATA
# ---------------------------------------------------------
def split_data(
    df,
    target_col="ProdTaken",
    test_size=0.20,
    random_state=42,
    repo_id=None
):
    """
    Split the cleaned dataset into stratified training and test sets,
    save both datasets as CSV files, and upload them to Hugging Face.
    """

    print("\n STEP 2.1: SPLIT DATA ")

    # Separate predictors and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nPredictor features: {X.shape[1]}")
    print(f"Target variable: {target_col}")

    # Stratified train/test split
    print("\nSplitting data using stratification...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTraining features: {X_train.shape}")
    print(f"Test features:     {X_test.shape}")

    # Check target distribution
    print("\nTraining target distribution (%):")
    print(
        y_train.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nTest target distribution (%):")
    print(
        y_test.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # Combine features and target
    print("\nCombining features and target for saving...")
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save CSV files
    os.makedirs("data", exist_ok=True)

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain saved: {train_path} {train_df.shape}")
    print(f"Test saved:  {test_path} {test_df.shape}")

    # Upload split datasets to Hugging Face
    if repo_id:
        print(f"\nUploading split datasets to Hugging Face: {repo_id}")

        upload_to_hf(
            local_path=[train_path, test_path],
            repo_id=repo_id
        )

        print("Train and test datasets uploaded successfully.")
    else:
        print("\nNo repo_id provided - upload skipped.")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 2. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------
def build_preprocessor(X_train):
    """
    Build preprocessing steps from the training features.

    Numerical variables are scaled and categorical
    variables are one-hot encoded.
    """

    print("\n STEP : PREPROCESSING CLEAN DATA ")

    numeric_features = (
        X_train
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_features = (
        X_train
        .select_dtypes(include=["object", "string"])
        .columns
        .tolist()
    )

    print("\nNumerical features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    print("\nBuilding preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )
    print("Preprocessing pipeline created successfully.")

    return preprocessor


# ---------------------------------------------------------
# 3. MODEL DEFINITIONS AND PARAMETER GRIDS
# ---------------------------------------------------------
def get_models_and_params():
    """
    Define candidate classification models and
    hyperparameter search grids.
    """

    models = {
        "decision_tree": DecisionTreeClassifier(random_state=42),

        "bagging": BaggingClassifier(random_state=42),

        "random_forest": RandomForestClassifier(random_state=42),

        "adaboost": AdaBoostClassifier(random_state=42),

        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
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
# 4. TRAIN, TUNE AND LOG MODEL
# ---------------------------------------------------------
def train_and_tune(
    model_name,
    model,
    params,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train and tune a model using GridSearchCV and
    log its performance in MLflow.
    """

    print(f"\n TRAINING: {model_name}")

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

        # Train only on training data
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Evaluate on unseen test data
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log results
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(f"\nBest Parameters: {grid.best_params_}")
        display(f"Accuracy: {accuracy:.4f}")
        display(f"F1 Score: {f1:.4f}")

    return best_model, f1, accuracy


# ---------------------------------------------------------
# 5. MODEL PERFORMANCE SUMMARY
# ---------------------------------------------------------
def summarize_models(results):
    """
    Create a summary table of model performance.
    """

    summary = []

    for name, metrics in results.items():

        summary.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "F1 Score": round(metrics["f1"], 4)
        })

    df_summary = (
        pd.DataFrame(summary)
        .sort_values(
            "F1 Score",
            ascending=False
        )
    )

    print("\n==========Step 5: MODEL PERFORMANCE ==========")
   # display(df_summary.to_string(index=False))

    return df_summary


# ---------------------------------------------------------
# 6. SELECT BEST MODEL
# ---------------------------------------------------------
def select_best_model(results):
    """
    Select the model with the highest F1 score.
    """

    best_name = max(
        results,
        key=lambda model: results[model]["f1"]
    )

    best_model = results[best_name]["model"]

    print("\n==========Step 6: BEST MODEL ==========")
    print(f"Model: {best_name}")
    print(f"F1 Score: {results[best_name]['f1']:.4f}")

    return best_name, best_model


# ---------------------------------------------------------
# 7. SAVE AND UPLOAD BEST MODEL
# ---------------------------------------------------------
def register_model_hf(
    model,
    model_name,
    model_repo,
    token=None
):
    """
    Save the best model and upload it to Hugging Face Model Hub.
    """

    if token is None:
        token = os.getenv("HF_TOKEN_ML")

    print("\n========== Step 7: REGISTER MODEL ==========")

    os.makedirs("models", exist_ok=True)

    save_path = f"models/{model_name}.joblib"

    # Save model locally
    joblib.dump(model, save_path)

    print(f"Model saved: {save_path}")

    # Upload model to Hugging Face Model Hub
    upload_to_hf(
        local_path=save_path,
        repo_id=model_repo,
        repo_type="model",
        token=token
    )

    print(
        f"Model uploaded to Hugging Face Model Hub: "
        f"{model_repo}/{model_name}.joblib"
    )

    return save_path
