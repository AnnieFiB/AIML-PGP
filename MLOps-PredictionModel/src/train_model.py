
import os
import warnings
from dotenv import load_dotenv

from src.data_prep import main as prepare_data
from src.train_utils import (
    split_data,
    build_preprocessor,
    get_models_and_params,
    train_and_tune,
    summarize_models,
    select_best_model,
    register_model_hf
)
from src.hf_utils import (load_from_hf, upload_to_hf)


warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

DATASET_REPO = os.getenv("HF_DATASET_REPO")
MODEL_REPO = os.getenv("HF_MODEL_REPO")

TARGET_COL = "ProdTaken"
FILENAME = "tourism.csv"


def main(repo_id=None, model_id=None):
    """
    Run the complete model-building workflow:
    load and clean data, split, preprocess, train,
    evaluate, select and register the best model.
    """

    repo_id = repo_id or DATASET_REPO
    model_id = model_id or MODEL_REPO

    # 1. Load and clean data
    print("\n========== 1. DATA PREPARATION ==========")

    df = prepare_data(
        FILENAME,
        repo_id=repo_id
    )

    # 2. Split data
    print("\n========== 2. TRAIN-TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = split_data(
    df,
    target_col=TARGET_COL,
    repo_id=repo_id
)

    # Load split datasets from Hugging Face
    dfs = load_from_hf(
        ["train.csv", "test.csv"],
        repo_id=DATASET_REPO
    )

    train_df = dfs["train.csv"]
    test_df = dfs["test.csv"]

    # Separate X and y
    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]

    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]


    # 3. Build preprocessing pipeline
    print("\n========== 3. PREPROCESSING ==========")

    preprocessor = build_preprocessor(X_train)

    # 4. Load models and parameter grids
    print("\n========== 4. MODEL TRAINING ==========")

    models, param_grids = get_models_and_params()
    results = {}

    # 5. Train and tune all models
    for name, model in models.items():

        best_model, f1, accuracy = train_and_tune(
            model_name=name,
            model=model,
            params=param_grids[name],
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results[name] = {
            "model": best_model,
            "f1": f1,
            "accuracy": accuracy
        }

    # 6. Compare models
    print("\n========== 5. MODEL COMPARISON ==========")

    summarize_models(results)

    # 7. Select best model
    print("\n========== 6. BEST MODEL ==========")

    best_name, best_model = select_best_model(results)

    # 8. Save and upload best model
    print("\n========== 7. MODEL REGISTRATION ==========")

    register_model_hf(
        model=best_model,
        model_name=best_name,
        model_repo=model_id
    )

    # Save outputs for GitHub Actions

    print("\nSaving model details for GitHub Actions...")

    local_model_path = f"models/{best_name}.joblib"

    model_filename = best_name + ".joblib"

    with open("deployment/model_output.txt", "w") as f:
        f.write(model_filename)

    with open("deployment/model_repo.txt", "w") as f:
        f.write(model_id)

    print(
        f"Saved model name: {model_filename} "
        "→ deployment/model_output.txt"
    )

    print(
        f"Saved model repo: {model_id} "
        "→ deployment/model_repo.txt"
    )

    print("\n========== PIPELINE COMPLETED ==========")

    return best_model, local_model_path, best_name, model_id


if __name__ == "__main__":
    main()
