import mlflow
import mlflow.sklearn
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from src.train_utils import *



def main(repo_id=None,model_id=None):
    print("\n=== STEP 6: MODEL TRAINING STARTED ===\n")

    # ---------------------------------------------------------
    # 1. Load train/test splits from Hugging Face (Step 4 output)
    # ---------------------------------------------------------
    print("Loading train/test splits from Hugging Face...")

    dfs = load_from_hf(["train.csv", "test.csv"], repo_id=repo_id)

    train_df = dfs["train.csv"]
    test_df = dfs["test.csv"]

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")

    # ---------------------------------------------------------
    # 2. Build preprocessing pipeline (OneHotEncoder + StandardScaler)
    # ---------------------------------------------------------
    print("\n Building preprocessing pipeline...")
    preprocessor = build_preprocessor(train_df)

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]

    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    # ---------------------------------------------------------
    # 3. Load models + parameter grids
    # ---------------------------------------------------------
    print("\n Loading models and parameter grids...")

    models, param_grids = get_models_and_params()

    # ---------------------------------------------------------
    # 4. Train + tune + log each model
    # ---------------------------------------------------------
    results = {}

    for name, model in models.items():
        print(f"\n  =========== Model: {name} ===========")
        best_model, f1 ,acc = train_and_tune(
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
            "accuracy": acc
        }


    # ---------------------------------------------------------
    # 5. Display summary table of all models
    # ---------------------------------------------------------
    summarize_models(results)

    # ---------------------------------------------------------
    # 6. Select best model
    # ---------------------------------------------------------
    print("\n Selecting best model based on F1 score...")

    best_name, best_model = select_best_model(results)

    # ---------------------------------------------------------
    # 7. Save + upload best model to Hugging Face Model Hub
    # ---------------------------------------------------------
    print("\n Uploading best model to Hugging Face Model Hub...")

    local_model_path = f"models/{best_name}.joblib"
    register_model_hf(
        model=best_model,
        model_name=best_name,
        model_repo=model_id
    )


    print("\n=== STEP 6 COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
