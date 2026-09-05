import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

import src.hf_utils as hf_utils

load_dotenv()

# Default dataset repo from .env
DATASET_REPO = os.getenv("HF_DATASET_REPO")


# ---------------------------------------------------------
# CLEANING FUNCTION
# ---------------------------------------------------------
def clean_data(df):
    print("Cleaning data...")

    index_like_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    id_like_cols = [col for col in df.columns if "id" in col.lower()]
    drop_cols = index_like_cols + id_like_cols

    print(f"1. Dropping unnecessary columns: {drop_cols if drop_cols else 'None'}")
    df = df.drop(columns=drop_cols, errors="ignore")

    print("2. Filling missing numerical values (if any)...")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    print("3. Data cleaning completed.\n")
    return df


# ---------------------------------------------------------
# SPLITTING FUNCTION
# ---------------------------------------------------------
def split_data(df, target_col="ProdTaken", test_size=0.2, random_state=42):
    print("Splitting data into train/test sets...")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"  Total samples: {len(df)}")
    print(f"  Features shape: {X.shape}")

    print("\n  Target distribution BEFORE split:")
    print(y.value_counts(normalize=True).rename("proportion"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("\n  Target distribution AFTER split:")
    print("  Train:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\n  Test:")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    os.makedirs("data", exist_ok=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain saved: {train_path} ({train_df.shape})")
    print(f"Test saved: {test_path} ({test_df.shape})\n")

    return train_path, test_path


# ---------------------------------------------------------
# UPLOAD FUNCTION
# ---------------------------------------------------------
def upload_splits(filenames, repo_id=None):
    """
    Upload train/test splits to Hugging Face dataset repo.
    If repo_id is not provided, fall back to .env HF_DATASET_REPO.
    """

    # Fallback to .env repo
    if repo_id is None:
        repo_id = DATASET_REPO


    hf_utils.upload_to_hf(filenames, repo_type="dataset", repo_id=repo_id)
    # hf_utils.upload_to_hf(test_path, repo_type="dataset", repo_id=repo_id)


# ---------------------------------------------------------
# MAIN FUNCTION (USER INPUT + OPTIONAL REPO OVERRIDE)
# ---------------------------------------------------------
def main(filenames, repo_id=None):
    """
    Step 1-5: Load raw data, clean, split, upload.

    Parameters
    ----------
    filenames : str or list
        Raw CSV filename(s) inside the HF dataset repo.

    repo_id : str, optional
        Override HF dataset repo. If None, uses HF_DATASET_REPO from .env.
    """

    print("=== STEP 1-5: DATA PREPARATION STARTED ===\n")

    # Normalize filenames to list
    if isinstance(filenames, str):
        filenames = [filenames]

    # Use .env repo if none provided
    if repo_id is None:
        repo_id = DATASET_REPO

    print(f"Using dataset repo: {repo_id}")
    print(f"Loading raw file(s): {filenames}\n")

    # Load raw files from HF
    dfs = hf_utils.load_from_hf(filenames)
    df = dfs[filenames[0]]

    print(f"Raw data loaded. Shape: {df.shape}\n")
    display(df.head()) # inspect first few rows
    display(df.info()) # inspect data types and non-null counts
    display(df.describe()) # inspect summary statistics
    display(df.isna().sum()) # inspect missing values


    # Clean
    df = clean_data(df)

    # Split
    train_path, test_path = split_data(df)
    filenames = [train_path, test_path]

    # Upload
    print("Uploading train/test splits to Hugging Face...")
    upload_splits(filenames, repo_id=repo_id)


    print("\n=== STEP 5 COMPLETED SUCCESSFULLY ===")


# ---------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Example usage:
    # python src/data_prep.py tourism.csv
    # python src/data_prep.py tourism.csv metadata.csv Omotayof/new-repo

    args = sys.argv[1:]

    if len(args) == 0:
        raise ValueError("Please provide at least one raw filename.")

    # If last argument looks like a repo (contains '/'), treat it as repo override
    if "/" in args[-1]:
        repo = args[-1]
        filenames = args[:-1]
    else:
        repo = None
        filenames = args

    main(filenames, repo_id=repo)
