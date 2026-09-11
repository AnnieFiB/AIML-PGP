
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display

import src.hf_utils as hf_utils


load_dotenv()
DATASET_REPO = os.getenv("HF_DATASET_REPO")


def inspect_data(df):
    """
    Inspect and print the structure and quality of the raw dataset.
    """
    print("\n========== RAW DATA INSPECTION ==========")

    print(f"\nDataset shape: {df.shape}")

    print("\nFirst 5 rows:")
    display(df.head())

    print("\nData types and non-null values:")
    df.info()

    print("\nMissing values:")
    display(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def target_distribution(df, target="ProdTaken"):
    """
    Print the count and percentage distribution of the target variable.
    """
    distribution = pd.DataFrame({
        "Count": df[target].value_counts(),
        "Percentage (%)": (
            df[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    })

    print("\n========== TARGET DISTRIBUTION ==========")
    display(distribution)


def clean_data(df):
    """
    Clean the dataset by removing unnecessary columns and
    standardising categorical values before model building.
    """
    df = df.copy()

    print("\n========== DATA CLEANING ==========")

    # Step 1: Check and remove true duplicates
    duplicates = df.duplicated().sum()

    print(f"\nStep 1 - Duplicate rows found: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate rows removed.")

    print(f"Dataset shape: {df.shape}")

    # Step 2: Remove non-predictive columns
    drop_cols = [
        col for col in df.columns
        if col.lower().startswith("unnamed") or col == "CustomerID"
    ]

    print(f"\nStep 2 - Columns removed: {drop_cols}")

    df = df.drop(columns=drop_cols, errors="ignore")

    print(f"Remaining columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"Dataset shape: {df.shape}")

    # Step 3: Standardise categorical values
    cat_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns

    print("\nStep 3 - Categorical columns:")
    print(cat_cols.tolist())

    df[cat_cols] = df[cat_cols].apply(
        lambda col: col.astype("string").str.strip()
    )

    print("Leading/trailing spaces removed.")

    # Step 4: Correct known inconsistent Gender label
    if "Gender" in df.columns:
        print("\nStep 4 - Gender values before correction:")
        print(df["Gender"].value_counts())

        df["Gender"] = df["Gender"].replace(
            "Fe Male", "Female"
        )

        print("\nGender values after correction:")
        print(df["Gender"].value_counts())

    # Step 5: Final missing-value check
    print("\nStep 5 - Missing values after cleaning:")
    display(df.isna().sum())

    print("\n========== CLEANING COMPLETE ==========")
    print(f"Final dataset shape: {df.shape}")
    print(f"Total missing values: {df.isna().sum().sum()}")

    return df


def main(filename, repo_id=None):
    """
    Load raw data from Hugging Face, inspect and clean it,
    then return the cleaned dataframe for model building.
    """
    repo_id = repo_id or DATASET_REPO

    if not repo_id:
        raise ValueError(
            "Set HF_DATASET_REPO in .env or provide repo_id."
        )

   # ========== STEP 1: LOAD DATA ==========

    print(f"Repository: {repo_id}")
    print(f"Filename: {filename}")

    dfs = hf_utils.load_from_hf(
        filenames=filename,
        repo_id=repo_id
    )

    df = dfs[filename]

    print("\nData loaded successfully.")
    print(f"Loaded shape: {df.shape}")

    # ========== STEP 2: INSPECT DATA ==========
    inspect_data(df)

    # ========== STEP 3: CHECK TARGET ==========
    target_distribution(df)

    # ========== STEP 4: CLEAN DATA ==========
    cleaned_df = clean_data(df)

    # ========== FINAL CLEANED DATA ==========

    display("\nFirst 5 rows:")
    display(cleaned_df.head())

    print("\nFinal data types:")
    cleaned_df.info()

    print("\nFinal target distribution:")
    target_distribution(cleaned_df)

    return cleaned_df


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python -m src.data_prep tourism.csv"
        )

    filename = sys.argv[1]
    repo_id = sys.argv[2] if len(sys.argv) > 2 else None

    main(filename, repo_id)
