import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import src.hf_utils as hf_utils
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

RAW_DATASET = os.getenv("HF_DATASET_REPO")


def load_raw_data():
    print("Loading raw data from Hugging Face...")
    print(f"Raw data loaded successfully. Dataset shape: {df.shape}")  
    ds = load_dataset(RAW_DATASET)
    df = ds["train"].to_pandas()
     
    return df

def clean_data(df):
    print("Cleaning data...")

    # Dynamically detect index-like and ID-like columns
    index_like_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    id_like_cols = [col for col in df.columns if "id" in col.lower()]

    drop_cols = index_like_cols + id_like_cols

    print(f"1. Dropping unnecessary columns: {drop_cols if drop_cols else 'None'}")
    df = df.drop(columns=drop_cols, errors="ignore")

    # Numerical fill (kept for robustness even though your dataset has no missing values)
    print("2. Filling missing values for numerical columns (if any)...")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    missing_before = df[num_cols].isnull().sum().sum()

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    missing_after = df[num_cols].isnull().sum().sum()
    filled = missing_before - missing_after

    print(f"{filled} missing values filled with median for numerical columns.")
    print("3. Data cleaning completed.")

    return df


def split_data(df, target_col="ProdTaken", test_size=0.2, random_state=42):
    print("\nSplitting data into train and test sets...")

    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"  Total samples: {len(df)}")
    print(f"  Features shape: {X.shape}")

    # Class distribution BEFORE split
    print("\n  Target distribution BEFORE split:")
    print(y.value_counts(normalize=True).rename("proportion"))

    print("\n  NOTE: This dataset is binary classification and is imbalanced.")
    print("  Stratified splitting ensures class proportions remain consistent in train/test.\n")

    # Stratified split (handles imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Combine back into train/test DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Class distribution AFTER split
    print("  Target distribution AFTER split:")
    print("  Train:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print("\n  Test:")
    print(y_test.value_counts(normalize=True).rename("proportion"))

    # Save to /data folder
    os.makedirs("data", exist_ok=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Train set saved to: {train_path} ({train_df.shape})")
    print(f"  Test set saved to: {test_path} ({test_df.shape})")
    print("Data splitting completed.\n")

    return train_path, test_path


def upload_splits(train_path, test_path):
    hf_utils.upload_to_hf(train_path)
    hf_utils.upload_to_hf(test_path)

def main():
    print("Loading raw data from Hugging Face...")
    df = load_raw_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting and saving locally...")
    train_path, test_path = split_data(df)

    print("Uploading train/test splits to Hugging Face...")
    upload_splits(train_path, test_path)

    print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
