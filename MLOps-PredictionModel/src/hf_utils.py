from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi, upload_file
from datasets import load_dataset
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()  # loads .env file


def init_hf(dataset_repo=None, model_repo=None):
    """
    One-step Hugging Face initializer:
    - Check token
    - Check or create dataset repo
    - Check or create model repo
    - Inject/eject repo IDs into .env if missing
    Returns: (token, dataset_repo, model_repo)
    """

    api = HfApi()

    # -------------------------------
    # 1. TOKEN
    # -------------------------------
    token = os.getenv("HF_TOKEN_ML")
    if not token:
        print("\n HF_TOKEN_ML is missing.")
        print("Create one at: https://huggingface.co/settings/tokens")
        print("Steps:")
        print("  1. Click 'New Token'")
        print("  2. Choose 'Write' permission")
        print("  3. Add to .env as: HF_TOKEN_ML=your_token_here\n")
        raise ValueError("HF_TOKEN_ML missing.")

    # -------------------------------
    # 2. DATASET REPO (inject)
    # -------------------------------
    if dataset_repo is None:
        dataset_repo = os.getenv("HF_DATASET_REPO")

    if not dataset_repo:
        print("\n No HF_DATASET_REPO found. Creating a new dataset repo...\n")
        api.create_repo(dataset_repo, repo_type="dataset", token=token, exist_ok=True)
        print(f" Created dataset repo: {dataset_repo}")

        # Write back to .env
        with open(".env", "a") as f:
            f.write(f"\n HF_DATASET_REPO={dataset_repo}\n")
        print(" Added HF_DATASET_REPO to .env\n")

    # Ensure dataset repo exists
    try:
        api.repo_info(dataset_repo, repo_type="dataset", token=token)
        print(f"\n  Dataset repo exists: {dataset_repo}")
    except Exception:
        print(f" Dataset repo not found. Creating: {dataset_repo}")
        api.create_repo(dataset_repo, repo_type="dataset", token=token, exist_ok=True)
        print(f" Dataset repo created: {dataset_repo}")

    # -------------------------------
    # 3. MODEL REPO (inject)
    # -------------------------------
    if model_repo is None:
        model_repo = os.getenv("HF_MODEL_REPO")

    if not model_repo:
        print("\n No HF_MODEL_REPO found. Creating a new model repo...\n")
        api.create_repo(model_repo, repo_type="model", token=token, exist_ok=True)
        print(f" Created model repo: {model_repo}")

        # Write back to .env
        with open(".env", "a") as f:
            f.write(f"\n HF_MODEL_REPO={model_repo}\n")
        print(" Added HF_MODEL_REPO to .env\n")

    # Ensure model repo exists
    try:
        api.repo_info(model_repo, repo_type="model", token=token)
        print(f" Model repo exists: {model_repo}")
    except Exception:
        print(f" Model repo not found. Creating: {model_repo}")
        api.create_repo(model_repo, repo_type="model", token=token, exist_ok=True)
        print(f" Model repo created: {model_repo}")

    return token, dataset_repo, model_repo



# ---------------------------------------------------------
# 4. UPLOAD FILES TO DATASET REPO
# ---------------------------------------------------------

def upload_to_hf(local_path, token=None,repo_type="dataset", repo_id=None):
    """
    Upload one or multiple files to a Hugging Face dataset or model repository.
    """

    if token is None:
        token = os.getenv("HF_TOKEN_ML")

    if repo_id is None:
            repo_id = os.getenv("HF_DATASET_REPO")

    # If multiple files are provided
    if isinstance(local_path, list):
        for file in local_path:
            remote_name = os.path.basename(file)
            upload_file(
                path_or_fileobj=file,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
            print(f"Uploaded {file} → {remote_name} in HF {repo_type} repo: {repo_id}")
        return

    # Single file
    remote_name = os.path.basename(local_path)
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )

    print(f"Uploaded {local_path} → {remote_name} in HF {repo_type} repo: {repo_id}")


# ---------------------------------------------------------
# 5. UPLOAD MODEL FILES
# ---------------------------------------------------------
def upload_model_files(filepaths, model_repo, token):
    """
    Upload one or multiple model files to HF model repo.
    """
    for path in filepaths:
        remote_name = os.path.basename(path)
        upload_file(
            path_or_fileobj=path,
            path_in_repo=remote_name,
            repo_id=model_repo,
            repo_type="model",
            token=token
        )
        print(f"Uploaded model file: {path} → {model_repo}/{remote_name}")

# ---------------------------------------------------------
# 6. LOAD FILES FROM HUGGING FACE DATASET REPO
# ---------------------------------------------------------

def load_from_hf(filenames, repo_id=None):
    """
    Load one or multiple CSV files from a Hugging Face dataset repo.

    Parameters
    ----------
    filenames : str or list
        One filename or a list of filenames to load.
        Example: "train.csv" or ["train.csv", "test.csv"]

    repo_id : str, optional
        Hugging Face repo ID. If not provided, uses DATASET_REPO from .env.

    Returns
    -------
    dict : {filename: pandas.DataFrame}
        A dictionary mapping each filename to its loaded DataFrame.
    """

    # Use .env repo if none provided
    if repo_id is None:
        repo_id = os.getenv("HF_DATASET_REPO")

    # Normalize input to list
    if isinstance(filenames, str):
        filenames = [filenames]

    dfs = {}

    for fname in filenames:
        print(f"  Downloading {fname} ...")

        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset"
        )

        df = pd.read_csv(file_path)
        print(f"  Loaded {fname}: {df.shape}")

        dfs[fname] = df

    print("All requested files loaded successfully.\n")
    return dfs
