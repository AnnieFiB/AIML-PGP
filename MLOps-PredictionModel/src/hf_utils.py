from huggingface_hub import HfApi, upload_file
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file

DATASET_REPO = os.getenv("HF_DATASET_REPO")

if DATASET_REPO is None:
    raise ValueError("HF_DATASET_REPO not found. Add it to your .env file or GitHub Secrets.")

def get_token():
    token = os.getenv("HF_TOKEN_ML")
    if token is None:
        raise ValueError("HF_TOKEN_ML not found. Add it to your .env file or GitHub Secrets.")
    return token

def upload_to_hf(local_path):
    api = HfApi()
    token = get_token()

    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=os.path.basename(local_path),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=token,
    )

    print(f"Uploaded {local_path} → {os.path.basename(local_path)} in HF dataset.")

