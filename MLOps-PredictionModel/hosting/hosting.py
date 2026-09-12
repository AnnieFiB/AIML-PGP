
import os

from huggingface_hub import HfApi, create_repo


# ---------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN_ML")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO")


if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN_ML environment variable is not set."
    )

if not HF_SPACE_REPO:
    raise ValueError(
        "HF_SPACE_REPO environment variable is not set."
    )


# ---------------------------------------------------------
# CREATE / CONNECT TO SPACE
# ---------------------------------------------------------

print(
    f"Preparing Hugging Face Space: "
    f"{HF_SPACE_REPO}"
)

create_repo(
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    token=HF_TOKEN,
    exist_ok=True
)


# ---------------------------------------------------------
# HUGGING FACE API
# ---------------------------------------------------------

api = HfApi(
    token=HF_TOKEN
)


# ---------------------------------------------------------
# ADD MODEL HUB TOKEN TO SPACE
# ---------------------------------------------------------

# Allows the deployed Streamlit application to download
# a private model from Hugging Face Model Hub.
api.add_space_secret(
    repo_id=HF_SPACE_REPO,
    key="HF_TOKEN_ML",
    value=HF_TOKEN
)


# ---------------------------------------------------------
# UPLOAD DEPLOYMENT FILES
# ---------------------------------------------------------

print(
    "\nUploading deployment files..."
)

api.upload_folder(
    folder_path="deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    path_in_repo=""
)


print(
    f"\nDeployment successful: "
    f"{HF_SPACE_REPO}"
)
