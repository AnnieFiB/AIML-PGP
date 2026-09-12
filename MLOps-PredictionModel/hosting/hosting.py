import os

from huggingface_hub import HfApi


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
# CONNECT TO HUGGING FACE
# ---------------------------------------------------------

api = HfApi(
    token=HF_TOKEN
)


# ---------------------------------------------------------
# ADD TOKEN TO SPACE SECRETS
# ---------------------------------------------------------

# Required if the model repository is private.
api.add_space_secret(
    repo_id=HF_SPACE_REPO,
    key="HF_TOKEN_ML",
    value=HF_TOKEN
)


# ---------------------------------------------------------
# UPLOAD DEPLOYMENT FILES
# ---------------------------------------------------------

print(
    f"Uploading deployment files to "
    f"Hugging Face Space: {HF_SPACE_REPO}"
)


api.upload_folder(
    folder_path="deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    path_in_repo=""
)


print(
    f"Deployment completed successfully: "
    f"{HF_SPACE_REPO}"
)
