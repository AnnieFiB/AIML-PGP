
import os
from huggingface_hub import HfApi

# Get Hugging Face credentials and Space repository
token = os.getenv("HF_TOKEN_ML")
space_repo = os.getenv("HF_SPACE_REPO")


# Connect to Hugging Face
api = HfApi(token=token)


# Upload deployment files to Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=space_repo,
    repo_type="space",
    path_in_repo=""
)

print(
    f"Deployment files uploaded successfully to "
    f"Hugging Face Space: {space_repo}"
)
