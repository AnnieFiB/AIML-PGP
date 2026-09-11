import joblib
from huggingface_hub import hf_hub_download
from src.train_model import main
from src.hf_utils import init_hf

# Set the Hugging Face dataset and model repository names
DATASET_REPO = "Omotayof/wellness-tourism-customers"
MODEL_REPO = "Omotayof/wellness-tourism-model"


# Run training pipeline and get dynamic model info
best_model, model_path, model_name, model_repo = main(
    repo_id=DATASET_REPO,
    model_id=MODEL_REPO
)


def load_model(repo_id: str, filename: str):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )
    return joblib.load(path)

