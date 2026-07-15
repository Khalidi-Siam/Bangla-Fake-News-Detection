from pathlib import Path
import os

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))

class DataDownloadConfig(BaseModel):
    hf_dataset_name: str = "hrshihab/BanFakeNews-2.0"
    cleaned_dataset_path: Path = BASE_DIR / "Artifacts" / "data.csv"

class OfflineTokenizeConfig(BaseModel):
    cache_dir: Path = BASE_DIR / "Artifacts" / "tokenized_cache_1024"
    max_length: int = 1024  #for bangla-bert keep 512 for ssm 512, 768, 1024 etc.
    class_weight_path: Path = BASE_DIR / "Artifacts" / "class_weights.npy"
    short_test_subset_path: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset_path: Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class BertFineTuneConfig(BaseModel):
    """File-system paths produced / consumed by BanglaBERT fine-tuning."""
    bert_cache_dir:    Path = BASE_DIR / "Artifacts" / "tokenized_cache_512"
    checkpoint_dir:    Path = BASE_DIR / "Artifacts" / "checkpoints" / "banglabert"
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / "banglabert"
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / "banglabert_results.json"
    short_test_subset: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset:  Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class BertEvaluateConfig(BaseModel):
    """File-system paths consumed / produced by the BanglaBERT evaluation step."""
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / "banglabert"
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / "banglabert_results.json"
    short_test_subset: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset:  Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"

MAMBA_MAX_LENGTH = 1024
PARAMETER_SIZE = 5  # if you change paramter size then need to update d_model and n_layer in params.py 
MAMBA_NAME = f"mamba_{PARAMETER_SIZE}_{MAMBA_MAX_LENGTH}"
class MambaTrainConfig(BaseModel):
    """File-system paths produced / consumed by the Bangla-Mamba training."""
    cache_dir:         Path = BASE_DIR / "Artifacts" / f"tokenized_cache_{MAMBA_MAX_LENGTH}"
    checkpoint_dir:    Path = BASE_DIR / "Artifacts" / "checkpoints" / MAMBA_NAME
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / MAMBA_NAME
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / f"{MAMBA_NAME}_results.json"
    short_test_subset: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset:  Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class MambaEvaluateConfig(BaseModel):
    """File-system paths consumed / produced by the Bangla-Mamba evaluation step."""
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / MAMBA_NAME
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / f"{MAMBA_NAME}_results.json"
    short_test_subset: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset:  Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class BertMLflowConfig(BaseModel):
    """MLflow experiment tracking configuration.

    DagsHub credentials are read from environment variables:
      MLFLOW_TRACKING_USERNAME  — your DagsHub username
      MLFLOW_TRACKING_PASSWORD  — your DagsHub access token / password
    """
    tracking_uri:    str  = "https://dagshub.com/Khalidi-Siam/Bangla-Fake-News-Detection.mlflow"
    experiment_name: str  = "BanglaBERT-Fake-News"
    run_name:        str  = "banglabert_finetune"
    log_model:       bool = True   # set False to skip uploading the ~500 MB model weights


class MambaMLflowConfig(BaseModel):
    """MLflow experiment tracking configuration for Bangla-Mamba.

    DagsHub credentials are read from environment variables:
      MLFLOW_TRACKING_USERNAME  — your DagsHub username
      MLFLOW_TRACKING_PASSWORD  — your DagsHub access token / password
    """
    tracking_uri:    str  = "https://dagshub.com/Khalidi-Siam/Bangla-Fake-News-Detection.mlflow"
    experiment_name: str  = "Bangla-Mamba-Fake-News"
    run_name:        str  = MAMBA_NAME
    log_model:       bool = True   # set False to skip uploading the model weights


class Settings(BaseSettings):
    seed: int = 42
    logging_dir: Path = BASE_DIR / "logs"
    data: DataDownloadConfig = DataDownloadConfig()
    offline_tokenize: OfflineTokenizeConfig = OfflineTokenizeConfig()
    bert_finetune: BertFineTuneConfig = BertFineTuneConfig()
    bert_evaluate: BertEvaluateConfig = BertEvaluateConfig()
    bert_mlflow: BertMLflowConfig = BertMLflowConfig()
    mamba_train: MambaTrainConfig = MambaTrainConfig()
    mamba_evaluate: MambaEvaluateConfig = MambaEvaluateConfig()
    mamba_mlflow: MambaMLflowConfig = MambaMLflowConfig()

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),   # absolute path — works regardless of CWD
        env_file_encoding="utf-8",
        env_file_required=False,           # no error if .env doesn't exist
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


settings = Settings()