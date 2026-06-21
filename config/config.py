from pathlib import Path
import os

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(os.getenv("BASE_DIR", "."))

class DataDownloadConfig(BaseModel):
    hf_dataset_name: str = "hrshihab/BanFakeNews-2.0"
    cleaned_dataset_path: Path = BASE_DIR / "Artifacts" / "data.csv"

class OfflineTokenizeConfig(BaseModel):
    cache_dir: Path = BASE_DIR / "Artifacts" / "tokenized_cache_bert"
    max_length: int = 768  #for bangla-bert keep 512 for ssm 512, 768, 1024 etc.
    class_weight_path: Path = BASE_DIR / "Artifacts" / "class_weights.npy"
    short_test_subset_path: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset_path: Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class BertFineTuneConfig(BaseModel):
    """File-system paths produced / consumed by BanglaBERT fine-tuning."""
    bert_cache_dir:    Path = BASE_DIR / "Artifacts" / "tokenized_cache_bert"
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


class MambaTrainConfig(BaseModel):
    """File-system paths produced / consumed by the Bangla-Mamba training."""
    cache_dir:         Path = BASE_DIR / "Artifacts" / "tokenized_cache_mamba_768"
    checkpoint_dir:    Path = BASE_DIR / "Artifacts" / "checkpoints" / "mamba_768"
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / "mamba_768"
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / "mamba_768_results.json"
    short_test_subset: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset:  Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"


class MambaEvaluateConfig(BaseModel):
    """File-system paths consumed / produced by the Bangla-Mamba evaluation step."""
    best_model_dir:    Path = BASE_DIR / "Artifacts" / "best_model" / "mamba_768"
    results_file:      Path = BASE_DIR / "Artifacts" / "logs" / "mamba_768_results.json"
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
    run_name:        str  = "mamba_train"
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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


settings = Settings()