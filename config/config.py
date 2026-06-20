from pathlib import Path
import os

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(os.getenv("BASE_DIR", "."))

class DataDownloadConfig(BaseModel):
    hf_dataset_name: str = "hrshihab/BanFakeNews-2.0"
    cleaned_dataset_path: Path = BASE_DIR / "Artifacts" / "data.csv"

class OfflineTokenizeConfig(BaseModel):
    cache_dir: Path = BASE_DIR / "Artifacts" / "tokenized_cache_mamba_768"
    max_length: int = 768  #for bangla-bert keep 512 for ssm 512, 768, 1024 etc.
    class_weight_path: Path = BASE_DIR / "Artifacts" / "class_weights.npy"
    short_test_subset_path: Path = BASE_DIR / "Artifacts" / "short_test_subset.csv"
    long_test_subset_path: Path = BASE_DIR / "Artifacts" / "long_test_subset.csv"  



class Settings(BaseSettings):
    seed: int = 42
    logging_dir: Path = BASE_DIR / "logs"
    data: DataDownloadConfig = DataDownloadConfig()
    offline_tokenize: OfflineTokenizeConfig = OfflineTokenizeConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


settings = Settings()