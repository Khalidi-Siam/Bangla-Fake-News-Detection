from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataDownloadConfig(BaseModel):
    hf_dataset_name: str = "hrshihab/BanFakeNews-2.0"
    cleaned_dataset_path: Path = Path("Artifacts/data.csv")




class Settings(BaseSettings):
    seed: int = 42
    data: DataDownloadConfig = DataDownloadConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )


settings = Settings()