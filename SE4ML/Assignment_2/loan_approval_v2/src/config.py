# Centralised settings — all paths and tunables in one place.
# Override any value via environment variable or a .env file.

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent  # loan_approval_v2/


class Settings(BaseSettings):
    # Data
    data_path: str = str(
        _REPO_ROOT.parent.parent / "Assignment_1" / "loan_approval" / "model" / "loan_data.csv"
    )

    # Artifacts
    model_dir: str = str(_REPO_ROOT / "model_artifacts")

    # Training
    test_size: float = 0.10
    val_size: float = 0.111
    random_state: int = 42
    max_iter: int = 1000

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Single shared instance — import this everywhere instead of os.environ.get()
settings = Settings()
