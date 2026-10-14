"""Environment-backed settings and project paths.

Paths are derived from this file's location so the app does not depend on
the machine-specific directory it was cloned into.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.urls import normalize_base_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WHISPER_MODELS: tuple[str, ...] = ("tiny", "base", "small")
DEVICES: tuple[str, ...] = ("cpu", "cuda")
COMPUTE_TYPES: tuple[str, ...] = (
    "int8",
    "int8_float16",
    "int8_float32",
    "float16",
    "float32",
)
CPU_COMPUTE_TYPES: tuple[str, ...] = ("int8", "float32")
CUDA_COMPUTE_TYPES: tuple[str, ...] = ("float16", "int8_float16", "int8", "float32")

_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:-]+(?:/[A-Za-z0-9._:-]+)?$")


def project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def data_dir() -> Path:
    """Return the local data directory."""
    return project_root() / "data"


def upload_dir() -> Path:
    """Return the directory for original uploads."""
    return data_dir() / "uploads"


def output_dir() -> Path:
    """Return the directory for transcripts and reports."""
    return data_dir() / "outputs"


def log_dir() -> Path:
    """Return the directory for application logs."""
    return data_dir() / "logs"


def database_path() -> Path:
    """Return the SQLite metadata database path."""
    return data_dir() / "meetings.db"


def log_file() -> Path:
    """Return the application log file path."""
    return log_dir() / "app.log"


def ensure_directories(root: Path | None = None) -> None:
    """Create the data directories used by the app."""
    base = (root or project_root()) / "data"
    for name in ("uploads", "outputs", "logs"):
        (base / name).mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper_model: str = "base"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    max_upload_mb: int = Field(default=200, ge=1, le=4096)
    device: str = "cpu"
    compute_type: str = "int8"
    ollama_timeout_seconds: float = Field(default=180, gt=0, le=3600)
    log_level: str = "INFO"

    @field_validator("whisper_model", "ollama_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not _MODEL_NAME.fullmatch(cleaned):
            raise ValueError(
                "Model names may contain letters, numbers, and . _ : - characters."
            )
        return cleaned

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        return normalize_base_url(value)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in DEVICES:
            allowed = ", ".join(DEVICES)
            raise ValueError(f"DEVICE must be one of: {allowed}.")
        return cleaned

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in COMPUTE_TYPES:
            allowed = ", ".join(COMPUTE_TYPES)
            raise ValueError(f"COMPUTE_TYPE must be one of: {allowed}.")
        return cleaned

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be a standard Python logging level.")
        return cleaned


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    env_file = project_root() / ".env"
    if env_file.is_file():
        return Settings(_env_file=env_file)
    return Settings(_env_file=None)
