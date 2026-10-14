"""Tests for environment settings and local paths."""

import pytest
from pydantic import ValidationError

from app.config import Settings, ensure_directories, get_settings
from app.utils.urls import normalize_base_url


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WHISPER_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "MAX_UPLOAD_MB",
        "DEVICE",
        "COMPUTE_TYPE",
        "OLLAMA_TIMEOUT_SECONDS",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    assert settings.whisper_model == "base"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:3b"
    assert settings.max_upload_mb == 200
    assert settings.device == "cpu"
    assert settings.compute_type == "int8"


def test_settings_reject_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_normalize_base_url_strips_slash() -> None:
    assert normalize_base_url("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_base_url_rejects_path() -> None:
    with pytest.raises(ValueError):
        normalize_base_url("http://localhost:11434/api")


def test_ensure_directories(tmp_path) -> None:
    ensure_directories(tmp_path)
    assert (tmp_path / "data" / "uploads").is_dir()
    assert (tmp_path / "data" / "outputs").is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
