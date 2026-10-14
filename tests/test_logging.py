"""Tests for structured log redaction."""

import json
import logging

from app.utils.logging import JsonFormatter, RedactionFilter, redact_log_message


def test_redacts_secrets_env_and_long_transcripts() -> None:
    transcript = "We discussed the budget. " * 40
    message = redact_log_message(
        f"password=hunter2 Authorization: Bearer sk-live-secret OLLAMA_MODEL=qwen2.5:3b {transcript}"
    )
    assert "hunter2" not in message
    assert "sk-live-secret" not in message
    assert "qwen2.5:3b" not in message
    assert "password=[redacted]" in message
    assert "OLLAMA_MODEL=[redacted]" in message
    assert message.endswith("...[truncated]")


def test_json_log_record_is_structured() -> None:
    record = logging.LogRecord(
        name="app.services.llm",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=abc123",
        args=(),
        exc_info=None,
    )
    assert RedactionFilter().filter(record) is True
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.services.llm"
    assert payload["message"] == "token=[redacted]"
    assert "time" in payload
