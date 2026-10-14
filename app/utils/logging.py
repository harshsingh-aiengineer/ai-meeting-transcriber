"""Structured application logging.

Log records are JSON lines. The filter redacts credentials and truncates very
long messages so transcripts, environment dumps, and secrets are not written
to the log. Callers should log meeting ids, durations, and error types only.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED = False
_MAX_MESSAGE = 500

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization|credential)"
    r"\s*[=:]\s*\S+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]+")
_ENV_ASSIGNMENT = re.compile(
    r"(?i)\b(WHISPER_MODEL|OLLAMA_[A-Z0-9_]+|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|"
    r"OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY)\s*=\s*\S+"
)


class RedactionFilter(logging.Filter):
    """Remove secrets and oversized text from a log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = redact_log_message(record.getMessage())
        record.msg = message
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """Format one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_log_message(record.getMessage()),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def redact_log_message(message: str) -> str:
    """Return a log message with secrets removed and length capped."""
    cleaned = _BEARER.sub("Bearer [redacted]", message)
    cleaned = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[redacted]", cleaned)
    cleaned = _ENV_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[redacted]", cleaned)
    if len(cleaned) > _MAX_MESSAGE:
        return cleaned[:_MAX_MESSAGE] + "...[truncated]"
    return cleaned


def configure_logging(level: str, log_path: Path) -> None:
    """Attach console and file handlers once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = []
    formatter = JsonFormatter()
    redaction = RedactionFilter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redaction)
    handlers.append(stream_handler)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction)
        handlers.append(file_handler)
    except OSError:
        print(
            "File logging is unavailable. Continuing with console logging only.",
            file=sys.stderr,
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    for handler in handlers:
        root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("Logging is configured at %s", level.upper())
    _CONFIGURED = True
