"""Safe handling of uploaded meeting files."""

from __future__ import annotations

import re
from pathlib import Path

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".wma",
        ".mpga",
        ".mp4",
        ".m4v",
        ".mkv",
        ".webm",
        ".mov",
        ".avi",
        ".mpeg",
        ".mpg",
    }
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def upload_type_labels() -> list[str]:
    """Return uploader extensions without a leading dot."""
    return sorted(extension.lstrip(".") for extension in ALLOWED_EXTENSIONS)


def sanitize_filename(name: str) -> str:
    """Reduce a client filename to a safe ASCII file name."""
    cleaned = _SAFE_NAME.sub("_", Path(name).name).strip("._")
    if not cleaned:
        return "upload"
    return cleaned[:120]


def save_upload(
    content: bytes,
    original_name: str,
    upload_directory: Path,
    meeting_id: str,
    max_bytes: int,
) -> Path:
    """Validate and store an upload under a generated meeting id.

    Raises:
        ValueError: The file is empty, too large, or an unsupported type.
    """
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > max_bytes:
        size_mb = len(content) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(
            f"File is {size_mb:.1f} MB, which is above the {limit_mb:.0f} MB limit."
        )

    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(upload_type_labels())
        raise ValueError(
            f"Unsupported file type '{suffix or 'unknown'}'. Supported types: {allowed}."
        )
    if not re.fullmatch(r"[0-9a-f]{32}", meeting_id):
        raise ValueError("Meeting id is invalid.")

    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / f"{meeting_id}_{sanitize_filename(original_name)}"
    destination.write_bytes(content)
    return destination


_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*")
_POSIX_PATH = re.compile(r"(?<![\w:])/(?:[\w.-]+/)+[\w.-]+")


def redact_user_message(message: str) -> str:
    """Remove local filesystem paths from a message shown in the UI."""
    cleaned = _WINDOWS_PATH.sub("a local file", message)
    cleaned = _POSIX_PATH.sub("a local file", cleaned)
    return " ".join(cleaned.split())


def resolve_under_directory(root: Path, candidate: Path) -> Path:
    """Resolve a stored path and require it to stay inside ``root``."""
    resolved = candidate.resolve()
    base = root.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("Path is outside the allowed directory.")
    if not resolved.is_file():
        raise ValueError("Saved file is missing.")
    return resolved
