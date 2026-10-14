"""Validation helpers for local service URLs."""

from urllib.parse import urlparse


def normalize_base_url(url: str) -> str:
    """Return a base URL without a trailing slash.

    Args:
        url: Candidate Ollama base URL.

    Raises:
        ValueError: The URL is not an HTTP(S) origin.
    """
    cleaned = url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Ollama base URL must start with http:// or https://.")
    if parsed.path not in {"", "/"}:
        raise ValueError(
            "Ollama base URL should not include a path. Example: http://localhost:11434"
        )
    if parsed.username or parsed.password:
        raise ValueError("Ollama base URL must not include credentials.")
    return cleaned
