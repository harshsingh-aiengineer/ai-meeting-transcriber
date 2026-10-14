"""Optional Ollama client.

The rest of the app treats a failed health check as a normal result.
Transcription does not use this module.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.models.schemas import LLMStatus
from app.utils.urls import normalize_base_url

logger = logging.getLogger(__name__)

OLLAMA_UNAVAILABLE_MESSAGE = (
    "AI analysis unavailable — start Ollama to enable meeting intelligence"
)


class LLMError(RuntimeError):
    """The local model was reachable, but the completion request failed."""


class LLMClient(Protocol):
    """Minimal client used by meeting analysis."""

    def health(self) -> LLMStatus:
        """Report whether a completion can be attempted."""

    def complete_json(self, prompt: str) -> str:
        """Return the raw model response for a JSON prompt."""


class OllamaClient:
    """Talk to a local Ollama server over HTTP."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = normalize_base_url(base_url)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    def health(self) -> LLMStatus:
        """Check that Ollama is up and the configured model is present."""
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=3.0,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            return LLMStatus(available=False, reachable=False, message=OLLAMA_UNAVAILABLE_MESSAGE)
        except httpx.HTTPError as exc:
            logger.info("Ollama health check failed: %s", exc)
            return LLMStatus(available=False, reachable=False, message=OLLAMA_UNAVAILABLE_MESSAGE)
        except ValueError:
            return LLMStatus(
                available=False,
                reachable=True,
                message="Ollama responded, but the model list was not valid JSON.",
            )

        names = _model_names(payload)
        if not _model_present(self.model, names):
            available = ", ".join(names) if names else "none"
            return LLMStatus(
                available=False,
                reachable=True,
                message=(
                    f"Ollama is running, but model '{self.model}' is not installed. "
                    f"Installed models: {available}. "
                    f"Pull it with: ollama pull {self.model}"
                ),
            )
        return LLMStatus(
            available=True,
            reachable=True,
            message=f"Ollama is ready with model {self.model}.",
            model=self.model,
        )

    def complete_json(self, prompt: str) -> str:
        """Request one JSON completion.

        Raises:
            LLMError: The server returned an error or an empty message.
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                timeout=self.timeout_seconds,
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.2, "num_predict": 2048},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise LLMError(
                "The local model did not finish before the timeout."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError("Ollama returned a response that was not JSON.") from exc

        message = payload.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama returned an empty analysis.")
        return content


def _model_names(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _model_present(requested: str, names: list[str]) -> bool:
    accepted = {requested, f"{requested}:latest"}
    return any(name in accepted for name in names)
