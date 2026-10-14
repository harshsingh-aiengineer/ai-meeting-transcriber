"""Meeting analysis through an optional local LLM.

This module never raises for an unavailable model. Callers receive
``MeetingAnalysis(available=False, message=...)`` and can keep showing the
transcript.
"""

from __future__ import annotations

import json
import logging

from app.models.schemas import MeetingAnalysis, Transcript, TranscriptSegment
from app.services.llm import LLMClient, LLMError, OLLAMA_UNAVAILABLE_MESSAGE, OllamaClient
from app.services.summarization import (
    analysis_chunks,
    build_analysis_prompt,
    merge_analyses,
    normalize_analysis,
    parse_model_json,
)
from app.utils.urls import normalize_base_url

logger = logging.getLogger(__name__)


def analyze_meeting(
    transcript: Transcript,
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    client: LLMClient | None = None,
    glossary: str | None = None,
) -> MeetingAnalysis:
    """Summarize a transcript and extract decisions, actions, and follow-ups."""
    if not transcript.text.strip() and not transcript.segments:
        return MeetingAnalysis(
            available=False,
            message="The transcript is empty, so there is nothing to analyze.",
        )

    try:
        normalized_url = normalize_base_url(base_url)
    except ValueError as exc:
        return MeetingAnalysis(available=False, message=str(exc))

    llm = client or OllamaClient(normalized_url, model, timeout_seconds)
    try:
        status = llm.health()
    except Exception as exc:
        logger.exception("LLM health check failed meeting_id=%s", transcript.meeting_id)
        return MeetingAnalysis(
            available=False,
            message=OLLAMA_UNAVAILABLE_MESSAGE,
        )
    if not status.available:
        message = OLLAMA_UNAVAILABLE_MESSAGE if not status.reachable else status.message
        return MeetingAnalysis(available=False, message=message)

    chunks, truncated = analysis_chunks(transcript)
    if not chunks:
        return MeetingAnalysis(
            available=False,
            message="The transcript is empty, so there is nothing to analyze.",
        )

    parts: list[MeetingAnalysis] = []
    try:
        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                "Analysis chunk %s/%s meeting_id=%s",
                index,
                len(chunks),
                transcript.meeting_id,
            )
            raw = llm.complete_json(build_analysis_prompt(chunk, glossary=glossary))
            payload = parse_model_json(raw)
            parts.append(
                normalize_analysis(payload, model=status.model or model)
            )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Analysis failed meeting_id=%s error=%s",
            transcript.meeting_id,
            exc,
        )
        return MeetingAnalysis(
            available=False,
            message=(
                "The local model could not produce a readable analysis. "
                f"{exc} The transcript is unchanged."
            ),
        )
    except Exception as exc:
        logger.exception(
            "Unexpected analysis failure meeting_id=%s",
            transcript.meeting_id,
        )
        return MeetingAnalysis(
            available=False,
            message=f"Analysis failed: {exc} The transcript is unchanged.",
        )

    return merge_analyses(
        parts,
        model=status.model or model,
        truncated=truncated,
        part_count=len(chunks),
    )


class MeetingAnalysisService:
    """Create a meeting brief from transcript text when a local model is available.

    Callers pass plain transcript text or a ``Transcript``. Ollama is checked
    before any completion request. A missing server returns a normal result
    instead of an exception.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        client: LLMClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = client

    def analyze(
        self,
        transcript: str | Transcript,
        glossary: str | None = None,
    ) -> MeetingAnalysis:
        """Return structured meeting intelligence for ``transcript``."""
        document = transcript if isinstance(transcript, Transcript) else _transcript_from_text(transcript)
        return analyze_meeting(
            document,
            base_url=self.base_url,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            client=self._client,
            glossary=glossary,
        )


def _transcript_from_text(text: str) -> Transcript:
    cleaned = " ".join(text.split())
    segments = [TranscriptSegment(start=0.0, end=0.0, text=cleaned)] if cleaned else []
    return Transcript(
        meeting_id="transcript",
        source_name="transcript",
        model="none",
        device="cpu",
        compute_type="int8",
        segments=segments,
        text=cleaned,
    )
