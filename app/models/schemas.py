"""Domain models for transcripts, analysis, and meeting metadata."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """One timestamped piece of recognized speech."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None


class Transcript(BaseModel):
    """A complete transcript for one uploaded recording."""

    meeting_id: str
    source_name: str
    language: str | None = None
    language_probability: float | None = None
    duration_seconds: float | None = None
    model: str
    device: str
    compute_type: str
    speaker_mode: Literal["none", "pause_estimate"] = "none"
    segments: list[TranscriptSegment] = Field(default_factory=list)
    text: str = ""


class ActionItem(BaseModel):
    """A task mentioned in the meeting."""

    task: str
    owner: str | None = None
    deadline: str | None = None
    priority: Literal["high", "medium", "low"] | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_names(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not str(data.get("task") or "").strip():
            legacy = data.get("description") or data.get("text")
            if isinstance(legacy, str):
                data["task"] = legacy
        if not str(data.get("deadline") or "").strip() and data.get("due"):
            data["deadline"] = data.get("due")
        priority = data.get("priority")
        if isinstance(priority, str):
            cleaned = priority.strip().lower()
            data["priority"] = cleaned if cleaned in {"high", "medium", "low"} else None
        elif priority is not None and not isinstance(priority, str):
            data["priority"] = None
        return data


class MeetingAnalysis(BaseModel):
    """Structured meeting brief, or a message explaining why it is unavailable."""

    available: bool
    message: str
    title: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    suggested_follow_ups: list[str] = Field(default_factory=list)
    model: str | None = None
    truncated: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_follow_ups(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "follow_up_questions" not in data and "follow_ups" in data:
            data["follow_up_questions"] = data.get("follow_ups") or []
        return data


class MeetingRecord(BaseModel):
    """SQLite row describing one transcription job."""

    id: str
    filename: str
    created_at: datetime
    whisper_model: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    status: Literal["processing", "completed", "failed"]
    transcript_path: str | None = None
    report_path: str | None = None
    analysis_status: Literal["pending", "completed", "unavailable", "failed"] | None = None
    error: str | None = None


class LLMStatus(BaseModel):
    """Result of a local LLM health check."""

    available: bool
    message: str
    model: str | None = None
    reachable: bool = False
