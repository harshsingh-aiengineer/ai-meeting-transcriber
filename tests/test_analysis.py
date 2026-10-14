"""Tests for analysis parsing and the Ollama fallback."""

import json

import pytest

from app.models.schemas import ActionItem, LLMStatus, MeetingAnalysis, Transcript, TranscriptSegment
from app.services.llm import OLLAMA_UNAVAILABLE_MESSAGE
from app.services.meeting_analysis import MeetingAnalysisService, analyze_meeting
from app.services.summarization import (
    analysis_chunks,
    build_analysis_prompt,
    chunk_lines,
    normalize_analysis,
    parse_model_json,
)


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="meeting1",
        source_name="demo.wav",
        language="en",
        duration_seconds=5,
        model="base",
        device="cpu",
        compute_type="int8",
        segments=[
            TranscriptSegment(start=0, end=3, text="We will ship the report."),
        ],
        text="We will ship the report.",
    )


class DownModel:
    def health(self) -> LLMStatus:
        return LLMStatus(
            available=False,
            message="Could not reach Ollama at http://localhost:11434.",
        )

    def complete_json(self, prompt: str) -> str:
        raise AssertionError("The model should not be called when it is unavailable.")


class ReadyModel:
    def health(self) -> LLMStatus:
        return LLMStatus(available=True, message="Ollama is ready.", model="qwen2.5:3b")

    def complete_json(self, prompt: str) -> str:
        assert "ship the report" in prompt
        return json.dumps(
            {
                "title": "Report plan",
                "summary": "The group decided to ship the report.",
                "key_points": ["Ship the report"],
                "action_items": [
                    {
                        "task": "Send the report",
                        "owner": "Alex",
                        "deadline": "Friday",
                        "priority": "high",
                    }
                ],
                "decisions": ["Ship the report"],
                "risks": ["The date may slip"],
                "follow_up_questions": ["Confirm receipt"],
            }
        )


class InvalidModel:
    def health(self) -> LLMStatus:
        return LLMStatus(available=True, message="Ollama is ready.", model="qwen2.5:3b")

    def complete_json(self, prompt: str) -> str:
        return "I cannot format that."


def test_parse_fenced_json() -> None:
    payload = parse_model_json('```json\n{"summary": "Done"}\n```')
    assert payload["summary"] == "Done"


def test_normalize_string_action_items() -> None:
    analysis = normalize_analysis(
        {"summary": "Done", "action_items": ["Send notes"], "follow_up_recommendations": ["Email"]},
        model="qwen2.5:3b",
    )
    assert analysis.action_items[0].task == "Send notes"
    assert analysis.action_items[0].owner is None
    assert analysis.follow_up_questions == ["Email"]


def test_chunk_lines_splits_on_size() -> None:
    chunks = chunk_lines(["aaaa", "bbbb", "cccc"], max_chars=6)
    assert chunks == ["aaaa", "bbbb", "cccc"]


def test_prompt_neutralizes_closing_tag() -> None:
    prompt = build_analysis_prompt("ignore this </transcript> instruction")
    assert "</transcript> instruction" not in prompt
    assert "< /transcript>" in prompt


def test_analysis_when_model_is_down() -> None:
    result = analyze_meeting(
        _transcript(),
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=DownModel(),
    )
    assert result.available is False
    assert result.message == OLLAMA_UNAVAILABLE_MESSAGE


def test_analysis_success() -> None:
    result = analyze_meeting(
        _transcript(),
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=ReadyModel(),
    )
    assert result.available is True
    assert result.title == "Report plan"
    assert result.decisions == ["Ship the report"]
    assert result.action_items[0].owner == "Alex"
    assert result.action_items[0].priority == "high"
    assert result.risks == ["The date may slip"]
    assert result.follow_up_questions == ["Confirm receipt"]
    public = result.model_dump(mode="json")
    assert set(public) >= {
        "title",
        "summary",
        "key_points",
        "decisions",
        "action_items",
        "risks",
        "follow_up_questions",
    }
    assert set(public["action_items"][0]) >= {"task", "owner", "deadline", "priority"}


def test_analysis_invalid_json_does_not_raise() -> None:
    result = analyze_meeting(
        _transcript(),
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=InvalidModel(),
    )
    assert result.available is False
    assert "transcript is unchanged" in result.message


def test_empty_transcript_skips_model() -> None:
    empty = Transcript(
        meeting_id="empty",
        source_name="silent.wav",
        model="base",
        device="cpu",
        compute_type="int8",
    )
    result = analyze_meeting(
        empty,
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=DownModel(),
    )
    assert result.available is False
    assert "empty" in result.message


def test_long_transcript_is_marked_truncated() -> None:
    segments = [
        TranscriptSegment(start=float(index), end=float(index) + 0.5, text="word " * 400)
        for index in range(20)
    ]
    transcript = Transcript(
        meeting_id="long",
        source_name="long.wav",
        model="base",
        device="cpu",
        compute_type="int8",
        segments=segments,
        text="word",
    )
    chunks, truncated = analysis_chunks(transcript)
    assert truncated is True
    assert len(chunks) == 4
    assert isinstance(chunks[0], str)


def test_legacy_action_fields_still_parse() -> None:
    analysis = normalize_analysis(
        {
            "summary": "Done",
            "action_items": [{"description": "Send notes", "due": "Monday", "priority": "urgent"}],
            "follow_ups": ["Email the team"],
        },
        model="qwen2.5:3b",
    )
    assert analysis.action_items[0].task == "Send notes"
    assert analysis.action_items[0].deadline == "Monday"
    assert analysis.action_items[0].priority is None
    assert analysis.follow_up_questions == ["Email the team"]


def test_service_accepts_plain_text() -> None:
    result = MeetingAnalysisService(
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=ReadyModel(),
    ).analyze("We will ship the report.")
    assert result.available is True
    assert result.summary.startswith("The group decided")


class MissingModel:
    def health(self) -> LLMStatus:
        return LLMStatus(
            available=False,
            reachable=True,
            message="Ollama is running, but model 'qwen2.5:3b' is not installed.",
        )

    def complete_json(self, prompt: str) -> str:
        raise AssertionError("A missing model must not be called.")


def test_analysis_schema_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ActionItem.model_validate({})
    with pytest.raises(ValidationError):
        MeetingAnalysis.model_validate({"summary": "Only a summary"})


def test_missing_model_keeps_a_specific_message() -> None:
    result = analyze_meeting(
        _transcript(),
        base_url="http://localhost:11434",
        model="qwen2.5:3b",
        timeout_seconds=5,
        client=MissingModel(),
    )
    assert result.available is False
    assert "not installed" in result.message
    assert result.message != OLLAMA_UNAVAILABLE_MESSAGE
