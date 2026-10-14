"""Tests for meeting metadata storage."""

import sqlite3
from datetime import datetime, timezone

from app.models.schemas import MeetingRecord
from app.services.storage import MeetingStore


def _record(meeting_id: str, filename: str, minute: int = 0) -> MeetingRecord:
    return MeetingRecord(
        id=meeting_id,
        filename=filename,
        created_at=datetime(2026, 10, 13, 12, minute, tzinfo=timezone.utc),
        whisper_model="base",
        status="processing",
    )


def test_store_round_trip(tmp_path) -> None:
    store = MeetingStore(tmp_path / "meetings.db")
    store.create(_record("one", "first.wav"))
    store.update(
        "one",
        status="completed",
        language="en",
        duration_seconds=12.5,
        transcript_path="one.json",
        report_path="one.md",
    )
    loaded = store.get("one")
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.language == "en"
    assert loaded.duration_seconds == 12.5

    store.create(_record("two", "second.wav", minute=5))
    recent = store.list_recent(limit=10)
    assert [record.id for record in recent] == ["two", "one"]


def test_analysis_status_and_no_audio_blob(tmp_path) -> None:
    store = MeetingStore(tmp_path / "meetings.db")
    store.create(_record("one", "first.wav"))
    created = store.get("one")
    assert created is not None
    assert created.analysis_status == "pending"
    store.update("one", analysis_status="completed", duration_seconds=3.0, whisper_model="tiny")
    loaded = store.get("one")
    assert loaded is not None
    assert loaded.analysis_status == "completed"
    assert loaded.whisper_model == "tiny"
    with sqlite3.connect(tmp_path / "meetings.db") as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(meetings)")]
    assert "analysis_status" in columns
    assert "transcript_path" in columns
    assert "audio" not in columns
    assert "blob" not in columns


def test_existing_database_gains_analysis_status(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE meetings (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                whisper_model TEXT,
                language TEXT,
                duration_seconds REAL,
                status TEXT NOT NULL,
                transcript_path TEXT,
                report_path TEXT,
                error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO meetings (
                id, filename, created_at, status
            ) VALUES ('legacy', 'old.wav', '2026-10-13T12:00:00+00:00', 'completed')
            """
        )
    store = MeetingStore(path)
    loaded = store.get("legacy")
    assert loaded is not None
    assert loaded.filename == "old.wav"
    assert loaded.analysis_status is None
    store.update("legacy", analysis_status="unavailable")
    updated = store.get("legacy")
    assert updated is not None
    assert updated.analysis_status == "unavailable"


def test_update_rejects_unknown_columns(tmp_path) -> None:
    store = MeetingStore(tmp_path / "meetings.db")
    store.create(_record("one", "first.wav"))
    try:
        store.update("one", filename="changed.wav")
    except ValueError as exc:
        assert "filename" in str(exc)
    else:
        raise AssertionError("Unknown columns should be rejected.")
