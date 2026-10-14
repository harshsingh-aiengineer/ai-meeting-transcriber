"""SQLite storage for meeting metadata.

The database stores filenames, status, and output paths. Transcript text
lives in the JSON files under the output directory.
"""

from __future__ import annotations

import sqlite3
from datetime import timezone
from pathlib import Path

from app.models.schemas import MeetingRecord

_UPDATABLE = {
    "status",
    "language",
    "duration_seconds",
    "transcript_path",
    "report_path",
    "analysis_status",
    "error",
    "whisper_model",
}


class MeetingStore:
    """Small repository around the meetings table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create(self, record: MeetingRecord) -> None:
        """Insert a new meeting row."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO meetings (
                    id, filename, created_at, whisper_model, language,
                    duration_seconds, status, transcript_path, report_path,
                    analysis_status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.filename,
                    record.created_at.astimezone(timezone.utc).isoformat(),
                    record.whisper_model,
                    record.language,
                    record.duration_seconds,
                    record.status,
                    record.transcript_path,
                    record.report_path,
                    record.analysis_status or "pending",
                    record.error,
                ),
            )

    def update(self, meeting_id: str, **fields: object) -> None:
        """Update whitelisted columns for one meeting."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported meeting fields: {names}")
        if not fields:
            return
        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [*fields.values(), meeting_id]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE meetings SET {assignments} WHERE id = ?",
                values,
            )

    def get(self, meeting_id: str) -> MeetingRecord | None:
        """Return one meeting, or None when the id is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM meetings WHERE id = ?",
                (meeting_id,),
            ).fetchone()
        if row is None:
            return None
        return MeetingRecord.model_validate(dict(row))

    def list_recent(self, limit: int = 10) -> list[MeetingRecord]:
        """Return the newest meetings."""
        bounded = min(max(limit, 1), 50)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM meetings
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (bounded,),
            ).fetchall()
        return [MeetingRecord.model_validate(dict(row)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS meetings (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    whisper_model TEXT,
                    language TEXT,
                    duration_seconds REAL,
                    status TEXT NOT NULL,
                    transcript_path TEXT,
                    report_path TEXT,
                    analysis_status TEXT,
                    error TEXT
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(meetings)").fetchall()
            }
            if "analysis_status" not in columns:
                connection.execute("ALTER TABLE meetings ADD COLUMN analysis_status TEXT")
