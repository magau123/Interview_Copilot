from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from interview_copilot.models import Source

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    source_text TEXT NOT NULL,
    translation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    vector BLOB,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self.connection:
            self.connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def upsert_document(self, name: str, path: str, sha256: str) -> int:
        with self._lock, self.connection:
            existing = self.connection.execute(
                "SELECT id, sha256 FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if existing:
                if existing["sha256"] == sha256:
                    return int(existing["id"])
                self.connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?", (existing["id"],)
                )
                self.connection.execute(
                    "UPDATE documents SET name = ?, sha256 = ?, created_at = ? WHERE id = ?",
                    (name, sha256, _now(), existing["id"]),
                )
                return int(existing["id"])
            cursor = self.connection.execute(
                "INSERT INTO documents(name, path, sha256, created_at) VALUES (?, ?, ?, ?)",
                (name, path, sha256, _now()),
            )
            return int(cursor.lastrowid)

    def replace_chunks(
        self, document_id: int, chunks: list[str], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        rows = [
            (
                document_id,
                ordinal,
                text,
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for ordinal, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.executemany(
                "INSERT INTO chunks(document_id, ordinal, text, vector) VALUES (?, ?, ?, ?)",
                rows,
            )

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self.connection.execute(
                """SELECT d.id, d.name, d.path, d.created_at, COUNT(c.id) AS chunk_count
                   FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                   GROUP BY d.id ORDER BY d.created_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query_vector: list[float], limit: int = 5) -> list[Source]:
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if not query_norm:
            return []
        with self._lock:
            rows = self.connection.execute(
                """SELECT c.id, c.text, c.vector, d.name
                   FROM chunks c JOIN documents d ON d.id = c.document_id"""
            ).fetchall()
        scored: list[Source] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.shape != query.shape:
                continue
            denominator = float(np.linalg.norm(vector)) * query_norm
            score = float(np.dot(vector, query) / denominator) if denominator else 0.0
            scored.append(
                Source(
                    chunk_id=int(row["id"]),
                    document_name=str(row["name"]),
                    text=str(row["text"]),
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def start_session(self) -> int:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO sessions(started_at) VALUES (?)", (_now(),)
            )
            return int(cursor.lastrowid)

    def end_session(self, session_id: int, summary: str = "") -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                (_now(), summary, session_id),
            )

    def add_turn(
        self, session_id: int, role: str, source_text: str, translation: str = ""
    ) -> int:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """INSERT INTO turns(session_id, role, source_text, translation, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, source_text, translation, _now()),
            )
            return int(cursor.lastrowid)

    def update_turn_translation(self, turn_id: int, translation: str) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE turns SET translation = ? WHERE id = ?", (translation, turn_id)
            )

    def recent_context(self, session_id: int, limit: int = 8) -> str:
        with self._lock:
            rows = self.connection.execute(
                """SELECT role, source_text, translation FROM turns
                   WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return "\n".join(
            f"{row['role']}: {row['source_text']}"
            + (f" / {row['translation']}" if row["translation"] else "")
            for row in reversed(rows)
        )

    def add_memory(
        self, text: str, source: str, confidence: float, vector: list[float] | None = None
    ) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes() if vector else None
        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO memories(text, source, confidence, vector, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(text) DO UPDATE SET source=excluded.source,
                   confidence=excluded.confidence, vector=excluded.vector""",
                (text, source, confidence, blob, _now()),
            )
