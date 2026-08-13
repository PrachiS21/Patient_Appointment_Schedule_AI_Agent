"""Durable record of finished chats — SQLite, not a service.

Deliberately the simplest thing that persists: one table, one row per
finished session, written exactly once (when a session reaches
`stage == "done"` — see main.py). Every function opens, uses, and closes its
own short-lived connection rather than holding one open across the app's
lifetime — sqlite3 connections aren't safe to share across threads, and
FastAPI runs sync route handlers (like the ones in this module) in a
thread pool, so a shared connection would be a real bug, not a
hypothetical one. Short-lived connections make that a non-issue: nothing is
ever contended for longer than one query.

This is the *durable* record of completed chats. It is separate from
`SessionStore`, which stays the fast, in-memory home for *in-progress*
sessions (see session_store.py) — the two aren't meant to merge.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def init_db(db_path: Path) -> None:
    """Create the chats table if it doesn't exist yet. Safe to call every
    app startup — CREATE TABLE IF NOT EXISTS, no migration framework needed
    at this scale."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                session_id TEXT PRIMARY KEY,
                final_summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_chat(db_path: Path, session_id: str, final_summary: dict) -> None:
    """Persist one finished chat's structured summary. INSERT OR REPLACE so
    a re-used session_id (shouldn't normally happen with UUIDs) overwrites
    rather than errors."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chats (session_id, final_summary_json, created_at) "
            "VALUES (?, ?, ?)",
            (session_id, json.dumps(final_summary), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def list_chats(db_path: Path) -> list[dict]:
    """Lightweight listing — enough to show a list of past chats without
    shipping every full summary blob. Use get_chat() for the full record."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT session_id, final_summary_json, created_at FROM chats "
            "ORDER BY created_at DESC"
        ).fetchall()

    result = []
    for row in rows:
        summary = json.loads(row["final_summary_json"])
        result.append(
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "chief_complaint": summary.get("chief_complaint"),
                "risk_level": summary.get("risk_level"),
                "requires_human": summary.get("requires_human"),
            }
        )
    return result


def get_chat(db_path: Path, session_id: str) -> dict | None:
    """Full structured summary for one finished chat, or None if unknown."""
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT final_summary_json FROM chats WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    if row is None:
        return None
    return json.loads(row[0])
