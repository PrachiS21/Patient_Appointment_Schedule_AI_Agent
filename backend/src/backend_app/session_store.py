"""Tiny in-memory session store.

Not thread-safe beyond what CPython's GIL already gives plain dict ops —
fine for a single-process MVP where each WebSocket connection processes its
own messages sequentially (see main.py's `while True: receive -> invoke ->
send` loop; two messages for the same session are never handled
concurrently). No persistence: a server restart loses every in-progress
conversation, and every session shares one process-wide PatientAgentBench
sandbox (see agent/src/patient_intake_agent/sandbox_setup.py) — both are
documented as known limitations in the root README.
"""

from __future__ import annotations

from patient_intake_agent import PatientState


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, PatientState] = {}

    def get_or_create(self, session_id: str, factory) -> PatientState:
        if session_id not in self._sessions:
            self._sessions[session_id] = factory()
        return self._sessions[session_id]

    def set(self, session_id: str, state: PatientState) -> None:
        self._sessions[session_id] = state

    def peek(self, session_id: str) -> PatientState | None:
        return self._sessions.get(session_id)
