# patient-intake-backend

FastAPI wrapper around `patient-intake-agent`.

- `WS /ws/chat/{session_id}` — send `{"message": "..."}`, receive a `{"type": "typing"}`
  event immediately, then a `{"type": "message", "content", "stage", "awaiting_patient",
  "risk_level", "emergency_flag", "final_summary"?}` reply once the graph finishes that
  turn. `final_summary` is only present once `stage == "done"` — at which point the
  summary is also persisted to SQLite (see below).
- `GET /api/sessions/{session_id}/summary` — the same structured JSON summary, from the
  **in-memory** session store (409 while still in progress, 404 for an unknown session
  — including after a server restart, since this store doesn't survive one).
- `GET /api/chats` — lightweight list of every *finished* chat (`session_id`, `created_at`,
  `chief_complaint`, `risk_level`, `requires_human`), from **SQLite** — survives a restart.
- `GET /api/chats/{session_id}` — full structured summary for one finished chat, from
  SQLite (404 if unknown or never finished).
- `GET /api/health` — liveness check.

Run it:

```bash
uv run --package patient-intake-backend uvicorn backend_app.main:app --app-dir src --port 8000
```

(from the `backend/` directory — or add `--app-dir backend/src` from the repo root).

## State: two separate stores, on purpose

- **In-memory** (`session_store.py`) — the *live* `PatientState` for in-progress
  sessions, one per `session_id`. Fast, no I/O, but gone on restart. Backs
  `/ws/chat/*` and `/api/sessions/*`.
- **SQLite** (`chat_store.py`, `chats.db` in this directory, gitignored) — the
  *durable* record of finished chats only, written exactly once per session
  (when it reaches `stage == "done"`). One table, no migration framework, every
  function opens/closes its own short-lived connection (sqlite3 connections
  aren't thread-safe to share, and FastAPI runs sync routes in a thread pool).
  Backs `/api/chats/*`.

These two intentionally don't merge — an in-progress conversation was never
meant to be durable; a finished one always is now.

Every session still shares one process-wide PatientAgentBench sandbox (see
`agent/src/patient_intake_agent/sandbox_setup.py` for why) — that limitation
is unrelated to the above and still documented in the root README's Known
Limitations.

`create_app(graph_factory=..., db_path=...)` is what makes `backend/tests/`
fully AWS-and-side-effect-free: the graph is only built lazily on the first
WebSocket message, and tests point `db_path` at a temp file (`tmp_path`)
instead of the real `chats.db`, so the module can be imported and fully
exercised with zero AWS configuration and zero pollution of real data.
