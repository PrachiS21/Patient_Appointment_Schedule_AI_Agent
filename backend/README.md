# patient-intake-backend

FastAPI wrapper around `patient-intake-agent`.

- `WS /ws/chat/{session_id}` — send `{"message": "..."}`, receive a `{"type": "typing"}`
  event immediately, then a `{"type": "message", "content", "stage", "awaiting_patient",
  "risk_level", "emergency_flag", "final_summary"?}` reply once the graph finishes that
  turn. `final_summary` is only present once `stage == "done"`.
- `GET /api/sessions/{session_id}/summary` — the same structured JSON summary, once
  the conversation has finished (409 while still in progress, 404 for an unknown session).
- `GET /api/health` — liveness check.

Run it:

```bash
uv run --package patient-intake-backend uvicorn backend_app.main:app --app-dir src --port 8000
```

(from the `backend/` directory — or add `--app-dir backend/src` from the repo root).

Session state (`PatientState`) lives in an in-memory dict, one per
`session_id` the client supplies; every session shares one process-wide
PatientAgentBench sandbox (see `agent/src/patient_intake_agent/sandbox_setup.py`
for why). No persistence across restarts. Both are documented as known
limitations in the root README.

`create_app(graph_factory=...)` is what makes `backend/tests/` fully
AWS-free: the graph is only built lazily, on the first WebSocket message, so
tests inject a fake-LLM graph and the module can even be imported with zero
AWS configuration present.
