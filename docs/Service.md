## Build status

| Phase | Status |
|---|---|
| Study PatientAgentBench source | ✅ done — see [`docs/exploration/patientagentbench-notes.md`](docs/exploration/patientagentbench-notes.md) |
| Minimal standalone conversation script, real sandbox + tools | ✅ tool mechanics verified (`--verify-tools-only`, no LLM needed), see [`docs/exploration/minimal_conversation.py`](docs/exploration/minimal_conversation.py) |
| Repo scaffold (`agent/`, `backend/`, `frontend/`, `docs/`) | ✅ done |
| Five-node graph, stub nodes, wiring tests | ✅ done — `uv run --package patient-intake-agent pytest agent/tests/` |
| Real Intake / Emergency Guard / Triage / Scheduling / Summary node logic | ✅ done, 26 tests against fake LLMs + the real PatientAgentBench sandbox, **plus a live end-to-end run against the real Gemini API** confirming the whole chain (extraction → question) works against actual model output |
| FastAPI backend (WebSocket chat + REST summary) | ✅ done, 5 tests + a real (non-`TestClient`) uvicorn boot, **plus a live WebSocket smoke test against the running backend + real Gemini** |
| React + TypeScript frontend | ✅ built (chat window, typing indicator, summary panel) — `npm run build` is clean; manually verified |
| Example transcripts, architecture diagram | ✅ done — see [`docs/architecture.md`](docs/architecture.md) and [`docs/transcripts/`](docs/transcripts/) |
