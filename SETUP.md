# Setup Instructions

Standalone setup guide for Project — the patient intake assistant.
(High-level architecture, understanding of PatientAgentBench, and design
notes live in [`README.md`](README.md); this file is just "how do I get it
running.")

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| [Python](https://www.python.org) | 3.12 | Runs `agent/` and `backend/`. (3.11+ is technically declared in `pyproject.toml`, but 3.12 is what this project was actually built and tested against.) |
| [uv](https://docs.astral.sh/uv/) | any recent | Manages the Python workspace (`agent` + `backend` as one `uv` workspace). |
| [Node.js](https://nodejs.org) | 18+ | Runs `frontend/` (Vite + React + TypeScript). |
| Git | any | To clone this repo and PatientAgentBench. |

You'll also need credentials for **1** LLM provider — see [Step 3](#step-3-configure-an-llm-provider).
Optional: the [AWS CLI](https://aws.amazon.com/cli/) if you use Bedrock, or [Ollama](https://ollama.com) if you want to run fully offline/free.

## Step 1 — Clone PatientAgentBench alongside this repo

The `agent` package depends on PatientAgentBench directly (it's imported as
a real dependency, not forked — see `README.md`'s "Your understanding of
PatientAgentBench"). By default it's configured as a **sibling directory**:

```
101genai-assignment/
├── Patient_Appointment_Schedule_AI_Agent/            <- this repo
└── PatientAgentBench/   <- sibling clone, referenced by ../../PatientAgentBench
```

```bash
git clone https://github.com/amazon-science/PatientAgentBench.git ../PatientAgentBench
```

If you're reading this inside the original assignment workspace, it's
likely already there — check before cloning.

**Don't want a sibling directory?** Swap the dependency source in
`agent/pyproject.toml` from the local path to PatientAgentBench's git URL
(the alternative is commented inline right next to it) and this repo
becomes fully self-contained.

## Step 2 — Install the workspace

From the `Patient_Appointment_Schedule_AI_Agent/` root:

```bash
uv sync --python 3.12
```

This installs `agent/`, `backend/`, and PatientAgentBench itself into one
shared `.venv`. No network calls to any LLM provider happen during install —
this step alone needs no credentials at all.

## Step 3 — Configure an LLM provider

```bash
cp .env.example .env
```

Then edit `Patient_Appointment_Schedule_AI_Agent/.env` (**not** `.env.example` — that file is a template
only, nothing reads it directly; `agent/src/patient_intake_agent/llm.py`
loads `.env` explicitly). Set `LLM_PROVIDER` to one of four options:

| Provider | `.env` needs | Cost | Notes |
|---|---|---|---|
| `bedrock` (default) | `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, plus `AWS_REGION` | Paid, no free tier | Also needs the model explicitly enabled in the AWS Console under Bedrock → Model access. Verify with `aws sts get-caller-identity`. |
| `anthropic` | `ANTHROPIC_API_KEY` | Paid, no free tier | Direct Claude API, no AWS involved. |
| `gemini` | `GEMINI_API_KEY` | **Has a free tier** | What this project was primarily developed and verified against live. Free tier has a low daily request quota per model (we hit it during development — see Troubleshooting). Uses `gemini-flash-latest`, a Google-maintained alias, rather than a pinned version, to avoid models that quietly lose free-tier access. |
| `ollama` | `OLLAMA_MODEL_ID` (needs `ollama serve` running locally, with that model pulled — e.g. `ollama pull llama3.1:8b`) | Free, fully local | **Known reliability gap:** in testing, local models (llama3.1:8b, qwen2.5:7b) failed to reliably populate structured-output fields (e.g. patient age was extracted as an empty value even when stated plainly), causing the assistant to repeat questions already answered. Fine for proving the wiring works end-to-end; not recommended for judging actual conversation quality — use `gemini` for that. |

Each option is documented inline in `.env.example` with the exact variable
names.

## Step 4 — Run the tests (no credentials needed for this step)

Every test in this repo runs against a fake LLM plus the real
PatientAgentBench sandbox — none of them call any real provider, so this
works with zero LLM configuration at all:

```bash
uv run --package patient-intake-agent pytest agent/tests/ -v
uv run --package patient-intake-backend pytest backend/tests/ -v
```

You should see 34 + 13 = 47 passing tests.

## Step 5 — Verify PatientAgentBench's real sandbox directly (optional, no credentials needed)

```bash
uv run --package patient-intake-agent python docs/exploration/minimal_conversation.py --verify-tools-only
```

Lists doctors, searches availability, and books a real appointment against
the real `HealthcareSandbox` — proves the PatientAgentBench integration
works independent of any LLM.

## Step 6 — Run the backend

```bash
cd backend
uv run --package patient-intake-backend uvicorn backend_app.main:app --app-dir src --port 8000
```

Confirm it's up: `curl http://127.0.0.1:8000/api/health` should return
`{"status":"ok"}`. The LLM provider you configured is only actually used
once the first chat message arrives (it's built lazily), so this step
succeeds even with bad credentials — the failure would show up on your
first message instead.

**Whenever you edit `.env`, restart this process.** The LLM client is
cached for the life of the backend once built; editing `.env` while it's
running has no effect until you stop (`Ctrl+C`) and start it again.

## Step 7 — Run the frontend

In a separate terminal:

```bash
cd frontend
npm install   # first time only
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). If you don't
have a `frontend/.env.local`, it defaults to talking to the backend at
`http://localhost:8000` — override with `VITE_BACKEND_URL` if yours is
elsewhere (see Troubleshooting below for a specific case where you need this).

## Verifying it actually works

1. Open the frontend in a browser.
2. Type a message describing a symptom (e.g. "I've had a headache since this morning").
3. You should see a typing indicator, then a follow-up question.
4. Keep answering — after a few turns the assistant recites what it's
   gathered and asks you to confirm before booking an appointment.
5. Try an obvious emergency phrase (e.g. "I have severe chest pain and can't
   breathe") in a fresh session — it should short-circuit straight to a
   summary recommending you call **112** and seek emergency care, with no
   further questions asked.
6. Once a conversation finishes, `GET http://127.0.0.1:8000/api/chats`
   should list it (this is durable — it survives a backend restart, unlike
   the live session itself).

## Troubleshooting

**Chat input stays disabled / WebSocket never connects.**
On macOS with Docker Desktop running, its own background service can end up
listening on `*:8000` and `*:5173` — ports this project also uses, purely
by coincidence. Since `localhost` resolves to both `127.0.0.1` and `::1`,
and browsers prefer IPv6, the frontend can end up talking to Docker instead
of your backend. Fix: create `frontend/.env.local` with
```
VITE_BACKEND_URL=http://127.0.0.1:8000
```
to force IPv4, restart the frontend dev server (env files are only read at
startup), and hard-refresh the page. If it's still not connecting, run
`lsof -iTCP -sTCP:LISTEN -P` and check exactly what's bound to the ports
you expect the backend/frontend to be using.

**`ValidationError ... Could not load credentials`, or the wrong provider
seems to be in use even though `.env` looks right.**
Two likely causes: (1) you edited `.env.example` instead of `.env` — only
`.env` is actually loaded; or (2) you edited `.env` correctly but never
restarted the backend process (see Step 6's note above — the LLM client is
cached for the process's lifetime).

**A message takes 30+ seconds, or seems to hang entirely.**
LLM calls — especially the structured/tool-calling ones every node
uses — can be genuinely slow, and occasionally providers return transient
errors (rate limits, server overload) that used to hang silently for
minutes before failing. Two things address this: `llm.py`'s Gemini branch
sets explicit `timeout=30`/`max_retries=2` (rather than the client
library's defaults of no timeout and 6 retries, which is what caused the
worst hangs), and the backend now catches any LLM failure and sends a
visible in-chat error message instead of dropping the connection silently —
if you see a red error bubble, just try again; nothing is lost.

**Ollama repeats a question you already answered.**
This is a known, diagnosed limitation, not a setup mistake — see the
`ollama` row in the provider table above. Switch to `gemini` if this
matters for what you're testing.

## Project layout, at a glance

```
Patient_Appointment_Schedule_AI_Agent/
├── agent/       patient-intake-agent — the LangGraph graph itself (agent/README.md)
├── backend/     patient-intake-backend — FastAPI WS + REST wrapper (backend/README.md)
├── frontend/    React + TypeScript UI (frontend/README.md)
├── docs/        architecture diagrams, PatientAgentBench notes, example transcripts
├── .env.example copy to .env, see Step 3
└── README.md    architecture, design rationale, build status
```

Each subdirectory's own `README.md` goes deeper on that piece specifically.
