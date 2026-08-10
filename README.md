# AI-Powered Patient Intake Assistant

An MVP patient-intake agent built on top of [PatientAgentBench](https://github.com/amazon-science/PatientAgentBench)'s
healthcare sandbox and appointment tools. Conducts a conversational intake,
detects emergencies and short-circuits to an urgent-care recommendation,
maps symptoms to a medical specialty, schedules an appointment against real
doctor availability, and produces a structured JSON summary.

> **Status:** in progress. This README is updated as each build phase lands
> — see [Build status](#build-status) for exactly what's implemented vs.
> stubbed right now. It is being built in phases on purpose, with a wiring
> checkpoint before any real (LLM-backed) node logic, so structural bugs and
> logic bugs stay separable.

## Build status

| Phase | Status |
|---|---|
| Study PatientAgentBench source | ✅ done — see [`docs/exploration/patientagentbench-notes.md`](docs/exploration/patientagentbench-notes.md) |
| Minimal standalone conversation script, real sandbox + tools | ✅ tool mechanics verified (`--verify-tools-only`, no AWS needed); full console chat is written but **unverified — needs Bedrock credentials**, see [`docs/exploration/minimal_conversation.py`](docs/exploration/minimal_conversation.py) |
| Repo scaffold (`agent/`, `backend/`, `frontend/`, `docs/`) | ✅ done |
| Five-node graph, stub nodes, wiring tests | ✅ done — `uv run --package patient-intake-agent pytest agent/tests/` |
| Real Intake / Emergency Guard / Triage / Scheduling / Summary node logic | ✅ done, 26 tests, all against fake LLMs + the real PatientAgentBench sandbox — **unverified against real Bedrock output**, see [Known limitations](#known-limitations) |
| FastAPI backend (WebSocket chat + REST summary) | ✅ done, 5 tests + a real (non-`TestClient`) uvicorn boot check — same Bedrock caveat as above |
| React + TypeScript frontend | ✅ built (chat window, typing indicator, summary panel) — `npm run build` is clean and every module round-trips through the Vite dev server, but **not yet clicked through in a real browser** (no browser automation tool in this environment) |
| Example transcripts, architecture diagram | ⬜ not started |

**Blocked on:** AWS Bedrock credentials, for one specific kind of
verification only — an actual end-to-end conversation against a real model.
Everything else needed zero AWS access: the graph, all 31 backend+agent
tests (against fake LLMs and the real PatientAgentBench sandbox/tools), a
real uvicorn boot, and the frontend build all run and pass with no
credentials at all. What's *not* yet verified is real Bedrock output
quality — whether the prompts actually produce sensible
extractions/classifications from a real model, not just from the canned
responses the tests supply — and a real click-through of the UI.

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Clone PatientAgentBench as a sibling directory of this repo (the agent
#    package depends on it via a local path source — see agent/pyproject.toml).
#    If you're reading this inside the assignment workspace, it's already there.
git clone https://github.com/amazon-science/PatientAgentBench.git ../PatientAgentBench

# 2. Install the workspace (agent + backend, and PatientAgentBench itself)
uv sync --python 3.12

# 3. AWS credentials for Bedrock (only needed once real node logic lands)
cp .env.example .env   # fill in AWS_PROFILE or AWS_ACCESS_KEY_ID/SECRET, AWS_REGION
aws sts get-caller-identity   # sanity check

# 4. Run all tests (no AWS needed — every test uses a fake LLM + the real sandbox)
uv run --package patient-intake-agent pytest agent/tests/ -v
uv run --package patient-intake-backend pytest backend/tests/ -v

# 5. Verify PatientAgentBench's real sandbox + tools directly (no AWS needed)
uv run --package patient-intake-agent python docs/exploration/minimal_conversation.py --verify-tools-only

# 6. Run the backend for real (needs AWS credentials from step 3 — the graph
#    is only built lazily, on the first chat message)
cd backend && uv run --package patient-intake-backend uvicorn backend_app.main:app --app-dir src --port 8000

# 7. In a separate terminal, run the frontend (needs Node.js)
cd frontend && npm install && npm run dev
```

If you don't have PatientAgentBench as a sibling directory and want this repo
fully self-contained, swap the dependency source in `agent/pyproject.toml`
from the local path to the git URL (commented inline in that file).

## Your understanding of PatientAgentBench

Full notes in [`docs/exploration/patientagentbench-notes.md`](docs/exploration/patientagentbench-notes.md).
Short version:

- PatientAgentBench is a **benchmark**, not an app — a generate → converse →
  evaluate pipeline. We reuse one corner of it: the assistant-side harness
  and the healthcare sandbox/tools. None of the benchmark runner, patient
  simulator, or LLM-jury evaluation is used.
- **Conversation management has no memory of its own.** `DefaultAssistantAgent.invoke()`
  takes the full message history as an argument every call and builds a
  brand-new LangGraph ReAct agent from scratch each time — there's no
  checkpointer or thread ID anywhere. The caller (`ConversationRunner`) owns
  the growing message list; state *is* that list.
- **Prompts** are one template per role with `{current_datetime}`/`{user_profile}`
  placeholders, filled per-turn via a "only substitute what's present"
  formatter. Safety ("never diagnose", "recognize red flags") is handled
  entirely as prompt instruction — there is no code-level emergency check
  anywhere in the harness. That gap is exactly what our Emergency Guard node
  fills.
- **Patient state** is a fixed, pre-existing medical record (nested
  dataclasses → XML, re-injected unchanged into the system prompt every
  turn), separate from the sandbox's *mutable* world state (appointment
  slots, bookings), which tools mutate as side effects. PatientAgentBench
  assumes an already-onboarded patient; it has no concept of intake,
  emergency triage, or specialty routing at all — those three nodes are net
  new, built on top of the sandbox/tools layer we did reuse directly.

## Architecture

```
[Intake] → [Emergency Guard] → [Triage/Routing] → [Scheduling] → [Summary]
```

A `StateGraph` over a shared `PatientState` (see
`agent/src/patient_intake_agent/state.py`), compiled with **no cycles and no
checkpointer** — deliberately mirroring PatientAgentBench's own
stateless-per-call pattern (see above). The graph is invoked once per
incoming patient message; the caller (FastAPI layer) holds `PatientState`
between calls. Full routing/pause logic is documented in the module
docstring of [`agent/src/patient_intake_agent/graph.py`](agent/src/patient_intake_agent/graph.py).

One deliberate departure from the diagram above: **Emergency Guard is the
graph's actual entry point**, running before Intake rather than after it.
Emergency Guard's whole design is to be a fast, free, deterministic safety
gate (see [Prompting strategy](#prompting-strategy)) — if Intake ran first,
an obviously dangerous message would still burn an Intake LLM call before
the guard ever got a chance to short-circuit it. Emergency Guard only needs
the raw patient message plus whatever was already known from prior turns,
both available before Intake touches anything, so nothing is lost by moving
it first — and it still runs unconditionally on every single turn, which is
what actually satisfies "runs on every patient turn, not just at the end."

- `emergency_guard` runs first, always. A conditional edge after it either
  short-circuits straight to `summary` (emergency), or routes to whichever
  stage the conversation is actually in (`intake` or `scheduling`).
- `intake` has its own conditional edge: pause (`END`, waiting for another
  patient reply), or continue into `triage` once enough has been gathered.
- `triage → scheduling` is unconditional (triage never needs patient input:
  it only uses what's already been collected) — the only edge not gated by
  a turn boundary.
- `scheduling` has its own conditional edge: pause for availability, or
  continue to `summary` once an appointment is resolved (booked, or
  explicitly no-availability).

A full visual architecture diagram (five-node graph + how it wraps
PatientAgentBench's sandbox) is still to do.

## Prompting strategy

Every LLM-backed node uses **structured output** (`llm.with_structured_output(PydanticModel)`,
see `agent/src/patient_intake_agent/llm.py::structured_call`) rather than
free-text parsing — each call returns a typed, validated object
(`IntakeExtraction`, `EmergencyClassification`, `TriageClassification`,
`SlotSelection`), so a node never has to guess how to parse a reply. Two
patterns repeat across every prompt:

- **Cheapest-check-first.** Emergency Guard and Triage both try a
  deterministic, free check before ever calling an LLM: a red-flag keyword
  scan (`nodes/emergency_guard.py::RED_FLAG_KEYWORDS`) and a symptom-keyword
  rules table (`nodes/triage.py::RULES_TABLE`) respectively. The LLM is only
  invoked for what the cheap tier didn't already resolve. This is also why
  Emergency Guard is the graph's actual entry point rather than sitting
  after Intake as the assignment's diagram shows it — see the "why" in
  `agent/src/patient_intake_agent/graph.py`'s module docstring.
- **Ground generation in real data, never let the LLM invent it.** Triage
  constrains its classifier to `patient_agent_bench.tools.appointment_tools.VALID_SPECIALTIES`
  and clamps anything outside that vocabulary to "Primary Care" rather than
  passing a hallucinated specialty downstream. Scheduling shows the LLM the
  *actual* list of open slots from the real sandbox and asks it to pick a
  `slot_id` from that list (or say none match) — it never asks the LLM to
  invent a time, so nothing it returns can be unbookable.

Safety language is a fixed constant, not LLM-generated: `URGENT_CARE_MESSAGE`
in `emergency_guard.py` is the same literal string both the keyword tier and
the LLM tier return, and it's what Summary's `recommendation` uses verbatim
for any emergency case — so there's no path where phrasing drifts into
something that reads as a diagnosis.

## Conversation flow

One `graph.invoke(state)` call per incoming patient message; the caller
(FastAPI's WebSocket handler) holds `PatientState` between calls. Concretely,
for the assignment's own example ("I've had a fever since yesterday"):

1. **Turn 1** — Emergency Guard finds no red flags → Intake extracts a chief
   complaint, asks one follow-up question (e.g. age and severity), and
   pauses.
2. **Turn 2** — patient answers → Emergency Guard re-checks (runs on *every*
   turn) → Intake decides enough is known → Triage classifies a specialty
   and looks up real candidate doctors → Scheduling asks for availability
   and pauses, all within the same invocation (only Intake and Scheduling
   ever pause; Triage never does, since it only uses what's already been
   collected).
3. **Turn 3** — patient states their availability → Scheduling reconciles it
   against real open slots for the candidate doctor(s), books one (or falls
   back to a "no availability" outcome — never retries in a loop) →
   Summary serializes the final JSON.

If a red-flag symptom appears on *any* turn — including mid-scheduling, after
a specialist has already been picked — Emergency Guard short-circuits
straight to Summary with an urgent-care recommendation, skipping whatever
stage the conversation was in. See `agent/tests/test_graph_wiring.py::test_emergency_short_circuits_even_mid_scheduling`
for this exact case.

## State management approach

`PatientState` (`agent/src/patient_intake_agent/state.py`) is a single flat
`TypedDict` threaded through every node — no separate memory subsystem.
Nodes are pure functions `PatientState -> dict` (partial update, merged by
LangGraph); they never mutate the incoming state in place, which is what
keeps them independently testable with hardcoded inputs. Two field groups:
the eight fields required by the assignment's structured-summary schema
(`chief_complaint` .. `risk_level`), and a few orchestration-only fields
(`stage`, `turn_input`, `awaiting_patient`, `assistant_message`) that exist
purely to drive turn-taking between graph invocations and are stripped out
before the final JSON summary is produced.

## Assumptions

- One patient per session, one chief complaint per conversation (no
  mid-conversation topic switching to a second, unrelated complaint).
- The patient scheduling directly, in English, is the one being seen (no
  proxy/caregiver-on-behalf-of flow).
- AWS Bedrock is reachable with access to the configured model; no fallback
  LLM provider.
- PatientAgentBench's sandbox data (offices, doctors, appointment slots) is
  synthetic and hardcoded for this MVP rather than LLM-generated per session
  (see `docs/exploration/patientagentbench-notes.md` for why) — doctor
  availability in the demo is fixed, not connected to a real scheduling
  system.

## Known limitations

- **Research-only foundation.** PatientAgentBench is CC-BY-NC-4.0,
  research-only, and explicitly not clinically validated. This MVP inherits
  that limitation — it is not a medical device and is not suitable for real
  patient use.
- **No diagnosis, by design.** The agent maps symptoms to a specialty/care
  level, never to a condition. This is a hard requirement, not a
  configurable behavior.
- **Emergency detection has not been validated against a real model yet.**
  The keyword tier is deterministic and thoroughly tested; the LLM
  classifier fallback (for emergencies phrased without hitting a listed
  keyword) has only been exercised against fake, canned LLM responses — its
  real-world recall against actual Bedrock output is unverified. Until it
  is, treat the LLM tier as unproven, not as a safety guarantee.
- No persistence: state lives in memory for the duration of a session; a
  server restart loses in-progress conversations.
- Single hardcoded sandbox, shared by every session on the backend (see
  Assumptions and `sandbox_setup.py`) — not connected to any real
  EHR/scheduling backend, and two concurrent users can compete for the same
  appointment slots.
- Wide-open CORS (`allow_origins=["*"]`) on the backend — fine for local
  development, not for a real deployment.
- The frontend has not been exercised in a real browser in this environment
  (no browser automation tool available) — see `frontend/README.md`.
