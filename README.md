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

## Features

- Conversational patient intake with follow-up questions tailored to
  the reported symptom (fever, pain, etc.)
- Code-enforced emergency detection that short-circuits the conversation
  and recommends urgent care
- Symptom-to-specialty triage and appointment scheduling against real
  doctor availability (via PatientAgentBench's sandbox)
- Structured JSON summary at the end of every conversation, persisted to SQLite
- Supports four LLM providers (Bedrock, Anthropic, Gemini, Ollama) via one env var


An LLM provider (Gemini, via `LLM_PROVIDER=gemini`
in `.env`) is configured and has been exercised live: a real message got a
real, sensible follow-up question back through the full agent → backend →
WebSocket chain. What's still genuinely unverified: Bedrock/Anthropic
specifically (only Gemini has been live-tested so far), the *emergency LLM
classifier's* real-world recall on non-keyword phrasing (only the
deterministic keyword tier has been exercised live).

## Setup

See **[`SETUP.md`](SETUP.md)** for the full standalone setup guide
(prerequisites, all four LLM provider options with their tradeoffs, running
tests, running the backend/frontend, verifying it works, and a
troubleshooting section covering every real issue hit during development —
a macOS/Docker port conflict, `.env` vs. `.env.example`, provider switches
needing a restart, and slow/hanging LLM calls).

Fastest path if you just want it running:

```bash
git clone https://github.com/amazon-science/PatientAgentBench.git ../PatientAgentBench
uv sync --python 3.12
cp .env.example .env   # then fill in one provider — see SETUP.md
uv run --package patient-intake-agent pytest agent/tests/ -v
cd backend && uv run --package patient-intake-backend uvicorn backend_app.main:app --app-dir src --port 8000
# separate terminal:
cd frontend && npm install && npm run dev
```

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
PatientAgentBench's sandbox) is in [`docs/architecture.md`](docs/architecture.md).

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
something that reads as a diagnosis. (Emergency number is 112.)

**What Intake actually converges on.** Rather than stopping after any single
exchange, Intake's prompt (`nodes/intake.py::_EXTRACTION_PROMPT_TEMPLATE`)
gives the LLM a priority-ordered checklist and tells it not to rush — apply
only whichever categories are relevant to what the patient actually
described:

1. Age — always required, every conversation. This is the one category
   enforced at the *code* level too, not just prompted: if the model sets
   `ready_for_triage=true` without age having been captured, `intake_node`
   overrides it and forces a direct age question instead of trusting it.
2. Symptom-specific detail: a temperature reading + associated symptoms
   (cold, cough, vomiting, chills, etc.) for fever; a location + character
   (muscle/joint/bone/gland/organ/nerve/skin) for pain.
3. Onset and severity of the chief complaint.
4. History — recurring before, currently on medication for it.
5. Any other symptoms alongside the main complaint.

Once that's satisfied, Intake doesn't hand off to Triage immediately —
it recites everything gathered back to the patient and waits for them to
confirm before moving on (`nodes/intake.py::_confirmation_recap` /
`_confirmation_phase`). Whether a reply counts as confirming vs. correcting
something is itself an LLM judgment call (`IntakeExtraction.patient_confirmed`),
not a keyword match — "actually I'm 30, not 29" gets merged like a real
correction, re-recited, and confirmation is asked for again, rather than
being misread as a "no". This whole exchange (gathering + confirming)
routinely takes 3-5 conversational turns in practice (verified live against
Gemini), not one — see `docs/transcripts/` for the shape of it.

## Conversation flow

One `graph.invoke(state)` call per incoming patient message; the caller
(FastAPI's WebSocket handler) holds `PatientState` between calls. Concretely,
for the assignment's own example ("I've had a fever since yesterday"):

1. **Turn 1** — Emergency Guard finds no red flags → Intake extracts a chief
   complaint, asks one follow-up question (e.g. age and severity), and
   pauses.
2. **Turns 2..N** — patient answers, Emergency Guard re-checks every turn,
   Intake keeps gathering (symptom-specific detail, history, medication —
   see [Prompting strategy](#prompting-strategy)) until it judges the
   picture well-rounded, *then* recites everything back and asks the
   patient to confirm — that confirmation exchange is itself one or more
   turns (a correction re-recites and asks again).
3. **The turn confirmation succeeds** — Intake hands off to Triage, which
   classifies a specialty and looks up real candidate doctors, then
   Scheduling asks for availability and pauses, all within that same
   invocation (Triage never pauses; it only uses what's already been
   collected).
4. **The next turn** — patient states their availability → Scheduling
   reconciles it against real open slots for the candidate doctor(s), books
   one (or falls back to a "no availability" outcome — never retries in a
   loop) → Summary serializes the final JSON.

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
- One of AWS Bedrock, the Anthropic API directly, or Gemini is reachable
  with access to the configured model — selectable via `LLM_PROVIDER` in
  `.env` (see `agent/src/patient_intake_agent/llm.py`). No other providers
  (e.g. OpenAI) are wired up.
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
- **Partial persistence.** *Finished* chats are durable — saved to SQLite
  (`backend/chats.db`) the moment a session reaches `stage == "done"`,
  readable via `GET /api/chats` / `GET /api/chats/{id}` even after a
  restart (see `backend/README.md`). *In-progress* conversations are still
  only in memory — a server restart mid-conversation loses that session's
  state, same as before.
- Single hardcoded sandbox, shared by every session on the backend (see
  Assumptions and `sandbox_setup.py`) — not connected to any real
  EHR/scheduling backend, and two concurrent users can compete for the same
  appointment slots.
- Wide-open CORS (`allow_origins=["*"]`) on the backend — fine for local
  development, not for a real deployment.
