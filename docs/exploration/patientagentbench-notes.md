# Understanding PatientAgentBench

Notes from studying the [PatientAgentBench](https://github.com/amazon-science/PatientAgentBench)
source (`src/patient_agent_bench/`) before writing any of our own code, and
from building [`minimal_conversation.py`](minimal_conversation.py) — a
standalone script that reuses its real sandbox and tools directly. This is
the source material for the "Your understanding of PatientAgentBench" section
of the root README.

## What PatientAgentBench actually is

It's a *benchmark*, not an app: a three-phase pipeline (generate a case →
run a scored multi-turn conversation between a simulated patient and an
assistant agent → evaluate the transcript with an LLM jury). We only reuse
one corner of it — the assistant-side harness and the healthcare sandbox —
and none of the benchmark/simulation/evaluation machinery.

## How patient scenarios are represented

A `PatientProfile` (`patient/patient_profile.py`) is a nested dataclass tree:
`account_info`, `personal_info`, `address`, `phone`, `emergency_contact`,
`insurance`, `pharmacy`, `care_team`, `medications`. Every leaf dataclass
carries a class-level `SCHEMA` dict describing its own fields in
prompt-friendly English (e.g. `"dob": "string (YYYY-MM-DD)"`). Two things
that schema is used for: (1) building the JSON-schema block inside the LLM
prompt that *generates* a synthetic profile (`patient/generator.py`,
`ENRICHMENT_PROMPT`), and (2) nothing else at runtime — once built, the
profile is just data.

For the assistant's system prompt, the profile is serialized to XML
(`PatientProfile.to_xml()`), not JSON — `_dict_to_xml` walks the `to_dict()`
output and skips empty/`None` fields entirely, which keeps the prompt shorter
than a full JSON dump with a lot of `null`s.

We reuse `PatientProfile` and its nested dataclasses verbatim (see
`build_hardcoded_patient_profile()` in `minimal_conversation.py`) but skip
`patient/generator.py`'s LLM-based enrichment — the assignment calls for one
hardcoded profile, and generation is a benchmark-seeding concern we don't
have.

## How the sandbox is represented and generated

`HealthcareSandbox` (`sandbox/sandbox.py`) is a thread-safe, in-memory store
of `OfficeLocation`, `Doctor`, `AppointmentSlot`, and `BookedAppointment`
dicts (all dataclasses with matching `SCHEMA`/`to_dict`/`from_dict`), plus a
reference to the `PatientProfile` (medications live there, not in the
sandbox). It's built once per benchmark conversation by
`sandbox/generator.py::initialize_sandbox`, which makes **its own separate
LLM call** to invent offices and doctors for the patient's city/state
(`generate_sandbox_data`), then generates two weeks of appointment slots in
plain Python (`_generate_appointment_slots` — weekdays only, 3-6 slots/doctor/day,
40-70% marked available, mixed in-person/telehealth). So a PatientAgentBench
conversation actually involves *two* LLM roles before the assistant even
sees a message: one to enrich the patient profile, one to generate the
sandbox inventory.

We reuse `HealthcareSandbox` and `_generate_appointment_slots` verbatim, but
hardcode the offices/doctors ourselves (`build_hardcoded_sandbox()`) instead
of calling `generate_sandbox_data` — one less hidden LLM dependency, and a
deterministic environment that's easier to demo and test against.

## How tools are wired to the sandbox

Each tool category (`tools/appointment_tools.py`, `prescription_tools.py`,
`profile_tools.py`, `telehealth_tools.py`) exposes a `get_*_tools(sandbox)`
factory that returns `StructuredTool`s built with `StructuredTool.from_function`,
each one a closure capturing the *same* `sandbox` instance. `create_tool_registry(sandbox)`
(`tools/registry.py`) calls all four factories and returns a `ToolRegistry`
(name → tool, with categories). This is a clean, directly reusable pattern —
our Scheduling node uses the exact same `list_doctors` / `get_available_appointments`
/ `schedule_appointment` / `cancel_appointment` / `list_appointments` tools,
bound to our own sandbox instance, with zero modification.

`appointment_tools.py::VALID_SPECIALTIES` (`Primary Care, Cardiology,
Dermatology, Endocrinology, Orthopedics, Psychiatry, OB/GYN`) is the fixed
vocabulary our Triage node's specialty classifier must target — `list_doctors`
and `get_available_appointments` reject anything outside this list.

## How conversations are managed (the important one)

This is the part most worth internalizing, because it directly shapes our
own graph design. **There is no persistent conversation memory anywhere in
PatientAgentBench's assistant harness — no LangGraph checkpointer, no
thread ID, no session store.**

`DefaultAssistantAgent.invoke()` (`assistant_agent/default_agent.py`) takes
the *entire* message history as a parameter on every call, builds a brand
new LangGraph ReAct agent from scratch (`langchain.agents.create_agent`,
`_create_agent()`), and invokes it once. The returned message list is the
input list plus whatever new messages this turn produced. The caller
(`runner/conversation_runner.py::_run_conversation`) owns a `Conversation`
object (`runner/conversation.py`, a thin wrapper around a
`list[HumanMessage | AIMessage | ToolMessage]`) and just keeps appending: on
turn *N*, it calls
`assistant_agent.invoke(messages=conversation.messages, user_profile=...)`
with everything from turns `1..N-1` still in the list, and appends only the
new tail (`Conversation.extend_from_agent_result`).

In other words: "conversation state" *is* the growing message list, full
stop. The assistant has no memory beyond what's literally in that list —
which is why the system prompt has to re-inject the *entire* patient profile
on every single turn (see below), and why nothing about "what have we asked
already" is tracked anywhere except by the LLM re-reading the transcript
each time.

This is the single biggest design decision we carried into our own graph
(see the module docstring in `agent/src/patient_intake_agent/graph.py`): our
five-node `StateGraph` also has no checkpointer and no cycles. It's invoked
fresh, once per patient message, with the caller (the FastAPI layer)
holding the authoritative `PatientState` dict between calls — the same
"stateless graph, stateful caller" shape PatientAgentBench uses, just with
an explicit typed dict standing in for the message list.

## How prompts are structured

One system prompt template per agent role (`assistant_agent/default_prompt.py`
for the assistant), a plain Python string with `{current_datetime}` and
`{user_profile}` placeholders, filled per-turn by `format_prompt_safe()`
(`config.py`) — which only substitutes placeholders that actually exist in
the template and warns (doesn't error) on unused kwargs, so swapping prompt
templates can't silently break on a missing placeholder. The default prompt
itself is organized into four tagged sections: `<current_datetime>`,
`<patient_profile>` (the XML dump), `<capabilities>` (what the four tool
categories let it do), `<guidelines>` (tone/behavior), and `<safety_rules>`
— the last of which is where "never provide definitive medical diagnoses"
and "recognize red flags that require immediate medical attention" live.
Notably, safety is handled *entirely* as prompt instruction, with no
code-level guard — there's no keyword scan or separate emergency-check step
anywhere in the harness. That's the specific gap our Emergency Guard node
fills: a code-level, always-runs check that doesn't depend on the LLM
choosing to follow the safety-rules paragraph.

## How patient state evolves through a conversation

Two genuinely different kinds of "state" coexist, and it's worth being
precise about which is which:

1. **The message list** (`Conversation.messages`) — this is what actually
   gives the assistant memory of the conversation. It grows monotonically;
   nothing is ever removed or summarized.
2. **The sandbox's mutable world state** — `patient_profile.medications`,
   `sandbox.slots[*].available`, `sandbox.booked_appointments` — which tools
   mutate as *side effects* of being called (e.g. `schedule_appointment`
   flips a slot's `available` flag and adds a `BookedAppointment`). This
   state is invisible to the LLM except through tool call results; the LLM
   never sees the sandbox directly, only what a tool chooses to print back.

The patient's *medical/demographic* profile, by contrast, does **not**
evolve during the conversation in the benchmark's normal flow — it's fixed
at generation time and re-injected unchanged into the system prompt every
turn (only `update_profile`-family tools mutate it, e.g. changing a
pharmacy). This is a real difference from our use case: PatientAgentBench
assumes an *already-onboarded* patient with a complete record, asking for
help with something concrete ("refill my prescription", "reschedule my
Tuesday appointment"). Our assignment is the opposite — a *new* patient with
an empty record, whose chief complaint and symptoms have to be extracted
from scratch through conversation. That's the main thing we had to add
rather than reuse: PatientAgentBench has no Intake, Emergency Guard, or
Triage concept at all — those three nodes are net-new, built on top of the
sandbox/tools layer we did reuse.

## Summary: reused vs. built

| Reused verbatim | Built new |
|---|---|
| `HealthcareSandbox`, `OfficeLocation`, `Doctor`, `AppointmentSlot`, `BookedAppointment` | Intake node (symptom/demographic extraction, no PatientAgentBench equivalent) |
| `PatientProfile` + nested dataclasses | Emergency Guard node (code-level safety check; PatientAgentBench only has prompt-level safety rules) |
| `create_tool_registry` / the 4 tool-category factories, esp. `appointment_tools.py` | Triage/Routing node (symptom → specialty classification; no PatientAgentBench equivalent) |
| `_generate_appointment_slots` | `PatientState` five-node `StateGraph` orchestration (PatientAgentBench's assistant is a single flat ReAct loop, not a multi-node graph) |
| The "stateless graph, full-history-per-call" conversation pattern | The stub-then-real, node-isolated-testing build process |
| `Conversation` helper's message-extraction logic (adapted for our console script) | The structured JSON summary (no PatientAgentBench equivalent — it never produces a terminal structured output) |
