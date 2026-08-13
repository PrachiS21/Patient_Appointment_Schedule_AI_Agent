# Architecture

Two views: the five-node graph itself, and how the whole application wraps
around PatientAgentBench. A print-ready PDF of this page (for the
assignment's separate "architecture diagram" deliverable) is at
[`architecture.pdf`](architecture.pdf) in this same directory.

## The five-node graph

Compiled by `agent/src/patient_intake_agent/graph.py::build_graph()`. No
cycles, no checkpointer — invoked once per incoming patient message (see
[Conversation flow](../README.md#conversation-flow) in the root README).
Emergency Guard is the actual entry point, not Intake — see the "why" in the
[Architecture](../README.md#architecture) section of the root README.

```mermaid
flowchart TD
    START(["patient message arrives"]) --> EG["Emergency Guard<br/>(keyword scan, then LLM fallback)"]

    EG -- "red flag found" --> SUM["Summary<br/>(serialize + validate JSON)"]
    EG -- "stage == scheduling" --> SCHED
    EG -- "otherwise" --> IN["Intake<br/>(extract symptoms/demographics,<br/>ask one follow-up question,<br/>then confirm before handoff)"]

    IN -- "still missing info,<br/>or awaiting confirmation<br/>(pause)" --> END1(["wait for next<br/>patient message"])
    IN -- "confirmed" --> TRI["Triage<br/>(rules table, then LLM;<br/>list_doctors on sandbox)"]

    TRI --> SCHED["Scheduling<br/>(get_available_appointments,<br/>reconcile, schedule_appointment)"]

    SCHED -- "no availability yet<br/>(pause)" --> END2(["wait for next<br/>patient message"])
    SCHED -- "booked, or<br/>no-overlap fallback" --> SUM

    SUM --> DONE(["structured JSON summary<br/>returned to caller"])

    classDef pause fill:#fef9c3,stroke:#854d0e,color:#111
    classDef terminal fill:#dcfce7,stroke:#166534,color:#111
    classDef emergency fill:#fee2e2,stroke:#7f1d1d,color:#111
    class END1,END2 pause
    class DONE terminal
    class EG emergency
```

Two things this diagram makes visible that the assignment's own linear
diagram (`Intake → Emergency Guard → Triage → Scheduling → Summary`)
doesn't: Emergency Guard sits on *every* path, including a direct route from
mid-scheduling straight to Summary; and only two nodes (Intake, Scheduling)
ever pause for another patient message — Triage always completes in the
same turn as whatever handed control to it. Intake's pause covers two
different sub-states now (still gathering, or reciting a confirmation
recap and waiting on a yes/correction) — both look identical to the graph
itself (`stage` stays `"intake"`), the distinction lives inside the node.

## How the application wraps PatientAgentBench

```mermaid
flowchart LR
    subgraph client["Browser"]
        FE["React + TypeScript<br/>chat window · typing indicator · summary panel"]
    end

    subgraph server["patient-intake-backend (FastAPI)"]
        WS["WS /ws/chat/{session_id}"]
        RESTS["GET /api/sessions/{id}/summary<br/>(in-memory, live sessions only)"]
        RESTC["GET /api/chats, /api/chats/{id}<br/>(SQLite, durable)"]
        SESS[("in-memory<br/>SessionStore")]
        DB[("SQLite<br/>chats.db")]
    end

    subgraph agent["patient-intake-agent"]
        GRAPH["five-node StateGraph"]
        LLM["ChatBedrockConverse · ChatAnthropic ·<br/>ChatGoogleGenerativeAI · ChatOllama<br/>(LLM_PROVIDER env var picks one)"]
    end

    subgraph pab["PatientAgentBench (reused, not forked)"]
        SANDBOX["HealthcareSandbox<br/>offices · doctors · slots · bookings"]
        TOOLS["appointment_tools<br/>(list_doctors, get_available_appointments,<br/>schedule_appointment, ...)"]
    end

    FE <-- "JSON over WebSocket" --> WS
    FE -. "REST (browse past chats)" .-> RESTC
    WS --> SESS
    WS -- "on stage == done" --> DB
    WS -- "graph.invoke(state)<br/>(failures caught -> error event,<br/>connection stays open)" --> GRAPH
    GRAPH -- "structured_call()" --> LLM
    GRAPH -- "get_doctors(), get_available_slots(),<br/>book_appointment()" --> SANDBOX
    TOOLS -.->|"same sandbox methods,<br/>reused directly"| SANDBOX
    RESTS --> SESS
    RESTC --> DB

    classDef reused fill:#e0e7ff,stroke:#3730a3,color:#111
    class SANDBOX,TOOLS reused
```

The two PatientAgentBench boxes (blue) are literally imported, not
reimplemented — `agent/src/patient_intake_agent/sandbox_setup.py` builds a
real `HealthcareSandbox` and Triage/Scheduling call its methods
(`get_doctors`, `get_available_slots`, `book_appointment`) directly, the same
methods PatientAgentBench's own `appointment_tools.py` calls internally. See
[`docs/exploration/patientagentbench-notes.md`](exploration/patientagentbench-notes.md)
for the full reused-vs-built breakdown.

Two data stores, on purpose, not one merged into the other: `SessionStore`
is fast, in-memory, and only ever holds *live* conversations — gone on
restart. `chats.db` (SQLite) is the durable record, written exactly once per
session, the moment it reaches `stage == "done"` (see `backend/README.md`).

A `graph.invoke()` failure — a provider rate limit, a transient server
error, a timeout, none of which are within this app's control — is caught
per-turn in the WebSocket handler and reported to the browser as a visible,
in-transcript error message rather than left to kill the connection
silently.
