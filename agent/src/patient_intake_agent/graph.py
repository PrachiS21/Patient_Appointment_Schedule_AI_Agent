"""The five-node patient intake state graph.

    START -> emergency_guard -[route_after_guard]-> intake -[route_after_intake]-> triage -> scheduling -[route_after_scheduling]-> summary -> END
                    |                                    |                                                    |
                    +-------------> summary               +------------------ END (paused) <-------------------+
                                       (emergency)

Emergency Guard is the true entry point, not Intake — a deliberate departure
from the assignment's diagram, which lists Intake first. Reasoning: Emergency
Guard's whole point is to be a fast, free, deterministic safety gate (see its
module docstring — the keyword tier costs no LLM call at all). If Intake ran
first, an obviously dangerous message ("severe chest pain, can't breathe")
would still burn an Intake LLM call before the guard ever got a chance to
short-circuit it — the requirement "runs on every turn" would technically
still hold, but the guard would no longer be the cheap first line of defense
it's supposed to be. Emergency Guard only needs the raw `turn_input` plus
whatever was already known from prior turns (both available before Intake
touches anything), so nothing is lost by moving it first.

Turn-taking model
------------------
This graph has NO cycles and NO checkpointer. It is a straight-line DAG that
is invoked once per incoming patient message, exactly the way
PatientAgentBench's own DefaultAssistantAgent is stateless per call and
replays the full message history on every turn (see
docs/exploration/patientagentbench-notes.md). The caller (the FastAPI
WebSocket handler, or a test) owns the authoritative `PatientState` between
calls:

1. Append the patient's new message, set it as `turn_input`, call `graph.invoke(state)`.
2. emergency_guard runs unconditionally on every single turn (this is what
   makes it run "on every patient turn, not just at the end" — it is not a
   special case, it is the entry point). If it trips, the graph short-circuits
   straight to summary, skipping intake/triage/scheduling entirely.
3. Otherwise, a conditional edge routes to whichever stage the conversation
   is actually in (`intake` or `scheduling` — `triage` is never re-entered
   this way, see below).
4. intake has its own conditional edge: stop (`END`) and hand control back
   to the caller because it needs another reply, or continue into triage
   once enough has been gathered.
5. triage never pauses (it only uses information already collected), so it
   always falls through into scheduling in the same invocation — this is
   the only edge that isn't gated by a turn boundary.
6. scheduling has its own conditional edge: pause for the patient's
   availability, or continue to summary once an appointment is resolved
   (booked, or explicitly no-availability).
7. summary always ends the graph.

Because there are no cycles, a single `.invoke()` call can never loop
forever regardless of what a node returns — the multi-turn "loop" only
exists across separate `.invoke()` calls made by the caller.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from .llm import get_llm
from .nodes import (
    emergency_guard_node,
    intake_node,
    scheduling_node,
    summary_node,
    triage_node,
)
from .sandbox_setup import get_default_sandbox
from .state import PatientState


def route_after_guard(state: PatientState) -> str:
    if state["emergency_flag"]:
        return "summary"
    if state["stage"] == "scheduling":
        return "scheduling"
    return "intake"


def route_after_intake(state: PatientState) -> str:
    return END if state["awaiting_patient"] else "triage"


def route_after_scheduling(state: PatientState) -> str:
    return END if state["awaiting_patient"] else "summary"


def build_graph(llm=None, sandbox=None):
    """Compile the five-node graph. No checkpointer — see module docstring.

    `llm` and `sandbox` are injectable so tests can build a graph over a fake
    LLM and/or an isolated sandbox instance (see
    agent/tests/test_graph_wiring.py) without touching AWS. Defaults to the
    real Bedrock model and the shared demo sandbox for actual use (the
    backend calls this with no arguments).
    """
    llm = llm or get_llm()
    sandbox = sandbox or get_default_sandbox()

    workflow = StateGraph(PatientState)

    workflow.add_node("emergency_guard", partial(emergency_guard_node, llm=llm))
    workflow.add_node("intake", partial(intake_node, llm=llm))
    workflow.add_node("triage", partial(triage_node, llm=llm, sandbox=sandbox))
    workflow.add_node("scheduling", partial(scheduling_node, llm=llm, sandbox=sandbox))
    workflow.add_node("summary", summary_node)

    workflow.add_edge(START, "emergency_guard")
    workflow.add_conditional_edges(
        "emergency_guard",
        route_after_guard,
        {"intake": "intake", "scheduling": "scheduling", "summary": "summary"},
    )
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {"triage": "triage", END: END},
    )
    workflow.add_edge("triage", "scheduling")
    workflow.add_conditional_edges(
        "scheduling",
        route_after_scheduling,
        {"summary": "summary", END: END},
    )
    workflow.add_edge("summary", END)

    return workflow.compile()
