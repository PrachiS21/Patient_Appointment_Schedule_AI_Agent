"""Shared graph state for the patient intake agent.

`PatientState` is the single object threaded through every node in the
five-node graph (agent/src/patient_intake_agent/graph.py). It is intentionally
a plain TypedDict, not a pydantic model or dataclass with methods: LangGraph
merges node return values into this dict by key on every step, so it has to
stay a plain mapping. Node functions read from it and return partial updates;
they never mutate it in place.

Fields fall into two groups:

- The eight fields required by the assignment spec (chief_complaint ..
  risk_level) — these also appear, unchanged, in the final summary JSON.
- A handful of orchestration fields (stage, turn_input, awaiting_patient,
  assistant_message) that exist only to drive turn-taking between graph
  invocations. See graph.py's module docstring for why the graph is invoked
  fresh per patient message rather than run once end-to-end.
"""

from __future__ import annotations

from typing import Literal, TypedDict

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

Stage = Literal["intake", "triage", "scheduling", "summary", "done"]


class Symptom(TypedDict, total=False):
    name: str
    onset: str
    severity: str
    notes: str


class CandidateDoctor(TypedDict, total=False):
    id: str
    name: str
    specialty: str
    credentials: str
    office: str


class AvailabilityWindow(TypedDict, total=False):
    date: str  # YYYY-MM-DD
    time_of_day: str  # "morning" | "afternoon" | "evening", or a specific HH:MM
    raw_text: str  # the patient's original phrasing, kept for the transcript


class SelectedAppointment(TypedDict, total=False):
    doctor: str
    specialty: str
    date: str
    time: str
    appointment_id: str
    status: str  # "booked" | "no_availability" | "pending"


class ConversationTurn(TypedDict):
    role: Literal["patient", "assistant"]
    content: str


class PatientState(TypedDict):
    # --- Required by the assignment's structured-summary schema ---
    chief_complaint: str | None
    symptoms: list[Symptom]
    demographics: dict
    conversation_history: list[ConversationTurn]
    emergency_flag: bool
    suggested_specialty: str | None
    candidate_doctors: list[CandidateDoctor]
    patient_availability: list[AvailabilityWindow]
    selected_appointment: SelectedAppointment | None
    missing_information: list[str]
    risk_level: RiskLevel

    # --- Orchestration-only fields (not part of the summary schema) ---
    stage: Stage
    turn_input: str | None
    awaiting_patient: bool
    assistant_message: str | None
    final_summary: dict | None


def new_patient_state() -> PatientState:
    """Build an empty PatientState with every field at its zero value.

    Use this to seed a new session; never construct the TypedDict literal
    inline elsewhere, so every new field added here is picked up everywhere.
    """
    return PatientState(
        chief_complaint=None,
        symptoms=[],
        demographics={},
        conversation_history=[],
        emergency_flag=False,
        suggested_specialty=None,
        candidate_doctors=[],
        patient_availability=[],
        selected_appointment=None,
        missing_information=[],
        risk_level="LOW",
        stage="intake",
        turn_input=None,
        awaiting_patient=False,
        assistant_message=None,
        final_summary=None,
    )
