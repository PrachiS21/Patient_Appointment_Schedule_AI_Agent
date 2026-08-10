"""Intake node.

Runs the conversational symptom/demographic interview. Only acts during the
"intake" stage — the graph's entry point runs this node unconditionally on
every turn (see graph.py), including turns where the conversation has
already moved on to scheduling, so this node must recognize when it's not
its turn and pass through untouched.

Each turn it does one of:
  - fold the latest patient message into state (chief complaint, symptoms,
    demographics) via a single LLM extraction call, then ask exactly one
    more follow-up question and pause; or
  - recognize enough has been gathered and hand off to triage.

"Never re-ask something already in state" is enforced two ways: the prompt
is shown everything already known and told not to ask about it again, and —
as a code-level backstop, since prompts can be ignored — anything the LLM
lists as still-missing is dropped if it's already a key in `demographics`
before it's used to decide what to ask.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_llm, structured_call
from ..state import ConversationTurn, PatientState, Symptom


class SymptomExtraction(BaseModel):
    name: str
    onset: str = ""
    severity: str = ""
    notes: str = ""


class IntakeExtraction(BaseModel):
    chief_complaint: str | None = Field(
        default=None, description="One-sentence chief complaint, patient's own words where possible."
    )
    new_symptoms: list[SymptomExtraction] = Field(default_factory=list)
    demographics_update: dict[str, str] = Field(
        default_factory=dict, description="e.g. {'age': '34'}"
    )
    still_missing: list[str] = Field(
        default_factory=list, description="Fields still needed before triage, e.g. ['age', 'onset']."
    )
    ready_for_triage: bool = Field(
        description="True once chief complaint, onset, and severity are known."
    )
    next_question: str = Field(
        default="", description="Exactly one follow-up question. Empty when ready_for_triage."
    )


def _merge_symptoms(existing: list[Symptom], new: list[SymptomExtraction]) -> list[Symptom]:
    merged: dict[str, dict] = {s["name"].lower(): dict(s) for s in existing}
    for symptom in new:
        key = symptom.name.lower()
        merged.setdefault(key, {})
        merged[key].update({k: v for k, v in symptom.model_dump().items() if v})
    return list(merged.values())


def _known_summary(state: PatientState) -> str:
    lines = [f"- chief complaint: {state['chief_complaint'] or '(not yet known)'}"]
    for symptom in state["symptoms"]:
        lines.append(f"- symptom: {symptom}")
    for key, value in state["demographics"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def intake_node(state: PatientState, llm=None) -> dict:
    if state["stage"] != "intake":
        return {}

    llm = llm or get_llm()

    prompt = (
        "You are conducting a medical intake interview. You are gathering "
        "symptom and demographic information only — you must never diagnose "
        "a condition or suggest a specific illness.\n\n"
        "Known so far:\n"
        f"{_known_summary(state)}\n\n"
        f"Patient's latest message: {state['turn_input']}\n\n"
        "Extract any new information from the latest message. Do not ask "
        "about anything already listed as known above. Ask exactly one "
        "clear follow-up question if more is needed (age, onset, and "
        "severity are the minimum before triage); otherwise set "
        "ready_for_triage=true and leave next_question empty."
    )

    extraction = structured_call(llm, IntakeExtraction, prompt)

    symptoms = _merge_symptoms(state["symptoms"], extraction.new_symptoms)
    demographics = {**state["demographics"], **extraction.demographics_update}
    chief_complaint = extraction.chief_complaint or state["chief_complaint"]

    known_keys = {k.lower() for k in demographics}
    still_missing = [f for f in extraction.still_missing if f.lower() not in known_keys]

    history_update: list[ConversationTurn] = []
    if state["turn_input"]:
        history_update.append({"role": "patient", "content": state["turn_input"]})

    if extraction.ready_for_triage and chief_complaint:
        return {
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "demographics": demographics,
            "missing_information": [],
            "awaiting_patient": False,
            "stage": "triage",
            "conversation_history": state["conversation_history"] + history_update,
        }

    question = extraction.next_question or "Can you tell me more about your symptoms?"
    history_update.append({"role": "assistant", "content": question})

    return {
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        "demographics": demographics,
        "missing_information": still_missing,
        "awaiting_patient": True,
        "assistant_message": question,
        "conversation_history": state["conversation_history"] + history_update,
    }
