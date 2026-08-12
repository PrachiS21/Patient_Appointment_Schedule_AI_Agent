"""Summary node.

Serializes the final `PatientState` into the structured JSON schema the
assignment requires, and validates it (via `PatientSummary`, a pydantic
model — see below) before returning. Always the last node reached on any
path through the graph: emergency short-circuit, no-availability fallback,
or a normal booked appointment.

Deliberately does NOT call an LLM. By this point every field it needs was
already produced by an upstream node (Intake extracted symptoms/complaint,
Emergency Guard set risk_level, Triage set specialty, Scheduling set the
appointment) — this node's job is formatting and validation, not generation,
matching how the assignment spec itself describes it ("serializes...
validates against the schema").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..state import PatientState
from .emergency_guard import URGENT_CARE_MESSAGE


class ScheduledAppointmentSummary(BaseModel):
    doctor: str | None = None
    time: str | None = None
    status: str | None = None


class PatientSummary(BaseModel):
    # Both optional: on the emergency short-circuit path, Intake never runs
    # at all (Emergency Guard is the graph's entry point — see graph.py's
    # module docstring), so neither is ever collected there.
    age: int | None = None
    sex: str | None = None
    chief_complaint: str | None
    symptoms: list[dict]
    summary: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: str
    requires_human: bool
    missing_information: list[str]
    specialty: str | None = None
    scheduled_appointment: ScheduledAppointmentSummary | None = None


def _parse_age(demographics: dict) -> int | None:
    """Best-effort int parse — `demographics["age"]` is free text an LLM
    populated (e.g. "34", "34 years old"), not a guaranteed clean integer."""
    raw = demographics.get("age")
    if not raw:
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else None


def _extract_sex(demographics: dict) -> str | None:
    for key in ("sex", "gender"):
        value = demographics.get(key)
        if value:
            return str(value)
    return None


def _compose_summary_text(state: PatientState) -> str:
    parts = [f"Chief complaint: {state['chief_complaint'] or 'not captured'}."]
    if state["symptoms"]:
        symptom_bits = "; ".join(
            symptom.get("name", "symptom")
            + (f" (onset {symptom['onset']})" if symptom.get("onset") else "")
            + (f", severity {symptom['severity']}" if symptom.get("severity") else "")
            for symptom in state["symptoms"]
        )
        parts.append(f"Reported symptoms: {symptom_bits}.")
    if state["demographics"]:
        demo_bits = "; ".join(f"{k}: {v}" for k, v in state["demographics"].items())
        parts.append(f"Patient details: {demo_bits}.")
    if state["emergency_flag"]:
        parts.append("Emergency red-flag symptoms were detected during intake.")
    return " ".join(parts)


def _recommendation(state: PatientState) -> str:
    if state["emergency_flag"]:
        return URGENT_CARE_MESSAGE

    appointment = state["selected_appointment"]
    specialty = state["suggested_specialty"] or "a specialist"

    if appointment and appointment.get("status") == "booked":
        return (
            f"Scheduled with {appointment.get('doctor', 'a provider')} "
            f"({appointment.get('specialty', specialty)}) on "
            f"{appointment.get('date', '')} at {appointment.get('time', '')}."
        )
    if appointment and appointment.get("status") == "no_availability":
        return (
            f"No matching {specialty} availability was found; a staff member "
            "will follow up to schedule directly."
        )
    return f"Recommend evaluation with {specialty}."


def _requires_human(state: PatientState) -> bool:
    if state["emergency_flag"] or state["risk_level"] == "HIGH":
        return True
    appointment = state["selected_appointment"]
    if appointment and appointment.get("status") == "no_availability":
        return True
    return False


def _scheduled_appointment_payload(state: PatientState) -> dict | None:
    appointment = state["selected_appointment"]
    if not appointment:
        return None
    time_str = None
    if appointment.get("date") and appointment.get("time"):
        time_str = f"{appointment['date']} {appointment['time']}"
    return {
        "doctor": appointment.get("doctor"),
        "time": time_str,
        "status": appointment.get("status"),
    }


def summary_node(state: PatientState) -> dict:
    payload = {
        "age": _parse_age(state["demographics"]),
        "sex": _extract_sex(state["demographics"]),
        "chief_complaint": state["chief_complaint"],
        "symptoms": state["symptoms"],
        "summary": _compose_summary_text(state),
        "risk_level": state["risk_level"],
        "recommendation": _recommendation(state),
        "requires_human": _requires_human(state),
        "missing_information": state["missing_information"],
        "specialty": state["suggested_specialty"],
        "scheduled_appointment": _scheduled_appointment_payload(state),
    }
    # Raises if malformed — fail loudly here rather than ship a broken summary.
    validated = PatientSummary.model_validate(payload)

    return {
        "stage": "done",
        "awaiting_patient": False,
        "assistant_message": validated.summary,
        "final_summary": validated.model_dump(),
    }
