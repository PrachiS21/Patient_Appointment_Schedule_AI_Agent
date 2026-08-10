"""Triage/Routing node.

Classifies collected symptoms into a specialty, then looks up real candidate
doctors for that specialty from the PatientAgentBench sandbox. Never pauses
for patient input — it only uses information Intake already collected.

Two-tier classification, cheapest first:
  1. A small rules table for obvious symptom combinations (e.g. chest pain +
     shortness of breath -> Cardiology). Deterministic, free, no LLM call.
  2. An LLM classifier for everything else, constrained to
     `appointment_tools.VALID_SPECIALTIES` (imported directly from
     PatientAgentBench — the same vocabulary its `list_doctors` and
     `get_available_appointments` tools enforce). If the LLM returns
     anything outside that vocabulary, we clamp to "Primary Care" rather
     than pass a value downstream that would break those tool calls.

Candidate doctors are read directly from `HealthcareSandbox.get_doctors()`
(the same sandbox method the real `list_doctors` tool calls internally) —
structured data, not a re-parse of that tool's human-readable string output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from patient_agent_bench.tools.appointment_tools import VALID_SPECIALTIES

from ..llm import get_llm, structured_call
from ..sandbox_setup import get_default_sandbox
from ..state import CandidateDoctor, PatientState

# (keywords, specialty) — first match wins. Deliberately narrow: this only
# needs to catch combinations unambiguous enough that a full LLM call would
# be overkill, not to be exhaustive.
RULES_TABLE: list[tuple[list[str], str]] = [
    (["chest pain", "shortness of breath", "palpitation"], "Cardiology"),
    (["rash", "skin lesion", "itchy skin", "hives"], "Dermatology"),
    (["joint pain", "fracture", "sprain", "back pain"], "Orthopedics"),
    (["excessive thirst", "blood sugar", "thyroid"], "Endocrinology"),
    (["anxiety", "depression", "panic attack"], "Psychiatry"),
    (["pregnan", "menstrual", "pelvic pain"], "OB/GYN"),
]


class TriageClassification(BaseModel):
    specialty: str = Field(description=f"One of: {', '.join(VALID_SPECIALTIES)}")
    confidence: float = Field(ge=0, le=1)
    reasoning: str


def _symptom_text(state: PatientState) -> str:
    parts = [state["chief_complaint"] or ""]
    for symptom in state["symptoms"]:
        parts.append(" ".join(str(v) for v in symptom.values()))
    return " ".join(parts)


def _rules_fallback(text: str) -> str | None:
    lowered = text.lower()
    for keywords, specialty in RULES_TABLE:
        if any(keyword in lowered for keyword in keywords):
            return specialty
    return None


def _candidate_doctors(sandbox, specialty: str) -> list[CandidateDoctor]:
    doctors = [d for d in sandbox.get_doctors() if d.specialty.lower() == specialty.lower()]
    result: list[CandidateDoctor] = []
    for doctor in doctors:
        office = sandbox.get_office(doctor.office_id)
        result.append(
            {
                "id": doctor.id,
                "name": doctor.name,
                "specialty": doctor.specialty,
                "credentials": doctor.credentials,
                "office": office.name if office else "",
            }
        )
    return result


def triage_node(state: PatientState, llm=None, sandbox=None) -> dict:
    text = _symptom_text(state)

    specialty = _rules_fallback(text)
    if specialty is None:
        llm = llm or get_llm()
        result = structured_call(
            llm,
            TriageClassification,
            (
                "Classify the medical specialty this patient should be routed "
                "to, based only on the symptoms below. Choose exactly one "
                f"specialty from this list: {', '.join(VALID_SPECIALTIES)}. "
                "Do not diagnose a condition — only recommend a care specialty.\n\n"
                f"Chief complaint and symptoms: {text.strip()}"
            ),
        )
        specialty = result.specialty if result.specialty in VALID_SPECIALTIES else "Primary Care"

    sandbox = sandbox or get_default_sandbox()

    return {
        "suggested_specialty": specialty,
        "candidate_doctors": _candidate_doctors(sandbox, specialty),
        "stage": "scheduling",
    }
