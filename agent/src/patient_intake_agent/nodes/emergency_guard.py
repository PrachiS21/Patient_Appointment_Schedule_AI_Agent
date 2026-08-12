"""Emergency Guard node.

Sits unconditionally between Intake and everything else in the graph (see
graph.py) — it runs on every single patient turn, not just the first one or
the last one, because the edge into it has no condition on it.

Two tiers, cheapest first:

1. Keyword scan (RED_FLAG_KEYWORDS) over the latest patient message plus
   every symptom collected so far. Deterministic, free, no LLM call — most
   obvious emergencies (severe chest pain, can't breathe, suicidal ideation)
   are caught here and the LLM is never even invoked.
2. LLM classifier fallback, only reached when the keyword tier found
   nothing, for phrasing that doesn't hit a listed phrase but is still
   clearly urgent (e.g. "the left side of my face stopped moving" — a
   stroke sign with no keyword match).

Non-negotiable, enforced by construction: neither tier ever names a
condition. Both only ever produce the same fixed "seek urgent care" message
— there is no code path where this node's output could read as a diagnosis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_llm, structured_call
from ..state import PatientState

RED_FLAG_KEYWORDS = [
    "chest pain", "chest pressure", "crushing pain",
    "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "trouble breathing",
    "face drooping", "slurred speech", "sudden numbness", "sudden weakness",
    "severe bleeding", "uncontrolled bleeding", "bleeding a lot",
    "coughing blood", "vomiting blood",
    "suicidal", "kill myself", "want to die", "end my life", "hurt myself",
    "unconscious", "unresponsive", "passed out", "seizure", "convulsions",
    "anaphylaxis", "severe allergic reaction", "throat closing", "can't swallow",
    "severe abdominal pain", "worst headache of my life", "sudden severe headache",
]

URGENT_CARE_MESSAGE = (
    "Based on what you've described, these symptoms warrant urgent medical evaluation. "
    "Please call 112 or go to the nearest emergency room now. This is not a diagnosis — "
    "it's a recommendation to be seen urgently, out of caution."
)


class EmergencyClassification(BaseModel):
    is_emergency: bool = Field(
        description="True only for symptoms that plausibly need emergency-level care right now."
    )
    reasoning: str = Field(description="One sentence, symptom-based. Never names a condition.")


def _keyword_hit(text: str) -> str | None:
    lowered = text.lower()
    for phrase in RED_FLAG_KEYWORDS:
        if phrase in lowered:
            return phrase
    return None


def _collected_text(state: PatientState) -> str:
    parts = [state["turn_input"] or "", state["chief_complaint"] or ""]
    for symptom in state["symptoms"]:
        parts.append(" ".join(str(v) for v in symptom.values()))
    return " ".join(parts)


def _emergency_result() -> dict:
    return {
        "emergency_flag": True,
        "risk_level": "HIGH",
        "assistant_message": URGENT_CARE_MESSAGE,
    }


def emergency_guard_node(state: PatientState, llm=None) -> dict:
    text = _collected_text(state)

    if _keyword_hit(text):
        return _emergency_result()

    if not text.strip():
        return {}

    llm = llm or get_llm()
    result = structured_call(
        llm,
        EmergencyClassification,
        (
            "You are a safety classifier for a patient-intake assistant. Decide "
            "whether the following patient-reported information describes a "
            "potential medical emergency requiring urgent or emergency-level "
            "care. Do not diagnose a condition — only assess urgency.\n\n"
            f"Patient-reported information so far:\n{text.strip()}"
        ),
    )

    if not result.is_emergency:
        return {}

    return _emergency_result()
