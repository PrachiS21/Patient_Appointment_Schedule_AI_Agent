"""Intake node.

Runs the conversational symptom/demographic interview. Only acts during the
"intake" stage — the graph's entry point runs this node unconditionally on
every turn (see graph.py), including turns where the conversation has
already moved on to scheduling, so this node must recognize when it's not
its turn and pass through untouched.

Two phases, both while `stage` stays "intake" (no graph-level routing
change needed — see below):

1. **Gathering.** Each turn: fold the latest patient message into state
   (chief complaint, symptoms, demographics) via one LLM extraction call,
   then either ask one more follow-up message and pause, or — once a
   well-rounded picture has been gathered — move to phase 2.

   "Well-rounded", in priority order, applying only whichever categories
   are relevant to what the patient described:
     1. Age and sex/gender — always asked, every conversation.
     2. Symptom-specific detail: temperature + associated symptoms for
        fever; location + character (muscle/joint/bone/gland/organ/nerve/
        skin) for pain.
     3. Onset and severity.
     4. History: recurring before, currently on medication for it.
     5. Any other symptoms alongside the main complaint.
   This is prompt-driven, not a hardcoded decision tree. Age specifically is
   the one category enforced at the code level too: even if the model says
   ready_for_triage=true, the node overrides with a direct age question if
   age isn't actually in `demographics` yet. Sex/gender is prompted the same
   way but not code-enforced — it doesn't block the transition to Triage the
   way a missing age does.

2. **Confirming.** Once phase 1 decides it's done, instead of handing off
   to Triage immediately, it recites everything gathered back to the
   patient and asks them to confirm before moving on. The next turn's
   reply is interpreted as either a confirmation (→ hand off to Triage) or
   a correction/addition (→ merge it, then re-recite and ask again) — this
   is itself an LLM judgment call (`IntakeExtraction.patient_confirmed`),
   not a keyword match on the reply, so "actually my age is 30, not 29"
   is handled the same way a real correction would be.

`awaiting_confirmation` (PatientState) is what lets the node tell phase 1
replies apart from phase 2 replies — both look identical otherwise (just
another patient message while `stage == "intake"`). No graph-level routing
change was needed for this: `stage` doesn't change until the patient
actually confirms, so the existing intake pause/continue conditional edge
(graph.py::route_after_intake) already does the right thing throughout both
phases.

"Never re-ask something already in state" is enforced two ways: the prompt
is shown everything already known and told not to ask about it again, and —
as a code-level backstop — anything the LLM lists as still-missing is
dropped if it's already a key in `demographics` before it's used to decide
what to ask.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_llm, structured_call
from ..state import ConversationTurn, PatientState, Symptom

FORCED_AGE_QUESTION = "Before I go any further — what is your age?"
CONFIRMATION_PROMPT_SUFFIX = "Does this all look correct? Reply OK to continue, or let me know if anything needs fixing."


class SymptomExtraction(BaseModel):
    name: str
    onset: str = ""
    severity: str = ""
    location: str = Field(default="", description="Where it hurts, for pain-type symptoms.")
    character: str = Field(
        default="",
        description="Type of symptom, e.g. muscle, joint, bone, gland/lymph node, organ/internal, nerve, skin.",
    )
    notes: str = Field(
        default="",
        description="Anything else relevant: temperature reading, associated symptoms, "
        "whether it's recurring, current medication for this problem, etc.",
    )


class IntakeExtraction(BaseModel):
    chief_complaint: str | None = Field(
        default=None, description="One-sentence chief complaint, patient's own words where possible."
    )
    new_symptoms: list[SymptomExtraction] = Field(default_factory=list)
    demographics_update: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'age': '34', 'recurring': 'yes, monthly', 'current_medication': 'none'}",
    )
    still_missing: list[str] = Field(
        default_factory=list, description="Fields still needed before triage, e.g. ['age', 'temperature']."
    )
    ready_for_triage: bool = Field(
        description="True once age is known AND a well-rounded picture has been gathered "
        "(symptom-specific detail, onset, severity, history/medication) — see the "
        "priority list in the prompt. Don't rush this after a single exchange."
    )
    next_question: str = Field(
        default="",
        description="One follow-up message, which may bundle a couple of short related "
        "questions. Empty when ready_for_triage.",
    )
    patient_confirmed: bool = Field(
        default=False,
        description="Only meaningful when the patient was just shown a summary and asked "
        "to confirm it: true if this reply confirms it as correct (e.g. 'ok', 'yes, "
        "that's right'), false if it adds or corrects something instead.",
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


def _confirmation_recap(chief_complaint: str | None, symptoms: list[Symptom], demographics: dict) -> str:
    """Conversational recap shown to the patient for confirmation.

    Deliberately separate from summary.py's `_compose_summary_text` — that
    one is the formal, terminal JSON report; this one is a mid-conversation
    UX device meant to be read and okayed by the patient.
    """
    complaint_text = chief_complaint or "no chief complaint captured"
    if not complaint_text.endswith((".", "!", "?")):
        complaint_text += "."
    parts = [f"Here's what I have so far: {complaint_text}"]
    if demographics:
        parts.append("Details: " + "; ".join(f"{k}: {v}" for k, v in demographics.items()) + ".")
    for symptom in symptoms:
        bits = [symptom.get("name", "symptom")]
        for field in ("location", "character", "onset", "severity"):
            if symptom.get(field):
                bits.append(f"{field}: {symptom[field]}")
        if symptom.get("notes"):
            bits.append(symptom["notes"])
        parts.append("Symptom — " + ", ".join(bits) + ".")
    return " ".join(parts)


_EXTRACTION_PROMPT_TEMPLATE = """\
You are conducting a medical intake interview. You are gathering symptom \
and demographic information only — you must never diagnose a condition or \
suggest a specific illness.

Known so far:
{known}

Patient's latest message: {turn_input}

Extract any new information from the latest message, then decide what to \
ask next. Ask one follow-up message per turn (it may bundle a couple of \
short related questions), prioritized in this order — apply only the \
categories that are actually relevant to what the patient described:

1. Age is always required — if it isn't known yet, ask for it. Sex/gender \
is also part of the standard baseline — ask for it too if not yet known \
(it's fine to bundle with the age question).
2. Symptom-specific detail:
   - Fever mentioned: ask for a temperature reading if not given, and \
whether it comes with other symptoms (cold, cough, vomiting, chills, body \
aches, etc.).
   - Pain mentioned: ask exactly where it hurts, and try to narrow down \
what kind of pain it is (muscle, joint, bone, gland/lymph node, \
internal/organ, skin, nerve, etc.).
3. Onset (when it started) and severity (how bad it is right now).
4. History: has this happened before / is it recurring, and are they \
currently taking any medication for this specific problem.
5. Any other symptoms alongside the main complaint.

Only set ready_for_triage=true once age is known and you've covered as much \
of the above as is relevant and as the patient is willing to share — don't \
rush to triage after a single exchange; a well-rounded picture gathered \
over a few turns is better than stopping early. If the patient is clearly \
done answering or has nothing more to add, it's fine to stop there.

IMPORTANT — re-read "Known so far" above before writing next_question. If \
something is already listed there (including anything mentioned in the \
patient's latest message just now, e.g. "I am 34 years old" means age is \
now known), you must NOT ask about it again. Only ask about genuinely \
still-missing items.\
"""

_CONFIRMATION_PROMPT_TEMPLATE = """\
You are conducting a medical intake interview. You just showed the patient \
a recap of everything gathered and asked them to confirm it.

Recap the patient was shown:
{recap}

Patient's reply: {turn_input}

Decide: does this reply confirm the recap as correct (patient_confirmed=true), \
or does it add or correct something (patient_confirmed=false)? Either way, \
extract any new or corrected information from the reply the same as normal \
— e.g. "actually I'm 30, not 29" should update demographics_update even \
though it's also clearly not a plain confirmation. Never diagnose a \
condition.\
"""


def _extraction_phase(state: PatientState, llm) -> dict:
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
        known=_known_summary(state),
        turn_input=state["turn_input"],
    )
    extraction = structured_call(llm, IntakeExtraction, prompt)

    symptoms = _merge_symptoms(state["symptoms"], extraction.new_symptoms)
    demographics = {**state["demographics"], **extraction.demographics_update}
    chief_complaint = extraction.chief_complaint or state["chief_complaint"]

    known_keys = {k.lower() for k in demographics}
    age_known = "age" in known_keys
    still_missing = [f for f in extraction.still_missing if f.lower() not in known_keys]

    history_update: list[ConversationTurn] = []
    if state["turn_input"]:
        history_update.append({"role": "patient", "content": state["turn_input"]})

    if extraction.ready_for_triage and chief_complaint and age_known:
        recap = _confirmation_recap(chief_complaint, symptoms, demographics)
        question = f"{recap} {CONFIRMATION_PROMPT_SUFFIX}"
        history_update.append({"role": "assistant", "content": question})
        return {
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "demographics": demographics,
            "missing_information": [],
            "awaiting_patient": True,
            "awaiting_confirmation": True,
            "assistant_message": question,
            "conversation_history": state["conversation_history"] + history_update,
        }

    # Age is a hard requirement: even if the model thought it was done,
    # never let a missing age slip through.
    if extraction.ready_for_triage and not age_known:
        question = FORCED_AGE_QUESTION
        if "age" not in still_missing:
            still_missing = still_missing + ["age"]
    else:
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


def _confirmation_phase(state: PatientState, llm) -> dict:
    recap = _confirmation_recap(state["chief_complaint"], state["symptoms"], state["demographics"])
    prompt = _CONFIRMATION_PROMPT_TEMPLATE.format(recap=recap, turn_input=state["turn_input"])
    extraction = structured_call(llm, IntakeExtraction, prompt)

    symptoms = _merge_symptoms(state["symptoms"], extraction.new_symptoms)
    demographics = {**state["demographics"], **extraction.demographics_update}
    chief_complaint = extraction.chief_complaint or state["chief_complaint"]

    history_update: list[ConversationTurn] = []
    if state["turn_input"]:
        history_update.append({"role": "patient", "content": state["turn_input"]})

    if extraction.patient_confirmed:
        return {
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "demographics": demographics,
            "missing_information": [],
            "awaiting_patient": False,
            "awaiting_confirmation": False,
            "stage": "triage",
            "conversation_history": state["conversation_history"] + history_update,
        }

    # Not confirmed — a correction/addition came in. Merge it, re-recite,
    # and ask again rather than assuming what changed is trivial.
    new_recap = _confirmation_recap(chief_complaint, symptoms, demographics)
    question = f"Got it, updated. {new_recap} {CONFIRMATION_PROMPT_SUFFIX}"
    history_update.append({"role": "assistant", "content": question})

    return {
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        "demographics": demographics,
        "missing_information": [],
        "awaiting_patient": True,
        "awaiting_confirmation": True,
        "assistant_message": question,
        "conversation_history": state["conversation_history"] + history_update,
    }


def intake_node(state: PatientState, llm=None) -> dict:
    if state["stage"] != "intake":
        return {}

    llm = llm or get_llm()

    if state["awaiting_confirmation"]:
        return _confirmation_phase(state, llm)

    return _extraction_phase(state, llm)
