"""Scheduling node.

Collects the patient's available time windows conversationally, reconciles
them against real open slots for the candidate doctors, confirms, and books
— using the real PatientAgentBench sandbox, not a re-implementation.

Turn 1 (no `patient_availability` yet, and we haven't already asked — i.e.
`awaiting_patient` is still False on the way in): ask and pause.

Turn 2 (no `patient_availability` yet, but `awaiting_patient` is True —
meaning we asked last turn and this turn's `turn_input` is the patient's
answer to that question): treat `turn_input` as the availability text and
proceed to reconciliation. `awaiting_patient` is what lets this node tell
"haven't asked yet" apart from "just got the reply" — both look like an
empty `patient_availability` list otherwise, since nothing else populates
that field between turns.

From there: pull every open slot for the candidate doctors via
`HealthcareSandbox.get_available_slots()` (the same method the real
`get_available_appointments` tool calls internally), then ask the LLM to
pick the best-matching slot_id from that *real, grounded* list against what
the patient said — or say none match. This avoids ever inventing a time
that isn't actually bookable. A match is booked directly via
`HealthcareSandbox.book_appointment()` (again, the same method the real
`schedule_appointment` tool calls internally).

No-overlap handling: if there are no open slots for the candidate doctors at
all, or the LLM finds no match between the real slots and what the patient
described, this does NOT loop asking again — it falls straight through to
Summary with `selected_appointment.status = "no_availability"`, which the
Summary node turns into `requires_human = true`. A production version would
retry with alternate windows; for this MVP, one clean human handoff is
simpler and impossible to get stuck in a loop from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_llm, structured_call
from ..sandbox_setup import get_default_sandbox
from ..state import ConversationTurn, PatientState, SelectedAppointment

ASK_AVAILABILITY_MESSAGE = "What days and times generally work best for your appointment?"

NO_AVAILABILITY_MESSAGE = (
    "I couldn't find an appointment time that matches what you described. "
    "I'll flag this for a staff member to help schedule you directly."
)


class SlotSelection(BaseModel):
    slot_id: str | None = Field(
        description="The chosen slot_id from the list, or null if none reasonably "
        "overlap with the patient's stated availability."
    )
    reasoning: str


def _format_slots(slots) -> str:
    if not slots:
        return "(no open slots)"
    return "\n".join(f"- {s.id}: {s.date} at {s.time} ({s.appointment_type})" for s in slots)


def _no_availability_result(
    state: PatientState,
    history_update: list[ConversationTurn],
    patient_availability: list,
) -> dict:
    return {
        "awaiting_patient": False,
        "patient_availability": patient_availability,
        "selected_appointment": {"status": "no_availability"},
        "stage": "summary",
        "assistant_message": NO_AVAILABILITY_MESSAGE,
        "conversation_history": state["conversation_history"] + history_update + [
            {"role": "assistant", "content": NO_AVAILABILITY_MESSAGE}
        ],
    }


def scheduling_node(state: PatientState, llm=None, sandbox=None) -> dict:
    if not state["patient_availability"] and not state["awaiting_patient"]:
        return {
            "awaiting_patient": True,
            "assistant_message": ASK_AVAILABILITY_MESSAGE,
            "conversation_history": state["conversation_history"] + [
                {"role": "assistant", "content": ASK_AVAILABILITY_MESSAGE}
            ],
        }

    patient_availability = state["patient_availability"] or [
        {"raw_text": state["turn_input"] or ""}
    ]

    sandbox = sandbox or get_default_sandbox()
    doctor_ids = [d["id"] for d in state["candidate_doctors"]]
    all_slots = []
    for doctor_id in doctor_ids:
        all_slots.extend(sandbox.get_available_slots(doctor_id=doctor_id))
    all_slots.sort(key=lambda s: (s.date, s.time))

    history_update: list[ConversationTurn] = []
    if state["turn_input"]:
        history_update.append({"role": "patient", "content": state["turn_input"]})

    if not all_slots:
        return _no_availability_result(state, history_update, patient_availability)

    availability_text = "; ".join(w.get("raw_text", "") for w in patient_availability)

    llm = llm or get_llm()
    selection = structured_call(
        llm,
        SlotSelection,
        (
            "Match the patient's stated availability to one of these real open "
            "appointment slots. Only choose a slot_id that is listed below; if "
            "none reasonably overlap with what the patient said, return null.\n\n"
            f"Patient's stated availability: {availability_text}\n\n"
            f"Open slots:\n{_format_slots(all_slots)}"
        ),
    )

    chosen = next((s for s in all_slots if s.id == selection.slot_id), None)
    if chosen is None:
        return _no_availability_result(state, history_update, patient_availability)

    booked = sandbox.book_appointment(chosen.id, reason=state["chief_complaint"])
    doctor = sandbox.get_doctor(chosen.doctor_id)

    appointment: SelectedAppointment = {
        "doctor": doctor.name if doctor else "",
        "specialty": doctor.specialty if doctor else (state["suggested_specialty"] or ""),
        "date": chosen.date,
        "time": chosen.time,
        "appointment_id": booked.id if booked else "",
        "status": "booked" if booked else "no_availability",
    }

    message = (
        f"You're booked with {appointment['doctor']} on {appointment['date']} at {appointment['time']}."
        if booked
        else "That slot was just taken — I'll have a staff member follow up to find another time."
    )

    return {
        "awaiting_patient": False,
        "patient_availability": patient_availability,
        "selected_appointment": appointment,
        "stage": "summary",
        "assistant_message": message,
        "conversation_history": state["conversation_history"] + history_update + [
            {"role": "assistant", "content": message}
        ],
    }
