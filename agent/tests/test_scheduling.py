from fakes import FakeStructuredLLM, PoisonLLM
from patient_agent_bench.sandbox.sandbox import AppointmentSlot, Doctor, HealthcareSandbox, OfficeLocation
from patient_intake_agent.nodes.scheduling import (
    ASK_AVAILABILITY_MESSAGE,
    NO_AVAILABILITY_MESSAGE,
    SlotSelection,
    scheduling_node,
)
from patient_intake_agent.state import new_patient_state

DOCTOR = Doctor(id="doc_001", name="Dr. Alice Nguyen", specialty="Primary Care", credentials="MD", office_id="office_001")
CANDIDATE = [{"id": "doc_001", "name": "Dr. Alice Nguyen", "specialty": "Primary Care", "credentials": "MD", "office": "Test Clinic"}]


def _sandbox_with_slots(*slots: AppointmentSlot) -> HealthcareSandbox:
    office = OfficeLocation(
        id="office_001", name="Test Clinic", address="1 Test St",
        city="Boston", state="MA", zip_code="02118", phone="555-000-0000",
        hours="Mon-Fri 8AM-5PM",
    )
    sandbox = HealthcareSandbox()
    sandbox.offices = {office.id: office}
    sandbox.doctors = {DOCTOR.id: DOCTOR}
    sandbox.slots = {s.id: s for s in slots}
    return sandbox


def test_first_arrival_asks_and_pauses_without_touching_sandbox_or_llm():
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["candidate_doctors"] = CANDIDATE
    state["turn_input"] = "ok"
    state["awaiting_patient"] = False  # haven't asked yet

    result = scheduling_node(state, llm=PoisonLLM(), sandbox=object())

    assert result["awaiting_patient"] is True
    assert result["assistant_message"] == ASK_AVAILABILITY_MESSAGE


def test_no_open_slots_at_all_falls_back_to_summary_without_calling_llm():
    """The required no-availability-overlap case: candidate doctor exists but has zero open slots."""
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["candidate_doctors"] = CANDIDATE
    state["awaiting_patient"] = True  # we already asked; this is the reply
    state["turn_input"] = "Anytime next week."
    sandbox = _sandbox_with_slots()  # no slots registered

    result = scheduling_node(state, llm=PoisonLLM(), sandbox=sandbox)

    assert result["selected_appointment"] == {"status": "no_availability"}
    assert result["stage"] == "summary"
    assert result["assistant_message"] == NO_AVAILABILITY_MESSAGE


def test_llm_finds_no_overlap_between_real_slots_and_stated_availability():
    """The required no-availability-overlap case: slots exist, but none match what the patient said."""
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["candidate_doctors"] = CANDIDATE
    state["awaiting_patient"] = True
    state["turn_input"] = "Only late nights after 9pm work for me."
    sandbox = _sandbox_with_slots(
        AppointmentSlot(id="slot_001", doctor_id="doc_001", office_id="office_001", date="2026-08-10", time="09:00", duration_minutes=30, available=True, appointment_type="in_person")
    )
    fake_llm = FakeStructuredLLM([SlotSelection(slot_id=None, reasoning="No evening slots available.")])

    result = scheduling_node(state, llm=fake_llm, sandbox=sandbox)

    assert result["selected_appointment"] == {"status": "no_availability"}
    assert result["stage"] == "summary"
    # The slot must remain untouched — nothing was actually booked.
    assert sandbox.slots["slot_001"].available is True
    assert sandbox.get_booked_appointments() == []


def test_matched_slot_is_actually_booked_via_the_real_sandbox():
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["chief_complaint"] = "fever"
    state["candidate_doctors"] = CANDIDATE
    state["suggested_specialty"] = "Primary Care"
    state["awaiting_patient"] = True  # we asked last turn
    state["turn_input"] = "Monday morning works great."
    sandbox = _sandbox_with_slots(
        AppointmentSlot(id="slot_001", doctor_id="doc_001", office_id="office_001", date="2026-08-10", time="09:00", duration_minutes=30, available=True, appointment_type="in_person")
    )
    fake_llm = FakeStructuredLLM([SlotSelection(slot_id="slot_001", reasoning="Monday morning matches.")])

    result = scheduling_node(state, llm=fake_llm, sandbox=sandbox)

    assert result["stage"] == "summary"
    assert result["awaiting_patient"] is False
    assert result["patient_availability"] == [{"raw_text": "Monday morning works great."}]
    appointment = result["selected_appointment"]
    assert appointment["status"] == "booked"
    assert appointment["doctor"] == "Dr. Alice Nguyen"
    assert appointment["date"] == "2026-08-10"
    assert appointment["time"] == "09:00"
    assert appointment["appointment_id"]
    # Really booked in the sandbox, not just described in the return value.
    assert sandbox.slots["slot_001"].available is False
    assert len(sandbox.get_booked_appointments()) == 1
