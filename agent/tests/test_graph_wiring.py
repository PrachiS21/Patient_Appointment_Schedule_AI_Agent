"""Graph-level integration tests, over the REAL node logic (not stubs).

Per-node behavior has its own isolated tests (test_intake.py,
test_emergency_guard.py, test_triage.py, test_scheduling.py, test_summary.py)
with hardcoded inputs, per the project's build order. These tests instead
exercise the actual `StateGraph` routing (graph.py) end to end: multiple real
`.invoke()` calls in sequence, exactly the way the FastAPI backend will drive
it, over a real (but test-local, isolated) PatientAgentBench sandbox and a
`FakeStructuredLLM` queued with one canned response per LLM call the turn is
expected to make — so a wrong response count is itself a assertion that the
routing took an unexpected path.
"""

from fakes import FakeStructuredLLM, PoisonLLM
from patient_agent_bench.sandbox.sandbox import AppointmentSlot, Doctor, HealthcareSandbox, OfficeLocation
from patient_intake_agent.graph import build_graph
from patient_intake_agent.nodes.emergency_guard import EmergencyClassification
from patient_intake_agent.nodes.intake import IntakeExtraction, SymptomExtraction
from patient_intake_agent.nodes.scheduling import SlotSelection
from patient_intake_agent.nodes.triage import TriageClassification
from patient_intake_agent.state import new_patient_state

NOT_EMERGENCY = EmergencyClassification(is_emergency=False, reasoning="Nothing urgent reported.")


def _sandbox_with_one_slot() -> HealthcareSandbox:
    office = OfficeLocation(
        id="office_001", name="Test Clinic", address="1 Test St",
        city="Boston", state="MA", zip_code="02118", phone="555-000-0000",
        hours="Mon-Fri 8AM-5PM",
    )
    doctor = Doctor(id="doc_001", name="Dr. Alice Nguyen", specialty="Primary Care", credentials="MD", office_id=office.id)
    slot = AppointmentSlot(
        id="slot_001", doctor_id=doctor.id, office_id=office.id,
        date="2026-08-10", time="09:00", duration_minutes=30,
        available=True, appointment_type="in_person",
    )
    sandbox = HealthcareSandbox()
    sandbox.offices = {office.id: office}
    sandbox.doctors = {doctor.id: doctor}
    sandbox.slots = {slot.id: slot}
    return sandbox


def test_full_three_turn_conversation_books_a_real_appointment():
    sandbox = _sandbox_with_one_slot()
    fake_llm = FakeStructuredLLM(
        [
            # Turn 1: "I've had a fever since yesterday." (guard runs before intake)
            NOT_EMERGENCY,
            IntakeExtraction(
                chief_complaint="Fever since yesterday",
                new_symptoms=[SymptomExtraction(name="fever", onset="yesterday")],
                still_missing=["age", "severity"],
                ready_for_triage=False,
                next_question="What is your age, and how severe is the fever?",
            ),
            # Turn 2: "I'm 34 and it's mild."
            NOT_EMERGENCY,
            IntakeExtraction(
                demographics_update={"age": "34"},
                new_symptoms=[SymptomExtraction(name="fever", severity="mild")],
                still_missing=[],
                ready_for_triage=True,
            ),
            TriageClassification(specialty="Primary Care", confidence=0.9, reasoning="Routine fever."),
            # Turn 3: "Monday morning works great."
            NOT_EMERGENCY,
            SlotSelection(slot_id="slot_001", reasoning="Matches Monday morning."),
        ]
    )
    graph = build_graph(llm=fake_llm, sandbox=sandbox)

    state = new_patient_state()
    state["turn_input"] = "I've had a fever since yesterday."
    state = graph.invoke(state)
    assert state["stage"] == "intake"
    assert state["awaiting_patient"] is True

    state["turn_input"] = "I'm 34 and it's mild."
    state = graph.invoke(state)
    assert state["stage"] == "scheduling"
    assert state["awaiting_patient"] is True
    assert state["suggested_specialty"] == "Primary Care"

    state["turn_input"] = "Monday morning works great."
    state = graph.invoke(state)

    assert state["stage"] == "done"
    assert state["selected_appointment"]["status"] == "booked"
    assert state["final_summary"]["scheduled_appointment"]["status"] == "booked"
    assert sandbox.get_booked_appointments()  # really booked, not just described


def test_emergency_keyword_short_circuits_straight_to_summary_no_llm_needed():
    graph = build_graph(llm=PoisonLLM(), sandbox=_sandbox_with_one_slot())
    state = new_patient_state()
    state["turn_input"] = "I have severe chest pain and can't breathe."

    result = graph.invoke(state)

    assert result["emergency_flag"] is True
    assert result["risk_level"] == "HIGH"
    assert result["stage"] == "done"
    assert result["suggested_specialty"] is None
    assert result["selected_appointment"] is None
    assert result["final_summary"]["requires_human"] is True


def test_emergency_short_circuits_even_mid_scheduling():
    """Emergency Guard must run on every turn, not just the first one."""
    graph = build_graph(llm=PoisonLLM(), sandbox=_sandbox_with_one_slot())
    state = new_patient_state()
    state["chief_complaint"] = "fever since yesterday"
    state["stage"] = "scheduling"
    state["awaiting_patient"] = True
    state["candidate_doctors"] = [{"id": "doc_001", "name": "Dr. Alice Nguyen", "specialty": "Primary Care"}]
    state["turn_input"] = "Actually now I have severe chest pain and can't breathe."

    result = graph.invoke(state)

    assert result["emergency_flag"] is True
    assert result["stage"] == "done"
    assert result["selected_appointment"] is None


def test_no_availability_overlap_flows_through_the_whole_graph_to_summary():
    """Required no-availability-overlap case, exercised through full graph routing."""
    empty_sandbox = HealthcareSandbox()  # candidate doctor has zero registered slots
    # One LLM call expected: the emergency guard's classifier tier (the stated
    # text has no keyword hit, so it can't skip the LLM the way keyword cases
    # do); scheduling itself finds zero slots and never reaches its own LLM call.
    fake_llm = FakeStructuredLLM([NOT_EMERGENCY])
    graph = build_graph(llm=fake_llm, sandbox=empty_sandbox)
    state = new_patient_state()
    state["chief_complaint"] = "fever since yesterday"
    state["stage"] = "scheduling"
    state["awaiting_patient"] = True
    state["candidate_doctors"] = [{"id": "doc_001", "name": "Dr. Alice Nguyen", "specialty": "Primary Care"}]
    state["turn_input"] = "Anytime next week works."

    result = graph.invoke(state)

    assert result["stage"] == "done"
    assert result["selected_appointment"]["status"] == "no_availability"
    assert result["final_summary"]["requires_human"] is True
