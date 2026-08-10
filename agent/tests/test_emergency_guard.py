from fakes import FakeStructuredLLM, PoisonLLM
from patient_intake_agent.nodes.emergency_guard import (
    URGENT_CARE_MESSAGE,
    EmergencyClassification,
    emergency_guard_node,
)
from patient_intake_agent.state import new_patient_state


def test_obvious_emergency_caught_by_keyword_tier_without_calling_llm():
    """The required 'obvious emergency' case — must not need an LLM call at all."""
    state = new_patient_state()
    state["turn_input"] = "I have severe chest pain and I can't breathe."

    result = emergency_guard_node(state, llm=PoisonLLM())

    assert result["emergency_flag"] is True
    assert result["risk_level"] == "HIGH"
    assert result["assistant_message"] == URGENT_CARE_MESSAGE


def test_keyword_scan_also_covers_symptoms_already_in_state_not_just_latest_message():
    state = new_patient_state()
    state["turn_input"] = "just checking in"
    state["symptoms"] = [{"name": "chest pain", "severity": "severe"}]

    result = emergency_guard_node(state, llm=PoisonLLM())

    assert result["emergency_flag"] is True


def test_empty_input_skips_llm_entirely():
    state = new_patient_state()
    state["turn_input"] = None

    result = emergency_guard_node(state, llm=PoisonLLM())

    assert result == {}


def test_benign_symptom_falls_through_to_llm_and_is_cleared():
    state = new_patient_state()
    state["turn_input"] = "I have a mild runny nose."
    fake_llm = FakeStructuredLLM(
        [EmergencyClassification(is_emergency=False, reasoning="Mild, non-urgent symptom.")]
    )

    result = emergency_guard_node(state, llm=fake_llm)

    assert result == {}
    assert len(fake_llm.prompts) == 1


def test_llm_tier_catches_non_keyword_emergency_phrasing():
    """A real emergency described without hitting any listed keyword phrase."""
    state = new_patient_state()
    state["turn_input"] = "The left side of my face suddenly stopped moving an hour ago."
    fake_llm = FakeStructuredLLM(
        [EmergencyClassification(is_emergency=True, reasoning="Possible stroke sign.")]
    )

    result = emergency_guard_node(state, llm=fake_llm)

    assert result["emergency_flag"] is True
    assert result["risk_level"] == "HIGH"
    assert result["assistant_message"] == URGENT_CARE_MESSAGE
