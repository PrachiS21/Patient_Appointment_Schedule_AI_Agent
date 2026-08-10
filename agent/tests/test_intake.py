from fakes import FakeStructuredLLM, PoisonLLM
from patient_intake_agent.nodes.intake import IntakeExtraction, SymptomExtraction, intake_node
from patient_intake_agent.state import new_patient_state


def test_first_turn_extracts_and_asks_for_missing_info():
    state = new_patient_state()
    state["turn_input"] = "I've had a fever since yesterday."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                chief_complaint="Fever since yesterday",
                new_symptoms=[SymptomExtraction(name="fever", onset="yesterday")],
                demographics_update={},
                still_missing=["age", "severity"],
                ready_for_triage=False,
                next_question="What is your age?",
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert result["chief_complaint"] == "Fever since yesterday"
    assert result["symptoms"] == [{"name": "fever", "onset": "yesterday"}]
    assert result["missing_information"] == ["age", "severity"]
    assert result["awaiting_patient"] is True
    assert result["assistant_message"] == "What is your age?"
    assert "stage" not in result  # stays in "intake" implicitly
    assert result["conversation_history"][-2] == {
        "role": "patient",
        "content": "I've had a fever since yesterday.",
    }
    assert result["conversation_history"][-1] == {
        "role": "assistant",
        "content": "What is your age?",
    }


def test_does_not_reask_a_field_already_known_even_if_llm_lists_it_again():
    """Code-level backstop: still_missing is filtered against demographics
    already in state, regardless of what the (possibly wrong) LLM output says.
    """
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday"}]
    state["demographics"] = {"age": "34"}
    state["turn_input"] = "It's pretty mild."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                chief_complaint=None,
                new_symptoms=[SymptomExtraction(name="fever", severity="mild")],
                demographics_update={},
                still_missing=["age", "severity"],  # LLM incorrectly re-lists "age"
                ready_for_triage=False,
                next_question="On a scale of 1-10, how severe is it?",
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert result["missing_information"] == ["severity"]
    assert result["symptoms"] == [{"name": "fever", "onset": "yesterday", "severity": "mild"}]


def test_ready_for_triage_transitions_stage_and_stops_asking():
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday", "severity": "mild"}]
    state["demographics"] = {"age": "34"}
    state["turn_input"] = "That's everything."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                chief_complaint=None,
                new_symptoms=[],
                demographics_update={},
                still_missing=[],
                ready_for_triage=True,
                next_question="",
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert result["stage"] == "triage"
    assert result["awaiting_patient"] is False
    assert result["missing_information"] == []


def test_passthrough_when_not_in_intake_stage_never_calls_llm():
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["turn_input"] = "Monday morning works for me."

    result = intake_node(state, llm=PoisonLLM())

    assert result == {}
