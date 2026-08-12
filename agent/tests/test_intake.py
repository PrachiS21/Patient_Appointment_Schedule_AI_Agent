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


def test_ready_for_triage_asks_for_confirmation_instead_of_transitioning_immediately():
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

    assert "stage" not in result  # must NOT jump straight to triage
    assert result["awaiting_patient"] is True
    assert result["awaiting_confirmation"] is True
    assert "Fever since yesterday" in result["assistant_message"]
    assert "age: 34" in result["assistant_message"]
    assert "correct" in result["assistant_message"].lower()


def test_confirmation_recap_does_not_double_up_a_trailing_period():
    state = new_patient_state()
    state["chief_complaint"] = "I have a fever since this morning."  # already punctuated
    state["symptoms"] = []
    state["demographics"] = {"age": "40"}
    state["turn_input"] = "That's all."
    fake_llm = FakeStructuredLLM([IntakeExtraction(still_missing=[], ready_for_triage=True)])

    result = intake_node(state, llm=fake_llm)

    assert ".." not in result["assistant_message"]


def test_passthrough_when_not_in_intake_stage_never_calls_llm():
    state = new_patient_state()
    state["stage"] = "scheduling"
    state["turn_input"] = "Monday morning works for me."

    result = intake_node(state, llm=PoisonLLM())

    assert result == {}


def test_age_is_a_hard_requirement_even_if_the_model_thinks_its_done():
    """Code-level backstop: the model can say ready_for_triage=True, but
    without age actually known, the node must override and force an age
    question rather than trust it (and must not even reach the confirmation
    phase).
    """
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday", "severity": "mild"}]
    state["demographics"] = {}  # age never captured
    state["turn_input"] = "No, nothing else."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                chief_complaint=None,
                new_symptoms=[],
                demographics_update={},
                still_missing=[],
                ready_for_triage=True,  # model incorrectly thinks it's done
                next_question="",
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert "stage" not in result
    assert "awaiting_confirmation" not in result  # unset -> stays False, never reaches confirmation
    assert result["awaiting_patient"] is True
    assert "age" in result["assistant_message"].lower()
    assert "age" in result["missing_information"]


def test_captures_pain_location_and_character():
    state = new_patient_state()
    state["turn_input"] = "My knee hurts a lot."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                chief_complaint="Knee pain",
                new_symptoms=[
                    SymptomExtraction(name="pain", location="knee", character="joint", severity="severe")
                ],
                still_missing=["age"],
                ready_for_triage=False,
                next_question="What is your age?",
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert result["symptoms"] == [
        {"name": "pain", "location": "knee", "character": "joint", "severity": "severe"}
    ]


def test_patient_confirming_the_recap_transitions_to_triage():
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday", "severity": "mild"}]
    state["demographics"] = {"age": "34"}
    state["awaiting_confirmation"] = True
    state["awaiting_patient"] = True
    state["turn_input"] = "Yes, that's all correct."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                still_missing=[],
                ready_for_triage=True,
                patient_confirmed=True,
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert result["stage"] == "triage"
    assert result["awaiting_patient"] is False
    assert result["awaiting_confirmation"] is False


def test_patient_correcting_the_recap_merges_and_asks_again():
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday", "severity": "mild"}]
    state["demographics"] = {"age": "29"}
    state["awaiting_confirmation"] = True
    state["awaiting_patient"] = True
    state["turn_input"] = "Actually I'm 30, not 29."
    fake_llm = FakeStructuredLLM(
        [
            IntakeExtraction(
                demographics_update={"age": "30"},
                still_missing=[],
                ready_for_triage=True,
                patient_confirmed=False,
            )
        ]
    )

    result = intake_node(state, llm=fake_llm)

    assert "stage" not in result  # still not handed off
    assert result["awaiting_confirmation"] is True
    assert result["awaiting_patient"] is True
    assert result["demographics"]["age"] == "30"
    assert "age: 30" in result["assistant_message"]
    assert result["assistant_message"].startswith("Got it, updated.")
