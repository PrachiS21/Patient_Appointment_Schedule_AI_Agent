import pytest
from pydantic import ValidationError

from patient_intake_agent.nodes.emergency_guard import URGENT_CARE_MESSAGE
from patient_intake_agent.nodes.summary import PatientSummary, summary_node
from patient_intake_agent.state import new_patient_state


def test_normal_booked_appointment_produces_a_schema_valid_summary():
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["symptoms"] = [{"name": "fever", "onset": "yesterday", "severity": "mild"}]
    state["demographics"] = {"age": "34", "sex": "female"}
    state["risk_level"] = "LOW"
    state["suggested_specialty"] = "Primary Care"
    state["selected_appointment"] = {
        "doctor": "Dr. Alice Nguyen",
        "specialty": "Primary Care",
        "date": "2026-08-10",
        "time": "09:00",
        "appointment_id": "appt_001",
        "status": "booked",
    }

    result = summary_node(state)

    summary = result["final_summary"]
    PatientSummary.model_validate(summary)  # re-validate the exact output shape
    assert summary["age"] == 34
    assert summary["sex"] == "female"
    assert summary["requires_human"] is False
    assert summary["risk_level"] == "LOW"
    assert summary["specialty"] == "Primary Care"
    assert summary["scheduled_appointment"] == {
        "doctor": "Dr. Alice Nguyen",
        "time": "2026-08-10 09:00",
        "status": "booked",
    }
    assert "Dr. Alice Nguyen" in summary["recommendation"]
    assert result["stage"] == "done"


def test_age_parses_from_messy_free_text():
    state = new_patient_state()
    state["demographics"] = {"age": "34 years old"}

    result = summary_node(state)

    assert result["final_summary"]["age"] == 34


def test_sex_falls_back_to_gender_key():
    state = new_patient_state()
    state["demographics"] = {"gender": "male"}

    result = summary_node(state)

    assert result["final_summary"]["sex"] == "male"


def test_age_and_sex_are_null_when_never_collected():
    """The emergency short-circuit path never runs Intake at all, so these
    are legitimately unknown — must not raise, must not fake a value."""
    state = new_patient_state()
    state["emergency_flag"] = True
    state["risk_level"] = "HIGH"

    result = summary_node(state)

    assert result["final_summary"]["age"] is None
    assert result["final_summary"]["sex"] is None


def test_emergency_summary_requires_human_and_uses_the_fixed_urgent_care_message():
    state = new_patient_state()
    state["turn_input"] = "severe chest pain"
    state["emergency_flag"] = True
    state["risk_level"] = "HIGH"

    result = summary_node(state)

    summary = result["final_summary"]
    assert summary["requires_human"] is True
    assert summary["recommendation"] == URGENT_CARE_MESSAGE
    assert summary["risk_level"] == "HIGH"


def test_no_availability_summary_requires_human():
    state = new_patient_state()
    state["chief_complaint"] = "Fever since yesterday"
    state["suggested_specialty"] = "Primary Care"
    state["selected_appointment"] = {"status": "no_availability"}

    result = summary_node(state)

    summary = result["final_summary"]
    assert summary["requires_human"] is True
    assert summary["scheduled_appointment"]["status"] == "no_availability"
    assert "staff" in summary["recommendation"].lower()


def test_missing_information_passes_through_unchanged():
    state = new_patient_state()
    state["missing_information"] = ["severity", "duration"]

    result = summary_node(state)

    assert result["final_summary"]["missing_information"] == ["severity", "duration"]


def test_schema_rejects_an_invalid_risk_level():
    with pytest.raises(ValidationError):
        PatientSummary.model_validate(
            {
                "chief_complaint": None,
                "symptoms": [],
                "summary": "x",
                "risk_level": "CRITICAL",  # not one of LOW | MEDIUM | HIGH
                "recommendation": "x",
                "requires_human": False,
                "missing_information": [],
            }
        )
