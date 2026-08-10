from fakes import FakeStructuredLLM, PoisonLLM
from patient_agent_bench.sandbox.sandbox import Doctor, HealthcareSandbox, OfficeLocation
from patient_intake_agent.nodes.triage import TriageClassification, triage_node
from patient_intake_agent.state import new_patient_state


def _sandbox_with(*doctors: Doctor) -> HealthcareSandbox:
    """A minimal real HealthcareSandbox — no slots needed for triage tests."""
    office = OfficeLocation(
        id="office_001", name="Test Clinic", address="1 Test St",
        city="Boston", state="MA", zip_code="02118", phone="555-000-0000",
        hours="Mon-Fri 8AM-5PM",
    )
    sandbox = HealthcareSandbox()
    sandbox.offices = {office.id: office}
    sandbox.doctors = {d.id: d for d in doctors}
    return sandbox


def test_rules_table_matches_obvious_case_without_calling_llm():
    state = new_patient_state()
    state["chief_complaint"] = "chest pain"
    state["symptoms"] = [{"name": "shortness of breath"}]
    sandbox = _sandbox_with(
        Doctor(id="doc_001", name="Dr. Marcus Ito", specialty="Cardiology", credentials="MD", office_id="office_001"),
    )

    result = triage_node(state, llm=PoisonLLM(), sandbox=sandbox)

    assert result["suggested_specialty"] == "Cardiology"
    assert result["stage"] == "scheduling"
    assert result["candidate_doctors"] == [
        {
            "id": "doc_001",
            "name": "Dr. Marcus Ito",
            "specialty": "Cardiology",
            "credentials": "MD",
            "office": "Test Clinic",
        }
    ]


def test_llm_classifies_ambiguous_symptoms_not_covered_by_rules_table():
    state = new_patient_state()
    state["chief_complaint"] = "constant fatigue and weight gain"
    sandbox = _sandbox_with(
        Doctor(id="doc_005", name="Dr. Lena Fischer", specialty="Endocrinology", credentials="MD", office_id="office_001"),
    )
    fake_llm = FakeStructuredLLM(
        [TriageClassification(specialty="Endocrinology", confidence=0.8, reasoning="Fatigue + weight gain.")]
    )

    result = triage_node(state, llm=fake_llm, sandbox=sandbox)

    assert result["suggested_specialty"] == "Endocrinology"
    assert result["candidate_doctors"][0]["name"] == "Dr. Lena Fischer"


def test_llm_hallucinating_an_invalid_specialty_is_clamped_to_primary_care():
    state = new_patient_state()
    state["chief_complaint"] = "vague, hard to categorize symptoms"
    sandbox = _sandbox_with(
        Doctor(id="doc_001", name="Dr. Alice Nguyen", specialty="Primary Care", credentials="MD", office_id="office_001"),
    )
    fake_llm = FakeStructuredLLM(
        [TriageClassification(specialty="Neurology", confidence=0.4, reasoning="Not a real option in our vocabulary.")]
    )

    result = triage_node(state, llm=fake_llm, sandbox=sandbox)

    assert result["suggested_specialty"] == "Primary Care"


def test_no_candidate_doctors_for_specialty_returns_empty_list_not_an_error():
    state = new_patient_state()
    state["chief_complaint"] = "chest pain"
    state["symptoms"] = [{"name": "shortness of breath"}]
    sandbox = _sandbox_with()  # no doctors registered at all

    result = triage_node(state, llm=PoisonLLM(), sandbox=sandbox)

    assert result["suggested_specialty"] == "Cardiology"
    assert result["candidate_doctors"] == []
