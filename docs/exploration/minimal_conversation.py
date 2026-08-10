"""Minimal standalone PatientAgentBench conversation loop.

Build phase 3 deliverable: strip PatientAgentBench down to the smallest
script that still holds a real console conversation using its *real*
components — no benchmark runner, no simulated-patient (user) agent, one
hardcoded patient.

What's reused verbatim from `patient_agent_bench` (installed as a real
dependency of this workspace, see agent/pyproject.toml):

  - `HealthcareSandbox`               (patient_agent_bench.sandbox.sandbox)
  - `PatientProfile` + nested dataclasses (patient_agent_bench.patient)
  - `create_tool_registry`            (patient_agent_bench.tools.registry)
  - `DefaultAssistantAgent`           (patient_agent_bench.assistant_agent.default_agent)
  - `ModelConfig` / `create_chat_model` (patient_agent_bench.config)

What's deliberately NOT reused:

  - The LLM-based sandbox generator (`generate_sandbox_data` /
    `initialize_sandbox`). It makes its own Bedrock call to invent offices
    and doctors. For a *minimal* script we don't want a second hidden LLM
    dependency, so offices/doctors are hardcoded below and slots are built
    with the sandbox module's own `_generate_appointment_slots` helper (the
    same code the real sandbox uses, just fed fixed doctors instead of
    LLM-generated ones).
  - `ConversationRunner` / the user (patient) simulator agent — this script
    is a human typing at a console, not a benchmark.

Two ways to run this file:

  --verify-tools-only   Exercises the real sandbox + tools with zero LLM
                         calls (list doctors, search availability, book,
                         cancel). Works right now, no AWS credentials needed.
                         This is what was actually run to validate this
                         script during development (no Bedrock access was
                         configured yet at write time).

  (no flag)              Starts a real console conversation against Bedrock,
                         via DefaultAssistantAgent's LangGraph ReAct agent
                         and the real appointment/prescription/profile/
                         telehealth tools. Requires AWS credentials with
                         Bedrock model access — see repo root README.

Run from the `Mysource/` directory so the workspace's synced venv (which has
patient_agent_bench installed) is used:

    uv run --package patient-intake-agent python docs/exploration/minimal_conversation.py --verify-tools-only
    uv run --package patient-intake-agent python docs/exploration/minimal_conversation.py
"""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage

from patient_agent_bench.assistant_agent.default_agent import DefaultAssistantAgent
from patient_agent_bench.config import ModelConfig
from patient_agent_bench.patient import (
    AccountInfo,
    Address,
    Medication,
    MedicationStatus,
    PatientProfile,
    PersonalInfo,
)
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.sandbox.generator import _generate_appointment_slots
from patient_agent_bench.sandbox.sandbox import Doctor, HealthcareSandbox, OfficeLocation
from patient_agent_bench.tools.registry import create_tool_registry

CURRENT_DATETIME = "Sunday, August 09, 2026 at 10:00 AM"
CURRENT_DATE = "2026-08-09"


def build_hardcoded_patient_profile() -> PatientProfile:
    """One fixed patient, in place of patient/generator.py's LLM enrichment."""
    return PatientProfile(
        account_info=AccountInfo(age_in_years="34", timezone="America/New_York"),
        personal_info=PersonalInfo(
            first_name="Jordan",
            last_name="Ellis",
            dob="1992-03-14",
            sex="female",
            gender="female",
            pronouns="she/her",
            email="jordan.ellis@example.com",
        ),
        address=Address(address1="12 Elm St", city="Boston", state="MA", zip="02118"),
        medications=[
            Medication(
                id="med_001",
                name="Levothyroxine",
                dosage="50mcg",
                frequency="once daily",
                status=MedicationStatus.ACTIVE,
                prescribed_date="2025-01-10",
                pharmacy="CVS Boston Downtown",
                refills_remaining=3,
                reason="Hypothyroidism",
            )
        ],
    )


def build_hardcoded_sandbox() -> HealthcareSandbox:
    """A fixed sandbox, in place of sandbox/generator.py's LLM-generated inventory.

    Offices/doctors are hardcoded; slots reuse the real sandbox module's own
    `_generate_appointment_slots` (weekday-only, 3-6 slots/doctor/day, 40-70%
    available, mixed in-person/telehealth) so the appointment tools see the
    exact data shape they expect.
    """
    office = OfficeLocation(
        id="office_001",
        name="Downtown Primary Care",
        address="100 Main St",
        city="Boston",
        state="MA",
        zip_code="02118",
        phone="555-100-2000",
        hours="Mon-Fri 8AM-6PM",
    )

    doctors = {
        "doc_001": Doctor(
            id="doc_001", name="Dr. Alice Nguyen", specialty="Primary Care",
            credentials="MD", office_id=office.id,
        ),
        "doc_002": Doctor(
            id="doc_002", name="Dr. Marcus Ito", specialty="Cardiology",
            credentials="MD", office_id=office.id,
        ),
    }

    slots_list = _generate_appointment_slots(list(doctors.values()), current_date=CURRENT_DATE)

    sandbox = HealthcareSandbox()
    sandbox.offices = {office.id: office}
    sandbox.doctors = doctors
    sandbox.slots = {s.id: s for s in slots_list}
    sandbox.patient_profile = build_hardcoded_patient_profile()
    sandbox.pcp_id = "doc_001"
    sandbox.current_date = CURRENT_DATE
    sandbox.patient_profile.update_pcp("doc_001", "Dr. Alice Nguyen")
    sandbox._initialized = True
    return sandbox


def verify_tools_only() -> None:
    """Exercise the real sandbox tools directly — no LLM involved."""
    sandbox = build_hardcoded_sandbox()
    registry = create_tool_registry(sandbox)

    print(f"Registered {len(registry)} real PatientAgentBench tools:")
    print(" ", ", ".join(sorted(registry.list_tools())))
    print()

    list_doctors = registry.get("list_doctors")
    print("--- list_doctors(specialty='Cardiology') ---")
    print(list_doctors.invoke({"specialty": "Cardiology"}))

    get_avail = registry.get("get_available_appointments")
    print("--- get_available_appointments (next 3 days, Primary Care) ---")
    print(
        get_avail.invoke(
            {
                "start_date": CURRENT_DATE,
                "end_date": "2026-08-12",
                "specialty": "Primary Care",
            }
        )
    )

    schedule = registry.get("schedule_appointment")
    print("--- schedule_appointment(Primary Care, morning) ---")
    print(
        schedule.invoke(
            {
                "appointment_type": "in_person",
                "preferred_time": "morning",
                "provider_name": "PCP",
                "reason": "Fever since yesterday",
            }
        )
    )

    list_appts = registry.get("list_appointments")
    print("--- list_appointments ---")
    print(list_appts.invoke({}))


def run_console_conversation() -> None:
    """Real console chat against Bedrock, using the real ReAct harness + tools."""
    sandbox = build_hardcoded_sandbox()
    tool_registry = create_tool_registry(sandbox)

    model_config = ModelConfig(
        model_id="global.anthropic.claude-sonnet-5",
        max_tokens=4096,
        provider="bedrock",
    )

    assistant = DefaultAssistantAgent(
        model_config=model_config,
        current_datetime=CURRENT_DATETIME,
        tools=tool_registry.get_tools(),
    )

    conversation = Conversation()
    user_profile_xml = sandbox.patient_profile.to_xml()

    print("Console patient chat (Ctrl+C to quit). Patient: Jordan Ellis.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue

        conversation.add_message(HumanMessage(content=user_text))
        result = assistant.invoke(messages=conversation.messages, user_profile=user_profile_xml)
        pre_len = len(conversation)
        conversation.extend_from_agent_result(result["messages"])
        conversation.set_all_new_assistant_responses(from_index=pre_len)
        reply = conversation.get_all_new_assistant_text(pre_len)
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    if "--verify-tools-only" in sys.argv:
        verify_tools_only()
    else:
        run_console_conversation()
