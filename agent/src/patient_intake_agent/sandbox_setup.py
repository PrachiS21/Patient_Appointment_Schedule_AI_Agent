"""Builds the demo HealthcareSandbox this MVP's Triage and Scheduling nodes
run against.

Reuses PatientAgentBench's real `HealthcareSandbox`, `Doctor`,
`OfficeLocation` classes and its own `_generate_appointment_slots` helper
verbatim (patient_agent_bench.sandbox). Offices and doctors are hardcoded
here rather than LLM-generated the way PatientAgentBench's own
sandbox/generator.py does it — see
docs/exploration/patientagentbench-notes.md for why (one hardcoded patient,
one fewer hidden LLM dependency, a deterministic environment to test and
demo against). One doctor per specialty in `appointment_tools.VALID_SPECIALTIES`
so Triage always has somewhere to route to.
"""

from __future__ import annotations

from functools import lru_cache

from patient_agent_bench.sandbox.generator import _generate_appointment_slots
from patient_agent_bench.sandbox.sandbox import Doctor, HealthcareSandbox, OfficeLocation

CURRENT_DATE = "2026-08-09"

_OFFICES = [
    OfficeLocation(
        id="office_001", name="Downtown Primary Care", address="100 Main St",
        city="Boston", state="MA", zip_code="02118", phone="555-100-2000",
        hours="Mon-Fri 8AM-6PM",
    ),
    OfficeLocation(
        id="office_002", name="Riverside Specialty Center", address="55 River Rd",
        city="Boston", state="MA", zip_code="02128", phone="555-200-3000",
        hours="Mon-Fri 8AM-5PM",
    ),
]

_DOCTORS = [
    Doctor(id="doc_001", name="Dr. Alice Nguyen", specialty="Primary Care", credentials="MD", office_id="office_001"),
    Doctor(id="doc_002", name="Dr. Marcus Ito", specialty="Cardiology", credentials="MD", office_id="office_002"),
    Doctor(id="doc_003", name="Dr. Priya Shah", specialty="Dermatology", credentials="MD", office_id="office_002"),
    Doctor(id="doc_004", name="Dr. Sam O'Brien", specialty="Orthopedics", credentials="DO", office_id="office_002"),
    Doctor(id="doc_005", name="Dr. Lena Fischer", specialty="Endocrinology", credentials="MD", office_id="office_002"),
    Doctor(id="doc_006", name="Dr. Tomas Reyes", specialty="Psychiatry", credentials="MD", office_id="office_002"),
    Doctor(id="doc_007", name="Dr. Ines Torres", specialty="OB/GYN", credentials="MD", office_id="office_002"),
]


def build_demo_sandbox() -> HealthcareSandbox:
    """Fresh sandbox instance — use this in tests that book/cancel appointments
    so each test gets its own slot inventory instead of sharing mutable state.
    """
    sandbox = HealthcareSandbox()
    sandbox.offices = {o.id: o for o in _OFFICES}
    sandbox.doctors = {d.id: d for d in _DOCTORS}
    slots = _generate_appointment_slots(_DOCTORS, current_date=CURRENT_DATE)
    sandbox.slots = {s.id: s for s in slots}
    sandbox.pcp_id = "doc_001"
    sandbox.current_date = CURRENT_DATE
    sandbox._initialized = True
    return sandbox


@lru_cache(maxsize=1)
def get_default_sandbox() -> HealthcareSandbox:
    """Process-wide singleton sandbox for the real graph (graph.py::build_graph).

    PatientAgentBench scopes a sandbox to a single conversation; this MVP's
    backend simplifies that to one shared sandbox per process, since we're
    running a single-session demo, not a multi-tenant service. Documented as
    a known limitation in the root README.
    """
    return build_demo_sandbox()
