# patient-intake-agent

The five-node LangGraph state machine (`intake → emergency_guard → triage → scheduling → summary`)
described in the root [README](../README.md). Depends on `patient-agent-bench`
(the sibling `../../PatientAgentBench` checkout) for the healthcare sandbox
and appointment tools.

See the root README for setup, and `docs/exploration/patientagentbench-notes.md`
for how this design maps onto PatientAgentBench's own conversation/prompt/state
patterns.
