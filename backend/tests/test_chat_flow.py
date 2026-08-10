"""Backend API tests: WebSocket protocol shape, REST summary, session isolation.

Graph correctness (node logic, routing) is already covered exhaustively in
agent/tests/ — these tests exercise only what's new at this layer: the
WebSocket message shapes, the typing indicator, the REST endpoint's
before/after-done behavior, and that two sessions never share state. The
emergency flow is used as the driving example specifically because it's the
one path that needs zero LLM calls (keyword tier only), keeping these tests
fast and AWS-free like everything else in this repo so far.
"""

from fastapi.testclient import TestClient

from patient_intake_agent.graph import build_graph
from patient_intake_agent.nodes.emergency_guard import EmergencyClassification
from patient_intake_agent.nodes.intake import IntakeExtraction
from patient_intake_agent.sandbox_setup import build_demo_sandbox

from backend_app.main import create_app


class _PoisonLLM:
    def with_structured_output(self, schema):
        raise AssertionError("LLM should not have been called for an obvious keyword emergency")


class _BoundResponder:
    def __init__(self, schema, responses_by_schema):
        self._schema = schema
        self._responses_by_schema = responses_by_schema

    def invoke(self, prompt):
        try:
            return self._responses_by_schema[self._schema]
        except KeyError:
            raise AssertionError(f"No canned response for schema: {self._schema}") from None


class _SchemaDispatchLLM:
    """Dispatches by requested schema type rather than call order — good
    enough for a single in-progress turn where call order doesn't matter,
    without needing to hand-trace the exact multi-node call sequence the
    ordered FakeStructuredLLM in agent/tests/fakes.py requires."""

    def __init__(self, responses_by_schema):
        self._responses_by_schema = responses_by_schema

    def with_structured_output(self, schema):
        return _BoundResponder(schema, self._responses_by_schema)


def _emergency_only_app() -> TestClient:
    graph = build_graph(llm=_PoisonLLM(), sandbox=build_demo_sandbox())
    app = create_app(graph_factory=lambda: graph)
    return TestClient(app)


def test_health_endpoint():
    client = _emergency_only_app()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_endpoint_404s_for_unknown_session():
    client = _emergency_only_app()
    response = client.get("/api/sessions/does-not-exist/summary")
    assert response.status_code == 404


def test_websocket_emergency_flow_sends_typing_then_final_summary():
    client = _emergency_only_app()

    with client.websocket_connect("/ws/chat/session-a") as ws:
        ws.send_json({"message": "I have severe chest pain and can't breathe."})

        typing = ws.receive_json()
        assert typing == {"type": "typing"}

        reply = ws.receive_json()
        assert reply["type"] == "message"
        assert reply["stage"] == "done"
        assert reply["emergency_flag"] is True
        assert reply["risk_level"] == "HIGH"
        assert "final_summary" in reply
        assert reply["final_summary"]["requires_human"] is True

    # REST endpoint now reflects the same finished session.
    response = client.get("/api/sessions/session-a/summary")
    assert response.status_code == 200
    assert response.json()["requires_human"] is True


def test_summary_endpoint_409s_while_conversation_still_in_progress():
    llm = _SchemaDispatchLLM(
        {
            EmergencyClassification: EmergencyClassification(is_emergency=False, reasoning="Not urgent."),
            IntakeExtraction: IntakeExtraction(
                chief_complaint="Mild headache",
                still_missing=["duration", "severity"],
                ready_for_triage=False,
                next_question="How long have you had the headache, and how severe is it?",
            ),
        }
    )
    graph = build_graph(llm=llm, sandbox=build_demo_sandbox())
    client = TestClient(create_app(graph_factory=lambda: graph))

    with client.websocket_connect("/ws/chat/session-b") as ws:
        ws.send_json({"message": "I've had a mild headache, nothing urgent."})
        ws.receive_json()  # typing
        reply = ws.receive_json()
        assert reply["stage"] == "intake"
        assert reply["awaiting_patient"] is True

    response = client.get("/api/sessions/session-b/summary")
    assert response.status_code == 409


def test_two_sessions_do_not_share_state():
    client = _emergency_only_app()

    with client.websocket_connect("/ws/chat/session-x") as ws:
        ws.send_json({"message": "severe chest pain, can't breathe"})
        ws.receive_json()
        ws.receive_json()

    # A second, unrelated session must not see session-x's finished summary.
    response = client.get("/api/sessions/session-y/summary")
    assert response.status_code == 404
