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


def _emergency_only_app(tmp_path) -> TestClient:
    # tmp_path: every test gets its own isolated SQLite file, never the real
    # backend/chats.db — same reasoning as the fake LLM/sandbox above, kept
    # AWS-and-side-effect-free.
    graph = build_graph(llm=_PoisonLLM(), sandbox=build_demo_sandbox())
    app = create_app(graph_factory=lambda: graph, db_path=tmp_path / "chats.db")
    return TestClient(app)


def test_health_endpoint(tmp_path):
    client = _emergency_only_app(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summary_endpoint_404s_for_unknown_session(tmp_path):
    client = _emergency_only_app(tmp_path)
    response = client.get("/api/sessions/does-not-exist/summary")
    assert response.status_code == 404


def test_websocket_emergency_flow_sends_typing_then_final_summary(tmp_path):
    client = _emergency_only_app(tmp_path)

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


def test_summary_endpoint_409s_while_conversation_still_in_progress(tmp_path):
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
    client = TestClient(create_app(graph_factory=lambda: graph, db_path=tmp_path / "chats.db"))

    with client.websocket_connect("/ws/chat/session-b") as ws:
        ws.send_json({"message": "I've had a mild headache, nothing urgent."})
        ws.receive_json()  # typing
        reply = ws.receive_json()
        assert reply["stage"] == "intake"
        assert reply["awaiting_patient"] is True

    response = client.get("/api/sessions/session-b/summary")
    assert response.status_code == 409


def test_two_sessions_do_not_share_state(tmp_path):
    client = _emergency_only_app(tmp_path)

    with client.websocket_connect("/ws/chat/session-x") as ws:
        ws.send_json({"message": "severe chest pain, can't breathe"})
        ws.receive_json()
        ws.receive_json()

    # A second, unrelated session must not see session-x's finished summary.
    response = client.get("/api/sessions/session-y/summary")
    assert response.status_code == 404


def test_completed_chat_is_persisted_and_listed_in_chats_endpoint(tmp_path):
    client = _emergency_only_app(tmp_path)

    with client.websocket_connect("/ws/chat/session-persist") as ws:
        ws.send_json({"message": "severe chest pain, can't breathe"})
        ws.receive_json()  # typing
        ws.receive_json()  # final message

    listing = client.get("/api/chats").json()
    assert len(listing) == 1
    assert listing[0]["session_id"] == "session-persist"
    assert listing[0]["risk_level"] == "HIGH"
    assert listing[0]["requires_human"] is True

    detail = client.get("/api/chats/session-persist")
    assert detail.status_code == 200
    assert detail.json()["requires_human"] is True

    # Distinct from the in-memory /api/sessions endpoint — this one is
    # backed by SQLite and would survive a server restart.
    assert client.get("/api/chats/does-not-exist").status_code == 404


def test_unfinished_chat_is_not_persisted(tmp_path):
    llm = _SchemaDispatchLLM(
        {
            EmergencyClassification: EmergencyClassification(is_emergency=False, reasoning="Not urgent."),
            IntakeExtraction: IntakeExtraction(
                chief_complaint="Mild headache",
                still_missing=["duration"],
                ready_for_triage=False,
                next_question="How long has this been going on?",
            ),
        }
    )
    graph = build_graph(llm=llm, sandbox=build_demo_sandbox())
    client = TestClient(create_app(graph_factory=lambda: graph, db_path=tmp_path / "chats.db"))

    with client.websocket_connect("/ws/chat/session-unfinished") as ws:
        ws.send_json({"message": "I've had a mild headache."})
        ws.receive_json()
        ws.receive_json()

    assert client.get("/api/chats").json() == []
    assert client.get("/api/chats/session-unfinished").status_code == 404


class _ExplodingGraph:
    """Stands in for a compiled graph whose LLM call fails — provider rate
    limit, transient server error, timeout. Doesn't matter which; the
    handler is supposed to treat any of them the same way."""

    def invoke(self, state):
        raise RuntimeError("503 UNAVAILABLE: simulated transient provider failure")


def test_llm_failure_sends_an_error_event_instead_of_killing_the_connection(tmp_path):
    app = create_app(graph_factory=lambda: _ExplodingGraph(), db_path=tmp_path / "chats.db")
    client = TestClient(app)

    with client.websocket_connect("/ws/chat/session-error") as ws:
        ws.send_json({"message": "I have a headache."})

        typing = ws.receive_json()
        assert typing == {"type": "typing"}

        error = ws.receive_json()
        assert error["type"] == "error"
        assert isinstance(error["message"], str) and error["message"]

        # Connection must still be alive and usable — not torn down by the
        # failure. Send a second message and confirm we get typing + error
        # again, rather than the socket having silently closed.
        ws.send_json({"message": "Hello again?"})
        assert ws.receive_json() == {"type": "typing"}
        second_error = ws.receive_json()
        assert second_error["type"] == "error"

    # Nothing about a failed turn should have been persisted anywhere.
    assert client.get("/api/chats").json() == []
    assert client.get("/api/sessions/session-error/summary").status_code in (404, 409)
