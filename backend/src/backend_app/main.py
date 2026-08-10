"""FastAPI wrapper around patient-intake-agent.

One WebSocket endpoint drives the conversation (one graph.invoke() per
incoming patient message, matching the graph's own turn-taking model — see
agent/src/patient_intake_agent/graph.py's module docstring), plus a REST
endpoint exposing the finished structured summary once a session reaches the
"done" stage.

`create_app(graph_factory=...)` exists so tests can build an app wired to a
fake LLM / isolated sandbox without ever touching AWS — `graph_factory` is
only called lazily, on the first WebSocket message, so importing this module
(or even constructing the real `app` object) never requires AWS credentials.
Only actually handling a chat message does.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from patient_intake_agent import build_graph, new_patient_state

from .session_store import SessionStore


def create_app(graph_factory=None) -> FastAPI:
    app = FastAPI(title="Patient Intake Assistant")

    # MVP only: wide-open CORS so the Vite dev server (a different origin)
    # can talk to this API with no extra config. Not appropriate as-is for a
    # real deployment — see root README's Known Limitations.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    factory = graph_factory or build_graph
    graph_holder: dict = {"graph": None}
    sessions = SessionStore()

    def get_graph():
        if graph_holder["graph"] is None:
            graph_holder["graph"] = factory()
        return graph_holder["graph"]

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/sessions/{session_id}/summary")
    def get_summary(session_id: str):
        state = sessions.peek(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        if state.get("final_summary") is None:
            raise HTTPException(status_code=409, detail="Conversation not finished yet")
        return state["final_summary"]

    @app.websocket("/ws/chat/{session_id}")
    async def chat(websocket: WebSocket, session_id: str):
        await websocket.accept()
        graph = get_graph()
        try:
            while True:
                payload = await websocket.receive_json()
                message = (payload.get("message") or "").strip()
                if not message:
                    continue

                state = sessions.get_or_create(session_id, new_patient_state)
                state["turn_input"] = message

                # Graph calls make real (slow) LLM requests — tell the
                # frontend to show a typing indicator before we block on it.
                await websocket.send_json({"type": "typing"})

                state = await run_in_threadpool(graph.invoke, state)
                sessions.set(session_id, state)

                response = {
                    "type": "message",
                    "content": state.get("assistant_message"),
                    "stage": state["stage"],
                    "awaiting_patient": state["awaiting_patient"],
                    "risk_level": state["risk_level"],
                    "emergency_flag": state["emergency_flag"],
                }
                if state["stage"] == "done":
                    response["final_summary"] = state["final_summary"]

                await websocket.send_json(response)
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
