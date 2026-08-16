"""FastAPI wrapper around patient-intake-agent.

One WebSocket endpoint drives the conversation (one graph.invoke() per
incoming patient message, matching the graph's own turn-taking model — see
agent/src/patient_intake_agent/graph.py's module docstring), plus REST
endpoints exposing the finished structured summary once a session reaches
the "done" stage.

`create_app(graph_factory=..., db_path=...)` exists so tests can build an
app wired to a fake LLM / isolated sandbox / temp SQLite file without ever
touching AWS or the real chats.db — `graph_factory` is only called lazily,
on the first WebSocket message, so importing this module (or even
constructing the real `app` object) never requires AWS credentials. Only
actually handling a chat message does.

A `graph.invoke()` failure (provider rate limit, transient server error,
timeout — none of which are under our control) is caught per-turn and
reported to the client as a `{"type": "error"}` event rather than left to
kill the WebSocket connection silently — see the `chat()` handler.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from patient_intake_agent import build_graph, new_patient_state

from .chat_store import get_chat, init_db, list_chats, save_chat
from .session_store import SessionStore

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "chats.db"

USER_FACING_ERROR_MESSAGE = (
    "Something went wrong processing that message. Please try again — "
    "nothing you've said so far has been lost."
)


def create_app(graph_factory=None, db_path: Path | None = None) -> FastAPI:
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

    db_path = db_path or DEFAULT_DB_PATH
    init_db(db_path)

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

    @app.get("/api/chats")
    def get_chat_list():
        """Durable record of *finished* chats (SQLite), independent of
        whatever's still live in the in-memory SessionStore — survives a
        server restart, unlike /api/sessions/{id}/summary above."""
        return list_chats(db_path)

    @app.get("/api/chats/{session_id}")
    def get_saved_chat(session_id: str):
        summary = get_chat(db_path, session_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Unknown chat")
        return summary

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

                try:
                    state = await run_in_threadpool(graph.invoke, state)
                except Exception:
                    # LLM calls fail for reasons with nothing to do with our
                    # code — provider rate limits (429), transient server
                    # overload (503), timeouts. Without this, any of those
                    # kills the WS connection with zero message ever sent to
                    # the frontend, which looks indistinguishable from "the
                    # app is just stuck". `sessions.set()` below never
                    # runs on this path, so the session's last *successful*
                    # state is untouched — the user can just retry.
                    logger.exception(
                        "graph.invoke failed for session %s", session_id
                    )
                    await websocket.send_json(
                        {"type": "error", "message": USER_FACING_ERROR_MESSAGE}
                    )
                    continue

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
                    save_chat(db_path, session_id, state["final_summary"])

                await websocket.send_json(response)
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
