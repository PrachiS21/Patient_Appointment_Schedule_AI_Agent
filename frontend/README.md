# frontend

React + TypeScript (Vite). Talks to `patient-intake-backend` over one
WebSocket per browser tab (`useChatSession.ts` generates a `crypto.randomUUID()`
session id on mount and opens `ws://<backend>/ws/chat/{id}`); the backend's
inline `final_summary` payload on that same socket is what populates the
summary panel. The REST endpoint (`GET /api/sessions/{id}/summary`) isn't
called by this UI — the WS already delivers the summary the moment it's
ready — it exists for other clients / a page-refresh recovery flow that
isn't implemented here (see root README's Known Limitations).

```bash
npm install
cp .env.example .env.local   # only needed if the backend isn't on localhost:8000
npm run dev
```

- `ChatWindow` — message history (auto-scrolls), typing indicator, input box.
- `SummaryPanel` — appears once the backend sends `final_summary`; renders
  the structured fields plus a collapsible raw-JSON view.
- `useChatSession.ts` — the entire WebSocket lifecycle: connect, send,
  receive, and the small state machine (`isTyping`, `isDone`, `emergencyFlag`)
  the two components above read from.

**Not yet verified in a real browser** — this environment doesn't have a
browser automation tool available. What *has* been verified: `npm run build`
compiles cleanly, the dev server serves and transpiles every module with no
errors, and the WebSocket message shapes this code expects
(`type: "typing" | "message"`, `final_summary`, etc.) are exactly what
`backend/tests/test_chat_flow.py` proves the backend actually sends. Please
click through it once for real before relying on it.
