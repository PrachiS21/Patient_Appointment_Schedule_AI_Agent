import type { ChatMessage } from "../types";

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "error") {
    return (
      <div className="message-row error" role="alert">
        <div className="message-bubble error-bubble">⚠️ {message.content}</div>
      </div>
    );
  }

  const isPatient = message.role === "patient";
  return (
    <div className={`message-row ${isPatient ? "patient" : "assistant"}`}>
      <div className="message-bubble">{message.content}</div>
    </div>
  );
}
