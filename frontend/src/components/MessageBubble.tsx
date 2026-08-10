import type { ChatMessage } from "../types";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isPatient = message.role === "patient";
  return (
    <div className={`message-row ${isPatient ? "patient" : "assistant"}`}>
      <div className="message-bubble">{message.content}</div>
    </div>
  );
}
