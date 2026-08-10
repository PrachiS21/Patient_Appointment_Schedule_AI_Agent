import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

interface ChatWindowProps {
  messages: ChatMessage[];
  isTyping: boolean;
  isConnected: boolean;
  isDone: boolean;
  onSend: (text: string) => void;
}

export function ChatWindow({ messages, isTyping, isConnected, isDone, onSend }: ChatWindowProps) {
  const [draft, setDraft] = useState("");
  const historyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    historyRef.current?.scrollTo({ top: historyRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isTyping]);

  const disabled = !isConnected || isDone;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (disabled) return;
    onSend(draft);
    setDraft("");
  }

  return (
    <section className="chat-window" aria-label="Chat with the patient intake assistant">
      <div className="chat-history" ref={historyRef}>
        {messages.length === 0 && (
          <p className="chat-empty-state">
            Describe what's going on and I'll ask a few follow-up questions.
          </p>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {isTyping && <TypingIndicator />}
      </div>

      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={isDone ? "Conversation complete" : "Type your message..."}
          disabled={disabled}
          aria-label="Message"
        />
        <button type="submit" disabled={disabled || !draft.trim()}>
          Send
        </button>
      </form>
      {!isConnected && <p className="connection-warning">Connecting to the assistant...</p>}
    </section>
  );
}
