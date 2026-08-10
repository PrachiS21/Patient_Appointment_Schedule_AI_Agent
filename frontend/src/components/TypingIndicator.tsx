export function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="message-bubble typing-indicator" aria-label="Assistant is typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}
