import { ChatWindow } from "./components/ChatWindow";
import { SummaryPanel } from "./components/SummaryPanel";
import { useChatSession } from "./useChatSession";
import "./App.css";

export default function App() {
  const { messages, isTyping, isConnected, isDone, emergencyFlag, summary, sendMessage } =
    useChatSession();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Patient Intake Assistant</h1>
        <p className="disclaimer">
          For informational intake only — not a diagnosis. In an emergency, call 911.
        </p>
      </header>

      {emergencyFlag && (
        <div className="emergency-banner" role="alert">
          These symptoms warrant urgent evaluation. Please call 911 or go to the nearest
          emergency room.
        </div>
      )}

      <main className="app-main">
        <ChatWindow
          messages={messages}
          isTyping={isTyping}
          isConnected={isConnected}
          isDone={isDone}
          onSend={sendMessage}
        />
        {summary && <SummaryPanel summary={summary} />}
      </main>
    </div>
  );
}
