import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, PatientSummary, RiskLevel, ServerEvent } from "./types";

const BACKEND_HTTP_BASE =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? "http://localhost:8000";
const BACKEND_WS_BASE = BACKEND_HTTP_BASE.replace(/^http/, "ws");

function newId(): string {
  return crypto.randomUUID();
}

export interface ChatSession {
  messages: ChatMessage[];
  isTyping: boolean;
  isConnected: boolean;
  isDone: boolean;
  emergencyFlag: boolean;
  riskLevel: RiskLevel | null;
  summary: PatientSummary | null;
  sendMessage: (text: string) => void;
}

/** Owns the WebSocket connection for one intake session — one browser tab,
 * one session id, one PatientState on the backend. Reconnection/multi-tab
 * sync is out of scope for this MVP (see root README's Known Limitations).
 */
export function useChatSession(): ChatSession {
  const sessionIdRef = useRef<string>(newId());
  const wsRef = useRef<WebSocket | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [emergencyFlag, setEmergencyFlag] = useState(false);
  const [riskLevel, setRiskLevel] = useState<RiskLevel | null>(null);
  const [summary, setSummary] = useState<PatientSummary | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${BACKEND_WS_BASE}/ws/chat/${sessionIdRef.current}`);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);

    ws.onmessage = (event) => {
      const data: ServerEvent = JSON.parse(event.data);

      if (data.type === "typing") {
        setIsTyping(true);
        return;
      }

      setIsTyping(false);
      if (data.content) {
        setMessages((prev) => [...prev, { id: newId(), role: "assistant", content: data.content as string }]);
      }
      setRiskLevel(data.risk_level);
      setEmergencyFlag(data.emergency_flag);
      if (data.stage === "done") {
        setIsDone(true);
        if (data.final_summary) setSummary(data.final_summary);
      }
    };

    return () => ws.close();
  }, []);

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || wsRef.current?.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [...prev, { id: newId(), role: "patient", content: trimmed }]);
    wsRef.current.send(JSON.stringify({ message: trimmed }));
  }, []);

  return { messages, isTyping, isConnected, isDone, emergencyFlag, riskLevel, summary, sendMessage };
}
