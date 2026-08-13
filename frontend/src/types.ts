export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface ScheduledAppointmentSummary {
  doctor: string | null;
  time: string | null;
  status: string | null;
}

export interface PatientSummary {
  age: number | null;
  sex: string | null;
  chief_complaint: string | null;
  symptoms: Record<string, string>[];
  summary: string;
  risk_level: RiskLevel;
  recommendation: string;
  requires_human: boolean;
  missing_information: string[];
  specialty: string | null;
  scheduled_appointment: ScheduledAppointmentSummary | null;
}

/** One line in the transcript. `role: "assistant"` covers real assistant
 * replies; `role: "error"` is a failed-turn notice (see useChatSession.ts)
 * rendered inline so it's visible in context, not just a toast. The typing
 * placeholder is handled separately, not as a chat message. */
export interface ChatMessage {
  id: string;
  role: "patient" | "assistant" | "error";
  content: string;
}

/** Server -> client WebSocket payloads (see backend_app/main.py). "error"
 * means this turn's graph.invoke() failed (provider rate limit, transient
 * server error, timeout, etc.) — the session's last successful state is
 * unchanged server-side, so the patient can just retry. */
export type ServerEvent =
  | { type: "typing" }
  | { type: "error"; message: string }
  | {
      type: "message";
      content: string | null;
      stage: string;
      awaiting_patient: boolean;
      risk_level: RiskLevel;
      emergency_flag: boolean;
      final_summary?: PatientSummary;
    };
