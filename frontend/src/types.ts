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

/** One line in the transcript. `role: "assistant"` covers both real
 * assistant replies and the typing placeholder is handled separately, not
 * as a chat message. */
export interface ChatMessage {
  id: string;
  role: "patient" | "assistant";
  content: string;
}

/** Server -> client WebSocket payloads (see backend_app/main.py). */
export type ServerEvent =
  | { type: "typing" }
  | {
      type: "message";
      content: string | null;
      stage: string;
      awaiting_patient: boolean;
      risk_level: RiskLevel;
      emergency_flag: boolean;
      final_summary?: PatientSummary;
    };
