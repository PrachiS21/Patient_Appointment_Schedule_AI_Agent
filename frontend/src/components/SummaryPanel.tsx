import type { PatientSummary } from "../types";

const RISK_LABEL: Record<PatientSummary["risk_level"], string> = {
  LOW: "Low",
  MEDIUM: "Medium",
  HIGH: "High",
};

export function SummaryPanel({ summary }: { summary: PatientSummary }) {
  return (
    <aside className="summary-panel" aria-label="Visit summary">
      <h2>Visit Summary</h2>

      <span className={`risk-badge risk-${summary.risk_level.toLowerCase()}`}>
        {RISK_LABEL[summary.risk_level]} risk
      </span>

      <dl className="summary-fields">
        <dt>Chief complaint</dt>
        <dd>{summary.chief_complaint ?? "Not captured"}</dd>

        <dt>Symptoms</dt>
        <dd>
          {summary.symptoms.length === 0 ? (
            "None recorded"
          ) : (
            <ul>
              {summary.symptoms.map((symptom, i) => (
                <li key={i}>
                  {symptom.name}
                  {symptom.onset ? ` — onset ${symptom.onset}` : ""}
                  {symptom.severity ? `, severity ${symptom.severity}` : ""}
                </li>
              ))}
            </ul>
          )}
        </dd>

        <dt>Specialty</dt>
        <dd>{summary.specialty ?? "Not determined"}</dd>

        {summary.scheduled_appointment && (
          <>
            <dt>Appointment</dt>
            <dd>
              {summary.scheduled_appointment.status === "booked" ? (
                <>
                  {summary.scheduled_appointment.doctor} — {summary.scheduled_appointment.time}
                </>
              ) : (
                "No matching availability found"
              )}
            </dd>
          </>
        )}

        <dt>Recommendation</dt>
        <dd>{summary.recommendation}</dd>

        {summary.requires_human && (
          <dd className="human-follow-up-flag">A staff member will follow up on this case.</dd>
        )}

        {summary.missing_information.length > 0 && (
          <>
            <dt>Still needed</dt>
            <dd>{summary.missing_information.join(", ")}</dd>
          </>
        )}
      </dl>

      <details className="summary-json">
        <summary>Raw JSON</summary>
        <pre>{JSON.stringify(summary, null, 2)}</pre>
      </details>
    </aside>
  );
}
