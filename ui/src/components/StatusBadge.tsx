export type StatusKind =
  | "ok"
  | "active"
  | "pending"
  | "warn"
  | "danger"
  | "neutral";

type Props = {
  kind: StatusKind;
  label: string;
  detail?: string;
};

/** Colored status dot with a text label; color is never the only signal. */
export function StatusBadge({ kind, label, detail }: Props) {
  return (
    <span className={`status-badge ${kind}`}>
      <span className="sdot" aria-hidden="true" />
      <span>{label}</span>
      {detail && <span className="status-detail">{detail}</span>}
    </span>
  );
}

export function roundStatus(state: string): { kind: StatusKind; label: string } {
  switch (state) {
    case "CLOSED":
      return { kind: "ok", label: "Closed" };
    case "AGGREGATING":
      return { kind: "warn", label: "Aggregating" };
    case "COLLECTING":
      return { kind: "active", label: "Collecting" };
    case "OPEN":
      return { kind: "active", label: "Open" };
    default:
      return { kind: "neutral", label: state || "Unknown" };
  }
}

export function jobStatus(state: string): { kind: StatusKind; label: string } {
  switch (state) {
    case "COMPLETED":
      return { kind: "ok", label: "Completed" };
    case "ASSIGNED":
      return { kind: "active", label: "Assigned" };
    case "QUEUED":
      return { kind: "pending", label: "Queued" };
    case "FAILED":
      return { kind: "danger", label: "Failed" };
    case "CANCELLED":
      return { kind: "neutral", label: "Cancelled" };
    default:
      return { kind: "neutral", label: state || "Unknown" };
  }
}

export type Presence = {
  kind: StatusKind;
  label: string;
  detail?: string;
};

/** Presence derived from reputation last_seen relative to server time. */
export function clientPresence(
  lastSeen: number | undefined,
  serverTime: number | undefined
): Presence {
  if (!lastSeen) return { kind: "neutral", label: "Registered", detail: "no activity" };
  const now = serverTime ?? Date.now() / 1000;
  const ageSeconds = Math.max(0, now - lastSeen);
  if (ageSeconds < 120) return { kind: "ok", label: "Online" };
  if (ageSeconds < 900)
    return { kind: "active", label: "Idle", detail: `${Math.round(ageSeconds / 60)}m ago` };
  if (ageSeconds < 3600)
    return { kind: "warn", label: "Stale", detail: `${Math.round(ageSeconds / 60)}m ago` };
  return {
    kind: "danger",
    label: "Offline",
    detail: `${Math.round(ageSeconds / 3600)}h ago`,
  };
}
