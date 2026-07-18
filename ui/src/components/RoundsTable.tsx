import type { ClassicRound } from "../api";
import { StatusBadge, roundStatus } from "./StatusBadge";

type Props = {
  rounds: ClassicRound[];
  busyId: number | null;
  asyncEnabled?: boolean;
  onAggregate: (roundId: number) => void;
  filter?: string;
};

export function RoundsTable({
  rounds,
  busyId,
  asyncEnabled,
  onAggregate,
  filter,
}: Props) {
  const filtered = rounds.filter((r) => {
    if (!filter || filter === "all") return true;
    return r.state.toLowerCase() === filter.toLowerCase();
  });

  if (filtered.length === 0) {
    return (
      <div className="list">
        <div className="empty">No rounds match this filter.</div>
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <table className="table">
        <caption className="sr-only">
          Classic federated learning rounds with state and update counts
        </caption>
        <thead>
          <tr>
            <th scope="col">Round</th>
            <th scope="col">State</th>
            <th scope="col">Updates</th>
            <th scope="col">Model</th>
            <th scope="col">Clients</th>
            <th scope="col">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((r) => {
            const ready =
              r.total_updates > 0 &&
              r.total_updates >= r.total_clients &&
              r.state !== "CLOSED" &&
              r.state !== "UNKNOWN";
            const status = roundStatus(r.state);
            return (
              <tr key={r.round_id}>
                <td>
                  <span className="title">#{r.round_id}</span>
                </td>
                <td>
                  <StatusBadge kind={status.kind} label={status.label} />
                </td>
                <td className="mono">
                  {r.total_updates}/{r.total_clients}
                </td>
                <td className="mono">{r.model_version ?? "—"}</td>
                <td className="cell-clients">
                  {r.assigned_clients.length ? (
                    <span title={r.assigned_clients.join(", ")}>
                      {r.assigned_clients.slice(0, 2).join(", ")}
                      {r.assigned_clients.length > 2
                        ? ` +${r.assigned_clients.length - 2}`
                        : ""}
                    </span>
                  ) : (
                    <span className="muted">None assigned</span>
                  )}
                </td>
                <td className="cell-actions">
                  <button
                    type="button"
                    className="btn"
                    disabled={!ready || busyId === r.round_id}
                    title={
                      asyncEnabled
                        ? "Manual override — async aggregation is enabled"
                        : undefined
                    }
                    onClick={() => onAggregate(r.round_id)}
                  >
                    {busyId === r.round_id ? "Aggregating…" : "Aggregate"}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
