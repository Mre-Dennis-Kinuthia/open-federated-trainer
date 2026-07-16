import type { ClassicRound } from "../api";

type Props = {
  rounds: ClassicRound[];
  busyId: number | null;
  onAggregate: (roundId: number) => void;
  filter?: string;
};

export function RoundsTable({ rounds, busyId, onAggregate, filter }: Props) {
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
    <div className="list">
      {filtered.map((r, idx) => {
        const ready =
          r.total_updates > 0 &&
          r.total_updates >= r.total_clients &&
          r.state !== "CLOSED" &&
          r.state !== "UNKNOWN";
        const statusClass = r.state.toLowerCase();
        return (
          <div className="list-row" key={r.round_id}>
            <div>
              <div className="title">
                Round #{r.round_id} · model {r.model_version ?? "—"}
              </div>
              <div className="meta">
                {r.assigned_clients.length
                  ? r.assigned_clients.join(", ")
                  : "No clients assigned"}
              </div>
            </div>
            <div className={`status ${statusClass}`}>
              <span className="sdot" />
              {r.state === "CLOSED" ? "Ready" : r.state}
              <span className="mono" style={{ marginLeft: 4 }}>
                {r.total_updates}/{r.total_clients}
              </span>
            </div>
            <div>
              <span className={`badge ${idx === 0 ? "primary" : "outline"}`}>
                Classic
              </span>
            </div>
            <div className="mono">{r.model_version ?? "—"}</div>
            <div className="mono">#{r.round_id}</div>
            <div>
              <button
                className="btn"
                disabled={!ready || busyId === r.round_id}
                onClick={() => onAggregate(r.round_id)}
              >
                {busyId === r.round_id ? "…" : "Aggregate"}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
