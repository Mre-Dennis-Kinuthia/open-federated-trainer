import type { Incentive, Reputation } from "../api";

type Props = {
  reputations: Record<string, Reputation>;
  incentives: Record<string, Incentive>;
  registeredClients: string[];
};

export function ClientsPanel({
  reputations,
  incentives,
  registeredClients,
}: Props) {
  const ids = Array.from(
    new Set([
      ...registeredClients,
      ...Object.keys(reputations),
      ...Object.keys(incentives),
    ])
  ).sort();

  if (ids.length === 0) {
    return (
      <div className="list">
        <div className="empty">No clients registered yet.</div>
      </div>
    );
  }

  return (
    <div className="list">
      {ids.map((id, idx) => {
        const rep = reputations[id];
        const inc = incentives[id];
        return (
          <div className="list-row" key={id}>
            <div>
              <div className="title">{id}</div>
              <div className="meta">
                Acceptance{" "}
                {rep ? `${(rep.acceptance_rate * 100).toFixed(0)}%` : "—"}
              </div>
            </div>
            <div className="status ready">
              <span className="sdot" />
              Active
            </div>
            <div>
              <span className={`badge ${idx === 0 ? "primary" : "outline"}`}>
                Edge
              </span>
            </div>
            <div className="mono">
              {rep ? rep.reputation_score.toFixed(2) : "—"}
            </div>
            <div className="mono">
              {rep ? `${rep.updates_accepted}/${rep.updates_submitted}` : "—"}
            </div>
            <div className="mono">
              {inc ? Math.round(inc.current_balance) : "—"} tok
            </div>
          </div>
        );
      })}
    </div>
  );
}
