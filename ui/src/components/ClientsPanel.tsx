import type { Incentive, Reputation } from "../api";
import { StatusBadge, clientPresence } from "./StatusBadge";

type Props = {
  reputations: Record<string, Reputation>;
  incentives: Record<string, Incentive>;
  registeredClients: string[];
  serverTime?: number;
};

export function ClientsPanel({
  reputations,
  incentives,
  registeredClients,
  serverTime,
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
    <div className="table-wrap">
      <table className="table">
        <caption className="sr-only">
          Registered federated clients with presence, reputation, and rewards
        </caption>
        <thead>
          <tr>
            <th scope="col">Client</th>
            <th scope="col">Presence</th>
            <th scope="col">Reputation</th>
            <th scope="col">Acceptance</th>
            <th scope="col">Updates</th>
            <th scope="col">Balance</th>
          </tr>
        </thead>
        <tbody>
          {ids.map((id) => {
            const rep = reputations[id];
            const inc = incentives[id];
            const presence = clientPresence(rep?.last_seen, serverTime);
            return (
              <tr key={id}>
                <td>
                  <span className="title">{id}</span>
                </td>
                <td>
                  <StatusBadge
                    kind={presence.kind}
                    label={presence.label}
                    detail={presence.detail}
                  />
                </td>
                <td className="mono">
                  {rep ? rep.reputation_score.toFixed(2) : "—"}
                </td>
                <td className="mono">
                  {rep ? `${(rep.acceptance_rate * 100).toFixed(0)}%` : "—"}
                </td>
                <td className="mono">
                  {rep ? `${rep.updates_accepted}/${rep.updates_submitted}` : "—"}
                </td>
                <td className="mono">
                  {inc ? `${Math.round(inc.current_balance)} tokens` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
