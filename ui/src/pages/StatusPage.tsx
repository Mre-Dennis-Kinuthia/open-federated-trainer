import { useEffect, useState } from "react";
import {
  fetchActivity,
  fetchOverview,
  type Activity,
  type Overview,
} from "../api";
import { PrivacyDisclosure } from "../components/PrivacyDisclosure";

type Props = {
  onBack?: () => void;
};

export function StatusPage({ onBack }: Props) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchOverview(10), fetchActivity()])
      .then(([ov, act]) => {
        if (cancelled) return;
        setOverview(ov);
        setActivity(act);
        setFetchedAt(Date.now());
        setError(null);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load status");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const health = {
    operator_auth_required: overview?.operator_auth_required,
    registered_clients: overview?.registered_clients?.length,
  };
  const jobStats = overview?.job_stats;
  const queued = jobStats?.counts?.QUEUED ?? null;

  return (
    <div className="public-page" data-testid="status-page">
      <a className="skip-link" href="#status-main">
        Skip to status
      </a>
      <header className="public-page-header">
        <a href="#/" className="public-brand">
          fed-compute
        </a>
        <nav aria-label="Public">
          <a href="#/privacy">Privacy</a>
          <a href="#/overview">Console</a>
        </nav>
      </header>
      <main id="status-main" className="public-page-main">
        <h1>Coordinator status</h1>
        <p className="public-lede">
          Live view of <strong>this coordinator instance</strong> — not a global
          production fleet. Numbers come from the API you are connected to.
        </p>
        {error && (
          <p role="alert" className="public-alert">
            Coordinator unreachable: {error}
          </p>
        )}
        {!error && !overview && <p>Loading status…</p>}
        {overview && (
          <dl className="status-grid">
            <div>
              <dt>API</dt>
              <dd>{error ? "unreachable" : "reachable"}</dd>
            </div>
            <div>
              <dt>Operator auth required</dt>
              <dd>{health?.operator_auth_required ? "yes" : "no (dev-open)"}</dd>
            </div>
            <div>
              <dt>Registered clients</dt>
              <dd>{health?.registered_clients ?? "—"}</dd>
            </div>
            <div>
              <dt>Nodes online (activity)</dt>
              <dd>{activity?.online_count ?? "—"}</dd>
            </div>
            <div>
              <dt>Training rounds (seen)</dt>
              <dd>{overview.global?.total_rounds ?? "—"}</dd>
            </div>
            <div>
              <dt>Jobs completed</dt>
              <dd>{jobStats?.completed ?? "—"}</dd>
            </div>
            <div>
              <dt>Jobs queued</dt>
              <dd>{queued ?? "—"}</dd>
            </div>
            <div>
              <dt>Jobs tracked (all states)</dt>
              <dd>{jobStats?.total ?? "—"}</dd>
            </div>
            <div>
              <dt>Last fetch</dt>
              <dd>
                {fetchedAt
                  ? new Date(fetchedAt).toLocaleString()
                  : "—"}
              </dd>
            </div>
          </dl>
        )}
        <p className="public-footnote">
          “Healthy” in the operator console means the overview API responded and
          failed-update counters are zero — not full cluster HA or disk checks.
        </p>
        <PrivacyDisclosure workload="general" compact />
        {onBack && (
          <p>
            <button type="button" className="ghost" onClick={onBack}>
              Back
            </button>
          </p>
        )}
      </main>
    </div>
  );
}
