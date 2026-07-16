type Job = {
  job_id: string;
  job_type: string;
  state: string;
  payload?: Record<string, unknown>;
  result?: Record<string, unknown>;
  assigned_client?: string | null;
  error?: string | null;
};

type Props = {
  jobs: Job[];
  stats?: { total?: number; counts?: Record<string, number> };
  onCreate: (jobType: string, payload: Record<string, unknown>) => Promise<void>;
  creating: boolean;
};

export function JobsPanel({ jobs, stats, onCreate, creating }: Props) {
  return (
    <>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-hd">
          <h2>Enqueue job</h2>
        </div>
        <div className="card-bd" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button
            className="btn primary"
            disabled={creating}
            type="button"
            onClick={() =>
              void onCreate("compute", {
                formula: "monte_carlo_pi",
                seed: Date.now() % 10000,
                steps: 80000,
              })
            }
          >
            Compute (π)
          </button>
          <button
            className="btn"
            disabled={creating}
            type="button"
            onClick={() =>
              void onCreate("inference", {
                inputs: ["edge privacy matters", "share deltas not data"],
                model_id: "local-scorer",
              })
            }
          >
            Inference
          </button>
          <button
            className="btn"
            disabled={creating}
            type="button"
            onClick={() => void onCreate("label", { offset: 0, limit: 8 })}
          >
            Label chunk
          </button>
          {stats && (
            <span className="mono" style={{ marginLeft: "auto", alignSelf: "center" }}>
              queue total {stats.total ?? 0}
            </span>
          )}
        </div>
      </div>

      <div className="list">
        {jobs.length === 0 ? (
          <div className="empty">No jobs yet. Enqueue one above or via POST /jobs.</div>
        ) : (
          jobs.map((j, idx) => (
            <div className="list-row" key={j.job_id}>
              <div>
                <div className="title">
                  {j.job_type} · {j.job_id.slice(0, 8)}
                </div>
                <div className="meta">
                  {j.assigned_client || "unassigned"}
                  {j.error ? ` · ${j.error}` : ""}
                </div>
              </div>
              <div className={`status ${j.state.toLowerCase()}`}>
                <span className="sdot" />
                {j.state}
              </div>
              <div>
                <span className={`badge ${idx === 0 ? "primary" : "outline"}`}>
                  {j.job_type}
                </span>
              </div>
              <div className="mono">{j.assigned_client || "—"}</div>
              <div className="mono">{j.result ? "has result" : "—"}</div>
              <div className="mono">{j.job_id.slice(0, 7)}</div>
            </div>
          ))
        )}
      </div>
    </>
  );
}
