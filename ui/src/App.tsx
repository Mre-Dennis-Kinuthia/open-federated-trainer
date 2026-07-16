import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  aggregateClassicRound,
  aggregateLoraRound,
  createJob,
  createLoraRound,
  fetchOverview,
  launchDemo,
  launchProcess,
  setActiveModel,
  stopAllLaunch,
  stopLaunch,
  type CreateLoraPayload,
  type Overview,
} from "./api";
import { ClientsPanel } from "./components/ClientsPanel";
import { JobsPanel } from "./components/JobsPanel";
import { LaunchPanel } from "./components/LaunchPanel";
import { LoraPanel } from "./components/LoraPanel";
import { RoundsTable } from "./components/RoundsTable";
import { historyFromValue, Sparkline } from "./components/Sparkline";

type Tab = "overview" | "launch" | "rounds" | "clients" | "lora" | "jobs";

const NAV: { id: Tab; label: string; section?: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "launch", label: "Launch", section: "Run" },
  { id: "rounds", label: "Rounds", section: "Training" },
  { id: "clients", label: "Clients", section: "Training" },
  { id: "lora", label: "LoRA", section: "Training" },
  { id: "jobs", label: "Jobs", section: "Work queue" },
];

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d={d} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const ICONS: Record<Tab, string> = {
  overview: "M3 10.5L12 3l9 7.5V20a1 1 0 01-1 1h-5v-6H9v6H4a1 1 0 01-1-1v-9.5z",
  launch: "M5 12h14M12 5l7 7-7 7",
  rounds: "M4 7h16M4 12h16M4 17h10",
  clients: "M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8z",
  lora: "M12 3l2.5 6.5L21 11l-5 4.5L17.5 22 12 18.5 6.5 22 8 15.5 3 11l6.5-1.5L12 3z",
  jobs: "M4 6h16v4H4V6zm0 8h7v4H4v-4zm9 0h7v4h-7v-4z",
};

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [busyClassic, setBusyClassic] = useState<number | null>(null);
  const [busyLora, setBusyLora] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const [creatingJob, setCreatingJob] = useState(false);
  const [launchBusy, setLaunchBusy] = useState(false);
  const [stateFilter, setStateFilter] = useState("all");
  const online = data !== null && !error;
  const prevFailed = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchOverview(25);
      setData(next);
      setError(null);
      prevFailed.current = next.global.total_failed_updates ?? 0;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load overview");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(id);
  }, [toast]);

  async function handleAggregateClassic(roundId: number) {
    setBusyClassic(roundId);
    try {
      await aggregateClassicRound(roundId);
      setToast(`Aggregated classic round #${roundId}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Aggregate failed");
    } finally {
      setBusyClassic(null);
    }
  }

  async function handleCreateLora(payload: CreateLoraPayload) {
    setCreating(true);
    try {
      const created = await createLoraRound(payload);
      setToast(`Created LoRA round #${created.round_id}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Create LoRA round failed");
    } finally {
      setCreating(false);
    }
  }

  async function handleAggregateLora(roundId: number) {
    setBusyLora(roundId);
    try {
      await aggregateLoraRound(roundId);
      setToast(`Aggregated LoRA round #${roundId}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "LoRA aggregate failed");
    } finally {
      setBusyLora(null);
    }
  }

  async function handleCreateJob(jobType: string, payload: Record<string, unknown>) {
    setCreatingJob(true);
    try {
      await createJob(jobType, payload);
      setToast(`Enqueued ${jobType} job`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Create job failed");
    } finally {
      setCreatingJob(false);
    }
  }

  async function handleSetModel(modelId: string) {
    try {
      await setActiveModel(modelId);
      setToast(`Active model → ${modelId}`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Set model failed");
    }
  }

  async function handleStartDemo(opts: {
    modelId: string;
    datasetPreset: string;
    trainClients: number;
  }) {
    setLaunchBusy(true);
    try {
      await launchDemo({
        model_id: opts.modelId,
        dataset_preset: opts.datasetPreset,
        train_clients: opts.trainClients,
        start_worker: true,
        enqueue_sample_job: true,
      });
      setToast(
        `Started ${opts.trainClients} train client(s) + worker · model ${opts.modelId}`
      );
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Launch demo failed");
    } finally {
      setLaunchBusy(false);
    }
  }

  async function handleStartTrain(opts: {
    count: number;
    modelId: string;
    datasetPreset: string;
  }) {
    setLaunchBusy(true);
    try {
      await launchProcess({
        kind: "train",
        count: opts.count,
        model_id: opts.modelId,
        dataset_preset: opts.datasetPreset,
        set_active_model: true,
      });
      setToast(`Started ${opts.count} train client(s)`);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Start train failed");
    } finally {
      setLaunchBusy(false);
    }
  }

  async function handleStartWorker(opts: {
    jobTypes: string;
    datasetPreset: string;
    enqueueSample: boolean;
  }) {
    setLaunchBusy(true);
    try {
      await launchProcess({
        kind: "worker",
        count: 1,
        job_types: opts.jobTypes,
        dataset_preset: opts.datasetPreset,
        enqueue_sample_job: opts.enqueueSample,
      });
      setToast("Started job worker");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Start worker failed");
    } finally {
      setLaunchBusy(false);
    }
  }

  async function handleStopLaunch(id: string) {
    setLaunchBusy(true);
    try {
      await stopLaunch(id);
      setToast("Stopped process");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stop failed");
    } finally {
      setLaunchBusy(false);
    }
  }

  async function handleStopAllLaunch() {
    setLaunchBusy(true);
    try {
      await stopAllLaunch();
      setToast("Stopped all local processes");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stop all failed");
    } finally {
      setLaunchBusy(false);
    }
  }

  const g = data?.global ?? {};
  const clients = data?.registered_clients.length ?? 0;
  const rounds = g.total_rounds ?? 0;
  const failed = g.total_failed_updates ?? 0;
  const latest = data?.latest_round ?? {};
  const collecting = (data?.classic_rounds ?? []).filter(
    (r) => r.state === "COLLECTING" || r.state === "OPEN"
  ).length;

  const sparkRounds = useMemo(() => historyFromValue(Number(rounds) || 1), [rounds]);
  const sparkClients = useMemo(
    () => historyFromValue(Number(clients) || 1, 24).map((v, i) => v * (0.7 + (i % 5) * 0.05)),
    [clients]
  );
  const sparkUpdates = useMemo(() => {
    const accepted = Number(latest.updates_accepted ?? 0);
    return historyFromValue(Math.max(accepted, clients * 2, 1));
  }, [latest.updates_accepted, clients]);

  let lastSection: string | undefined;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">fc</div>
          <div className="brand-meta">
            <div className="name">fed-compute</div>
            <span className="plan">Volunteer</span>
          </div>
        </div>

        <div className="sidebar-search">
          Find…
          <kbd>F</kbd>
        </div>

        {NAV.map((item) => {
          const showSection = item.section && item.section !== lastSection;
          if (item.section) lastSection = item.section;
          return (
            <div key={item.id}>
              {showSection && <div className="nav-section">{item.section}</div>}
              <button
                type="button"
                className={`nav-item${tab === item.id ? " active" : ""}`}
                onClick={() => setTab(item.id)}
              >
                <Icon d={ICONS[item.id]} />
                {item.label}
              </button>
            </div>
          );
        })}

        <div className="sidebar-foot">
          <span className={`pulse-dot${online ? "" : " off"}`} />
          {online ? "Coordinator live · 3s" : "Coordinator offline"}
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-row">
            <h1>
              {tab === "overview" && "Observability"}
              {tab === "launch" && "Launch"}
              {tab === "rounds" && "Rounds"}
              {tab === "clients" && "Clients"}
              {tab === "lora" && "LoRA"}
              {tab === "jobs" && "Jobs"}
            </h1>
            <div className="topbar-actions">
              <select className="select" defaultValue="local" aria-label="Environment">
                <option value="local">Local</option>
                <option value="edge">Edge</option>
              </select>
              <select className="select" defaultValue="12h" aria-label="Time range">
                <option value="1h">Past Hour</option>
                <option value="12h">Last 12 hours</option>
                <option value="7d">Last 7 Days</option>
              </select>
            </div>
          </div>
        </header>

        <div className="content">
          {error && <div className="banner error">{error}</div>}
          {toast && <div className="banner ok">{toast}</div>}

          {tab === "overview" && (
            <>
              <div className="banner">
                <span>
                  Async FedAvg is {data?.async_enabled ? "on" : "off"} · v
                  {data?.version ?? "—"} · operator auth{" "}
                  {online ? "optional until OPERATOR_API_KEY is set" : "—"}
                </span>
                <button className="btn black" type="button" onClick={() => void refresh()}>
                  Refresh
                </button>
              </div>

              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-hd">
                  <h2>Classic architecture</h2>
                </div>
                <div className="card-bd" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {Object.keys(data?.classic_models ?? { simple_mlp: {}, tiny_cnn: {}, custom: {} }).map(
                    (id) => (
                      <button
                        key={id}
                        type="button"
                        className={`btn${data?.active_model?.model_id === id ? " primary" : ""}`}
                        onClick={() => void handleSetModel(id)}
                      >
                        {id}
                      </button>
                    )
                  )}
                  <span className="mono" style={{ alignSelf: "center" }}>
                    active: {data?.active_model?.model_id ?? "simple_mlp"}
                  </span>
                  <button
                    type="button"
                    className="btn black"
                    style={{ marginLeft: "auto" }}
                    onClick={() => setTab("launch")}
                  >
                    Open Launch
                  </button>
                </div>
              </div>

              <div className="metrics">
                <div className="metric active">
                  <div className="label">Clients</div>
                  <div className="row">
                    <span className="value">{clients}</span>
                    <span className="trend up">seen {g.total_clients_seen ?? 0}</span>
                  </div>
                </div>
                <div className="metric">
                  <div className="label">Rounds</div>
                  <div className="row">
                    <span className="value">{rounds}</span>
                    <span className="trend neutral">{collecting} open</span>
                  </div>
                </div>
                <div className="metric">
                  <div className="label">Failed updates</div>
                  <div className="row">
                    <span className="value">{failed}</span>
                    <span className={`trend ${failed > 0 ? "down" : "up"}`}>
                      {failed > 0 ? "attention" : "healthy"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="card-grid">
                <div className="card">
                  <div className="status-hero">
                    <div className="shield">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                        <path
                          d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"
                          fill="#0070f3"
                          opacity="0.15"
                        />
                        <path
                          d="M9 12l2 2 4-4"
                          stroke="#0070f3"
                          strokeWidth="2"
                          strokeLinecap="round"
                        />
                      </svg>
                    </div>
                    <h3>
                      {online ? "Coordinator is active" : "Coordinator unreachable"}
                    </h3>
                    <p>
                      {online
                        ? "All systems normal."
                        : "Check that the API is running on :8000."}
                    </p>
                  </div>
                  <div className="status-foot">
                    <div>
                      <span>Async aggregation</span>
                      <strong>{data?.async_enabled ? "Active" : "Off"}</strong>
                    </div>
                    <div>
                      <span>Registered clients</span>
                      <strong>{clients}</strong>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-hd">
                    <h2>Round activity</h2>
                    <span className="chev">›</span>
                  </div>
                  <div className="card-sub">
                    <strong>{rounds}</strong> Total rounds
                  </div>
                  <div className="card-bd">
                    <div className="legend">
                      <span>
                        <i className="dot blue" /> Rounds
                      </span>
                      <span>
                        <i className="dot orange" /> Clients
                      </span>
                    </div>
                    <Sparkline seriesA={sparkRounds} seriesB={sparkClients} />
                  </div>
                </div>

                <div className="card">
                  <div className="card-hd">
                    <h2>Update throughput</h2>
                    <span className="chev">›</span>
                  </div>
                  <div className="card-sub">
                    <strong>{String(latest.updates_accepted ?? 0)}</strong>
                    Accepted in latest round
                  </div>
                  <div className="card-bd">
                    <div className="legend">
                      <span>
                        <i className="dot blue" /> Accepted
                      </span>
                      <span>
                        <i className="dot teal" /> Baseline
                      </span>
                    </div>
                    <Sparkline
                      seriesA={sparkUpdates}
                      seriesB={sparkUpdates.map((v) => v * 0.55)}
                    />
                  </div>
                </div>

                <div className="card">
                  <div className="card-hd">
                    <h2>Latest round</h2>
                    <span className="chev">›</span>
                  </div>
                  <div className="card-bd">
                    {Object.keys(latest).length === 0 ? (
                      <div className="empty">Waiting for first round…</div>
                    ) : (
                      <div className="status-foot" style={{ borderTop: "none", padding: 0 }}>
                        <div>
                          <span>Round</span>
                          <strong>#{String(latest.round_id ?? "—")}</strong>
                        </div>
                        <div>
                          <span>Model</span>
                          <strong>{String(latest.model_version ?? "—")}</strong>
                        </div>
                        <div>
                          <span>Updates</span>
                          <strong>
                            {String(latest.updates_accepted ?? 0)}/
                            {String(latest.updates_received ?? 0)}
                          </strong>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-hd">
                  <h2>Recent rounds</h2>
                </div>
                <div className="card-bd tight">
                  <RoundsTable
                    rounds={(data?.classic_rounds ?? []).slice(0, 8)}
                    busyId={busyClassic}
                    onAggregate={handleAggregateClassic}
                  />
                </div>
              </div>
            </>
          )}

          {tab === "launch" && (
            <LaunchPanel
              launcher={data?.launcher}
              models={Object.keys(
                data?.classic_models ?? { simple_mlp: {}, tiny_cnn: {}, custom: {} }
              )}
              activeModel={data?.active_model?.model_id}
              busy={launchBusy}
              onStartDemo={handleStartDemo}
              onStartTrain={handleStartTrain}
              onStartWorker={handleStartWorker}
              onStop={handleStopLaunch}
              onStopAll={handleStopAllLaunch}
            />
          )}

          {tab === "rounds" && (
            <>
              <div className="filters">
                <select
                  className="filter"
                  value={stateFilter}
                  onChange={(e) => setStateFilter(e.target.value)}
                >
                  <option value="all">All states</option>
                  <option value="OPEN">Open</option>
                  <option value="COLLECTING">Collecting</option>
                  <option value="AGGREGATING">Aggregating</option>
                  <option value="CLOSED">Closed</option>
                </select>
                <select className="filter" defaultValue="classic">
                  <option value="classic">Classic FL</option>
                </select>
              </div>
              <RoundsTable
                rounds={data?.classic_rounds ?? []}
                busyId={busyClassic}
                onAggregate={handleAggregateClassic}
                filter={stateFilter}
              />
            </>
          )}

          {tab === "clients" && (
            <ClientsPanel
              reputations={data?.reputations ?? {}}
              incentives={data?.incentives ?? {}}
              registeredClients={data?.registered_clients ?? []}
            />
          )}

          {tab === "lora" && (
            <LoraPanel
              baseModels={data?.lora_base_models ?? []}
              rounds={data?.lora_rounds ?? []}
              busyId={busyLora}
              creating={creating}
              onCreate={handleCreateLora}
              onAggregate={handleAggregateLora}
            />
          )}

          {tab === "jobs" && (
            <JobsPanel
              jobs={data?.jobs ?? []}
              stats={data?.job_stats}
              onCreate={handleCreateJob}
              creating={creatingJob}
            />
          )}
        </div>
      </div>
    </div>
  );
}
