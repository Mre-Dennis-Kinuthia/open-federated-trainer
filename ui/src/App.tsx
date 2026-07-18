import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  aggregateClassicRound,
  aggregateLoraRound,
  cancelJob,
  createJob,
  createLoraRound,
  fetchOverview,
  getOperatorKey,
  launchDemo,
  launchProcess,
  setActiveModel,
  stopAllLaunch,
  stopLaunch,
  type CreateLoraPayload,
  type Overview,
} from "./api";
import { ClientsPanel } from "./components/ClientsPanel";
import { Landing } from "./landing/Landing";
import { PrivacyPage } from "./pages/PrivacyPage";
import { StatusPage } from "./pages/StatusPage";
import { ConfirmDialog, type ConfirmRequest } from "./components/ConfirmDialog";
import { JobsPanel } from "./components/JobsPanel";
import { LaunchPanel } from "./components/LaunchPanel";
import { LoraPanel } from "./components/LoraPanel";
import { RoundsTable } from "./components/RoundsTable";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sparkline } from "./components/Sparkline";

type Tab =
  | "overview"
  | "launch"
  | "rounds"
  | "clients"
  | "lora"
  | "jobs"
  | "settings";

const TABS: Tab[] = [
  "overview",
  "launch",
  "rounds",
  "clients",
  "lora",
  "jobs",
  "settings",
];

const NAV: { id: Tab; label: string; section?: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "launch", label: "Launch", section: "Run" },
  { id: "rounds", label: "Rounds", section: "Training" },
  { id: "clients", label: "Clients", section: "Training" },
  { id: "lora", label: "LoRA", section: "Training" },
  { id: "jobs", label: "Jobs", section: "Work queue" },
  { id: "settings", label: "Settings", section: "System" },
];

const PUBLIC_NAV = [
  { href: "#/status", label: "Status" },
  { href: "#/privacy", label: "Privacy" },
];

const TITLES: Record<Tab, string> = {
  overview: "Overview",
  launch: "Launch",
  rounds: "Rounds",
  clients: "Clients",
  lora: "LoRA",
  jobs: "Jobs",
  settings: "Settings",
};

function Icon({ d }: { d: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
    >
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
  settings:
    "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12a7 7 0 01-.1 1.2l2 1.6-2 3.4-2.4-1a7 7 0 01-2 1.2L14 21h-4l-.5-2.6a7 7 0 01-2-1.2l-2.4 1-2-3.4 2-1.6A7 7 0 015 12a7 7 0 01.1-1.2l-2-1.6 2-3.4 2.4 1a7 7 0 012-1.2L10 3h4l.5 2.6a7 7 0 012 1.2l2.4-1 2 3.4-2 1.6c.06.4.1.8.1 1.2z",
};

type Route = Tab | "landing" | "status" | "privacy";

function routeFromHash(): Route {
  const raw = window.location.hash.replace(/^#\/?/, "");
  if (raw === "") return "landing";
  if (raw === "status") return "status";
  if (raw === "privacy") return "privacy";
  return (TABS as string[]).includes(raw) ? (raw as Tab) : "overview";
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  const tab: Tab =
    route === "landing" || route === "status" || route === "privacy"
      ? "overview"
      : route;
  const [data, setData] = useState<Overview | null>(null);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<ConfirmRequest | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const [busyClassic, setBusyClassic] = useState<number | null>(null);
  const [busyLora, setBusyLora] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const [creatingJob, setCreatingJob] = useState(false);
  const [cancellingJob, setCancellingJob] = useState<string | null>(null);
  const [launchAction, setLaunchAction] = useState<string | null>(null);
  const [stateFilter, setStateFilter] = useState("all");
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const fetchSeq = useRef(0);

  const online = data !== null && !connectionError;

  const setTab = useCallback((next: Tab) => {
    window.location.hash = `/${next}`;
    setRoute(next);
    setNavOpen(false);
  }, []);

  useEffect(() => {
    function onHashChange() {
      setRoute(routeFromHash());
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const refresh = useCallback(async () => {
    const seq = ++fetchSeq.current;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8000);
    try {
      const next = await fetchOverview(25, controller.signal);
      if (seq !== fetchSeq.current) return;
      setData(next);
      setConnectionError(null);
      setLastUpdated(Date.now());
    } catch (e) {
      if (seq !== fetchSeq.current) return;
      setConnectionError(
        e instanceof Error ? e.message : "Failed to load overview"
      );
    } finally {
      window.clearTimeout(timeout);
      if (seq === fetchSeq.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (route === "landing" || route === "status" || route === "privacy") return;
    void refresh();
    const id = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, 3000);
    const onVisible = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh, route]);

  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 5000);
    return () => window.clearTimeout(id);
  }, [toast]);

  const reportActionError = useCallback(
    (e: unknown, fallback: string) => {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        setActionError(
          "Operator authentication failed. Set your operator key in Settings."
        );
        return;
      }
      setActionError(e instanceof Error ? e.message : fallback);
    },
    []
  );

  async function runAction(
    fn: () => Promise<void>,
    fallback: string
  ): Promise<void> {
    setActionError(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      reportActionError(e, fallback);
    }
  }

  function requestAggregateClassic(roundId: number) {
    setConfirm({
      title: `Aggregate round #${roundId}?`,
      body: "This averages all submitted updates and publishes a new global model version. It cannot be undone.",
      confirmLabel: "Aggregate",
      onConfirm: () => {
        setBusyClassic(roundId);
        void runAction(async () => {
          await aggregateClassicRound(roundId);
          setToast(`Aggregated classic round #${roundId}`);
        }, "Aggregate failed").finally(() => setBusyClassic(null));
      },
    });
  }

  function requestAggregateLora(roundId: number) {
    setConfirm({
      title: `Aggregate LoRA round #${roundId}?`,
      body: "This merges all submitted adapters, runs evaluation, and publishes a new adapter version. It cannot be undone.",
      confirmLabel: "Aggregate",
      onConfirm: () => {
        setBusyLora(roundId);
        void runAction(async () => {
          await aggregateLoraRound(roundId);
          setToast(`Aggregated LoRA round #${roundId}`);
        }, "LoRA aggregate failed").finally(() => setBusyLora(null));
      },
    });
  }

  async function handleCreateLora(payload: CreateLoraPayload) {
    setCreating(true);
    await runAction(async () => {
      const created = await createLoraRound(payload);
      setToast(
        `Created LoRA round #${created.round_id}. Start LoRA participants to submit adapters.`
      );
    }, "Create LoRA round failed");
    setCreating(false);
  }

  async function handleCreateJob(
    jobType: string,
    payload: Record<string, unknown>
  ) {
    setCreatingJob(true);
    await runAction(async () => {
      await createJob(jobType, payload);
      setToast(`Enqueued ${jobType} job`);
    }, "Create job failed");
    setCreatingJob(false);
  }

  async function handleCancelJob(jobId: string) {
    setCancellingJob(jobId);
    await runAction(async () => {
      await cancelJob(jobId);
      setToast("Job cancelled");
    }, "Cancel job failed");
    setCancellingJob(null);
  }

  function requestSetModel(modelId: string) {
    const current = data?.active_model?.model_id;
    if (modelId === current) return;
    setConfirm({
      title: `Switch active model to ${modelId}?`,
      body: "New training rounds will use this architecture. Clients already assigned to open rounds keep their current model.",
      confirmLabel: "Switch model",
      onConfirm: () => {
        void runAction(async () => {
          await setActiveModel(modelId);
          setToast(`Active model is now ${modelId}`);
        }, "Set model failed");
      },
    });
  }

  async function handleStartDemo(opts: {
    modelId: string;
    datasetPreset: string;
    trainClients: number;
  }) {
    setLaunchAction("demo");
    await runAction(async () => {
      await launchDemo({
        model_id: opts.modelId,
        dataset_preset: opts.datasetPreset,
        train_clients: opts.trainClients,
        start_worker: true,
        enqueue_sample_job: true,
      });
      setToast(
        `Started ${opts.trainClients} train client(s) and a worker with model ${opts.modelId}`
      );
    }, "Launch demo failed");
    setLaunchAction(null);
  }

  async function handleStartTrain(opts: {
    count: number;
    modelId: string;
    modelModule?: string;
    datasetPreset: string;
    datasetPath?: string;
  }) {
    setLaunchAction("train");
    await runAction(async () => {
      await launchProcess({
        kind: "train",
        count: opts.count,
        model_id: opts.modelId,
        model_module: opts.modelModule,
        dataset_preset: opts.datasetPreset,
        dataset_path: opts.datasetPath,
        set_active_model: true,
      });
      setToast(`Started ${opts.count} train client(s)`);
    }, "Start train failed");
    setLaunchAction(null);
  }

  async function handleStartWorker(opts: {
    jobTypes: string;
    datasetPreset: string;
    datasetPath?: string;
    enqueueSample: boolean;
  }) {
    setLaunchAction("worker");
    await runAction(async () => {
      await launchProcess({
        kind: "worker",
        count: 1,
        job_types: opts.jobTypes,
        dataset_preset: opts.datasetPreset,
        dataset_path: opts.datasetPath,
        enqueue_sample_job: opts.enqueueSample,
      });
      setToast("Started job worker");
    }, "Start worker failed");
    setLaunchAction(null);
  }

  async function handleStopLaunch(id: string) {
    setLaunchAction(`stop-${id}`);
    await runAction(async () => {
      await stopLaunch(id);
      setToast("Stopped process");
    }, "Stop failed");
    setLaunchAction(null);
  }

  function requestStopAll() {
    const running = data?.launcher?.running ?? 0;
    setConfirm({
      title: `Stop all ${running} local process(es)?`,
      body: "Every locally launched train client and job worker will be terminated. In-progress training rounds will stall until clients are restarted.",
      confirmLabel: "Stop all",
      tone: "danger",
      onConfirm: () => {
        setLaunchAction("stop-all");
        void runAction(async () => {
          await stopAllLaunch();
          setToast("Stopped all local processes");
        }, "Stop all failed").finally(() => setLaunchAction(null));
      },
    });
  }

  const g = data?.global ?? {};
  const clients = data?.registered_clients.length ?? 0;
  const rounds = g.total_rounds ?? 0;
  const failed = g.total_failed_updates ?? 0;
  const latest = data?.latest_round ?? {};
  const collecting = (data?.classic_rounds ?? []).filter(
    (r) => r.state === "COLLECTING" || r.state === "OPEN"
  ).length;
  const workerRunning = Boolean(
    data?.launcher?.processes?.some((p) => p.running && p.kind === "worker")
  );
  const healthy = online && failed === 0;
  const healthLabel = !online
    ? "Coordinator unreachable"
    : failed > 0
      ? "Attention needed"
      : "API reachable";
  const healthDetail = !online
    ? "Set the coordinator API URL in Settings (Vercel cannot proxy to localhost by itself)."
    : failed > 0
      ? `${failed} failed update(s) recorded — check client logs and round states.`
      : "Overview reachable and no failed updates recorded. This is not a full system health check.";
  const authRequired = data?.operator_auth_required ?? false;
  const hasKey = Boolean(getOperatorKey());

  const roundHistory = useMemo(
    () => [...(data?.classic_rounds ?? [])].reverse(),
    [data?.classic_rounds]
  );
  const sparkRounds = roundHistory.map((round) => Number(round.total_updates) || 0);
  const sparkClients = roundHistory.map((round) => Number(round.total_clients) || 0);
  const sparkAccepted = roundHistory.map(
    (round) => Number(round.metrics?.updates_accepted) || 0
  );
  const sparkReceived = roundHistory.map(
    (round) => Number(round.metrics?.updates_received) || round.total_updates || 0
  );

  if (route === "landing") {
    return <Landing onEnter={() => setTab("overview")} />;
  }
  if (route === "status") {
    return <StatusPage />;
  }
  if (route === "privacy") {
    return <PrivacyPage />;
  }

  let lastSection: string | undefined;

  const sidebar = (
    <>
      <a className="sidebar-brand" href="#/" title="Back to landing page">
        <div className="brand-mark" aria-hidden="true">
          fc
        </div>
        <div className="brand-meta">
          <div className="name">fed-compute</div>
          <span className="plan">Operator console</span>
        </div>
      </a>

      <nav aria-label="Primary">
        {NAV.map((item) => {
          const showSection = item.section && item.section !== lastSection;
          if (item.section) lastSection = item.section;
          return (
            <div key={item.id}>
              {showSection && <div className="nav-section">{item.section}</div>}
              <button
                type="button"
                className={`nav-item${tab === item.id ? " active" : ""}`}
                aria-current={tab === item.id ? "page" : undefined}
                onClick={() => setTab(item.id)}
              >
                <Icon d={ICONS[item.id]} />
                {item.label}
              </button>
            </div>
          );
        })}
        <div className="nav-section">Public</div>
        {PUBLIC_NAV.map((item) => (
          <a key={item.href} className="nav-item" href={item.href}>
            {item.label}
          </a>
        ))}
      </nav>

      <div className="sidebar-foot">
        <span className={`pulse-dot${online ? "" : " off"}`} aria-hidden="true" />
        {loading
          ? "Connecting…"
          : online
            ? "Coordinator connected"
            : "Coordinator offline"}
      </div>
    </>
  );

  return (
    <div className="shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>

      <aside className={`sidebar${navOpen ? " open" : ""}`}>{sidebar}</aside>
      {navOpen && (
        <div
          className="nav-overlay"
          aria-hidden="true"
          onClick={() => setNavOpen(false)}
        />
      )}

      <div className="main">
        <header className="topbar">
          <div className="topbar-row">
            <div className="topbar-lead">
              <button
                type="button"
                className="btn nav-toggle"
                aria-expanded={navOpen}
                aria-label="Toggle navigation"
                onClick={() => setNavOpen((v) => !v)}
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  aria-hidden="true"
                >
                  <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round" />
                </svg>
              </button>
              <h1>{TITLES[tab]}</h1>
            </div>
            <div className="topbar-actions">
              {lastUpdated && (
                <span className="muted">
                  Updated {new Date(lastUpdated).toLocaleTimeString()}
                </span>
              )}
              <button
                type="button"
                className="btn"
                onClick={() => void refresh()}
              >
                Refresh
              </button>
            </div>
          </div>
        </header>

        <main id="main-content" className="content">
          {connectionError && (
            <div className="banner error" role="alert">
              Coordinator unreachable: {connectionError}
              {data && " — showing last known data."}
            </div>
          )}
          {actionError && (
            <div className="banner error" role="alert">
              <span>{actionError}</span>
              <button
                type="button"
                className="btn"
                onClick={() => setActionError(null)}
              >
                Dismiss
              </button>
            </div>
          )}
          <div aria-live="polite">
            {toast && <div className="banner ok">{toast}</div>}
          </div>

          {loading ? (
            <div className="list">
              <div className="empty">Loading coordinator overview…</div>
            </div>
          ) : (
            <>
              {tab === "overview" && (
                <>
                  {authRequired && !hasKey && (
                    <div className="banner warn" role="status">
                      <span>
                        This coordinator requires an operator key for
                        privileged actions.
                      </span>
                      <button
                        type="button"
                        className="btn"
                        onClick={() => setTab("settings")}
                      >
                        Open Settings
                      </button>
                    </div>
                  )}

                  <div className="card spaced">
                    <div className="card-hd">
                      <h2>Classic architecture</h2>
                    </div>
                    <div className="card-bd model-picker">
                      {Object.keys(
                        data?.classic_models ?? {
                          simple_mlp: {},
                          tiny_cnn: {},
                          custom: {},
                        }
                      ).map((id) => (
                        <button
                          key={id}
                          type="button"
                          className={`btn${
                            data?.active_model?.model_id === id ? " primary" : ""
                          }`}
                          aria-pressed={data?.active_model?.model_id === id}
                          onClick={() => requestSetModel(id)}
                        >
                          {id}
                        </button>
                      ))}
                      <span className="muted">
                        Active: {data?.active_model?.model_id ?? "simple_mlp"}
                      </span>
                      <button
                        type="button"
                        className="btn push-right"
                        onClick={() => setTab("launch")}
                      >
                        Open Launch
                      </button>
                    </div>
                  </div>

                  <div className="metrics">
                    <div className="metric">
                      <div className="metric-label">Clients</div>
                      <div className="row">
                        <span className="value">{clients}</span>
                        <span className="trend neutral">
                          seen {g.total_clients_seen ?? 0}
                        </span>
                      </div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Rounds</div>
                      <div className="row">
                        <span className="value">{rounds}</span>
                        <span className="trend neutral">{collecting} open</span>
                      </div>
                    </div>
                    <div className="metric">
                      <div className="metric-label">Failed updates</div>
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
                        <div
                          className={`shield${healthy ? "" : " degraded"}`}
                          aria-hidden="true"
                        >
                          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                            <path
                              d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"
                              fill="currentColor"
                              opacity="0.15"
                            />
                            {healthy ? (
                              <path
                                d="M9 12l2 2 4-4"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                              />
                            ) : (
                              <path
                                d="M12 8v5m0 3h.01"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                              />
                            )}
                          </svg>
                        </div>
                        <h3>{healthLabel}</h3>
                        <p>{healthDetail}</p>
                      </div>
                      <div className="status-foot">
                        <div>
                          <span>Async aggregation</span>
                          <strong>{data?.async_enabled ? "Active" : "Off"}</strong>
                        </div>
                        <div>
                          <span>Operator auth</span>
                          <strong>
                            {authRequired
                              ? hasKey
                                ? "Required · key set"
                                : "Required · no key"
                              : "Not required"}
                          </strong>
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
                      </div>
                      <div className="card-sub">
                        <strong>{rounds}</strong> Total rounds
                      </div>
                      <div className="card-bd">
                        <div className="legend">
                          <span>
                            <i className="dot blue" aria-hidden="true" /> Updates
                          </span>
                          <span>
                            <i className="dot orange" aria-hidden="true" /> Clients
                          </span>
                        </div>
                        <Sparkline
                          seriesA={sparkRounds}
                          seriesB={sparkClients}
                          label="Updates and clients per recent round"
                        />
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-hd">
                        <h2>Update throughput</h2>
                      </div>
                      <div className="card-sub">
                        <strong>{String(latest.updates_accepted ?? 0)}</strong>
                        Accepted in latest round
                      </div>
                      <div className="card-bd">
                        <div className="legend">
                          <span>
                            <i className="dot blue" aria-hidden="true" /> Accepted
                          </span>
                          <span>
                            <i className="dot teal" aria-hidden="true" /> Received
                          </span>
                        </div>
                        <Sparkline
                          seriesA={sparkAccepted}
                          seriesB={sparkReceived}
                          label="Accepted and received updates per recent round"
                        />
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-hd">
                        <h2>Latest round</h2>
                      </div>
                      <div className="card-bd">
                        {Object.keys(latest).length === 0 ? (
                          <div className="empty">
                            Waiting for the first round. Start clients from the
                            Launch page.
                          </div>
                        ) : (
                          <div className="status-foot borderless">
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
                      <button
                        type="button"
                        className="btn ghost"
                        onClick={() => setTab("rounds")}
                      >
                        View all
                      </button>
                    </div>
                    <div className="card-bd tight">
                      <RoundsTable
                        rounds={(data?.classic_rounds ?? []).slice(0, 8)}
                        busyId={busyClassic}
                        asyncEnabled={data?.async_enabled}
                        onAggregate={requestAggregateClassic}
                      />
                    </div>
                  </div>
                </>
              )}

              {tab === "launch" && (
                <LaunchPanel
                  launcher={data?.launcher}
                  models={Object.keys(
                    data?.classic_models ?? {
                      simple_mlp: {},
                      tiny_cnn: {},
                      custom: {},
                    }
                  )}
                  activeModel={data?.active_model?.model_id}
                  busyAction={launchAction}
                  onStartDemo={handleStartDemo}
                  onStartTrain={handleStartTrain}
                  onStartWorker={handleStartWorker}
                  onStop={handleStopLaunch}
                  onStopAll={requestStopAll}
                />
              )}

              {tab === "rounds" && (
                <>
                  <div className="filters">
                    <label className="sr-only" htmlFor="round-state-filter">
                      Filter rounds by state
                    </label>
                    <select
                      id="round-state-filter"
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
                  </div>
                  <RoundsTable
                    rounds={data?.classic_rounds ?? []}
                    busyId={busyClassic}
                    asyncEnabled={data?.async_enabled}
                    onAggregate={requestAggregateClassic}
                    filter={stateFilter}
                  />
                </>
              )}

              {tab === "clients" && (
                <ClientsPanel
                  reputations={data?.reputations ?? {}}
                  incentives={data?.incentives ?? {}}
                  registeredClients={data?.registered_clients ?? []}
                  serverTime={data?.server_time}
                />
              )}

              {tab === "lora" && (
                <LoraPanel
                  baseModels={data?.lora_base_models ?? []}
                  adapterVersions={data?.lora_adapters ?? []}
                  rounds={data?.lora_rounds ?? []}
                  busyId={busyLora}
                  creating={creating}
                  onCreate={handleCreateLora}
                  onAggregate={requestAggregateLora}
                />
              )}

              {tab === "jobs" && (
                <JobsPanel
                  jobs={data?.jobs ?? []}
                  stats={data?.job_stats}
                  workerRunning={workerRunning}
                  onCreate={handleCreateJob}
                  onCancel={handleCancelJob}
                  creating={creatingJob}
                  cancellingId={cancellingJob}
                />
              )}

              {tab === "settings" && (
                <SettingsPanel
                  authRequired={authRequired}
                  onSaved={() => void refresh()}
                />
              )}
            </>
          )}
        </main>
      </div>

      <ConfirmDialog request={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}
