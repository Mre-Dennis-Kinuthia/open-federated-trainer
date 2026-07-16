import { useEffect, useState } from "react";

type Process = {
  id: string;
  kind: string;
  name: string;
  pid: number;
  running: boolean;
  exit_code?: number | null;
  env_summary?: Record<string, string>;
  uptime_seconds?: number | null;
  log_path?: string | null;
};

type LauncherStatus = {
  enabled?: boolean;
  running?: number;
  total?: number;
  by_kind?: { train?: number; worker?: number };
  processes?: Process[];
  dataset_presets?: string[];
};

type Props = {
  launcher?: LauncherStatus | null;
  models: string[];
  activeModel?: string;
  busy: boolean;
  onStartTrain: (opts: {
    count: number;
    modelId: string;
    datasetPreset: string;
  }) => Promise<void>;
  onStartWorker: (opts: {
    jobTypes: string;
    datasetPreset: string;
    enqueueSample: boolean;
  }) => Promise<void>;
  onStartDemo: (opts: {
    modelId: string;
    datasetPreset: string;
    trainClients: number;
  }) => Promise<void>;
  onStop: (id: string) => Promise<void>;
  onStopAll: () => Promise<void>;
};

export function LaunchPanel({
  launcher,
  models,
  activeModel,
  busy,
  onStartTrain,
  onStartWorker,
  onStartDemo,
  onStop,
  onStopAll,
}: Props) {
  const [modelId, setModelId] = useState(activeModel || models[0] || "tiny_cnn");
  const [datasetPreset, setDatasetPreset] = useState("sample_private");
  const [trainCount, setTrainCount] = useState(2);
  const [jobTypes, setJobTypes] = useState("inference,label,compute");

  useEffect(() => {
    if (activeModel) setModelId(activeModel);
  }, [activeModel]);

  const presets = launcher?.dataset_presets ?? ["none", "sample_private", "sample_tabular"];
  const processes = launcher?.processes ?? [];
  const enabled = launcher?.enabled !== false;

  return (
    <>
      {!enabled && (
        <div className="banner error" style={{ marginBottom: 16 }}>
          Local launcher is disabled. Set ENABLE_LOCAL_LAUNCHER=true on the coordinator.
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-hd">
          <h2>One-click demo</h2>
        </div>
        <div className="card-bd">
          <p className="help">
            Sets the active model, starts train clients on a private sample dataset, starts a job
            worker, and enqueues a compute job.
          </p>
          <div className="form-grid">
            <div className="field">
              <label htmlFor="demo-model">Model</label>
              <select
                id="demo-model"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="demo-ds">Dataset</label>
              <select
                id="demo-ds"
                value={datasetPreset}
                onChange={(e) => setDatasetPreset(e.target.value)}
              >
                {presets.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="demo-n">Train clients</label>
              <input
                id="demo-n"
                type="number"
                min={1}
                max={8}
                value={trainCount}
                onChange={(e) => setTrainCount(Number(e.target.value) || 1)}
              />
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn primary"
              disabled={busy || !enabled}
              onClick={() =>
                void onStartDemo({
                  modelId,
                  datasetPreset,
                  trainClients: trainCount,
                })
              }
            >
              Start from UI
            </button>
            <button
              type="button"
              className="btn"
              disabled={busy || !processes.some((p) => p.running)}
              onClick={() => void onStopAll()}
            >
              Stop all
            </button>
            <span className="mono" style={{ alignSelf: "center" }}>
              running {launcher?.running ?? 0}
              {launcher?.by_kind
                ? ` · train ${launcher.by_kind.train ?? 0} · worker ${launcher.by_kind.worker ?? 0}`
                : ""}
            </span>
          </div>
        </div>
      </div>

      <div className="card-grid" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="card-hd">
            <h2>Train clients</h2>
          </div>
          <div className="card-bd">
            <div className="form-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <div className="field">
                <label htmlFor="train-model">Model</label>
                <select
                  id="train-model"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="train-ds">Dataset</label>
                <select
                  id="train-ds"
                  value={datasetPreset}
                  onChange={(e) => setDatasetPreset(e.target.value)}
                >
                  {presets.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="train-n">Count</label>
                <input
                  id="train-n"
                  type="number"
                  min={1}
                  max={8}
                  value={trainCount}
                  onChange={(e) => setTrainCount(Number(e.target.value) || 1)}
                />
              </div>
            </div>
            <button
              type="button"
              className="btn primary"
              disabled={busy || !enabled}
              onClick={() =>
                void onStartTrain({
                  count: trainCount,
                  modelId,
                  datasetPreset,
                })
              }
            >
              Start train clients
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-hd">
            <h2>Job worker</h2>
          </div>
          <div className="card-bd">
            <div className="form-grid" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label htmlFor="job-types">Job types</label>
                <input
                  id="job-types"
                  value={jobTypes}
                  onChange={(e) => setJobTypes(e.target.value)}
                  placeholder="inference,label,compute"
                />
              </div>
              <div className="field">
                <label htmlFor="worker-ds">Dataset (for label jobs)</label>
                <select
                  id="worker-ds"
                  value={datasetPreset}
                  onChange={(e) => setDatasetPreset(e.target.value)}
                >
                  {presets.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <button
              type="button"
              className="btn primary"
              disabled={busy || !enabled}
              onClick={() =>
                void onStartWorker({
                  jobTypes,
                  datasetPreset,
                  enqueueSample: true,
                })
              }
            >
              Start worker + sample job
            </button>
          </div>
        </div>
      </div>

      <div className="list">
        {processes.length === 0 ? (
          <div className="empty">No local processes yet. Use Start from UI above.</div>
        ) : (
          processes.map((p) => (
            <div className="list-row" key={p.id}>
              <div>
                <div className="title">
                  {p.kind} · {p.name}
                </div>
                <div className="meta">
                  {p.env_summary?.MODEL_ID || p.env_summary?.JOB_TYPES || "—"}
                  {p.env_summary?.DATASET_PATH
                    ? ` · ${p.env_summary.DATASET_PATH.split("/").slice(-2).join("/")}`
                    : ""}
                </div>
              </div>
              <div className={`status ${p.running ? "collecting" : "ready"}`}>
                <span className="sdot" />
                {p.running ? "RUNNING" : `EXIT ${p.exit_code ?? "?"}`}
              </div>
              <div>
                <span className={`badge ${p.kind === "train" ? "primary" : "outline"}`}>
                  {p.kind}
                </span>
              </div>
              <div className="mono">pid {p.pid}</div>
              <div className="mono">
                {p.running && p.uptime_seconds != null ? `${Math.round(p.uptime_seconds)}s` : "—"}
              </div>
              <div>
                {p.running ? (
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() => void onStop(p.id)}
                  >
                    Stop
                  </button>
                ) : (
                  <span className="mono">stopped</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </>
  );
}
