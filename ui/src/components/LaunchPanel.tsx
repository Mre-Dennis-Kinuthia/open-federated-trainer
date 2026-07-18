import { useEffect, useState } from "react";
import type { LauncherStatus } from "../api";
import { StatusBadge } from "./StatusBadge";

type Props = {
  launcher?: LauncherStatus | null;
  models: string[];
  activeModel?: string;
  busyAction: string | null;
  onStartTrain: (opts: {
    count: number;
    modelId: string;
    modelModule?: string;
    datasetPreset: string;
    datasetPath?: string;
  }) => Promise<void>;
  onStartWorker: (opts: {
    jobTypes: string;
    datasetPreset: string;
    datasetPath?: string;
    enqueueSample: boolean;
  }) => Promise<void>;
  onStartDemo: (opts: {
    modelId: string;
    datasetPreset: string;
    trainClients: number;
  }) => Promise<void>;
  onStop: (id: string) => Promise<void>;
  onStopAll: () => void;
};

const CSV_PRESETS = new Set(["sample_private", "sample_tabular"]);

export function LaunchPanel({
  launcher,
  models,
  activeModel,
  busyAction,
  onStartTrain,
  onStartWorker,
  onStartDemo,
  onStop,
  onStopAll,
}: Props) {
  const [demoModelId, setDemoModelId] = useState(
    activeModel || models[0] || "simple_mlp"
  );
  const [demoPreset, setDemoPreset] = useState("sample_private");
  const [demoCount, setDemoCount] = useState(2);

  const [trainModelId, setTrainModelId] = useState(
    activeModel || models[0] || "simple_mlp"
  );
  const [trainModelModule, setTrainModelModule] = useState("");
  const [trainPreset, setTrainPreset] = useState("sample_private");
  const [trainPath, setTrainPath] = useState("");
  const [trainCount, setTrainCount] = useState(2);

  const [jobTypes, setJobTypes] = useState("inference,label,compute");
  const [workerPreset, setWorkerPreset] = useState("sample_private");
  const [workerPath, setWorkerPath] = useState("");

  useEffect(() => {
    if (activeModel) {
      setDemoModelId(activeModel);
      setTrainModelId(activeModel);
    }
  }, [activeModel]);

  const presets = launcher?.dataset_presets ?? [
    "none",
    "sample_private",
    "sample_tabular",
  ];
  const processes = launcher?.processes ?? [];
  const enabled = launcher?.enabled !== false;

  const demoCnnMismatch =
    demoModelId === "tiny_cnn" && CSV_PRESETS.has(demoPreset);
  const trainCnnMismatch =
    trainModelId === "tiny_cnn" && !trainPath && CSV_PRESETS.has(trainPreset);
  const trainCustomMissingModule =
    trainModelId === "custom" && !trainModelModule.trim();

  return (
    <>
      {!enabled && (
        <div className="banner error spaced" role="alert">
          Local launcher is disabled on this API. Set{" "}
          <code>ENABLE_LOCAL_LAUNCHER=true</code> on the coordinator (and mount{" "}
          <code>CLIENT_ROOT</code> in Docker) to start processes from this page.
          On the HA production profile, launcher is enabled on{" "}
          <code>coordinator-a</code>; compose <code>client-1</code>/
          <code>client-2</code> also train without using this panel.
        </div>
      )}

      <div className="card spaced">
        <div className="card-hd">
          <h2>One-click demo</h2>
        </div>
        <div className="card-bd">
          <p className="help">
            Demo mode uses the repository sample dataset and enqueues a sample
            science job. For real workloads use the train and worker controls
            below with your own dataset path.
          </p>
          {demoCnnMismatch && (
            <div className="banner warn" role="status">
              tiny_cnn expects image tensors; the CSV sample presets will fail.
              Choose simple_mlp or an image dataset.
            </div>
          )}
          <div className="form-grid">
            <div className="field">
              <label htmlFor="demo-model">Model</label>
              <select
                id="demo-model"
                value={demoModelId}
                onChange={(e) => setDemoModelId(e.target.value)}
              >
                {models
                  .filter((m) => m !== "custom")
                  .map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="demo-ds">Dataset preset</label>
              <select
                id="demo-ds"
                value={demoPreset}
                onChange={(e) => setDemoPreset(e.target.value)}
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
                value={demoCount}
                onChange={(e) => setDemoCount(Number(e.target.value) || 1)}
              />
            </div>
          </div>
          <div className="row-actions">
            <button
              type="button"
              className="btn primary"
              disabled={busyAction !== null || !enabled || demoCnnMismatch}
              onClick={() =>
                void onStartDemo({
                  modelId: demoModelId,
                  datasetPreset: demoPreset,
                  trainClients: demoCount,
                })
              }
            >
              {busyAction === "demo" ? "Starting…" : "Start demo"}
            </button>
            <button
              type="button"
              className="btn danger-outline"
              disabled={busyAction !== null || !processes.some((p) => p.running)}
              onClick={onStopAll}
            >
              {busyAction === "stop-all" ? "Stopping…" : "Stop all"}
            </button>
            <span className="muted">
              Running: {launcher?.running ?? 0}
              {launcher?.by_kind
                ? ` (train ${launcher.by_kind.train ?? 0}, worker ${launcher.by_kind.worker ?? 0})`
                : ""}
            </span>
          </div>
        </div>
      </div>

      <div className="card-grid spaced">
        <div className="card">
          <div className="card-hd">
            <h2>Train clients</h2>
          </div>
          <div className="card-bd">
            {trainCnnMismatch && (
              <div className="banner warn" role="status">
                tiny_cnn needs an image dataset; pick a dataset path with images
                or switch models.
              </div>
            )}
            <div className="form-grid two">
              <div className="field">
                <label htmlFor="train-model">Model</label>
                <select
                  id="train-model"
                  value={trainModelId}
                  onChange={(e) => setTrainModelId(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              {trainModelId === "custom" && (
                <div className="field">
                  <label htmlFor="train-module">
                    Custom trainer (MODEL_MODULE)
                  </label>
                  <input
                    id="train-module"
                    value={trainModelModule}
                    onChange={(e) => setTrainModelModule(e.target.value)}
                    placeholder="examples.custom_linear:CustomLinearTrainer"
                  />
                </div>
              )}
              <div className="field">
                <label htmlFor="train-ds">Dataset preset</label>
                <select
                  id="train-ds"
                  value={trainPreset}
                  onChange={(e) => setTrainPreset(e.target.value)}
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
              <div className="field">
                <label htmlFor="train-path">
                  Dataset path under client/ (overrides preset)
                </label>
                <input
                  id="train-path"
                  value={trainPath}
                  onChange={(e) => setTrainPath(e.target.value)}
                  placeholder="data/private/train.csv"
                />
              </div>
            </div>
            <button
              type="button"
              className="btn primary"
              disabled={
                busyAction !== null ||
                !enabled ||
                trainCustomMissingModule ||
                trainCnnMismatch
              }
              title={
                trainCustomMissingModule
                  ? "Custom model requires a MODEL_MODULE entrypoint"
                  : undefined
              }
              onClick={() =>
                void onStartTrain({
                  count: trainCount,
                  modelId: trainModelId,
                  modelModule: trainModelModule.trim() || undefined,
                  datasetPreset: trainPath ? "none" : trainPreset,
                  datasetPath: trainPath || undefined,
                })
              }
            >
              {busyAction === "train" ? "Starting…" : "Start train clients"}
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-hd">
            <h2>Job worker</h2>
          </div>
          <div className="card-bd">
            <div className="form-grid one">
              <div className="field">
                <label htmlFor="job-types">Job types, comma-separated</label>
                <input
                  id="job-types"
                  value={jobTypes}
                  onChange={(e) => setJobTypes(e.target.value)}
                  placeholder="inference,label,compute"
                />
              </div>
              <div className="field">
                <label htmlFor="worker-ds">Dataset preset (for label jobs)</label>
                <select
                  id="worker-ds"
                  value={workerPreset}
                  onChange={(e) => setWorkerPreset(e.target.value)}
                >
                  {presets.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="worker-path">
                  Dataset path under client/ (overrides preset)
                </label>
                <input
                  id="worker-path"
                  value={workerPath}
                  onChange={(e) => setWorkerPath(e.target.value)}
                  placeholder="data/private/labels.jsonl"
                />
              </div>
            </div>
            <button
              type="button"
              className="btn primary"
              disabled={busyAction !== null || !enabled}
              onClick={() =>
                void onStartWorker({
                  jobTypes,
                  datasetPreset: workerPath ? "none" : workerPreset,
                  datasetPath: workerPath || undefined,
                  enqueueSample: true,
                })
              }
            >
              {busyAction === "worker" ? "Starting…" : "Start worker + sample job"}
            </button>
          </div>
        </div>
      </div>

      {processes.length === 0 ? (
        <div className="list">
          <div className="empty">
            No local processes yet. Use the demo or launch controls above.
          </div>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <caption className="sr-only">
              Locally launched clients and workers with process status
            </caption>
            <thead>
              <tr>
                <th scope="col">Process</th>
                <th scope="col">Status</th>
                <th scope="col">Configuration</th>
                <th scope="col">PID</th>
                <th scope="col">Uptime</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {processes.map((p) => (
                <tr key={p.id}>
                  <td>
                    <span className="title">{p.name}</span>
                    <div className="meta">{p.kind}</div>
                  </td>
                  <td>
                    {p.running ? (
                      <StatusBadge kind="ok" label="Running" />
                    ) : (
                      <StatusBadge
                        kind={p.exit_code === 0 ? "neutral" : "danger"}
                        label={`Exited (${p.exit_code ?? "?"})`}
                      />
                    )}
                  </td>
                  <td className="cell-config">
                    <span className="mono">
                      {p.env_summary?.MODEL_ID || p.env_summary?.JOB_TYPES || "—"}
                    </span>
                    {p.env_summary?.DATASET_PATH && (
                      <div className="meta mono" title={p.env_summary.DATASET_PATH}>
                        {p.env_summary.DATASET_PATH.split("/").slice(-2).join("/")}
                      </div>
                    )}
                    {p.log_path && (
                      <div className="meta mono" title={p.log_path}>
                        log: {p.log_path.split("/").slice(-1)[0]}
                      </div>
                    )}
                  </td>
                  <td className="mono">{p.pid}</td>
                  <td className="mono">
                    {p.running && p.uptime_seconds != null
                      ? `${Math.round(p.uptime_seconds)}s`
                      : "—"}
                  </td>
                  <td className="cell-actions">
                    {p.running && (
                      <button
                        type="button"
                        className="btn"
                        disabled={busyAction !== null}
                        onClick={() => void onStop(p.id)}
                      >
                        {busyAction === `stop-${p.id}` ? "Stopping…" : "Stop"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
