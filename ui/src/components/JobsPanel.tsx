import { useState } from "react";

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

type JobType = "inference" | "label" | "compute";

export function JobsPanel({ jobs, stats, onCreate, creating }: Props) {
  const [jobType, setJobType] = useState<JobType>("inference");
  const [modelId, setModelId] = useState("");
  const [task, setTask] = useState("text-classification");
  const [inputs, setInputs] = useState("");
  const [candidateLabels, setCandidateLabels] = useState("");
  const [entrypoint, setEntrypoint] = useState(
    "examples.science_plugin:lennard_jones"
  );
  const [workUnit, setWorkUnit] = useState(
    '{"positions":[[0,0,0],[1.2,0,0],[0,1.2,0]],"steps":250,"dt":0.001}'
  );
  const [formError, setFormError] = useState<string | null>(null);

  function submit() {
    try {
      setFormError(null);
      if (jobType === "compute") {
        const parsed = JSON.parse(workUnit) as Record<string, unknown>;
        void onCreate("compute", { entrypoint, work_unit: parsed });
        return;
      }
      if (!modelId.trim()) {
        throw new Error("A local or Hugging Face model ID is required");
      }
      const payload: Record<string, unknown> = {
        model_id: modelId.trim(),
        task,
      };
      const inputValues = inputs
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean);
      if (inputValues.length) payload.inputs = inputValues;
      else {
        payload.offset = 0;
        payload.limit = 16;
      }
      if (jobType === "label" && candidateLabels.trim()) {
        payload.candidate_labels = candidateLabels
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean);
      }
      void onCreate(jobType, payload);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Invalid job");
    }
  }

  return (
    <>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-hd">
          <h2>Enqueue real workload</h2>
        </div>
        <div className="card-bd">
          <p className="help">
            Inference and labeling use an actual Transformers model on the worker.
            Compute invokes an operator-installed, allowlisted Python plugin.
          </p>
          {formError && <div className="banner error">{formError}</div>}
          <div className="form-grid">
            <div className="field">
              <label htmlFor="job-type">Type</label>
              <select
                id="job-type"
                value={jobType}
                onChange={(event) => setJobType(event.target.value as JobType)}
              >
                <option value="inference">Inference</option>
                <option value="label">Auto-label</option>
                <option value="compute">Science compute</option>
              </select>
            </div>

            {jobType === "compute" ? (
              <>
                <div className="field">
                  <label htmlFor="entrypoint">Allowlisted entrypoint</label>
                  <input
                    id="entrypoint"
                    value={entrypoint}
                    onChange={(event) => setEntrypoint(event.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="work-unit">Work unit JSON</label>
                  <input
                    id="work-unit"
                    value={workUnit}
                    onChange={(event) => setWorkUnit(event.target.value)}
                  />
                </div>
              </>
            ) : (
              <>
                <div className="field">
                  <label htmlFor="inference-model">Model ID or local path</label>
                  <input
                    id="inference-model"
                    value={modelId}
                    onChange={(event) => setModelId(event.target.value)}
                    placeholder="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                  />
                </div>
                <div className="field">
                  <label htmlFor="inference-task">Transformers task</label>
                  <select
                    id="inference-task"
                    value={task}
                    onChange={(event) => setTask(event.target.value)}
                  >
                    <option value="text-classification">Text classification</option>
                    <option value="text-generation">Text generation</option>
                    <option value="token-classification">Token classification</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="job-inputs">
                    Inputs sent via coordinator (blank uses worker DATASET_PATH)
                  </label>
                  <input
                    id="job-inputs"
                    value={inputs}
                    onChange={(event) => setInputs(event.target.value)}
                    placeholder="Leave blank to keep private inputs on worker"
                  />
                </div>
                {jobType === "label" && (
                  <div className="field">
                    <label htmlFor="candidate-labels">
                      Zero-shot labels, comma-separated (optional)
                    </label>
                    <input
                      id="candidate-labels"
                      value={candidateLabels}
                      onChange={(event) => setCandidateLabels(event.target.value)}
                      placeholder="support,billing,technical"
                    />
                  </div>
                )}
              </>
            )}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              className="btn primary"
              disabled={creating}
              type="button"
              onClick={submit}
            >
              {creating ? "Enqueuing…" : `Enqueue ${jobType}`}
            </button>
            <span className="mono">queue total {stats?.total ?? 0}</span>
          </div>
        </div>
      </div>

      <div className="list">
        {jobs.length === 0 ? (
          <div className="empty">No jobs yet.</div>
        ) : (
          jobs.map((job, index) => (
            <div className="list-row" key={job.job_id}>
              <div>
                <div className="title">
                  {job.job_type} · {job.job_id.slice(0, 8)}
                </div>
                <div className="meta">
                  {job.assigned_client || "unassigned"}
                  {job.error ? ` · ${job.error}` : ""}
                </div>
              </div>
              <div className={`status ${job.state.toLowerCase()}`}>
                <span className="sdot" />
                {job.state}
              </div>
              <div>
                <span className={`badge ${index === 0 ? "primary" : "outline"}`}>
                  {job.job_type}
                </span>
              </div>
              <div className="mono">{job.assigned_client || "—"}</div>
              <div className="mono">{job.result ? "has result" : "—"}</div>
              <div className="mono">{job.job_id.slice(0, 7)}</div>
            </div>
          ))
        )}
      </div>
    </>
  );
}
