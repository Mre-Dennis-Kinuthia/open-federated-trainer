import { useState } from "react";
import type { Job } from "../api";
import { StatusBadge, jobStatus } from "./StatusBadge";
import { PrivacyDisclosure } from "./PrivacyDisclosure";

type Props = {
  jobs: Job[];
  stats?: { total?: number; counts?: Record<string, number> };
  workerRunning: boolean;
  onCreate: (jobType: string, payload: Record<string, unknown>) => Promise<void>;
  onCancel: (jobId: string) => Promise<void>;
  creating: boolean;
  cancellingId: string | null;
};

type JobType = "inference" | "label" | "compute";

const CANCELLABLE_STATES = new Set(["QUEUED", "ASSIGNED", "FAILED"]);

export function JobsPanel({
  jobs,
  stats,
  workerRunning,
  onCreate,
  onCancel,
  creating,
  cancellingId,
}: Props) {
  const [jobType, setJobType] = useState<JobType>("inference");
  const [modelId, setModelId] = useState("");
  const [task, setTask] = useState("text-classification");
  const [inputs, setInputs] = useState("");
  const [candidateLabels, setCandidateLabels] = useState("");
  const [entrypoint, setEntrypoint] = useState(
    "examples.science_plugin:lennard_jones"
  );
  const [workUnit, setWorkUnit] = useState(
    '{\n  "positions": [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]],\n  "steps": 250,\n  "dt": 0.001\n}'
  );
  const [formError, setFormError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  function submit() {
    try {
      setFormError(null);
      if (jobType === "compute") {
        if (!entrypoint.trim().includes(":")) {
          throw new Error("Entrypoint must be module.path:function");
        }
        const parsed = JSON.parse(workUnit) as Record<string, unknown>;
        void onCreate("compute", { entrypoint: entrypoint.trim(), work_unit: parsed });
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
      <div className="card spaced">
        <div className="card-hd">
          <h2>Enqueue workload</h2>
        </div>
        <div className="card-bd">
          <p className="help">
            Inference and labeling run an actual Transformers model on a worker.
            Compute invokes an operator-installed, allowlisted Python plugin.
          </p>
          <PrivacyDisclosure
            workload={jobType}
            compact
            className="privacy-inline"
          />
          {!workerRunning && (
            <div className="banner warn" role="status">
              No running job worker detected. Jobs will stay queued until a
              worker with matching job types starts (see Launch).
            </div>
          )}
          {formError && (
            <div className="banner error" role="alert">
              {formError}
            </div>
          )}
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
                    placeholder="examples.science_plugin:lennard_jones"
                  />
                </div>
                <div className="field span-2">
                  <label htmlFor="work-unit">Work unit JSON</label>
                  <textarea
                    id="work-unit"
                    rows={5}
                    className="mono-input"
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
                <div className="field span-2">
                  <label htmlFor="job-inputs">
                    Inputs, one per line (blank keeps private data on the worker
                    via DATASET_PATH)
                  </label>
                  <textarea
                    id="job-inputs"
                    rows={3}
                    value={inputs}
                    onChange={(event) => setInputs(event.target.value)}
                    placeholder="Leave blank to keep private inputs on the worker"
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
          <div className="row-actions">
            <button
              className="btn primary"
              disabled={creating}
              type="button"
              onClick={submit}
            >
              {creating ? "Enqueuing…" : `Enqueue ${jobType}`}
            </button>
            <span className="muted">Queue total: {stats?.total ?? 0}</span>
          </div>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="list">
          <div className="empty">No jobs yet.</div>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <caption className="sr-only">
              Job queue with state, assignment, and actions
            </caption>
            <thead>
              <tr>
                <th scope="col">Job</th>
                <th scope="col">State</th>
                <th scope="col">Type</th>
                <th scope="col">Assigned to</th>
                <th scope="col">Result</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const status = jobStatus(job.state);
                const expanded = expandedId === job.job_id;
                const cancellable = CANCELLABLE_STATES.has(job.state);
                return [
                  <tr key={job.job_id}>
                    <td>
                      <span className="title mono">{job.job_id.slice(0, 8)}</span>
                      {job.error && (
                        <div className="meta error-text">{job.error}</div>
                      )}
                    </td>
                    <td>
                      <StatusBadge kind={status.kind} label={status.label} />
                    </td>
                    <td>{job.job_type}</td>
                    <td className="mono">{job.assigned_client || "—"}</td>
                    <td className="mono">{job.result ? "Available" : "—"}</td>
                    <td className="cell-actions">
                      <button
                        type="button"
                        className="btn"
                        aria-expanded={expanded}
                        onClick={() =>
                          setExpandedId(expanded ? null : job.job_id)
                        }
                      >
                        {expanded ? "Hide" : "Details"}
                      </button>
                      {cancellable && (
                        <button
                          type="button"
                          className="btn danger-outline"
                          disabled={cancellingId === job.job_id}
                          onClick={() => void onCancel(job.job_id)}
                        >
                          {cancellingId === job.job_id ? "Cancelling…" : "Cancel"}
                        </button>
                      )}
                    </td>
                  </tr>,
                  expanded ? (
                    <tr key={`${job.job_id}-details`} className="details-row">
                      <td colSpan={6}>
                        <div className="details-grid">
                          <div>
                            <h3>Payload</h3>
                            {job.payload &&
                            typeof job.payload === "object" &&
                            "redacted" in job.payload &&
                            (job.payload as { redacted?: boolean }).redacted ? (
                              <p role="status">
                                Redacted — set the operator key in Settings to
                                view job payloads.
                              </p>
                            ) : (
                              <pre>
                                {JSON.stringify(job.payload ?? {}, null, 2)}
                              </pre>
                            )}
                          </div>
                          <div>
                            <h3>Result</h3>
                            {job.result &&
                            typeof job.result === "object" &&
                            "redacted" in job.result &&
                            (job.result as { redacted?: boolean }).redacted ? (
                              <p role="status">
                                Redacted — set the operator key in Settings to
                                view results.
                              </p>
                            ) : (
                              <pre>
                                {job.result
                                  ? JSON.stringify(job.result, null, 2)
                                  : "No result yet."}
                              </pre>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  ) : null,
                ];
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
