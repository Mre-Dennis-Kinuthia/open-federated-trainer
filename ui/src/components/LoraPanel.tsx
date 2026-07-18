import { useState } from "react";
import type { CreateLoraPayload, LoraRound } from "../api";
import { StatusBadge, roundStatus } from "./StatusBadge";

type Props = {
  baseModels: string[];
  adapterVersions: string[];
  rounds: LoraRound[];
  busyId: number | null;
  creating: boolean;
  onCreate: (payload: CreateLoraPayload) => Promise<void>;
  onAggregate: (roundId: number) => void;
};

export function LoraPanel({
  baseModels,
  adapterVersions,
  rounds,
  busyId,
  creating,
  onCreate,
  onAggregate,
}: Props) {
  const models = baseModels.length > 0 ? baseModels : ["tiny-llama"];
  const [form, setForm] = useState<CreateLoraPayload>({
    base_model_id: models[0],
    adapter_version: "",
    lora_r: 8,
    lora_alpha: 16,
    max_steps: 50,
    learning_rate: 0.0002,
    batch_size: 4,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advanced, setAdvanced] = useState({
    lora_dropout: 0.1,
    target_modules: "q_proj,v_proj",
    gradient_accumulation_steps: 4,
    warmup_steps: 10,
    max_seq_length: 512,
  });

  function set<K extends keyof CreateLoraPayload>(
    key: K,
    value: CreateLoraPayload[K]
  ) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  return (
    <>
      <div className="card spaced">
        <div className="card-hd">
          <h2>Create LoRA round</h2>
        </div>
        <div className="card-bd">
          <p className="help">
            Creates a round that LoRA participants can join. Participants run{" "}
            <code>client/src/lora_client.py</code> with a configured
            DATASET_PATH; the local launcher does not start LoRA clients.
          </p>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const payload: CreateLoraPayload = {
                ...form,
                adapter_version: form.adapter_version || undefined,
              };
              if (showAdvanced) {
                payload.lora_dropout = advanced.lora_dropout;
                payload.target_modules = advanced.target_modules
                  .split(",")
                  .map((m) => m.trim())
                  .filter(Boolean);
                payload.gradient_accumulation_steps =
                  advanced.gradient_accumulation_steps;
                payload.warmup_steps = advanced.warmup_steps;
                payload.max_seq_length = advanced.max_seq_length;
              }
              await onCreate(payload);
            }}
          >
            <div className="form-grid">
              <div className="field">
                <label htmlFor="base_model_id">Base model</label>
                <select
                  id="base_model_id"
                  value={form.base_model_id}
                  onChange={(e) => set("base_model_id", e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="adapter_version">Continue adapter</label>
                <select
                  id="adapter_version"
                  value={form.adapter_version ?? ""}
                  onChange={(e) => set("adapter_version", e.target.value)}
                >
                  <option value="">Start fresh</option>
                  {adapterVersions.map((version) => (
                    <option key={version} value={version}>
                      {version}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="lora_r">LoRA rank (r)</label>
                <input
                  id="lora_r"
                  type="number"
                  min={1}
                  value={form.lora_r}
                  onChange={(e) => set("lora_r", Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="lora_alpha">LoRA alpha</label>
                <input
                  id="lora_alpha"
                  type="number"
                  min={1}
                  value={form.lora_alpha}
                  onChange={(e) => set("lora_alpha", Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="max_steps">Max steps</label>
                <input
                  id="max_steps"
                  type="number"
                  min={1}
                  value={form.max_steps}
                  onChange={(e) => set("max_steps", Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="learning_rate">Learning rate</label>
                <input
                  id="learning_rate"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.learning_rate}
                  onChange={(e) => set("learning_rate", Number(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="batch_size">Batch size</label>
                <input
                  id="batch_size"
                  type="number"
                  min={1}
                  value={form.batch_size}
                  onChange={(e) => set("batch_size", Number(e.target.value))}
                />
              </div>
            </div>

            <button
              type="button"
              className="btn ghost"
              aria-expanded={showAdvanced}
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? "Hide advanced options" : "Show advanced options"}
            </button>

            {showAdvanced && (
              <div className="form-grid" style={{ marginTop: 12 }}>
                <div className="field">
                  <label htmlFor="lora_dropout">LoRA dropout</label>
                  <input
                    id="lora_dropout"
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={advanced.lora_dropout}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        lora_dropout: Number(e.target.value),
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="target_modules">
                    Target modules, comma-separated
                  </label>
                  <input
                    id="target_modules"
                    value={advanced.target_modules}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        target_modules: e.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="grad_accum">Gradient accumulation steps</label>
                  <input
                    id="grad_accum"
                    type="number"
                    min={1}
                    value={advanced.gradient_accumulation_steps}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        gradient_accumulation_steps: Number(e.target.value),
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="warmup_steps">Warmup steps</label>
                  <input
                    id="warmup_steps"
                    type="number"
                    min={0}
                    value={advanced.warmup_steps}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        warmup_steps: Number(e.target.value),
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="max_seq_length">Max sequence length</label>
                  <input
                    id="max_seq_length"
                    type="number"
                    min={16}
                    value={advanced.max_seq_length}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        max_seq_length: Number(e.target.value),
                      }))
                    }
                  />
                </div>
              </div>
            )}

            <div className="row-actions">
              <button className="btn primary" type="submit" disabled={creating}>
                {creating ? "Creating…" : "Create round"}
              </button>
            </div>
          </form>
        </div>
      </div>

      {rounds.length === 0 ? (
        <div className="list">
          <div className="empty">There are no LoRA rounds yet.</div>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <caption className="sr-only">
              LoRA fine-tuning rounds with configuration and submissions
            </caption>
            <thead>
              <tr>
                <th scope="col">Round</th>
                <th scope="col">State</th>
                <th scope="col">Base model</th>
                <th scope="col">Config</th>
                <th scope="col">Adapters</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rounds.map((r) => {
                const status = roundStatus(r.state);
                return (
                  <tr key={r.round_id}>
                    <td>
                      <span className="title">#{r.round_id}</span>
                    </td>
                    <td>
                      <StatusBadge kind={status.kind} label={status.label} />
                    </td>
                    <td>{r.base_model_id}</td>
                    <td className="mono">
                      r={r.lora_r} α={r.lora_alpha} · {r.max_steps} steps
                    </td>
                    <td className="mono">{r.submission_count}</td>
                    <td className="cell-actions">
                      <button
                        type="button"
                        className="btn"
                        disabled={
                          r.state === "CLOSED" ||
                          r.submission_count === 0 ||
                          busyId === r.round_id
                        }
                        onClick={() => onAggregate(r.round_id)}
                      >
                        {busyId === r.round_id ? "Aggregating…" : "Aggregate"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
