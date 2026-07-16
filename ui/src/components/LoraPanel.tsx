import { useState } from "react";
import type { CreateLoraPayload, LoraRound } from "../api";

type Props = {
  baseModels: string[];
  rounds: LoraRound[];
  busyId: number | null;
  creating: boolean;
  onCreate: (payload: CreateLoraPayload) => Promise<void>;
  onAggregate: (roundId: number) => void;
};

export function LoraPanel({
  baseModels,
  rounds,
  busyId,
  creating,
  onCreate,
  onAggregate,
}: Props) {
  const models = baseModels.length > 0 ? baseModels : ["tiny-llama"];
  const [form, setForm] = useState<CreateLoraPayload>({
    base_model_id: models[0],
    lora_r: 8,
    lora_alpha: 16,
    max_steps: 50,
    learning_rate: 0.0002,
    batch_size: 4,
  });

  return (
    <>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-hd">
          <h2>Create LoRA round</h2>
        </div>
        <div className="card-bd">
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              await onCreate(form);
            }}
          >
            <div className="form-grid">
              <div className="field">
                <label htmlFor="base_model_id">Base model</label>
                <select
                  id="base_model_id"
                  value={form.base_model_id}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, base_model_id: e.target.value }))
                  }
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="lora_r">LoRA r</label>
                <input
                  id="lora_r"
                  type="number"
                  min={1}
                  value={form.lora_r}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, lora_r: Number(e.target.value) }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="lora_alpha">LoRA alpha</label>
                <input
                  id="lora_alpha"
                  type="number"
                  min={1}
                  value={form.lora_alpha}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      lora_alpha: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="max_steps">Max steps</label>
                <input
                  id="max_steps"
                  type="number"
                  min={1}
                  value={form.max_steps}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      max_steps: Number(e.target.value),
                    }))
                  }
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
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      learning_rate: Number(e.target.value),
                    }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="batch_size">Batch size</label>
                <input
                  id="batch_size"
                  type="number"
                  min={1}
                  value={form.batch_size}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      batch_size: Number(e.target.value),
                    }))
                  }
                />
              </div>
            </div>
            <button className="btn primary" type="submit" disabled={creating}>
              {creating ? "Creating…" : "Create round"}
            </button>
          </form>
        </div>
      </div>

      <div className="list">
        {rounds.length === 0 ? (
          <div className="empty">There are no LoRA rounds yet.</div>
        ) : (
          rounds.map((r, idx) => (
            <div className="list-row" key={r.round_id}>
              <div>
                <div className="title">
                  LoRA #{r.round_id} · {r.base_model_id}
                </div>
                <div className="meta">
                  r={r.lora_r} α={r.lora_alpha} · {r.submission_count} adapters
                </div>
              </div>
              <div className={`status ${r.state.toLowerCase()}`}>
                <span className="sdot" />
                {r.state}
              </div>
              <div>
                <span className={`badge ${idx === 0 ? "primary" : "outline"}`}>
                  LoRA
                </span>
              </div>
              <div className="mono">{r.max_steps} steps</div>
              <div className="mono">{r.submission_count} up</div>
              <div>
                <button
                  className="btn"
                  disabled={
                    r.state === "CLOSED" ||
                    r.submission_count === 0 ||
                    busyId === r.round_id
                  }
                  onClick={() => onAggregate(r.round_id)}
                >
                  {busyId === r.round_id ? "…" : "Aggregate"}
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </>
  );
}
