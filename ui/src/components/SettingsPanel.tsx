import { useState } from "react";
import { getOperatorKey, setOperatorKey } from "../api";

type Props = {
  authRequired?: boolean;
  onSaved: () => void;
};

export function SettingsPanel({ authRequired, onSaved }: Props) {
  const [key, setKey] = useState(getOperatorKey());
  const [saved, setSaved] = useState(false);

  return (
    <div className="card">
      <div className="card-hd">
        <h2>Operator access</h2>
      </div>
      <div className="card-bd">
        <p className="help">
          {authRequired
            ? "This coordinator requires an operator key for privileged actions (aggregate, launch, jobs, model changes)."
            : "This coordinator does not currently require an operator key. Set OPERATOR_API_KEY on the coordinator to lock privileged actions."}
        </p>
        <div className="form-grid two">
          <div className="field">
            <label htmlFor="operator-key">Operator key</label>
            <input
              id="operator-key"
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => {
                setKey(e.target.value);
                setSaved(false);
              }}
              placeholder="Paste OPERATOR_API_KEY"
            />
          </div>
        </div>
        <div className="row-actions">
          <button
            type="button"
            className="btn primary"
            onClick={() => {
              setOperatorKey(key.trim());
              setSaved(true);
              onSaved();
            }}
          >
            Save key
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setKey("");
              setOperatorKey("");
              setSaved(true);
              onSaved();
            }}
          >
            Clear
          </button>
          {saved && (
            <span className="muted" role="status">
              Saved for this browser session. The key is sent as an
              X-Operator-Key header, never in the URL.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
