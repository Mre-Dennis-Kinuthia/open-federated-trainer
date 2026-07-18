import { useState } from "react";
import { getApiBase, getOperatorKey, setApiBase, setOperatorKey } from "../api";

type Props = {
  authRequired?: boolean;
  onSaved: () => void;
};

export function SettingsPanel({ authRequired, onSaved }: Props) {
  const [key, setKey] = useState(getOperatorKey());
  const [apiBase, setApiBaseInput] = useState(getApiBase());
  const [saved, setSaved] = useState(false);

  return (
    <div className="card">
      <div className="card-hd">
        <h2>Connection</h2>
      </div>
      <div className="card-bd">
        <p className="help">
          On Vercel the UI is static — set the coordinator API URL so requests
          do not hit this site&apos;s HTML. For the HA lab stack use{" "}
          <code>https://127.0.0.1:8443</code> (accept the self-signed cert in a
          tab first). Local Vite already proxies <code>/api</code>.
        </p>
        <div className="form-grid one">
          <div className="field">
            <label htmlFor="api-base">Coordinator API URL</label>
            <input
              id="api-base"
              type="url"
              autoComplete="off"
              value={apiBase}
              onChange={(e) => {
                setApiBaseInput(e.target.value);
                setSaved(false);
              }}
              placeholder="https://127.0.0.1:8443"
            />
          </div>
        </div>

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
              setApiBase(apiBase);
              setOperatorKey(key.trim());
              setSaved(true);
              onSaved();
            }}
          >
            Save connection
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setApiBaseInput("");
              setKey("");
              setApiBase("");
              setOperatorKey("");
              setSaved(true);
              onSaved();
            }}
          >
            Clear
          </button>
          {saved && (
            <span className="muted" role="status">
              API URL is stored in this browser. The operator key is
              session-only and sent as X-Operator-Key, never in the URL.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
