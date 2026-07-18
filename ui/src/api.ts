export type ClassicRound = {
  round_id: number;
  model_version: string | null;
  state: string;
  assigned_clients: string[];
  updates_received: string[];
  total_clients: number;
  total_updates: number;
  metrics?: Record<string, unknown>;
};

export type Reputation = {
  client_id: string;
  reputation_score: number;
  updates_submitted: number;
  updates_accepted: number;
  updates_rejected: number;
  acceptance_rate: number;
  dropout_rate: number;
  last_seen: number;
};

export type Incentive = {
  client_id: string;
  total_tokens_earned: number;
  current_balance: number;
  speed_bonuses: number;
  consistency_bonuses: number;
  total_rewards: number;
};

export type LoraRound = {
  round_id: number;
  base_model_id: string;
  adapter_version: string | null;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  max_steps: number;
  learning_rate: number;
  batch_size: number;
  state: string;
  created_at: string;
  submission_count: number;
  submitters: string[];
};

export type Job = {
  job_id: string;
  job_type: string;
  state: string;
  payload?: Record<string, unknown>;
  result?: Record<string, unknown>;
  assigned_client?: string | null;
  error?: string | null;
  created_at?: number;
  attempts?: number;
};

export type Process = {
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

export type LauncherStatus = {
  enabled?: boolean;
  running?: number;
  total?: number;
  by_kind?: { train?: number; worker?: number };
  processes?: Process[];
  dataset_presets?: string[];
};

export type Overview = {
  version: string;
  async_enabled: boolean;
  operator_auth_required?: boolean;
  server_time?: number;
  global: {
    total_clients_seen?: number;
    total_failed_updates?: number;
    total_rounds?: number;
  };
  latest_round: Record<string, unknown>;
  classic_rounds: ClassicRound[];
  reputations: Record<string, Reputation>;
  incentives: Record<string, Incentive>;
  lora_base_models: string[];
  lora_adapters?: string[];
  lora_rounds: LoraRound[];
  registered_clients: string[];
  jobs?: Job[];
  job_stats?: {
    total?: number;
    completed?: number;
    counts?: Record<string, number>;
  };
  classic_models?: Record<string, unknown>;
  active_model?: { model_id?: string; model_config?: Record<string, unknown> };
  launcher?: LauncherStatus;
};

export type CreateLoraPayload = {
  base_model_id: string;
  adapter_version?: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout?: number;
  target_modules?: string[];
  max_steps: number;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps?: number;
  warmup_steps?: number;
  max_seq_length?: number;
};

// Dev: Vite proxies /api → coordinator. Prod: same-origin (nginx /ui),
// VITE_API_BASE at build time, or a runtime override in Settings (Vercel).
const BUILD_API_BASE = (
  (import.meta.env.VITE_API_BASE as string | undefined)?.trim() ||
  (import.meta.env.DEV ? "/api" : "")
).replace(/\/$/, "");
const API_BASE_STORAGE = "fed-compute.api-base";
const OPERATOR_KEY_STORAGE = "fed-compute.operator-key";
const BUILD_TIME_KEY =
  (import.meta.env.VITE_OPERATOR_API_KEY as string | undefined)?.trim() || "";

export function getApiBase(): string {
  try {
    const stored = window.localStorage.getItem(API_BASE_STORAGE);
    if (stored !== null) return stored.replace(/\/$/, "");
  } catch {
    /* storage unavailable */
  }
  return BUILD_API_BASE;
}

export function setApiBase(url: string): void {
  try {
    const cleaned = url.trim().replace(/\/$/, "");
    if (cleaned) window.localStorage.setItem(API_BASE_STORAGE, cleaned);
    else window.localStorage.removeItem(API_BASE_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

export function getOperatorKey(): string {
  try {
    const stored = window.sessionStorage.getItem(OPERATOR_KEY_STORAGE);
    if (stored !== null) return stored;
  } catch {
    /* storage unavailable */
  }
  return BUILD_TIME_KEY;
}

export function setOperatorKey(key: string): void {
  try {
    if (key) window.sessionStorage.setItem(OPERATOR_KEY_STORAGE, key);
    else window.sessionStorage.removeItem(OPERATOR_KEY_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const operatorKey = getOperatorKey();
  if (operatorKey) headers["X-Operator-Key"] = operatorKey;

  const base = getApiBase();
  const res = await fetch(`${base}${path}`, { ...init, headers });
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    const snippet = (await res.text()).slice(0, 80);
    if (snippet.trimStart().startsWith("<!") || snippet.includes("<html")) {
      throw new ApiError(
        res.status,
        base
          ? `Coordinator returned HTML instead of JSON (${res.status}). Check the API URL in Settings.`
          : "No coordinator URL set. Open Settings and set the API base (e.g. https://127.0.0.1:8443)."
      );
    }
    throw new ApiError(
      res.status,
      `Unexpected non-JSON response (${res.status}). Set the coordinator API URL in Settings.`
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(
      res.status,
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
  }
  return res.json() as Promise<T>;
}

export function fetchOverview(
  limit = 25,
  signal?: AbortSignal
): Promise<Overview> {
  return request<Overview>(`/dashboard/overview?limit=${limit}`, { signal });
}

export type ActivityNode = {
  lat: number;
  lng: number;
  city?: string | null;
  country?: string | null;
  last_seen: number;
  online: boolean;
};

export type Activity = {
  server_time: number;
  nodes: ActivityNode[];
  online_count: number;
};

export function fetchActivity(signal?: AbortSignal): Promise<Activity> {
  return request<Activity>(`/dashboard/activity`, { signal });
}

export function aggregateClassicRound(roundId: number): Promise<unknown> {
  return request(`/aggregate/${roundId}`);
}

export function createLoraRound(
  payload: CreateLoraPayload
): Promise<{ round_id: number; state: string }> {
  return request(`/rounds/create`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function aggregateLoraRound(roundId: number): Promise<unknown> {
  return request(`/rounds/${roundId}/aggregate`, {
    method: "POST",
    body: JSON.stringify({ round_id: roundId, weight_by_samples: true }),
  });
}

export function createJob(
  jobType: string,
  payload: Record<string, unknown>
): Promise<unknown> {
  return request(`/jobs`, {
    method: "POST",
    body: JSON.stringify({ job_type: jobType, payload }),
  });
}

export function cancelJob(jobId: string): Promise<unknown> {
  return request(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

export function setActiveModel(
  modelId: string,
  modelConfig?: Record<string, unknown>
): Promise<unknown> {
  return request(`/models/active`, {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      ...(modelConfig ? { model_config: modelConfig } : {}),
    }),
  });
}

export type LaunchPayload = {
  kind: "train" | "worker";
  count?: number;
  model_id?: string;
  model_module?: string;
  dataset_preset?: string;
  dataset_path?: string;
  job_types?: string;
  set_active_model?: boolean;
  enqueue_sample_job?: boolean;
};

export function launchProcess(payload: LaunchPayload): Promise<unknown> {
  return request(`/launch`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function launchDemo(payload: {
  model_id?: string;
  dataset_preset?: string;
  train_clients?: number;
  start_worker?: boolean;
  enqueue_sample_job?: boolean;
}): Promise<unknown> {
  return request(`/launch/demo`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stopLaunch(processId: string): Promise<unknown> {
  return request(`/launch/${processId}/stop`, { method: "POST" });
}

export function stopAllLaunch(kind?: string): Promise<unknown> {
  const q = kind ? `?kind=${encodeURIComponent(kind)}` : "";
  return request(`/launch/stop-all${q}`, { method: "POST" });
}
