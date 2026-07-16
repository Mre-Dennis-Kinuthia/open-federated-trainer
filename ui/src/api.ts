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

export type Overview = {
  version: string;
  async_enabled: boolean;
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
  lora_rounds: LoraRound[];
  registered_clients: string[];
  jobs?: Array<{
    job_id: string;
    job_type: string;
    state: string;
    assigned_client?: string | null;
    error?: string | null;
    result?: Record<string, unknown>;
  }>;
  job_stats?: { total?: number; counts?: Record<string, number> };
  classic_models?: Record<string, unknown>;
  active_model?: { model_id?: string; model_config?: Record<string, unknown> };
  launcher?: {
    enabled?: boolean;
    running?: number;
    total?: number;
    by_kind?: { train?: number; worker?: number };
    processes?: Array<{
      id: string;
      kind: string;
      name: string;
      pid: number;
      running: boolean;
      exit_code?: number | null;
      env_summary?: Record<string, string>;
      uptime_seconds?: number | null;
      log_path?: string | null;
    }>;
    dataset_presets?: string[];
  };
};

export type CreateLoraPayload = {
  base_model_id: string;
  lora_r: number;
  lora_alpha: number;
  max_steps: number;
  learning_rate: number;
  batch_size: number;
};

const API_BASE = import.meta.env.DEV ? "/api" : "";
const OPERATOR_KEY = (import.meta.env.VITE_OPERATOR_API_KEY as string | undefined)?.trim() || "";

function operatorQuery(): string {
  return OPERATOR_KEY ? `?operator_key=${encodeURIComponent(OPERATOR_KEY)}` : "";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export function fetchOverview(limit = 25): Promise<Overview> {
  return request<Overview>(`/dashboard/overview?limit=${limit}`);
}

export function aggregateClassicRound(roundId: number): Promise<unknown> {
  return request(`/aggregate/${roundId}${operatorQuery()}`);
}

export function createLoraRound(
  payload: CreateLoraPayload
): Promise<{ round_id: number; state: string }> {
  return request(`/rounds/create${operatorQuery()}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function aggregateLoraRound(roundId: number): Promise<unknown> {
  return request(`/rounds/${roundId}/aggregate${operatorQuery()}`, {
    method: "POST",
    body: JSON.stringify({ round_id: roundId, weight_by_samples: true }),
  });
}

export function createJob(
  jobType: string,
  payload: Record<string, unknown>
): Promise<unknown> {
  return request(`/jobs${operatorQuery()}`, {
    method: "POST",
    body: JSON.stringify({ job_type: jobType, payload }),
  });
}

export function setActiveModel(
  modelId: string,
  modelConfig?: Record<string, unknown>
): Promise<unknown> {
  return request(`/models/active${operatorQuery()}`, {
    method: "POST",
    body: JSON.stringify({ model_id: modelId, model_config: modelConfig ?? {} }),
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
  return request(`/launch${operatorQuery()}`, {
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
  return request(`/launch/demo${operatorQuery()}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function stopLaunch(processId: string): Promise<unknown> {
  return request(`/launch/${processId}/stop${operatorQuery()}`, {
    method: "POST",
  });
}

export function stopAllLaunch(kind?: string): Promise<unknown> {
  const params = new URLSearchParams();
  if (OPERATOR_KEY) params.set("operator_key", OPERATOR_KEY);
  if (kind) params.set("kind", kind);
  const q = params.toString() ? `?${params.toString()}` : "";
  return request(`/launch/stop-all${q}`, { method: "POST" });
}
