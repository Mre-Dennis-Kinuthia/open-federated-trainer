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
