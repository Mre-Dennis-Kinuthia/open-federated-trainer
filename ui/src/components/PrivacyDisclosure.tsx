import type { CSSProperties } from "react";

export type WorkloadKind =
  | "train"
  | "lora"
  | "inference"
  | "label"
  | "compute"
  | "general";

type Flag = {
  id: string;
  label: string;
  status: "on" | "off" | "partial" | "planned";
  note: string;
};

/** Capability flags aligned with docs/architecture/PRIVACY_MODEL.md (truthful today). */
export const PRIVACY_CAPABILITY_FLAGS: Flag[] = [
  {
    id: "data_locality",
    label: "Training data locality",
    status: "on",
    note: "Classic / LoRA training datasets stay on nodes by default.",
  },
  {
    id: "authenticated_nodes",
    label: "Authenticated nodes",
    status: "partial",
    note: "Per-node API keys; optional Ed25519 public keys.",
  },
  {
    id: "tls_transport",
    label: "TLS transport",
    status: "planned",
    note: "Depends on deployment (reverse proxy / ingress).",
  },
  {
    id: "secure_aggregation",
    label: "Secure aggregation",
    status: "off",
    note: "Not implemented — do not claim SecAgg.",
  },
  {
    id: "central_differential_privacy",
    label: "Central DP",
    status: "off",
    note: "Not implemented.",
  },
  {
    id: "local_differential_privacy",
    label: "Local DP",
    status: "off",
    note: "Not implemented.",
  },
  {
    id: "location_reporting_disabled",
    label: "Location reporting off",
    status: "partial",
    note: "Optional coarse geo; disable with GEO_LOOKUP_DISABLED=true.",
  },
  {
    id: "trusted_worker_pool",
    label: "Trusted worker pool",
    status: "partial",
    note: "Jobs may be claimed by any authenticated worker unless you restrict membership.",
  },
];

const WORKLOAD_COPY: Record<
  WorkloadKind,
  { stays: string; leaves: string; visible: string }
> = {
  train: {
    stays: "Raw training datasets remain on the participant.",
    leaves: "Weight deltas (and base weights in the update payload) go to the coordinator.",
    visible: "Anyone who can download published global models.",
  },
  lora: {
    stays: "Fine-tuning corpora remain on the participant.",
    leaves: "LoRA adapters and metrics go to the coordinator.",
    visible: "Coordinator operators; nodes with adapter download credentials.",
  },
  inference: {
    stays: "Nothing is local-only if you put text in the job payload.",
    leaves:
      "payload.inputs (prompts) are stored on the coordinator and delivered to claiming workers.",
    visible: "Operators (and unauthenticated readers if OPERATOR_API_KEY is unset).",
  },
  label: {
    stays: "Preferred: load via dataset_alias / local path on the worker.",
    leaves: "Labels and metrics; optional payload.inputs if supplied.",
    visible: "Coordinator operators with job access.",
  },
  compute: {
    stays: "Plugin code must already be installed and allowlisted on the worker.",
    leaves: "work_unit parameters and JSON results.",
    visible: "Coordinator and claiming workers.",
  },
  general: {
    stays: "Training datasets stay on nodes unless you put them in a job payload.",
    leaves: "Model updates, adapters, job inputs/results, and coarse geo presence.",
    visible: "This coordinator instance — not a private enclave by default.",
  },
};

const STATUS_LABEL: Record<Flag["status"], string> = {
  on: "Enabled",
  off: "Not available",
  partial: "Partial",
  planned: "Planned",
};

type Props = {
  workload?: WorkloadKind;
  showFlags?: boolean;
  compact?: boolean;
  className?: string;
  style?: CSSProperties;
};

export function PrivacyDisclosure({
  workload = "general",
  showFlags = false,
  compact = false,
  className,
  style,
}: Props) {
  const copy = WORKLOAD_COPY[workload];
  return (
    <div
      className={className}
      style={style}
      role="region"
      aria-label={`Privacy disclosure (${workload})`}
      data-testid="privacy-disclosure"
    >
      {!compact && <h3 className="privacy-disclosure-title">Privacy for this workload</h3>}
      <ul className="privacy-disclosure-list">
        <li>
          <strong>Stays local:</strong> {copy.stays}
        </li>
        <li>
          <strong>Leaves the node:</strong> {copy.leaves}
        </li>
        <li>
          <strong>Who can see it:</strong> {copy.visible}
        </li>
      </ul>
      <p className="privacy-disclosure-note">
        Federated learning alone does not guarantee privacy. Secure aggregation and
        differential privacy are not enabled on this build.
      </p>
      {showFlags && (
        <table className="privacy-flags-table">
          <caption className="sr-only">Privacy capability flags</caption>
          <thead>
            <tr>
              <th scope="col">Capability</th>
              <th scope="col">Status</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {PRIVACY_CAPABILITY_FLAGS.map((flag) => (
              <tr key={flag.id}>
                <td>{flag.label}</td>
                <td>
                  <span data-status={flag.status}>{STATUS_LABEL[flag.status]}</span>
                </td>
                <td>{flag.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
