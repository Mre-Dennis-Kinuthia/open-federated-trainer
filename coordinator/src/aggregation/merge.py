"""Isolated LoRA merge — never mutates a shared base module in-place."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

from .lora_math import pair_lora_ab_keys, reconstruct_delta, to_tensor


def merge_delta_into_weight(
    base_weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Return a new tensor: W' = W + scaling * (B @ A). Does not mutate W."""
    delta = reconstruct_delta(a, b) * float(scaling)
    if base_weight.shape != delta.shape:
        raise ValueError(
            f"Base weight {tuple(base_weight.shape)} != delta {tuple(delta.shape)}"
        )
    return base_weight.detach().float() + delta


def isolated_merge_state_dicts(
    base_state: Dict[str, Any],
    adapter_state: Dict[str, Any],
    *,
    lora_alpha: int,
    lora_r: int,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Produce a deep-copied base state with LoRA deltas applied.

    Only keys whose names contain a target module fragment and have A/B pairs
    are merged. The input ``base_state`` is never modified.
    """
    merged = deepcopy(base_state)
    scaling = float(lora_alpha) / float(max(lora_r, 1))
    pairs, _ = pair_lora_ab_keys(list(adapter_state.keys()))
    targets = set(target_modules or [])

    for stem, (a_key, b_key) in pairs.items():
        if targets:
            # stem still contains module path; require any target substring
            if not any(t in a_key for t in targets):
                continue
        # Map PEFT adapter key → base weight key heuristically:
        # "...q_proj.lora_A.weight" → find base key ending with "q_proj.weight"
        base_key = _infer_base_weight_key(a_key, list(merged.keys()))
        if base_key is None:
            continue
        a = to_tensor(adapter_state[a_key])
        b = to_tensor(adapter_state[b_key])
        base_w = to_tensor(merged[base_key])
        merged[base_key] = merge_delta_into_weight(
            base_w, a, b, scaling=scaling
        ).tolist()
    return merged


def apply_adapter_on_peft_copy(
    peft_model: Any,
    adapter_state: Dict[str, Any],
) -> Any:
    """
    Load adapter weights into ``peft_model`` after verifying we operate on this
    instance only (caller must pass a freshly constructed PEFT wrapper).
    """
    from peft import set_peft_model_state_dict

    state = {k: to_tensor(v) for k, v in adapter_state.items()}
    set_peft_model_state_dict(peft_model, state)
    return peft_model


def _infer_base_weight_key(lora_a_key: str, base_keys: List[str]) -> Optional[str]:
    # Strip peft prefixes and lora_A.weight → .weight
    candidate = lora_a_key
    for prefix in ("base_model.model.", "base_model.", "model."):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    candidate = candidate.replace("lora_A.weight", "weight").replace(
        "lora_A.default.weight", "weight"
    )
    if candidate in base_keys:
        return candidate
    # Fuzzy: endswith
    for key in base_keys:
        if key.endswith(candidate) or key.endswith(candidate.split(".", 1)[-1]):
            return key
    return None
