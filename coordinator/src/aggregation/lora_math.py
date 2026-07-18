"""LoRA aggregation math: ΔW FedAvg + SVD reparameterization.

Elementwise FedAvg on A and B does not preserve the product ΔW = B @ A.
The corrected path averages reconstructed ΔW (sample-weighted), then factors
back to rank-r factors via truncated SVD.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def pair_lora_ab_keys(keys: List[str]) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """
    Map module stem → (A_key, B_key). Unpaired keys returned separately.

    PEFT naming uses ``...lora_A.weight`` / ``...lora_B.weight`` (also ``lora_A.default``).
    """
    a_keys = {}
    b_keys = {}
    for key in keys:
        if "lora_A" in key:
            stem = key.replace("lora_A", "lora_X", 1)
            a_keys[stem] = key
        elif "lora_B" in key:
            stem = key.replace("lora_B", "lora_X", 1)
            b_keys[stem] = key
    pairs: Dict[str, Tuple[str, str]] = {}
    for stem, a_key in a_keys.items():
        if stem in b_keys:
            pairs[stem] = (a_key, b_keys[stem])
    paired = {a for a, b in pairs.values()} | {b for a, b in pairs.values()}
    unpaired = [k for k in keys if k not in paired]
    return pairs, unpaired


def reconstruct_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ΔW = B @ A for standard LoRA (B: [out, r], A: [r, in])."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"LoRA A/B must be 2D, got {tuple(a.shape)} / {tuple(b.shape)}")
    if b.shape[1] != a.shape[0]:
        raise ValueError(
            f"Incompatible LoRA shapes B{tuple(b.shape)} @ A{tuple(a.shape)}"
        )
    return b @ a


def svd_factorize_delta(
    delta: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate ΔW ≈ B @ A with rank ``rank`` via truncated SVD.

    Returns (A [r, in], B [out, r]) matching PEFT layout.
    """
    if delta.ndim != 2:
        raise ValueError("delta must be 2D")
    out_features, in_features = delta.shape
    r = max(1, min(int(rank), out_features, in_features))
    # torch.linalg.svd: U [out, out], S [min], Vh [in, in]
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    s_r = s[:r].clamp(min=0)
    sqrt_s = torch.sqrt(s_r)
    b = u[:, :r] * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh[:r, :]
    return a, b


def to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def infer_rank_from_pair(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(a.shape[0])
