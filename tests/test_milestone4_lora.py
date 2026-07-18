"""Milestone 4: corrected LoRA aggregation, manifests, isolated merge."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


@pytest.fixture()
def coord_path(monkeypatch):
    sys.path.insert(0, str(COORD_SRC))
    yield


def test_delta_svd_matches_averaged_product_not_param_fedavg(coord_path):
    from aggregation.fedavg_adapters import aggregate_lora_adapters
    from aggregation.lora_math import reconstruct_delta, to_tensor

    # Rank-1 LoRA: ΔW = B @ A
    # Client1: A=[[1]], B=[[2]] → ΔW=2
    # Client2: A=[[3]], B=[[4]] → ΔW=12
    # Uniform avg ΔW = 7
    # Param FedAvg: A=2, B=3 → product 6 ≠ 7
    submissions = {
        "c1": {
            "num_samples": 1,
            "adapter_state_dict": {
                "layer.q_proj.lora_A.weight": [[1.0]],
                "layer.q_proj.lora_B.weight": [[2.0]],
            },
        },
        "c2": {
            "num_samples": 1,
            "adapter_state_dict": {
                "layer.q_proj.lora_A.weight": [[3.0]],
                "layer.q_proj.lora_B.weight": [[4.0]],
            },
        },
    }

    param = aggregate_lora_adapters(
        submissions, weight_by_samples=False, strategy="param_fedavg"
    )
    delta = aggregate_lora_adapters(
        submissions, weight_by_samples=False, strategy="delta_svd"
    )
    assert param is not None and delta is not None

    param_product = reconstruct_delta(
        to_tensor(param["layer.q_proj.lora_A.weight"]),
        to_tensor(param["layer.q_proj.lora_B.weight"]),
    ).item()
    delta_product = reconstruct_delta(
        to_tensor(delta["layer.q_proj.lora_A.weight"]),
        to_tensor(delta["layer.q_proj.lora_B.weight"]),
    ).item()

    assert abs(param_product - 6.0) < 1e-4
    assert abs(delta_product - 7.0) < 1e-3
    assert abs(param_product - delta_product) > 0.5


def test_sample_weighted_delta_svd(coord_path):
    from aggregation.fedavg_adapters import aggregate_lora_adapters
    from aggregation.lora_math import reconstruct_delta, to_tensor

    submissions = {
        "small": {
            "num_samples": 1,
            "adapter_state_dict": {
                "m.lora_A.weight": [[1.0, 0.0], [0.0, 1.0]],
                "m.lora_B.weight": [[1.0, 0.0], [0.0, 1.0]],
            },
        },
        "large": {
            "num_samples": 3,
            "adapter_state_dict": {
                "m.lora_A.weight": [[2.0, 0.0], [0.0, 2.0]],
                "m.lora_B.weight": [[2.0, 0.0], [0.0, 2.0]],
            },
        },
    }
    # ΔW_small = I, ΔW_large = 4I; weighted avg = 0.25*I + 0.75*4I = 3.25 I
    result = aggregate_lora_adapters(submissions, weight_by_samples=True, strategy="delta_svd")
    assert result is not None
    product = reconstruct_delta(
        to_tensor(result["m.lora_A.weight"]),
        to_tensor(result["m.lora_B.weight"]),
    )
    assert torch.allclose(product, torch.eye(2) * 3.25, atol=1e-3)


def test_isolated_merge_does_not_mutate_base(coord_path):
    from aggregation.merge import isolated_merge_state_dicts

    base = {"q_proj.weight": [[1.0, 0.0], [0.0, 1.0]]}
    adapter = {
        "q_proj.lora_A.weight": [[1.0, 0.0]],
        "q_proj.lora_B.weight": [[2.0], [0.0]],
    }
    # ΔW = [[2,0],[0,0]]; scaling alpha/r = 16/8 = 2 → add [[4,0],[0,0]]
    merged = isolated_merge_state_dicts(
        base,
        adapter,
        lora_alpha=16,
        lora_r=8,
        target_modules=["q_proj"],
    )
    assert base["q_proj.weight"] == [[1.0, 0.0], [0.0, 1.0]]
    assert merged["q_proj.weight"][0][0] == pytest.approx(5.0)


def test_adapter_manifest_hash_stable(coord_path, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_STORE_ROOT", str(tmp_path / "arts"))
    from aggregation.adapter_manifest import (
        build_adapter_manifest,
        hash_adapter_state,
        register_adapter_manifest,
    )

    state = {"layer.lora_A.weight": [[1.0]], "layer.lora_B.weight": [[2.0]]}
    manifest = build_adapter_manifest(
        adapter_version="v1",
        adapter_state_dict=state,
        base_model_id="tiny-llama",
        round_id=1,
        lora_r=1,
        lora_alpha=2,
        target_modules=["q_proj"],
        aggregation_strategy="delta_svd",
        task_type="causal_lm",
    )
    assert manifest.content_hash == hash_adapter_state(state)
    assert manifest.artifact_type == "lora_adapter"
    register_adapter_manifest(manifest)


def test_task_aware_eval_skip_preserves_task_type(coord_path, monkeypatch):
    monkeypatch.delenv("LORA_EVAL_DATASET_PATH", raising=False)
    monkeypatch.delenv("LORA_REQUIRE_EVALUATION", raising=False)
    from evaluation.evaluate_adapter import evaluate_adapter

    result = evaluate_adapter(
        1,
        "v1",
        {"x": [[1.0]]},
        base_model_name="unused",
        lora_r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        max_seq_length=16,
        task_type="seq_cls",
    )
    assert result.evaluated is False
    assert result.task_type == "seq_cls"
