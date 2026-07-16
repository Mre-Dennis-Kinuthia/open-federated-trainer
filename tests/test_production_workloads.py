"""Tests that production paths do not silently use demo implementations."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_SRC = ROOT / "client" / "src"
COORDINATOR_SRC = ROOT / "coordinator" / "src"


def test_dataset_is_required_by_default(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    monkeypatch.delenv("DATASET_PATH", raising=False)
    monkeypatch.delenv("ALLOW_SYNTHETIC_DATA", raising=False)
    from private_datasets import DatasetConfigurationError, load_local_dataset

    try:
        load_local_dataset()
    except DatasetConfigurationError as exc:
        assert "DATASET_PATH is required" in str(exc)
    else:
        raise AssertionError("dataset loader silently created demo data")


def test_numeric_csv_uses_real_values(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    from private_datasets import load_local_dataset

    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(
        "a,b,label\n1.5,2.5,3\n4.5,5.5,6\n",
        encoding="utf-8",
    )
    dataset = load_local_dataset(str(dataset_path), "csv")
    features, labels = dataset.as_tensors(kind="tabular", input_dim=2)
    assert features.tolist() == [[1.5, 2.5], [4.5, 5.5]]
    assert labels.tolist() == [[3.0], [6.0]]


def test_image_folder_decodes_actual_pixels(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    from PIL import Image
    from private_datasets import load_local_dataset

    for class_name, value in (("dark", 0), ("light", 255)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        Image.new("L", (4, 4), color=value).save(class_dir / "sample.png")

    dataset = load_local_dataset(str(tmp_path), "folder")
    images, labels = dataset.as_tensors(kind="image", channels=1, size=4)
    assert images.shape == (2, 1, 4, 4)
    assert images[0].mean().item() == 0.0
    assert images[1].mean().item() == 1.0
    assert labels.tolist() == [0, 1]


def test_inference_requires_real_model(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    monkeypatch.delenv("INFERENCE_MODEL_ID", raising=False)
    from jobs import JobConfigurationError, run_job

    try:
        run_job(
            {"job_type": "inference", "payload": {"inputs": ["hello"]}},
            "client-1",
        )
    except JobConfigurationError as exc:
        assert "model_id" in str(exc)
    else:
        raise AssertionError("inference used a dummy scorer")


def test_allowlisted_science_plugin(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    monkeypatch.syspath_prepend(str(ROOT / "client"))
    monkeypatch.setenv("COMPUTE_PLUGIN_ALLOWLIST", "examples.science_plugin")
    from jobs import run_job

    result = run_job(
        {
            "job_type": "compute",
            "payload": {
                "entrypoint": "examples.science_plugin:lennard_jones",
                "work_unit": {
                    "positions": [[0, 0, 0], [1.2, 0, 0]],
                    "steps": 2,
                },
            },
        },
        "client-1",
    )
    assert result["backend"] == "python-plugin"
    assert result["result"]["particle_count"] == 2


def test_job_queue_survives_restart(tmp_path, monkeypatch):
    module_path = COORDINATOR_SRC / "jobs" / "__init__.py"
    spec = importlib.util.spec_from_file_location("coordinator_jobs_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    JobQueue = module.JobQueue

    path = tmp_path / "jobs.json"
    queue = JobQueue(state_path=str(path))
    job = queue.create_job("compute", {"entrypoint": "x:y"})
    claimed = queue.claim_next("client-1", {"compute"})
    assert claimed and claimed.job_id == job.job_id

    restored = JobQueue(state_path=str(path))
    loaded = restored.get(job.job_id)
    assert loaded is not None
    assert loaded.assigned_client == "client-1"


def test_lora_rounds_survive_restart(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(COORDINATOR_SRC))
    from rounds.create_round import LoRARoundManager

    path = tmp_path / "lora.json"
    manager = LoRARoundManager(state_path=str(path))
    created = manager.create_round("tiny-llama", lora_r=2)
    assert manager.submit_adapter(
        created.round_id,
        "client-1",
        {"layer.lora_A": [[1.0]]},
        num_samples=3,
        training_loss=0.5,
        adapter_hash="abc",
    )

    restored = LoRARoundManager(state_path=str(path))
    assert restored.get_round(created.round_id).state == "COLLECTING"
    assert "client-1" in restored.get_submissions(created.round_id)
    assert restored.next_round_id == created.round_id + 1


def test_lora_fedavg_weights_each_parameter_once(monkeypatch):
    monkeypatch.syspath_prepend(str(COORDINATOR_SRC))
    from aggregation.fedavg_adapters import aggregate_lora_adapters

    submissions = {
        "small": {
            "client_id": "small",
            "num_samples": 1,
            "adapter_state_dict": {"a": [0.0], "b": [0.0]},
        },
        "large": {
            "client_id": "large",
            "num_samples": 3,
            "adapter_state_dict": {"a": [4.0], "b": [8.0]},
        },
    }
    result = aggregate_lora_adapters(submissions, weight_by_samples=True)
    assert result == {"a": [3.0], "b": [6.0]}


def test_classic_clients_share_and_reload_global_weights(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_SRC))
    monkeypatch.syspath_prepend(str(COORDINATOR_SRC))
    import torch
    from core.aggregator import apply_weight_delta, fedavg_weight_deltas
    from trainer import train_local_model

    task = {
        "round_id": 1,
        "model_version": "v1",
        "model_id": "simple_mlp",
        "model_seed": 7,
    }
    data_a = (torch.ones(3, 2), torch.ones(3, 1))
    data_b = (torch.full((3, 2), 2.0), torch.full((3, 1), 4.0))
    update_a = json.loads(
        train_local_model(
            task,
            client_id="a",
            num_epochs=1,
            input_dim=2,
            hidden_dim=3,
            data=data_a,
        )
    )
    update_b = json.loads(
        train_local_model(
            task,
            client_id="b",
            num_epochs=1,
            input_dim=2,
            hidden_dim=3,
            data=data_b,
        )
    )
    assert update_a["base_weights"] == update_b["base_weights"]
    average = fedavg_weight_deltas(
        [update_a["weight_delta"], update_b["weight_delta"]]
    )
    global_weights = apply_weight_delta(update_a["base_weights"], average)

    next_update = json.loads(
        train_local_model(
            {**task, "round_id": 2, "model_version": "v2"},
            client_id="c",
            num_epochs=1,
            input_dim=2,
            hidden_dim=3,
            data=data_a,
            global_weights=global_weights,
        )
    )
    for loaded_layer, saved_layer in zip(
        next_update["base_weights"],
        global_weights,
    ):
        assert loaded_layer == pytest.approx(saved_layer, abs=1e-7)


def test_task_assigner_ignores_legacy_delta_only_models(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(COORDINATOR_SRC))
    from core.model_store import ModelStore
    from core.round_manager import RoundManager
    from core.task_assigner import TaskAssigner

    store = ModelStore(str(tmp_path))
    store.save_model("v5", {"averaged_weight_delta": [[0.1]]})
    assigner = TaskAssigner(RoundManager(), store)
    assert assigner.model_version == "v6"

    store.save_model(
        "v6",
        {
            "model_id": "simple_mlp",
            "model_config": {"input_dim": 2},
            "weights": [[1.0]],
        },
    )
    restored = TaskAssigner(RoundManager(), store)
    assert restored.model_version == "v6"
    assert restored.model_config == {"input_dim": 2}


def test_lora_eval_without_dataset_is_explicitly_skipped(monkeypatch):
    monkeypatch.syspath_prepend(str(COORDINATOR_SRC))
    monkeypatch.delenv("LORA_EVAL_DATASET_PATH", raising=False)
    monkeypatch.delenv("LORA_REQUIRE_EVALUATION", raising=False)
    from evaluation.evaluate_adapter import evaluate_adapter

    result = evaluate_adapter(
        1,
        "v1",
        {"adapter.weight": [[1.0]]},
        base_model_name="unused",
        lora_r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        max_seq_length=16,
    )
    assert result.evaluated is False
    assert result.evaluation_loss is None
    assert "not configured" in (result.reason or "")
