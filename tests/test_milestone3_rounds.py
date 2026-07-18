"""Milestone 3: strategies, idempotent aggregate, restart mid-round recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


@pytest.fixture()
def coord_path(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("CLASSIC_ROUNDS_PATH", str(tmp_path / "rounds.json"))
    sys.path.insert(0, str(COORD_SRC))
    yield tmp_path


def _payload(client_id: str, delta, *, samples: float = 1.0, base=None) -> str:
    base = base or [[0.0, 0.0], [0.0]]
    return json.dumps(
        {
            "client_id": client_id,
            "weight_delta": delta,
            "base_weights": base,
            "model_id": "simple_mlp",
            "model_config": {},
            "num_samples": samples,
            "final_loss": 0.1,
        }
    )


def test_strategies_fedavg_adaptive_robust(coord_path):
    from aggregation.strategies import (
        AdaptiveFedAvgStrategy,
        ClientContribution,
        FedAvgStrategy,
        RobustTrimmedMeanStrategy,
        get_strategy,
        list_strategies,
    )

    assert set(list_strategies()) == {"adaptive", "fedavg", "robust"}
    c1 = ClientContribution("a", [[1.0, 3.0], [2.0]], num_samples=1)
    c2 = ClientContribution("b", [[5.0, 7.0], [6.0]], num_samples=3)

    fed = FedAvgStrategy().aggregate([c1, c2])
    assert fed.averaged_delta[0] == [3.0, 5.0]

    adaptive = AdaptiveFedAvgStrategy().aggregate([c1, c2])
    # (1*1 + 5*3)/4 = 4.0 ; (3*1 + 7*3)/4 = 6.0 ; (2*1 + 6*3)/4 = 5.0
    assert adaptive.averaged_delta[0] == [4.0, 6.0]
    assert adaptive.averaged_delta[1] == [5.0]

    # Outlier should be trimmed/median-resistant with 3 clients
    c3 = ClientContribution("c", [[100.0, 100.0], [100.0]], num_samples=1)
    robust = RobustTrimmedMeanStrategy(trim_ratio=0.34).aggregate([c1, c2, c3])
    assert robust.averaged_delta[0][0] < 50.0

    assert get_strategy("fedavg").name == "fedavg"


def test_idempotent_aggregate_returns_same_version(coord_path, tmp_path):
    from core.aggregator import Aggregator
    from core.model_store import ModelStore
    from core.round_manager import RoundManager
    from core.state_store import StateStore
    from persistence.json_repos import JsonRoundRepository

    store = StateStore(path=str(tmp_path / "state.json"))
    models = ModelStore(models_dir=str(tmp_path / "models"))
    rm = RoundManager(
        state_store=store,
        round_repo=JsonRoundRepository(rounds_path=str(tmp_path / "rounds.json")),
    )
    agg = Aggregator(rm, model_store=models, state_store=store)

    rm.register_client("c1")
    rm.register_client("c2")
    rid = rm.assign_client_to_round("c1", "v1")
    rm.assign_client_to_round("c2", "v1")
    assert rid is not None

    base = [[1.0, 2.0], [3.0]]
    assert agg.submit_update("c1", rid, _payload("c1", [[0.1, 0.1], [0.1]], base=base))
    assert agg.submit_update("c2", rid, _payload("c2", [[0.3, 0.3], [0.3]], base=base))

    first = agg.aggregate(rid)
    assert first and first["status"] == "aggregated"
    version = first["model_version"]
    second = agg.aggregate(rid)
    assert second["status"] == "already_closed"
    assert second["replayed"] is True
    assert second["model_version"] == version
    assert second["aggregated_model"]["version"] == version


def test_restart_mid_aggregate_reconciles(coord_path, tmp_path):
    from core.aggregator import Aggregator
    from core.model_store import ModelStore
    from core.round_manager import RoundManager, RoundState
    from core.state_store import StateStore
    from persistence.json_repos import JsonRoundRepository

    state_path = str(tmp_path / "state.json")
    rounds_path = str(tmp_path / "rounds.json")
    models_dir = str(tmp_path / "models")

    store = StateStore(path=state_path)
    models = ModelStore(models_dir=models_dir)
    repo = JsonRoundRepository(rounds_path=rounds_path)
    rm = RoundManager(state_store=store, round_repo=repo)
    agg = Aggregator(rm, model_store=models, state_store=store)

    rm.register_client("c1")
    rm.register_client("c2")
    rid = rm.assign_client_to_round("c1", "v1")
    rm.assign_client_to_round("c2", "v1")
    base = [[0.0, 0.0], [0.0]]
    agg.submit_update("c1", rid, _payload("c1", [[1.0, 1.0], [1.0]], base=base))
    agg.submit_update("c2", rid, _payload("c2", [[3.0, 3.0], [3.0]], base=base))

    # Simulate crash after entering AGGREGATING, before publish
    rm.set_round_state(rid, RoundState.AGGREGATING)
    assert rid in agg.updates

    # Restart: new processes
    store2 = StateStore(path=state_path)
    rm2 = RoundManager(
        state_store=store2,
        round_repo=JsonRoundRepository(rounds_path=rounds_path),
    )
    # Crash recovery should have rolled AGGREGATING → COLLECTING
    assert rm2.rounds[rid].state == RoundState.COLLECTING
    assert rm2.rounds[rid].metadata.get("resume_after_crash") is True

    agg2 = Aggregator(rm2, model_store=ModelStore(models_dir=models_dir), state_store=store2)
    assert rid in agg2.updates
    results = agg2.reconcile_after_restart()
    assert results
    assert results[0]["status"] == "aggregated"
    assert rm2.rounds[rid].state == RoundState.CLOSED
    published = rm2.rounds[rid].metadata["published_version"]
    assert ModelStore(models_dir=models_dir).model_exists(published)

    # Second reconcile / aggregate is idempotent
    again = agg2.aggregate(rid)
    assert again["status"] == "already_closed"
    assert again["model_version"] == published
