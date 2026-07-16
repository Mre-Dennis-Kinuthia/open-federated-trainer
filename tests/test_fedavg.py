"""Unit tests for FedAvg weight-delta averaging."""

import sys
from pathlib import Path

# Allow importing coordinator core without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coordinator" / "src"))

from core.aggregator import fedavg_weight_deltas, _parse_weight_delta


def test_fedavg_two_clients():
    d1 = [[1.0, 3.0], [2.0]]
    d2 = [[5.0, 7.0], [6.0]]
    avg = fedavg_weight_deltas([d1, d2])
    assert avg[0] == [3.0, 5.0]
    assert avg[1] == [4.0]


def test_parse_weight_delta_payload():
    import json

    raw = json.dumps(
        {
            "client_id": "c1",
            "weight_delta": [[1.0, 2.0], [3.0]],
            "final_loss": 0.5,
        }
    )
    parsed = _parse_weight_delta(raw)
    assert parsed == [[1.0, 2.0], [3.0]]


if __name__ == "__main__":
    test_fedavg_two_clients()
    test_parse_weight_delta_payload()
    print("ok")
