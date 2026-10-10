"""An incomplete detailed inventory names its unclassified operations and the unselected host lane."""

from __future__ import annotations

from merlin_experiments.phase0.requirements import incomplete_inventory_blocker

DEQ = "quantized_decomposed.dequantize_per_tensor.default"


def _full(status="incomplete"):
    return {
        "status": status,
        "applications": {
            "cnn": {
                "status": "incomplete",
                "signatures": [
                    {"operation": DEQ, "disposition": "unclassified", "count": 4},
                    {"operation": DEQ, "disposition": "unclassified", "count": 3},
                    {"operation": "linalg.matmul", "disposition": "hardware_admitted", "count": 9},
                ],
            },
            "mlp": {"status": "inventoried", "signatures": []},
        },
    }


def test_a_complete_inventory_has_no_blocker():
    assert incomplete_inventory_blocker(_full("inventoried"), {}) is None


def test_the_blocker_names_each_unclassified_operation_and_the_missing_host_declaration():
    unselected = {"int8_w8a8": {"status": "unknown", "reason": "host_lane package 'p' is missing or unreadable"}}
    text = incomplete_inventory_blocker(_full(), unselected)
    assert f"cnn: {DEQ} x7" in text and "mlp" not in text
    assert "host lane's capability declaration was not selected (int8_w8a8: host_lane package 'p'" in text
    selected = {"int8_w8a8": {"capability_spec": {"operations": []}}}
    assert "not selected" not in incomplete_inventory_blocker(_full(), selected)
