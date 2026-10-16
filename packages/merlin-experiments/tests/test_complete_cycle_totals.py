"""Measured non-MAC work remains visible without inventing its achievable rate."""

from merlin_experiments.phase2.feedback_metrics import recoverable_cycles


def test_measured_host_cost_is_part_of_complete_corpus_total():
    result = recoverable_cycles(
        [
            {
                "family": "compute",
                "capsule": "matmul",
                "measured": True,
                "declared_macs": 500,
                "baseline_gsim_cycles": 100,
            },
            {"family": "host", "capsule": "copy", "measured": True, "declared_macs": None, "baseline_gsim_cycles": 900},
        ],
        10,
    )
    assert result["corpus_total_cycles"] == 1000
    assert result["total_recoverable_cycles"] == 50
    assert result["total_recoverable_share"] == 0.05
    assert result["priced_members"] == 1
    assert "UNKNOWN" in result["headroom_scope"]


def test_unknown_host_headroom_does_not_erase_measured_host_cycles():
    result = recoverable_cycles(
        [
            {"family": "host", "capsule": "copy", "measured": True, "declared_macs": None, "baseline_gsim_cycles": 900},
        ]
    )
    assert result["status"] == "unavailable"
    assert result["corpus_total_cycles"] == 900
    assert result["ranked"] == []
