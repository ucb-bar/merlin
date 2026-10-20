"""Adjacent conditional joint-store cases never imply emitted resource coverage."""

import pytest
from merlin_experiments.phase0 import development_decisive_roster as C
from merlin_experiments.phase0 import development_sources as D


def selected():
    return C.requests(
        basis={
            "operand_store_bytes": 2048,
            "accumulator_bytes": 512,
            "operand_bytes": 1,
            "accumulate_bytes": 4,
            "command_windows": {"execute": 6},
        },
        axis_granules={"M": 4, "N": 8, "K": 2},
        resource_sha256="a" * 64,
        placement_assumptions={"operand_reserved_bytes": 0, "accumulator_reserved_bytes": 0, "buffer_copies": [1, 2]},
    )


def test_rectangular_granules_and_exact_joint_live_capacity_pairs():
    result = selected()
    joints = [row for row in result["requests"] if row["family"] == "joint_sp_acc"]
    assert len(joints) == 4
    for fit, spill in zip(joints[::2], joints[1::2], strict=True):
        a, b = fit["joint_boundary"], spill["joint_boundary"]
        assert a["operand_bytes_including_reservation"] <= a["operand_capacity_bytes"]
        assert b["operand_bytes_including_reservation"] > b["operand_capacity_bytes"]
        assert a["accumulator_bytes_including_reservation"] == a["accumulator_capacity_bytes"]
        assert b["accumulator_bytes_including_reservation"] == b["accumulator_capacity_bytes"]
        assert spill["declaration"]["geometry"]["K"] - fit["declaration"]["geometry"]["K"] == 2
        assert not a["actual_placement_proven"] and fit["physical_mechanism_status"] == "UNEXERCISED"
    tails = [row for row in result["requests"] if row["family"] == "independent_axis_tail"]
    for row in tails:
        assert row["granule"] == {"M": 4, "N": 8, "K": 2}[row["axis"]]
        assert row["declaration"]["geometry"][row["axis"]] % row["granule"] == row["remainder"]


def test_all_materializable_controls_retain_exact_accelerator_work_and_unproved_obligations():
    result = selected()
    names = []
    for row in result["requests"]:
        if "declaration" not in row:
            assert row["status"] == "source_refused" and row["reason"]
            continue
        description = D.describe(row["declaration"])
        assert description["scalar_products"] > 0 and description["contractions"] > 0
        assert row["required_emitted_witness"] and row["physical_mechanism_status"] == "UNEXERCISED"
        names += [row["declaration"]["name"]]
    assert len(names) == len(set(names))
    queues = [row for row in result["requests"] if row["family"] == "queue_window"]
    assert [row["source_contraction_count"] for row in queues] == [5, 6, 7]
    assert not result["original_cohort_changed"]


@pytest.mark.parametrize(
    "assumptions",
    [
        {},
        {"operand_reserved_bytes": -1, "accumulator_reserved_bytes": 0, "buffer_copies": [1]},
        {"operand_reserved_bytes": 0, "accumulator_reserved_bytes": 0, "buffer_copies": [3]},
    ],
)
def test_missing_or_unbounded_placement_assumptions_refuse(assumptions):
    with pytest.raises(ValueError):
        C.requests(
            basis={"operand_store_bytes": 2048, "accumulator_bytes": 512, "operand_bytes": 1, "accumulate_bytes": 4},
            axis_granules={"M": 4, "N": 8, "K": 2},
            resource_sha256="a" * 64,
            placement_assumptions=assumptions,
        )
