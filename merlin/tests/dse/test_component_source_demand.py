"""Exact logical features have no timing, traffic or hardware interpretation."""

import copy

import pytest

from merlin.perf.component_source_demand import SourceDemandLimits, derive_source_demand
from merlin.targetgen import component_program

LIMITS = SourceDemandLimits(8, 32, 16, 64, 64, 256)


def profile(program, schedule=None, limits=LIMITS):
    return derive_source_demand(
        program=program,
        operand_dtype="i8",
        accumulator_dtype="i32",
        limits=limits,
        schedule=tuple(row["name"] for row in program["nodes"]) if schedule is None else schedule,
    )


def chains():
    return {
        "inputs": [
            {"name": "A", "role": "input", "shape": [8, 1], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [1, 8], "dtype": "i8"},
            {"name": "V", "role": "weight", "shape": [8, 1], "dtype": "i32"},
        ],
        "nodes": [
            {"name": "pa", "op": "matmul", "inputs": ["A", "W"]},
            {"name": "oa", "op": "matmul", "inputs": ["pa", "V"]},
            {"name": "pb", "op": "matmul", "inputs": ["A", "W"]},
            {"name": "ob", "op": "matmul", "inputs": ["pb", "V"]},
        ],
        "outputs": [{"name": "first", "value": "oa"}, {"name": "second", "value": "ob"}],
    }


def test_equal_work_and_operand_bytes_do_not_identify_logical_live_demand():
    program = chains()
    sequential = profile(program)
    interleaved = profile(program, ("pa", "pb", "oa", "ob"))
    for name in ("macs", "operand_payload_bytes", "input_payload_bytes", "publication_payload_bytes"):
        assert sequential["demand"][name] == interleaved["demand"][name]
    assert sequential["demand"]["macs"] == 256
    assert sequential["demand"]["peak_eager_logical_payload_bytes"] == 352
    assert interleaved["demand"]["peak_eager_logical_payload_bytes"] == 576
    assert sequential["source_program_sha256"] == interleaved["source_program_sha256"]
    assert sequential["value_intervals"] != interleaved["value_intervals"]
    assert sequential["authority"] == "none" and sequential["physical_costs"] == "UNKNOWN"


def test_original_alias_epoch_and_published_old_snapshot_stay_distinct():
    program = {
        "inputs": [{"name": "X", "role": "input", "shape": [1, 5], "dtype": "i8"}],
        "nodes": [
            {"name": "view", "op": "alias", "inputs": ["X"]},
            {"name": "snapshot", "op": "copy", "inputs": ["view"]},
            {"name": "next", "op": "update", "inputs": ["view", "snapshot"]},
        ],
        "outputs": [{"name": "old", "value": "snapshot"}, {"name": "new", "value": "X"}],
    }
    result = profile(program)
    assert result["logical_epochs"] == {"X": 1}
    assert result["logical_aliases"] == {"view": "X"}
    assert [row["logical_value"] for row in result["publication"]] == ["snapshot", "next"]
    assert {row["value"] for row in result["value_intervals"]} == {"X", "snapshot", "next"}
    assert result["actions"][0]["result_payload_bytes"] == 0
    assert result["demand"]["publication_payload_bytes"] == 10


def test_huge_checked_extents_use_only_bounded_integer_metadata(monkeypatch):
    program = {
        "inputs": [{"name": "X", "role": "input", "shape": [1 << 40, 1 << 40], "dtype": "i8"}],
        "nodes": [{"name": "Y", "op": "copy", "inputs": ["X"]}],
        "outputs": [{"name": "out", "value": "Y"}],
    }
    monkeypatch.setattr(component_program, "render", lambda *_args, **_kwargs: pytest.fail("rendered shaped data"))
    result = profile(program)
    assert result["status"] == "observed"
    assert result["demand"]["input_payload_bytes"] == 1 << 80
    assert result["demand"]["peak_eager_logical_payload_bytes"] == 1 << 81
    assert len(result["boundaries"]) == 3


@pytest.mark.parametrize(
    "schedule", [("oa", "pa", "pb", "ob"), ("pa", "pb", "ob"), ("pa", "pa", "oa", "ob"), ["pa", "oa", "pb", "ob"]]
)
def test_schedule_omission_duplicate_dependency_or_unclosed_type_refuses(schedule):
    with pytest.raises(ValueError, match="schedule"):
        profile(chains(), schedule)


@pytest.mark.parametrize("defect", ["dtype", "layout", "operation", "extent", "node_count", "bit_budget"])
def test_unsupported_or_over_budget_source_is_unknown_before_large_arithmetic(defect, monkeypatch):
    program, limits = chains(), LIMITS
    if defect == "dtype":
        program["inputs"][0]["dtype"] = "f32"
    elif defect == "layout":
        program["inputs"][0]["layout"] = "packed"
    elif defect == "operation":
        program["nodes"][0]["op"] = "unsupported"
    elif defect == "extent":
        program["inputs"][0]["shape"][0] = 1 << 128
    elif defect == "node_count":
        limits = SourceDemandLimits(8, 2, 16, 64, 64, 256)
    else:
        limits = SourceDemandLimits(8, 32, 16, 64, 64, 8)
    if defect != "operation":
        monkeypatch.setattr(component_program, "analyze", lambda *_a, **_k: pytest.fail("copied over-budget source"))
    result = profile(program, limits=limits)
    assert result["status"] == "UNKNOWN" and result["missing"]
    assert "demand" not in result


def test_changed_original_publication_changes_identity_and_last_needed_boundary():
    original = chains()
    changed = copy.deepcopy(original)
    changed["outputs"][1]["value"] = "pb"
    left, right = profile(original), profile(changed)
    assert left["source_program_sha256"] != right["source_program_sha256"]
    assert left["publication"] != right["publication"]
    assert left["demand"]["publication_payload_bytes"] == 64
    assert right["demand"]["publication_payload_bytes"] == 288
