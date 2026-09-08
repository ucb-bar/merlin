"""Focused tests for the fail-closed Atlas SmolVLA hybrid scheduler."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from submission.mlir_oot.frontend import parse_verified
from submission.mlir_oot.hybrid_runtime import allocate_intervals
from submission.mlir_oot.host_semantics import HostSemanticLane, UnsupportedHostRegion


ROOT = Path(__file__).resolve().parent
PLAN_ROOT = ROOT / "whole_capture_plan"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_lane() -> HostSemanticLane:
    capture = ROOT.parents[4] / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    return HostSemanticLane(parse_verified(capture.read_text(encoding="utf-8")))


def test_interval_allocator_respects_inclusive_lifetimes_and_reuses_storage() -> None:
    result = allocate_intervals([
        {"name": "a", "start": 0, "end": 2, "bytes": 33},
        {"name": "b", "start": 2, "end": 3, "bytes": 32},
        {"name": "c", "start": 3, "end": 4, "bytes": 16},
        {"name": "d", "start": 5, "end": 5, "bytes": 64},
    ], alignment=32)
    by_name = {row["name"]: row for row in result["allocations"]}
    assert by_name["a"]["offset"] == 0
    assert by_name["b"]["offset"] == 64  # endpoint 2 overlaps a
    assert by_name["c"]["offset"] == 0   # a is dead, b is still live
    assert by_name["d"]["offset"] == 0
    assert result["peak_bytes"] == 96
    assert result["saved_bytes"] > 0


def test_saved_hybrid_schedule_is_complete_ordered_and_fail_closed() -> None:
    schedule = load(PLAN_ROOT / "hybrid_schedule.json")
    assert schedule["status"] == "e2e_blocked_fail_closed"
    assert schedule["runnable_e2e"] is False
    assert schedule["coverage"] == {
        "host_signature_regions_by_semantic": {
            "add": 215,
            "arange": 63,
            "compare": 4,
            "cos": 57,
            "div": 56,
            "dtype_cast": 472,
            "elementwise": 3,
            "fill": 50,
            "gelu": 12,
            "minmax": 2,
            "mul": 536,
            "pow": 123,
            "rsqrt": 66,
            "select": 45,
            "sigmoid": 33,
            "sin": 57,
            "sub": 68,
        },
        "host_signature_regions_implemented": 1862,
        "layout_bridge_candidates": 2033,
        "materialized_copy_bridges": 112,
        "missing_host_semantics_reduction": 1840,
        "partition_host_region_overlap": ["conv_0"],
        "previous_bounded_host_regions_implemented": 22,
        "previous_missing_host_semantics": 2408,
        "proven_metadata_aliases": 1675,
        "qualified_accelerator_partitions": 3,
        "semantic_host_required_regions": 2430,
        "strided_broadcast_bridges": 246,
        "structural_accelerator_partitions": 391,
    }
    assert schedule["fail_closed"]["missing_host_semantics"] == 568
    assert schedule["fail_closed"]["unqualified_accelerator_partitions"] == 388
    assert schedule["fail_closed"]["unrealized_layout_bridges"] == 358
    assert schedule["conversion_boundaries"] == {
        "by_conversion": {
            "device_requantize": 88,
            "device_to_host_dequantize": 391,
            "host_to_device_quantize": 694,
            "host_to_device_quantize_bias": 77,
        },
        "count": 1250,
        "qualified": 12,
    }
    assert len(schedule["events"]) == 6104
    assert [row["event_index"] for row in schedule["events"]] == list(range(6104))
    qualified_host = [
        row for row in schedule["events"]
        if row["kind"] == "host_region" and row["executable"]
    ]
    assert len(qualified_host) == 1862
    assert all(len(row["operation_signature_sha256"]) == 64 for row in qualified_host)
    rejected_select = next(
        row for row in schedule["events"]
        if row.get("region_id") == "select_0"
    )
    assert rejected_select["executable"] is False
    assert "operation_signature_sha256" not in rejected_select
    assert schedule["device_activation_arena"]["allocation_count"] == 391
    assert schedule["device_activation_arena"]["reuse_count"] > 0
    assert schedule["device_activation_arena"]["peak_bytes"] < (
        schedule["device_activation_arena"]["naive_no_reuse_bytes"]
    )


def test_bounded_real_chain_replays_host_semantics_and_retains_scoped_evidence() -> None:
    chain = load(PLAN_ROOT / "hybrid_schedule.json")["bounded_chain"]
    assert chain["status"] == "host_replay_matches_retained_qualified_rtl_chain"
    assert chain["partition_path"] == ["atlas_p0243", "atlas_p0244"]
    assert chain["host_bridge_region_count"] == 27
    assert chain["p0244_activation_shape"] == [50, 1440]
    assert chain["p0244_activation_f32_sha256"] == (
        "23457ea06c994031379dcfc4491e866b1f9e5558a2ad4c5ad5a310366145bcce"
    )
    assert chain["retained_results"][1]["dispatches"] == 3
    assert "not whole-model execution" in chain["claim"]


def test_real_host_chain_is_capture_discovered_fresh_and_dependency_carrying() -> None:
    chain = load(PLAN_ROOT / "hybrid_schedule.json")["generic_host_chain"]
    assert chain["status"] == "fresh_numeric_execution_exactly_replayed"
    assert chain["selection"].startswith("first stable-ranked consecutive qualified run")
    assert chain["region_count"] == 16
    assert chain["dependency_edges"] == 17
    assert chain["semantics"] == [
        "arange", "compare", "dtype_cast", "mul", "add", "sub",
        "dtype_cast", "mul", "sub", "select", "pow", "mul",
        "elementwise", "mul", "mul", "mul",
    ]
    assert len(chain["fresh_inputs"]) == 1
    assert all(row["finite"] for row in chain["outputs"])
    assert chain["outputs"][1]["true_elements"] == 17
    assert chain["replay_hashes_equal"] is True
    assert "not whole-model E2E" in chain["claim"]


def test_new_scalar_families_have_fresh_real_capture_numeric_witnesses() -> None:
    witnesses = load(PLAN_ROOT / "hybrid_schedule.json")[
        "generic_host_numeric_witnesses"
    ]
    assert [row["label"] for row in witnesses] == [
        "pow_reciprocal", "rsqrt_normalization", "sigmoid_gate",
        "trigonometric_fanout", "arange_dependency", "fill_dependency",
        "gelu_standalone",
    ]
    covered = set()
    for row in witnesses:
        covered.update(row["semantics"])
        assert row["status"] == "fresh_numeric_execution_exactly_replayed"
        assert row["replay_hashes_equal"] is True
        assert all(output["finite"] for output in row["outputs"])
        if row["label"] != "gelu_standalone":
            assert row["dependency_edges"] > 0
    assert {
        "pow", "elementwise", "rsqrt", "sigmoid", "sin", "cos", "gelu",
        "arange", "fill",
    } <= covered


def test_all_real_constructor_regions_are_extracted_and_execute_exactly(
    real_lane: HostSemanticLane,
) -> None:
    constructors = [
        program for program in real_lane.programs.values()
        if program.semantic in {"arange", "fill"}
    ]
    assert Counter(program.semantic for program in constructors) == {
        "arange": 63,
        "fill": 50,
    }
    assert all(
        program.signature["schema"] == "atlas_host_constructor_signature_v1"
        and program.signature["scalar_constants"]
        for program in constructors
    )

    fractional = next(
        program for program in constructors
        if program.semantic == "arange" and program.signature["output_shape"] == [31]
    )
    np.testing.assert_array_equal(
        real_lane.execute(fractional.region_id, {}),
        (np.arange(31, dtype=np.float32) + np.float32(1)) / np.float32(32),
    )
    integer = next(
        program for program in constructors
        if (program.semantic == "arange"
            and program.signature["output_shape"] == [360])
    )
    np.testing.assert_array_equal(
        real_lane.execute(integer.region_id, {}), np.arange(360, dtype=np.int64)
    )

    for program in (row for row in constructors if row.semantic == "fill"):
        actual = real_lane.execute(program.region_id, {})
        assert tuple(actual.shape) == tuple(program.signature["output_shape"])
        if actual.dtype == np.bool_:
            assert np.all(actual)
        elif np.isneginf(actual).any():
            assert np.all(np.isneginf(actual))
        else:
            assert np.all(actual == actual.reshape(-1)[0])


def test_declared_arange_with_wrong_scalar_dag_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward() -> tensor<4xi64> {
        %empty = tensor.empty() : tensor<4xi64>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } outs(%empty : tensor<4xi64>)
          attrs = {prov.region_id = "false_arange", prov.op = "arange",
                   prov.family = "iota", prov.aten = "aten.arange.start_step"} {
        ^bb0(%old: i64):
          %index = linalg.index 0 : index
          %cast = arith.index_cast %index : index to i64
          linalg.yield %cast : i64
        } -> tensor<4xi64>
        return %result : tensor<4xi64>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_arange") is None
    assert lane.rejections["false_arange"] == (
        "scalar DAG does not match a captured constructor pattern"
    )


def test_fill_whose_splat_uses_an_unrecorded_constant_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward() -> tensor<4xf32> {
        %captured = arith.constant {
          prov.region_id = "false_fill", prov.op = "fill", prov.family = "fill",
          prov.aten = "aten.full.default"
        } 2.000000e+00 : f32
        %other = arith.constant 3.000000e+00 : f32
        %result = tensor.splat %other {
          prov.region_id = "false_fill", prov.op = "fill", prov.family = "fill",
          prov.aten = "aten.full.default"
        } : tensor<4xf32>
        return %result : tensor<4xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_fill") is None
    assert lane.rejections["false_fill"] == (
        "fill splat does not consume its captured constant"
    )


def test_declared_unary_semantic_with_wrong_scalar_dag_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%arg: tensor<4xf32>) -> tensor<4xf32> {
        %empty = tensor.empty() : tensor<4xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%arg : tensor<4xf32>) outs(%empty : tensor<4xf32>)
          attrs = {prov.region_id = "false_sigmoid", prov.op = "sigmoid",
                   prov.family = "elementwise", prov.aten = "aten.sigmoid.default"} {
        ^bb0(%value: f32, %old: f32):
          %wrong = arith.negf %value : f32
          linalg.yield %wrong : f32
        } -> tensor<4xf32>
        return %result : tensor<4xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_sigmoid") is None
    assert lane.rejections["false_sigmoid"] == (
        "scalar DAG does not match a captured semantic pattern"
    )


def test_generic_host_lane_executes_exact_affine_broadcast_not_numpy_shape_guessing() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%a: tensor<2x1xf32>, %b: tensor<1x3xf32>) -> tensor<2x3xf32> {
        %empty = tensor.empty() : tensor<2x3xf32>
        %result = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, 0)>,
            affine_map<(d0, d1) -> (0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a, %b : tensor<2x1xf32>, tensor<1x3xf32>)
          outs(%empty : tensor<2x3xf32>)
          attrs = {prov.region_id = "add_synthetic", prov.op = "add",
                   prov.family = "elementwise", prov.aten = "aten.add.Tensor"} {
        ^bb0(%lhs: f32, %rhs: f32, %old: f32):
          %sum = arith.addf %lhs, %rhs : f32
          linalg.yield %sum : f32
        } -> tensor<2x3xf32>
        return %result : tensor<2x3xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    program = lane.programs["add_synthetic"]
    values = {
        program.generic.inputs[0]: np.array([[1.0], [4.0]], dtype=np.float32),
        program.generic.inputs[1]: np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
    }
    actual = lane.execute("add_synthetic", values)
    np.testing.assert_array_equal(
        actual,
        np.array([[11.0, 21.0, 31.0], [14.0, 24.0, 34.0]], dtype=np.float32),
    )
    assert program.signature["operand_maps"][0][1] == {"kind": "constant", "value": 0}


def test_real_div_uses_its_extracted_cross_axis_broadcast(real_lane: HostSemanticLane) -> None:
    program = real_lane.programs["div_0"]
    lhs = (np.arange(113, dtype=np.float32) + 2).reshape(1, 113, 1)
    rhs = (np.arange(32, dtype=np.float32) + 1).reshape(1, 1, 32)
    actual = real_lane.execute(
        "div_0", {program.generic.inputs[0]: lhs, program.generic.inputs[1]: rhs}
    )
    np.testing.assert_array_equal(actual, lhs / rhs)


def test_select_provenance_does_not_qualify_a_slice_as_pointwise_where(
    real_lane: HostSemanticLane,
) -> None:
    assert real_lane.signature_for("select_0") is None
    assert "non-pointwise scaffold" in real_lane.rejections["select_0"]
    try:
        real_lane.execute("select_0", {})
    except UnsupportedHostRegion:
        pass
    else:
        raise AssertionError("a provenance-only tensor slice must fail closed")


def test_hybrid_schedule_is_byte_stable_across_rebuilds() -> None:
    paths = [PLAN_ROOT / "hybrid_schedule.json", PLAN_ROOT / "hybrid_schedule_summary.json"]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
