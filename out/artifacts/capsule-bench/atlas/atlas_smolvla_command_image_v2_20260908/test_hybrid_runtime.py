"""Focused tests for the fail-closed Atlas SmolVLA hybrid scheduler."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from submission.mlir_oot.frontend import parse_verified
from submission.mlir_oot.hybrid_runtime import allocate_intervals
from submission.mlir_oot.host_semantics import HostSemanticLane, LayoutBridgeLane
from submission.mlir_oot.accelerator_semantics import AcceleratorContractLane


ROOT = Path(__file__).resolve().parent
PLAN_ROOT = ROOT / "whole_capture_plan"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_workload():
    capture = ROOT.parents[4] / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    return parse_verified(capture.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_lane(real_workload) -> HostSemanticLane:
    return HostSemanticLane(real_workload)


@pytest.fixture(scope="module")
def real_layout_lane(real_workload) -> LayoutBridgeLane:
    return LayoutBridgeLane(real_workload)


@pytest.fixture(scope="module")
def real_partition_plan() -> dict:
    return load(PLAN_ROOT / "partition_plan.json")


@pytest.fixture(scope="module")
def real_command_buffers(real_partition_plan) -> dict[str, dict]:
    return {
        row["kernel_id"]: load(ROOT / row["command_buffer"])
        for row in real_partition_plan["kernel_library"]
    }


@pytest.fixture(scope="module")
def real_accelerator_lane(
    real_workload, real_partition_plan, real_command_buffers,
) -> AcceleratorContractLane:
    return AcceleratorContractLane(
        real_workload, real_partition_plan, real_command_buffers
    )


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
        "host_materialized_layout_bridges": 358,
        "host_materialized_layout_bridges_by_rule": {
            "contiguous_identity_copy": 291,
            "contiguous_zero_stride_broadcast": 67,
        },
        "host_signature_regions_by_semantic": {
            "add": 215,
            "arange": 63,
            "aten_min_dim": 8,
            "bitwise": 16,
            "bucketize": 2,
            "cat": 95,
            "compare": 4,
            "convolution_im2col_matmul": 1,
            "cos": 57,
            "cumsum": 4,
            "div": 56,
            "dtype_cast": 472,
            "elementwise": 3,
            "embedding": 2,
            "fill": 50,
            "gelu": 12,
            "index_gather": 1,
            "index_put": 1,
            "layer_norm": 25,
            "mask_gather": 1,
            "minmax": 2,
            "mul": 536,
            "pow": 123,
            "reduce_mean": 66,
            "reduce_sum": 3,
            "rsqrt": 66,
            "select": 47,
            "sigmoid": 33,
            "sin": 57,
            "slice": 129,
            "slice_scatter": 112,
            "softmax": 44,
            "split": 56,
            "sub": 68,
        },
        "host_signature_regions_implemented": 2430,
        "layout_bridge_candidates": 2033,
        "materialized_copy_bridges": 112,
        "missing_host_semantics_reduction": 2408,
        "partition_host_region_overlap": ["conv_0"],
        "previous_bounded_host_regions_implemented": 22,
        "previous_missing_host_semantics": 2408,
        "proven_metadata_aliases": 1675,
        "qualified_accelerator_partitions": 3,
        "qualified_layout_bridges": 2033,
        "semantic_host_required_regions": 2430,
        "static_command_contract_partitions_qualified": 302,
        "strided_broadcast_bridges": 246,
        "structural_accelerator_partitions": 391,
    }
    assert schedule["fail_closed"]["missing_host_semantics"] == 0
    assert schedule["fail_closed"]["missing_host_semantics_by_semantic"] == {}
    assert schedule["fail_closed"]["missing_physical_event_runtime"] == 1
    assert schedule["fail_closed"]["unqualified_accelerator_command_contracts"] == 89
    assert schedule["fail_closed"]["unqualified_accelerator_partitions"] == 388
    assert schedule["fail_closed"]["unrealized_layout_bridges"] == 0
    assert schedule["conversion_boundaries"] == {
        "by_conversion": {
            "device_requantize": 88,
            "device_to_host_dequantize": 391,
            "host_to_device_quantize": 694,
            "host_to_device_quantize_bias": 77,
        },
        "count": 1250,
        "qualified": 12,
        "semantics_qualified": 947,
    }
    assert len(schedule["events"]) == 6104
    assert [row["event_index"] for row in schedule["events"]] == list(range(6104))
    qualified_host = [
        row for row in schedule["events"]
        if row["kind"] == "host_region" and row["executable"]
    ]
    assert len(qualified_host) == 2430
    assert all(len(row["operation_signature_sha256"]) == 64 for row in qualified_host)
    qualified_convolution = next(
        row for row in schedule["events"]
        if row.get("region_id") == "conv_0" and row["kind"] == "host_region"
    )
    assert qualified_convolution["semantic"] == "convolution_im2col_matmul"
    assert qualified_convolution["executable"] is True
    assert len(qualified_convolution["operation_signature_sha256"]) == 64
    layout_events = [row for row in schedule["events"] if row["kind"] == "layout_bridge"]
    assert len(layout_events) == 2033
    assert all(row["executable"] for row in layout_events)
    materialized = [
        row for row in layout_events
        if row["status"] == "host_materialized_layout_bridge"
    ]
    assert len(materialized) == 358
    assert all(len(row["layout_signature_sha256"]) == 64 for row in materialized)
    assert schedule["device_activation_arena"]["allocation_count"] == 391
    assert schedule["device_activation_arena"]["reuse_count"] > 0
    assert schedule["device_activation_arena"]["peak_bytes"] < (
        schedule["device_activation_arena"]["naive_no_reuse_bytes"]
    )


def test_accelerator_contract_census_and_numeric_witnesses_are_exact() -> None:
    schedule = load(PLAN_ROOT / "hybrid_schedule.json")
    census = schedule["accelerator_contract_census"]
    assert census["exact_class_count"] == 31
    assert census["by_kind"] == {"matmul": 303, "matmul_batched": 88}
    assert census["contract_qualified_by_kind"] == {"matmul": 302}
    assert census["contract_qualified_partitions"] == 302
    assert census["contract_unqualified_partitions"] == 89
    assert census["rejections_by_reason"] == {
        "batched command consumes raw W after declaring an unused resident pack": 88,
        "rank-2 source requires capture-specific preprocessing outside the command": 1,
    }
    assert len(census["classes"]) == 31
    assert sum(row["count"] for row in census["classes"]) == 391
    assert sum(row["contract_qualified"] for row in census["classes"]) == 302

    witnesses = schedule["accelerator_contract_numeric_witnesses"]
    assert [row["label"] for row in witnesses] == [
        "rank2_bf16_no_bias", "rank2_f32_no_bias", "rank2_f32_bias",
    ]
    assert sum(row["class_partition_count"] for row in witnesses) == 302
    assert all(
        row["status"] == "fresh_device_domain_execution_matches_independent_oracle"
        and row["output_sha256"] == row["oracle_sha256"]
        and row["published_capture_sha256"] == row["published_oracle_sha256"]
        and len(row["command_contract_sha256"]) == 64
        and "physical partition" in row["claim"]
        for row in witnesses
    )


def test_all_real_rank2_contracts_qualify_without_promoting_physical_execution(
    real_accelerator_lane: AcceleratorContractLane,
) -> None:
    assert len(real_accelerator_lane.contracts) == 302
    assert len(real_accelerator_lane.rejections) == 89
    assert Counter(
        contract.signature["source_dtype"]
        for contract in real_accelerator_lane.contracts.values()
    ) == {"bf16": 208, "f32": 94}
    assert Counter(
        contract.signature["bias_fused"]
        for contract in real_accelerator_lane.contracts.values()
    ) == {False: 225, True: 77}
    assert all(
        contract.signature["command"]["command_sequence"] == [
            "RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT",
        ]
        and "not calibration" in contract.signature["qualification_scope"]
        for contract in real_accelerator_lane.contracts.values()
    )
    schedule = load(PLAN_ROOT / "hybrid_schedule.json")
    accelerator_events = [
        row for row in schedule["events"] if row["kind"] == "accelerator_partition"
    ]
    assert sum(row["command_contract_qualified"] for row in accelerator_events) == 302
    assert sum(row["executable"] for row in accelerator_events) == 3
    conversion_events = [
        row for row in schedule["events"] if row["kind"] == "conversion_boundary"
    ]
    assert sum(row["conversion_semantics_qualified"] for row in conversion_events) == 947
    assert sum(row["executable"] for row in conversion_events) == 12


def test_rank2_device_domain_execution_and_conversion_are_independently_exact(
    real_accelerator_lane: AcceleratorContractLane,
) -> None:
    partition_id = "atlas_p0098"
    contract = real_accelerator_lane.contracts[partition_id].signature
    m, k, n = (contract["geometry"][key] for key in ("M", "K", "N"))
    activation = np.zeros((m, k), dtype=np.float32)
    activation[0, 7] = np.float32(1)
    weight = (((np.arange(k)[:, None] * 3 + np.arange(n)[None, :]) % 5) - 2).astype(
        np.float32
    )
    bias = ((np.arange(n) % 3) - 1).astype(np.float32)
    values = {"A0": activation, "W": weight, "B": bias}
    actual = real_accelerator_lane.execute_device_domain(partition_id, dict(values))
    expected_f32 = np.asarray(weight[7:8] + bias, dtype=np.float32)
    bits = expected_f32.view(np.uint32)
    expected = (
        bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    np.testing.assert_array_equal(actual, expected.view(np.float32))
    converted = real_accelerator_lane.prepare_capture_inputs(partition_id, values)
    assert len(converted["preloads"]["A0"]) == m * k
    assert len(converted["preloads"]["W"]) == k * n
    assert len(converted["preloads"]["B"]) == 2 * n
    assert converted["record"]["bias_equation"] == "BF16_RNE(B / (sA * sW))"
    published = real_accelerator_lane.publish_device_output(
        partition_id, actual, converted["record"]["output_scale"]
    )
    np.testing.assert_array_equal(
        published,
        np.asarray(actual * np.float32(converted["record"]["output_scale"]), dtype=np.float32),
    )


def test_malformed_command_and_capture_binding_fail_closed(
    real_workload, real_partition_plan, real_command_buffers,
) -> None:
    partition = next(
        row for row in real_partition_plan["partitions"]
        if row["partition_id"] == "atlas_p0001"
    )
    receipt = next(
        row for row in real_partition_plan["kernel_library"]
        if row["kernel_id"] == partition["kernel_id"]
    )
    bounded_plan = {"partitions": [copy.deepcopy(partition)],
                    "kernel_library": [copy.deepcopy(receipt)]}
    malformed_commands = {
        partition["kernel_id"]: copy.deepcopy(real_command_buffers[partition["kernel_id"]])
    }
    malformed_commands[partition["kernel_id"]]["commands"][1]["operands"]["rhs"] = "W"
    lane = AcceleratorContractLane(real_workload, bounded_plan, malformed_commands)
    assert lane.contracts == {}
    assert lane.rejections[partition["partition_id"]] == (
        "rank-2 command dependency chain changed"
    )

    malformed_plan = copy.deepcopy(bounded_plan)
    malformed_plan["partitions"][0]["capture_regions"][0] = "add_3"
    lane = AcceleratorContractLane(real_workload, malformed_plan, {
        partition["kernel_id"]: real_command_buffers[partition["kernel_id"]]
    })
    assert lane.contracts == {}
    assert lane.rejections[partition["partition_id"]] == (
        "source region is not one isolated linalg.matmul"
    )


def test_batched_and_patch_command_contracts_remain_explicitly_blocked(
    real_accelerator_lane: AcceleratorContractLane,
) -> None:
    assert real_accelerator_lane.rejections["atlas_p0000"] == (
        "rank-2 source requires capture-specific preprocessing outside the command"
    )
    batched = [
        reason for partition_id, reason in real_accelerator_lane.rejections.items()
        if partition_id != "atlas_p0000"
    ]
    assert len(batched) == 88
    assert set(batched) == {
        "batched command consumes raw W after declaring an unused resident pack"
    }


def test_rank2_conversion_runtime_shape_and_dtype_fail_closed(
    real_accelerator_lane: AcceleratorContractLane,
) -> None:
    with pytest.raises(ValueError, match="capture A0/W shape or dtype differs"):
        real_accelerator_lane.prepare_capture_inputs(
            "atlas_p0098",
            {
                "A0": np.zeros((1, 32), dtype=np.float64),
                "W": np.zeros((32, 960), dtype=np.float32),
                "B": np.zeros((960,), dtype=np.float32),
            },
        )


def _independent_layout_materialization(
    source: np.ndarray, signature: dict,
) -> np.ndarray:
    """Small test oracle derived from affine coordinates, not lane internals."""
    output_shape = tuple(signature["output_shape"])
    result = np.empty(output_shape, dtype=source.dtype, order="C")
    mapping = signature["input_map"]
    for output_index in np.ndindex(output_shape):
        input_index = tuple(
            output_index[item["position"]]
            if item["kind"] == "dim" else 0
            for item in mapping
        )
        result[output_index] = source[input_index]
    return result


def test_layout_bridge_census_and_saved_numeric_witnesses_are_exact() -> None:
    schedule = load(PLAN_ROOT / "hybrid_schedule.json")
    census = schedule["layout_bridge_census"]
    assert census["shape_class_count"] == 25
    assert census["by_semantic"] == {"copy": 112, "expand": 246}
    assert census["by_materialization"] == {
        "contiguous_identity_copy": 291,
        "contiguous_zero_stride_broadcast": 67,
    }
    assert len(census["shape_classes"]) == 25
    assert sum(row["count"] for row in census["shape_classes"]) == 358

    witnesses = schedule["layout_bridge_numeric_witnesses"]
    assert [row["label"] for row in witnesses] == [
        "copy_f32_contiguous_identity_copy_rank4",
        "expand_bf16_contiguous_identity_copy_rank4",
        "expand_bf16_contiguous_zero_stride_broadcast_rank5",
        "expand_f32_contiguous_identity_copy_rank2",
        "expand_f32_contiguous_identity_copy_rank4",
        "expand_f32_contiguous_zero_stride_broadcast_rank3",
        "expand_f32_contiguous_zero_stride_broadcast_rank5",
        "expand_i1_contiguous_identity_copy_rank2",
        "expand_i1_contiguous_identity_copy_rank4",
        "expand_i1_contiguous_zero_stride_broadcast_rank2",
        "expand_i1_contiguous_zero_stride_broadcast_rank3",
    ]
    assert sum(row["class_region_count"] for row in witnesses) == 358
    assert all(
        row["status"] == "fresh_numeric_execution_matches_independent_oracle"
        and row["output_sha256"] == row["oracle_sha256"]
        and row["distinct_contiguous_storage"] is True
        and "physical DMA/event execution is absent" in row["claim"]
        for row in witnesses
    )


def test_all_real_layout_bridges_qualify_and_each_topology_executes_exactly(
    real_layout_lane: LayoutBridgeLane,
) -> None:
    programs = list(real_layout_lane.programs.values())
    assert len(programs) == 358
    assert real_layout_lane.rejections == {}
    assert Counter(program.semantic for program in programs) == {
        "copy": 112,
        "expand": 246,
    }
    assert Counter(
        program.signature["materialization"] for program in programs
    ) == {
        "contiguous_identity_copy": 291,
        "contiguous_zero_stride_broadcast": 67,
    }
    grouped = {}
    for program in programs:
        signature = program.signature
        key = (
            program.semantic,
            signature["dtype"],
            signature["materialization"],
            len(signature["output_shape"]),
            json.dumps(signature["input_map"], sort_keys=True),
        )
        grouped.setdefault(key, []).append(program)
    assert len(grouped) == 11

    for candidates in grouped.values():
        program = min(
            candidates,
            key=lambda row: (np.prod(row.signature["output_shape"]), row.region_id),
        )
        signature = program.signature
        input_shape = tuple(signature["input_shape"])
        if signature["dtype"] == "i1":
            source = (np.arange(np.prod(input_shape)).reshape(input_shape) % 3) == 0
        else:
            source = (
                np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
                % np.float32(31)
            )
        values = {program.input_values[0]: source}
        actual = real_layout_lane.execute(program.region_id, values)
        expected = _independent_layout_materialization(source, signature)
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.c_contiguous
        assert not np.shares_memory(actual, source)
        assert values[program.output_value] is actual


def test_layout_bridge_malformed_scalar_body_and_copy_broadcast_fail_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%source: tensor<1x1xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
        %first_empty = tensor.empty() : tensor<1x4xf32>
        %wrong_yield = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%source : tensor<1x1xf32>) outs(%first_empty : tensor<1x4xf32>)
          attrs = {prov.region_id = "false_expand", prov.op = "expand",
                   prov.family = "layout", prov.aten = "aten.expand.default"} {
        ^bb0(%value: f32, %old: f32):
          linalg.yield %old : f32
        } -> tensor<1x4xf32>
        %second_empty = tensor.empty() : tensor<1x4xf32>
        %copy_broadcast = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%source : tensor<1x1xf32>) outs(%second_empty : tensor<1x4xf32>)
          attrs = {prov.region_id = "false_copy", prov.op = "copy",
                   prov.family = "layout", prov.aten = "aten.copy.default"} {
        ^bb0(%value: f32, %old: f32):
          linalg.yield %value : f32
        } -> tensor<1x4xf32>
        return %wrong_yield, %copy_broadcast : tensor<1x4xf32>, tensor<1x4xf32>
      }
    }''')
    lane = LayoutBridgeLane(workload)
    assert lane.programs == {}
    assert lane.rejections == {
        "false_expand": "layout bridge scalar body is not an exact value copy",
        "false_copy": "copy bridge is not an exact identity materialization",
    }


def test_layout_bridge_runtime_shape_check_fails_closed(
    real_layout_lane: LayoutBridgeLane,
) -> None:
    program = real_layout_lane.programs["expand_49"]
    with pytest.raises(ValueError, match="runtime layout input shape differs"):
        real_layout_lane.execute(
            program.region_id,
            {program.input_values[0]: np.ones((1, 2), dtype=np.bool_)},
        )
    f32_program = real_layout_lane.programs["expand_148"]
    with pytest.raises(ValueError, match="runtime layout input dtype differs"):
        real_layout_lane.execute(
            f32_program.region_id,
            {f32_program.input_values[0]: np.ones((1, 50), dtype=np.float64)},
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
        "cumsum_reduce_mean", "masked_softmax", "argmin_successor",
        "reduce_sum_successor", "layer_norm_standalone",
        "static_split_slice", "slice_scatter_successor", "concat_successor",
        "bitwise_reduction", "bucketize_chain", "static_select_slice_chain",
        "gelu_standalone",
    ]
    covered = set()
    for row in witnesses:
        covered.update(row["semantics"])
        assert row["status"] == "fresh_numeric_execution_exactly_replayed"
        assert row["replay_hashes_equal"] is True
        assert all(output["finite"] for output in row["outputs"])
        if row["label"] not in {"gelu_standalone", "layer_norm_standalone"}:
            assert row["dependency_edges"] > 0
    assert {
        "pow", "elementwise", "rsqrt", "sigmoid", "sin", "cos", "gelu",
        "arange", "fill",
        "reduce_mean", "softmax", "layer_norm", "aten_min_dim", "cumsum",
        "reduce_sum", "slice", "split", "slice_scatter", "cat", "bitwise",
        "bucketize", "select",
    } <= covered


def test_final_indexed_regions_have_fresh_independent_numeric_witnesses() -> None:
    witnesses = load(PLAN_ROOT / "hybrid_schedule.json")[
        "final_indexed_host_numeric_witnesses"
    ]
    assert [row["label"] for row in witnesses] == [
        "embedding_0", "embedding_1", "index_gather",
        "mask_gather_index_put", "patch_embedding_convolution",
    ]
    assert [region_id for row in witnesses for region_id in row["region_ids"]] == [
        "gather_0", "gather_2", "gather_1",
        "mask_gather_0", "mask_scatter_0", "conv_0",
    ]
    assert all(
        row["status"] == "fresh_numeric_execution_matches_independent_oracle"
        and row["output_sha256"] == row["oracle_sha256"]
        and "not device or whole-model E2E" in row["claim"]
        for row in witnesses
    )


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


def _left_fold_last_axis(source: np.ndarray, initial, operation) -> np.ndarray:
    accumulator = np.full(source.shape[:-1], initial, dtype=source.dtype)
    for index in range(source.shape[-1]):
        accumulator = np.asarray(
            operation(source[..., index], accumulator), dtype=source.dtype
        )
    return accumulator


def test_all_real_reduction_signatures_qualify_and_representatives_are_exact(
    real_lane: HostSemanticLane,
) -> None:
    expected_counts = {
        "reduce_mean": 66,
        "softmax": 44,
        "layer_norm": 25,
        "aten_min_dim": 8,
        "cumsum": 4,
        "reduce_sum": 3,
    }
    reductions = [
        program for program in real_lane.programs.values()
        if program.semantic in expected_counts
    ]
    assert Counter(program.semantic for program in reductions) == expected_counts
    assert all(
        program.signature["schema"] == "atlas_host_reduction_signature_v1"
        and program.signature["accumulation_rule"].startswith("captured row-major")
        for program in reductions
    )

    mean = min(
        (program for program in reductions if program.semantic == "reduce_mean"),
        key=lambda program: program.signature["input_shapes"],
    )
    mean_input = (
        (np.arange(np.prod(mean.signature["input_shapes"][0]), dtype=np.float32) % 31)
        / np.float32(7)
    ).reshape(mean.signature["input_shapes"][0])
    mean_sum = _left_fold_last_axis(mean_input, np.float32(0), np.add)
    mean_expected = (mean_sum / np.float32(mean_input.shape[-1]))[..., None]
    np.testing.assert_array_equal(
        real_lane.execute(mean.region_id, {mean.input_values[0]: mean_input}), mean_expected
    )

    softmax = min(
        (program for program in reductions if program.semantic == "softmax"),
        key=lambda program: np.prod(program.signature["input_shapes"][0]),
    )
    softmax_input = (
        (np.arange(np.prod(softmax.signature["input_shapes"][0]), dtype=np.float32) % 19)
        / np.float32(5)
    ).reshape(softmax.signature["input_shapes"][0])
    maximum = _left_fold_last_axis(softmax_input, np.float32(-np.inf), np.maximum)
    shifted = np.asarray(softmax_input - maximum[..., None], dtype=np.float32)
    exponent = np.asarray(np.exp(shifted), dtype=np.float32)
    denominator = _left_fold_last_axis(exponent, np.float32(0), np.add)
    softmax_expected = np.asarray(exponent / denominator[..., None], dtype=np.float32)
    np.testing.assert_array_equal(
        real_lane.execute(
            softmax.region_id, {softmax.input_values[0]: softmax_input}
        ),
        softmax_expected,
    )

    layer_norm = next(
        program for program in reductions if program.semantic == "layer_norm"
    )
    source_shape = layer_norm.signature["input_shapes"][0]
    source = (
        (np.arange(np.prod(source_shape), dtype=np.float32) % 23) / np.float32(9)
    ).reshape(source_shape)
    gamma = np.linspace(
        np.float32(0.5), np.float32(1.5), source_shape[-1], dtype=np.float32
    )
    beta = np.linspace(
        np.float32(-0.25), np.float32(0.25), source_shape[-1], dtype=np.float32
    )
    layer_mean = np.asarray(
        _left_fold_last_axis(source, np.float32(0), np.add)
        / np.float32(source_shape[-1]),
        dtype=np.float32,
    )
    centered = np.asarray(source - layer_mean[..., None], dtype=np.float32)
    squared = np.asarray(centered * centered, dtype=np.float32)
    variance = np.asarray(
        _left_fold_last_axis(squared, np.float32(0), np.add)
        / np.float32(source_shape[-1]),
        dtype=np.float32,
    )
    inverse_std = np.asarray(
        np.reciprocal(np.sqrt(np.asarray(variance + np.float32(1e-6), dtype=np.float32))),
        dtype=np.float32,
    )
    layer_expected = np.asarray(
        np.asarray(centered * inverse_std[..., None], dtype=np.float32) * gamma + beta,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        real_lane.execute(
            layer_norm.region_id,
            {
                layer_norm.input_values[0]: source,
                layer_norm.input_values[1]: gamma,
                layer_norm.input_values[2]: beta,
            },
        ),
        layer_expected,
    )

    argmin = next(
        program for program in reductions if program.semantic == "aten_min_dim"
    )
    argmin_input = np.arange(50, 0, -1, dtype=np.int64).reshape(1, 50)
    argmin_values = {argmin.input_values[0]: argmin_input}
    np.testing.assert_array_equal(
        real_lane.execute(argmin.region_id, argmin_values),
        np.array([[1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        argmin_values[argmin.operations[-1].results[0]],
        np.array([[49]], dtype=np.int64),
    )

    cumsum = next(
        program for program in reductions
        if program.semantic == "cumsum" and program.signature["output_dtype"] == "f32"
    )
    cumsum_input = np.arange(1, 51, dtype=np.float32).reshape(1, 50)
    cumsum_expected = np.add.accumulate(cumsum_input, axis=1, dtype=np.float32)
    np.testing.assert_array_equal(
        real_lane.execute(cumsum.region_id, {cumsum.input_values[0]: cumsum_input}),
        cumsum_expected,
    )

    reduce_sum = next(
        program for program in reductions if program.semantic == "reduce_sum"
    )
    bool_input = (np.arange(32).reshape(1, 32) % 3) == 0
    np.testing.assert_array_equal(
        real_lane.execute(
            reduce_sum.region_id, {reduce_sum.input_values[0]: bool_input}
        ),
        np.array([11], dtype=np.int64),
    )


def _captured_slices(record: dict) -> tuple[slice, ...]:
    return tuple(
        slice(offset, offset + size * stride, stride)
        for offset, size, stride in zip(
            record["offsets"], record["sizes"], record["strides"]
        )
    )


def test_all_real_movement_signatures_qualify_and_representatives_are_exact(
    real_lane: HostSemanticLane,
) -> None:
    expected_counts = {
        "slice": 129,
        "split": 56,
        "slice_scatter": 112,
        "cat": 95,
        "select": 2,
        "bitwise": 16,
        "bucketize": 2,
    }
    movement = [
        program for program in real_lane.programs.values()
        if (program.semantic in expected_counts
            and program.signature["schema"] == "atlas_host_movement_signature_v1")
    ]
    assert Counter(program.semantic for program in movement) == expected_counts
    assert all(
        program.signature["materialization_rule"]
        == "execute captured static indexing/concat/generic operations in order"
        for program in movement
    )

    extraction = next(program for program in movement if program.semantic == "slice")
    extraction_input = np.arange(
        np.prod(extraction.signature["input_shapes"][0]), dtype=np.float32
    ).reshape(extraction.signature["input_shapes"][0])
    extraction_expected = np.array(
        extraction_input[_captured_slices(extraction.signature["static_slices"][0])],
        copy=True,
    )
    np.testing.assert_array_equal(
        real_lane.execute(
            extraction.region_id, {extraction.input_values[0]: extraction_input}
        ),
        extraction_expected,
    )

    split = next(program for program in movement if program.semantic == "split")
    split_input = np.arange(
        np.prod(split.signature["input_shapes"][0]), dtype=np.float32
    ).reshape(split.signature["input_shapes"][0])
    split_values = {split.input_values[0]: split_input}
    first = real_lane.execute(split.region_id, split_values)
    np.testing.assert_array_equal(
        first,
        split_input[_captured_slices(split.signature["static_slices"][0])],
    )
    np.testing.assert_array_equal(
        split_values[split.operations[1].results[0]],
        split_input[_captured_slices(split.signature["static_slices"][1])],
    )

    scatter = next(
        program for program in movement if program.semantic == "slice_scatter"
    )
    source = -np.arange(
        1, np.prod(scatter.signature["input_shapes"][0]) + 1, dtype=np.float32
    ).reshape(scatter.signature["input_shapes"][0])
    destination = np.arange(
        np.prod(scatter.signature["input_shapes"][1]), dtype=np.float32
    ).reshape(scatter.signature["input_shapes"][1])
    scatter_expected = destination.copy()
    scatter_expected[_captured_slices(scatter.signature["static_slices"][0])] = source
    np.testing.assert_array_equal(
        real_lane.execute(
            scatter.region_id,
            {
                scatter.input_values[0]: source,
                scatter.input_values[1]: destination,
            },
        ),
        scatter_expected,
    )

    concat = next(
        program for program in movement
        if (program.semantic == "cat"
            and program.signature["operation_sequence"]
            == ["linalg.generic", "tensor.concat"])
    )
    concat_inputs = [
        (np.arange(np.prod(shape), dtype=np.float32) + 1000 * index).reshape(shape)
        for index, shape in enumerate(concat.signature["input_shapes"])
    ]
    concat_sources = dict(zip(concat.input_values, concat_inputs))
    concat_sources[concat.operations[0].results[0]] = concat_inputs[0]
    expected_concat = np.concatenate(
        [concat_sources[value] for value in concat.operations[-1].operands],
        axis=concat.signature["concats"][0]["dimension"],
    )
    np.testing.assert_array_equal(
        real_lane.execute(
            concat.region_id,
            dict(zip(concat.input_values, concat_inputs)),
        ),
        expected_concat,
    )

    select = next(program for program in movement if program.semantic == "select")
    select_input = (
        np.arange(np.prod(select.signature["input_shapes"][0])).reshape(
            select.signature["input_shapes"][0]
        ) % 3
    ) == 0
    selected = select_input[
        _captured_slices(select.signature["static_slices"][0])
    ].reshape(select.signature["output_shape"])
    np.testing.assert_array_equal(
        real_lane.execute(select.region_id, {select.input_values[0]: select_input}),
        selected,
    )

    bitwise_and = next(
        program for program in movement
        if (program.semantic == "bitwise"
            and program.region_id == "bitwise_15")
    )
    lhs = (
        np.arange(np.prod(bitwise_and.signature["input_shapes"][0])).reshape(
            bitwise_and.signature["input_shapes"][0]
        ) % 2
    ) == 0
    rhs = (
        np.arange(np.prod(bitwise_and.signature["input_shapes"][1])).reshape(
            bitwise_and.signature["input_shapes"][1]
        ) % 3
    ) == 0
    np.testing.assert_array_equal(
        real_lane.execute(
            bitwise_and.region_id,
            {bitwise_and.input_values[0]: lhs, bitwise_and.input_values[1]: rhs},
        ),
        np.bitwise_and(lhs, rhs),
    )

    bitwise_not = next(
        program for program in movement
        if (program.semantic == "bitwise"
            and program.signature["generics"][0]["scalar_ops"]
            == ["arith.constant", "arith.xori"])
    )
    not_input = (
        np.arange(np.prod(bitwise_not.signature["input_shapes"][0])).reshape(
            bitwise_not.signature["input_shapes"][0]
        ) % 5
    ) == 0
    np.testing.assert_array_equal(
        real_lane.execute(
            bitwise_not.region_id, {bitwise_not.input_values[0]: not_input}
        ),
        np.logical_not(not_input),
    )

    bucketize = next(program for program in movement if program.semantic == "bucketize")
    bucket_values = np.linspace(-2, 2, 32, dtype=np.float32).reshape(1, 32)
    boundaries = np.linspace(-1, 1, 31, dtype=np.float32)
    np.testing.assert_array_equal(
        real_lane.execute(
            bucketize.region_id,
            {
                bucketize.input_values[0]: bucket_values,
                bucketize.input_values[1]: boundaries,
            },
        ),
        np.searchsorted(boundaries, bucket_values, side="right"),
    )


def test_declared_slice_with_two_extractions_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%arg: tensor<1x4xf32>) -> tensor<1x2xf32> {
        %first = "tensor.extract_slice"(%arg) <{
          static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 2>,
          static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
        }> {prov.region_id = "false_slice", prov.op = "slice",
             prov.family = "layout", prov.aten = "aten.slice.Tensor"}
          : (tensor<1x4xf32>) -> tensor<1x2xf32>
        %second = "tensor.extract_slice"(%arg) <{
          static_offsets = array<i64: 0, 2>, static_sizes = array<i64: 1, 2>,
          static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
        }> {prov.region_id = "false_slice", prov.op = "slice",
             prov.family = "layout", prov.aten = "aten.slice.Tensor"}
          : (tensor<1x4xf32>) -> tensor<1x2xf32>
        return %second : tensor<1x2xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_slice") is None
    assert lane.rejections["false_slice"] == "slice is not one exact static extraction"


def test_declared_bitwise_with_wrong_scalar_body_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%lhs: tensor<4xi1>, %rhs: tensor<4xi1>) -> tensor<4xi1> {
        %empty = tensor.empty() : tensor<4xi1>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%lhs, %rhs : tensor<4xi1>, tensor<4xi1>) outs(%empty : tensor<4xi1>)
          attrs = {prov.region_id = "false_bitwise", prov.op = "bitwise",
                   prov.family = "bitwise", prov.aten = "aten.bitwise_or.Tensor"} {
        ^bb0(%left: i1, %right: i1, %old: i1):
          %wrong = arith.ori %left, %right : i1
          linalg.yield %wrong : i1
        } -> tensor<4xi1>
        return %result : tensor<4xi1>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_bitwise") is None
    assert lane.rejections["false_bitwise"] == (
        "generic scalar body contains an unsupported operation"
    )


def test_final_six_real_indexed_regions_execute_exactly(real_lane: HostSemanticLane) -> None:
    final_ids = {
        "conv_0", "mask_gather_0", "mask_scatter_0",
        "gather_0", "gather_1", "gather_2",
    }
    assert final_ids <= real_lane.programs.keys()
    assert len(real_lane.programs) == 2430
    assert all(
        real_lane.programs[region_id].signature["schema"] in {
            "atlas_host_indexed_signature_v1",
            "atlas_host_im2col_convolution_signature_v1",
        }
        for region_id in final_ids
    )

    embedding = real_lane.programs["gather_0"]
    embedding_indices = np.arange(1023, -1, -1, dtype=np.int64).reshape(1, 1024)
    embedding_table = (
        np.arange(1024, dtype=np.float32)[:, None] * np.float32(1000)
        + np.arange(768, dtype=np.float32)[None, :]
    )
    np.testing.assert_array_equal(
        real_lane.execute(
            embedding.region_id,
            {
                embedding.input_values[0]: embedding_indices,
                embedding.input_values[1]: embedding_table,
            },
        ),
        embedding_table[embedding_indices],
    )

    bf16_embedding = real_lane.programs["gather_2"]
    bf16_indices = (np.arange(48, dtype=np.int64) * 101).reshape(1, 48)
    exact_bf16_row = np.arange(960, dtype=np.float32) % np.float32(32)
    bf16_table = np.broadcast_to(exact_bf16_row, (49280, 960))
    np.testing.assert_array_equal(
        real_lane.execute(
            bf16_embedding.region_id,
            {
                bf16_embedding.input_values[0]: bf16_indices,
                bf16_embedding.input_values[1]: bf16_table,
            },
        ),
        np.broadcast_to(exact_bf16_row, (1, 48, 960)),
    )

    index_gather = real_lane.programs["gather_1"]
    row_indices = np.zeros((1, 1, 1, 1), dtype=np.int64)
    column_indices = np.arange(1023, -1, -1, dtype=np.int64).reshape(1, 1, 1, 1024)
    bool_table = ((np.arange(1024) * 7) % 11 < 5).reshape(1, 1024)
    np.testing.assert_array_equal(
        real_lane.execute(
            index_gather.region_id,
            {
                index_gather.input_values[0]: row_indices,
                index_gather.input_values[1]: column_indices,
                index_gather.input_values[2]: bool_table,
            },
        ),
        bool_table[row_indices, column_indices],
    )

    mask_gather = real_lane.programs["mask_gather_0"]
    data = (np.arange(1024, dtype=np.int64) * 13 - 7).reshape(1, 1024)
    mask = ((np.arange(1024) * 5) % 17 < 6).reshape(1, 1024)
    chain_values = {
        mask_gather.input_values[0]: data,
        mask_gather.input_values[1]: mask,
    }
    compact = real_lane.execute(mask_gather.region_id, chain_values)
    np.testing.assert_array_equal(compact, data.reshape(-1)[mask.reshape(-1)])

    index_put = real_lane.programs["mask_scatter_0"]
    destination = np.full((1, 1024), -99, dtype=np.int64)
    chain_values[index_put.input_values[0]] = destination
    chain_values[index_put.input_values[1]] = mask
    scattered = real_lane.execute(index_put.region_id, chain_values)
    expected_scatter = destination.copy()
    expected_scatter[mask] = compact
    np.testing.assert_array_equal(scattered, expected_scatter)

    convolution = real_lane.programs["conv_0"]
    image = np.arange(1 * 3 * 512 * 512, dtype=np.float32).reshape(1, 3, 512, 512)
    weight = np.zeros((768, 3, 16, 16), dtype=np.float32)
    weight[0, 0, 0, 0] = np.float32(2)
    weight[1, 2, 15, 15] = np.float32(-1)
    bias = (np.arange(768, dtype=np.float32) - np.float32(384)) / np.float32(8)
    actual_conv = real_lane.execute(
        convolution.region_id,
        {
            convolution.input_values[0]: image,
            convolution.input_values[1]: weight,
            convolution.input_values[2]: bias,
        },
    )
    expected_conv = np.broadcast_to(
        bias.reshape(1, 768, 1, 1), actual_conv.shape
    ).copy()
    expected_conv[0, 0] += np.float32(2) * image[0, 0, 0::16, 0::16]
    expected_conv[0, 1] -= image[0, 2, 15::16, 15::16]
    np.testing.assert_array_equal(actual_conv, expected_conv)


def test_indexed_runtime_bounds_and_update_count_fail_closed(
    real_lane: HostSemanticLane,
) -> None:
    embedding = real_lane.programs["gather_0"]
    with pytest.raises(ValueError, match="runtime index is out of bounds"):
        real_lane.execute(
            embedding.region_id,
            {
                embedding.input_values[0]: np.full((1, 1024), 1024, dtype=np.int64),
                embedding.input_values[1]: np.zeros((1024, 768), dtype=np.float32),
            },
        )

    index_put = real_lane.programs["mask_scatter_0"]
    mask = np.zeros((1, 1024), dtype=np.bool_)
    mask[0, :3] = True
    with pytest.raises(ValueError, match="runtime update count differs from mask"):
        real_lane.execute(
            index_put.region_id,
            {
                index_put.input_values[0]: np.zeros((1, 1024), dtype=np.int64),
                index_put.input_values[1]: mask,
                index_put.input_values[2]: np.array([1, 2], dtype=np.int64),
            },
        )


def test_declared_embedding_with_wrong_feature_index_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%indices: tensor<1x2xi64>, %table: tensor<4x3xf32>)
          -> tensor<1x2x3xf32> {
        %empty = tensor.empty() : tensor<1x2x3xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%indices : tensor<1x2xi64>) outs(%empty : tensor<1x2x3xf32>)
          attrs = {prov.region_id = "false_embedding", prov.op = "embedding",
                   prov.family = "gather_scatter", prov.aten = "aten.embedding.default"} {
        ^bb0(%index: i64, %old: f32):
          %row = arith.index_cast %index : i64 to index
          %feature = linalg.index 1 : index
          %value = tensor.extract %table[%row, %feature] : tensor<4x3xf32>
          linalg.yield %value : f32
        } -> tensor<1x2x3xf32>
        return %result : tensor<1x2x3xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_embedding") is None
    assert lane.rejections["false_embedding"] == (
        "embedding scalar dataflow is not the captured indexed load"
    )


def test_declared_mask_and_convolution_with_incomplete_topology_fail_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%data: tensor<1x4xi64>, %image: tensor<1x1x2x2xf32>)
          -> (tensor<4xi64>, tensor<1x1x2x2xf32>) {
        %flat = tensor.collapse_shape %data [[0, 1]] {
          prov.region_id = "false_mask", prov.op = "mask_gather",
          prov.family = "gather_scatter", prov.aten = "aten.index.Tensor"
        } : tensor<1x4xi64> into tensor<4xi64>
        %empty = tensor.empty() : tensor<1x1x2x2xf32>
        %copy = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%image : tensor<1x1x2x2xf32>) outs(%empty : tensor<1x1x2x2xf32>)
          attrs = {prov.region_id = "false_conv", prov.op = "convolution_im2col_matmul",
                   prov.family = "contraction", prov.aten = "aten.convolution.default"} {
        ^bb0(%value: f32, %old: f32):
          linalg.yield %value : f32
        } -> tensor<1x1x2x2xf32>
        return %flat, %copy : tensor<4xi64>, tensor<1x1x2x2xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.rejections["false_mask"] == (
        "mask_gather does not match its complete captured topology"
    )
    assert lane.rejections["false_conv"] == (
        "im2col convolution does not match its captured topology"
    )


def test_declared_reduction_with_incomplete_topology_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%arg: tensor<1x4xf32>) -> tensor<1x4xf32> {
        %empty = tensor.empty() : tensor<1x4xf32>
        %result = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg : tensor<1x4xf32>) outs(%empty : tensor<1x4xf32>)
          attrs = {prov.region_id = "false_mean", prov.op = "reduce_mean",
                   prov.family = "reduce", prov.aten = "aten.mean.dim"} {
        ^bb0(%value: f32, %old: f32):
          linalg.yield %value : f32
        } -> tensor<1x4xf32>
        return %result : tensor<1x4xf32>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_mean") is None
    assert lane.rejections["false_mean"] == (
        "region does not match its complete captured reduction pattern"
    )


def test_reduce_sum_with_wrong_accumulator_body_fails_closed() -> None:
    workload = parse_verified(r'''builtin.module {
      func.func @forward(%arg: tensor<1x4xi1>) -> tensor<1xi64> {
        %converted_empty = tensor.empty() : tensor<1x4xi64>
        %converted = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg : tensor<1x4xi1>) outs(%converted_empty : tensor<1x4xi64>)
          attrs = {prov.region_id = "false_sum", prov.op = "reduce_sum",
                   prov.family = "reduce", prov.aten = "aten.sum.dim_IntList"} {
        ^bb0(%value: i1, %old: i64):
          %wide = arith.extui %value : i1 to i64
          linalg.yield %wide : i64
        } -> tensor<1x4xi64>
        %zero = arith.constant {
          prov.region_id = "false_sum", prov.op = "reduce_sum",
          prov.family = "reduce", prov.aten = "aten.sum.dim_IntList"
        } 0 : i64
        %init = tensor.splat %zero {
          prov.region_id = "false_sum", prov.op = "reduce_sum",
          prov.family = "reduce", prov.aten = "aten.sum.dim_IntList"
        } : tensor<1xi64>
        %result = linalg.reduce ins(%converted : tensor<1x4xi64>)
          outs(%init : tensor<1xi64>) dimensions = [1]
          (%value: i64, %acc: i64) {
            %wrong = arith.muli %value, %acc : i64
            linalg.yield %wrong : i64
          }
        return %result : tensor<1xi64>
      }
    }''')
    lane = HostSemanticLane(workload)
    assert lane.signature_for("false_sum") is None
    assert lane.rejections["false_sum"] == "linalg.reduce scalar body is unsupported"


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


def test_select_slice_is_qualified_only_as_an_exact_movement_chain(
    real_lane: HostSemanticLane,
) -> None:
    signature = real_lane.signature_for("select_0")
    assert signature is not None
    assert signature["schema"] == "atlas_host_movement_signature_v1"
    assert signature["operation_sequence"] == [
        "tensor.extract_slice", "tensor.collapse_shape", "tensor.expand_shape",
    ]


def test_hybrid_schedule_is_byte_stable_across_rebuilds() -> None:
    paths = [PLAN_ROOT / "hybrid_schedule.json", PLAN_ROOT / "hybrid_schedule_summary.json"]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    subprocess.run([sys.executable, str(ROOT / "build_hybrid_schedule.py")], check=True,
                   capture_output=True, text=True, timeout=60)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
