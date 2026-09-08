"""Tests for bounded real-capture Atlas calibration/dispatch boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.capture_bridge import (  # noqa: E402
    ATLAS_GSIM_DRAM_BYTES,
    FP8_MAX_FINITE,
    bridge_inputs,
    build_dispatch_manifest,
    calibrate_e4m3,
    comparison,
    f32_to_bf16_rne,
    passes_tolerance,
    read_f32_safetensor,
    sha256_bytes,
    validate_gsim_dram_window,
    validate_partition_abi,
)
from run_capture_partition import (  # noqa: E402
    PARTITIONS,
    _load_capture_values,
    _tiled_interface,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def state_proj_partition() -> dict:
    plan = load(ROOT / "whole_capture_plan/partition_plan.json")
    matches = [entry for entry in plan["partitions"]
               if entry["partition_id"] == "atlas_p0098"]
    assert len(matches) == 1
    return matches[0]


def partition_by_id(partition_id: str) -> dict:
    plan = load(ROOT / "whole_capture_plan/partition_plan.json")
    matches = [entry for entry in plan["partitions"]
               if entry["partition_id"] == partition_id]
    assert len(matches) == 1
    return matches[0]


def test_calibration_scale_equations_and_rounding_are_explicit_and_deterministic() -> None:
    values = np.asarray([[0.0, -1.0, 2.0, 0.25]], dtype=np.float32)
    first = calibrate_e4m3("A0", values)
    second = calibrate_e4m3("A0", values.copy())
    assert first["scale"] == 2.0 / FP8_MAX_FINITE
    assert first["record"] == second["record"]
    assert first["codes"].tobytes() == second["codes"].tobytes()
    assert np.array_equal(first["decoded"], second["decoded"])
    headroom = calibrate_e4m3("A0", values, code_cap=16.0)
    assert headroom["scale"] == 2.0 / 16.0
    assert headroom["record"]["code_cap"] == 16.0
    for invalid_cap in (0.0, FP8_MAX_FINITE + 1.0, math.nan):
        try:
            calibrate_e4m3("A0", values, code_cap=invalid_cap)
        except ValueError as error:
            assert "code cap" in str(error)
        else:
            raise AssertionError(f"invalid calibration cap {invalid_cap} was accepted")

    bias = np.asarray([0.125, -0.25, 1.0], dtype=np.float32)
    words, decoded = f32_to_bf16_rne(bias)
    assert words.dtype == np.dtype("<u2")
    assert np.array_equal(decoded, bias)


def test_bias_is_folded_in_quant_domain_and_output_scale_is_restored() -> None:
    activation = np.asarray([[1.0, -2.0]], dtype=np.float32)
    weight = np.asarray([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)
    bias = np.asarray([0.125, -0.375], dtype=np.float32)
    bridged = bridge_inputs(activation, weight, bias, {"M": 1, "K": 2, "N": 2})
    expected_scale = (2.0 / FP8_MAX_FINITE) * (2.0 / FP8_MAX_FINITE)
    assert bridged["record"]["output_scale"] == expected_scale
    assert bridged["record"]["bias"]["equation"] == "BF16_RNE(B / output_scale)"
    reconstructed_bias = (
        bridged["decoded"]["B_quant_domain"]
        * np.float32(bridged["record"]["output_scale"])
    )
    assert float(np.max(np.abs(reconstructed_bias - bias))) == (
        bridged["record"]["bias"]["max_abs_error_after_output_rescale"]
    )
    assert len(bridged["preloads"]["A0"]) == 2
    assert len(bridged["preloads"]["W"]) == 4
    assert len(bridged["preloads"]["B"]) == 4


def test_gsim_dram_window_rejects_full_p0244_and_accepts_declared_n_tiles() -> None:
    command_root = ROOT / "whole_capture_plan/command_buffers"
    full = load(command_root / "matmul_50_1440_720_bias.json")
    try:
        validate_gsim_dram_window(full)
    except ValueError as error:
        assert str(error) == (
            "Atlas GSIM DRAM allocation span exceeds its alias-free window: "
            "1182240 > 1048576 bytes (over by 133664)"
        )
    else:
        raise AssertionError("oversized p0244 command image was accepted")

    assert PARTITIONS["action_time_mlp_in"]["n_tiles"] == [256, 256, 208]
    for width in PARTITIONS["action_time_mlp_in"]["n_tiles"]:
        # These are the exact contiguous allocation sizes emitted by cmdbuf.py;
        # alignment can add at most 31 bytes between four tensors.
        allocated = 1440 * width + 50 * 1440 + 2 * width + 2 * 50 * width
        assert allocated + 3 * 31 < ATLAS_GSIM_DRAM_BYTES
        source = _tiled_interface(50, 1440, width)
        assert f"tensor<1440x{width}xf8E4M3FN>" in source
        assert f"tensor<50x{width}xbf16>" in source


def test_gsim_dram_window_rejects_masked_overlap_even_below_total_bytes() -> None:
    cb = {
        "tensors": {
            "A0": {"base": 0x90000000, "shape": [16], "dtype": "i8"},
            "Y0": {
                "base": 0x90000000 + ATLAS_GSIM_DRAM_BYTES,
                "shape": [16], "dtype": "i8",
            },
        }
    }
    try:
        validate_gsim_dram_window(cb)
    except ValueError as error:
        assert "allocation span exceeds" in str(error)
    else:
        raise AssertionError("masked address alias was accepted")


def test_dispatch_manifest_comes_from_planned_dependency_and_lifetime_abi() -> None:
    partition = state_proj_partition()
    dispatch = build_dispatch_manifest(partition)
    assert dispatch["partition_id"] == "atlas_p0098"
    assert dispatch["capture_regions"] == ["matmul_97", "add_99"]
    assert [event["capture_op_index"] for event in dispatch["events"]] == [
        2825, 2825, 2825, 2827, 2834
    ]
    assert dispatch["events"][-2]["phase"] == "dequantize_and_publish_Y0"
    assert dispatch["events"][-1]["phase"] == "release_Y0_after_frontier"
    assert dispatch["abi"] == partition["abi"]


def test_capture_bridge_fails_closed_on_semantic_or_value_drift() -> None:
    partition = state_proj_partition()
    bad_transpose = copy.deepcopy(partition)
    bad_transpose["abi"]["inputs"][1]["origin"]["bridges"] = []
    try:
        validate_partition_abi(bad_transpose)
    except ValueError as error:
        assert "transpose" in str(error)
    else:
        raise AssertionError("missing capture transpose was accepted")

    bad_output = copy.deepcopy(partition)
    bad_output["abi"]["outputs"][0]["device_dtype"] = "fp8_e4m3"
    try:
        validate_partition_abi(bad_output)
    except ValueError as error:
        assert "BF16" in str(error)
    else:
        raise AssertionError("changed device output dtype was accepted")

    try:
        bridge_inputs(
            np.asarray([[math.nan] * 32], dtype=np.float32),
            np.zeros((32, 960), dtype=np.float32),
            np.zeros(960, dtype=np.float32),
            partition["geometry"],
        )
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite capture input was accepted")


def test_saved_real_capture_qualification_is_scoped_and_passes_fixed_tolerance() -> None:
    result = load(ROOT / "capture_semantics_state_proj/result.json")
    calibration = load(ROOT / "capture_semantics_state_proj/calibration.json")
    dispatch = load(ROOT / "capture_semantics_state_proj/dispatch_manifest.json")
    contract = load(ROOT / "calibration_contract.json")
    raw_receipt = load(ROOT / result["raw_gsim_receipt"])
    raw_spec_path = ROOT / raw_receipt["spec"]
    raw_stdout_path = ROOT / raw_receipt["stdout"]
    raw_stderr_path = ROOT / raw_receipt["stderr"]
    raw_spec = load(raw_spec_path)
    raw_result_page = json.loads(raw_stdout_path.read_text().strip().splitlines()[-1])
    capture = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
    with np.load(capture / "inputs.npz", allow_pickle=False) as archive:
        activation = np.asarray(archive["in4"], dtype=np.float32).copy()
    source_weight, _ = read_f32_safetensor(
        capture / "weights.safetensors", "model.state_proj.weight"
    )
    bias, _ = read_f32_safetensor(
        capture / "weights.safetensors", "model.state_proj.bias"
    )
    weight = np.ascontiguousarray(source_weight.T)
    independently_bridged = bridge_inputs(
        activation, weight, bias, state_proj_partition()["geometry"]
    )
    raw_device_output = (ROOT / result["device_output"]["path"]).read_bytes()
    assert sha256_bytes(raw_device_output) == result["device_output"]["raw_sha256"]
    assert "golden" not in raw_spec and "expected" not in raw_spec
    assert raw_receipt["halted"] is True and raw_result_page["halted"] is True
    assert raw_receipt["assertion_clean"] is True
    assert raw_stderr_path.read_bytes() == b""
    assert raw_receipt["cycles"] == raw_result_page["cycles"] == result["cycles"]
    assert raw_receipt["final_pc_available"] is False
    assert raw_receipt["raw_output_sha256"] == sha256_bytes(
        bytes.fromhex(raw_result_page["outputs"][0])
    )
    assert raw_device_output == bytes.fromhex(raw_result_page["outputs"][0])
    assert raw_receipt["spec_sha256"] == hashlib.sha256(raw_spec_path.read_bytes()).hexdigest()
    assert raw_receipt["stdout_sha256"] == hashlib.sha256(raw_stdout_path.read_bytes()).hexdigest()
    assert raw_receipt["stderr_sha256"] == hashlib.sha256(raw_stderr_path.read_bytes()).hexdigest()
    submitted_binary = np.asarray(raw_spec["words"], dtype="<u4").tobytes()
    assert raw_receipt["kernel_binary_sha256"] == sha256_bytes(submitted_binary)
    assert raw_spec["reads"] == [[2415951776, 1920]]
    device_output = (
        np.frombuffer(raw_device_output, dtype="<u2").astype(np.uint32) << 16
    ).view(np.float32).reshape(1, 960)
    output_scale = np.float32(independently_bridged["record"]["output_scale"])
    actual = device_output * output_scale
    source_reference = np.matmul(activation, weight, dtype=np.float32) + bias
    quantized_reference = (
        np.matmul(
            independently_bridged["decoded"]["A0"],
            independently_bridged["decoded"]["W"],
            dtype=np.float32,
        ) + independently_bridged["decoded"]["B_quant_domain"]
    ) * output_scale
    floor = float(contract["tolerance"]["max_relative_denominator_floor"])
    recomputed_source = comparison(actual, source_reference, floor)
    recomputed_quantized = comparison(actual, quantized_reference, floor)
    assert result["partition_id"] == "atlas_p0098"
    assert result["capture_semantics_executable_partitions"] == 1
    assert result["structural_partitions_total"] == 391
    assert result["cycles"] == 154458
    assert result["image"]["instruction_words"] == 1509
    assert result["engine"]["derived_from_rtl"] is True
    assert result["acceptance"] == {"passed": True, "thresholds": contract["tolerance"]}
    assert result["source_f32_comparison"] == recomputed_source
    assert result["quantized_domain_reference_comparison"] == recomputed_quantized
    assert passes_tolerance(recomputed_source, contract["tolerance"])
    assert result["source_f32_comparison"]["max_abs_error"] < 0.071
    assert result["source_f32_comparison"]["cosine_similarity"] > 0.9994
    assert result["quantized_domain_reference_comparison"]["max_abs_error"] < 0.013
    assert result["quantized_domain_reference_comparison"]["cosine_similarity"] > 0.99999
    assert calibration["measurements"]["output_scale"] == (
        calibration["measurements"]["activation"]["scale"]
        * calibration["measurements"]["weight"]["scale"]
    )
    assert calibration["measurements"]["bias"]["max_abs_error_after_output_rescale"] < 0.00046
    assert dispatch["qualified_capture_semantics"] is True
    assert "whole-model numeric correctness" in result["not_claimed"]
    assert "source bit-exactness" in result["not_claimed"]

    # This check can fail: perturb one published output by a full f32 unit and
    # re-run the same fixed acceptance rule.  No saved verdict is trusted.
    perturbed = actual.copy()
    perturbed[0, 0] += np.float32(1.0)
    assert not passes_tolerance(
        comparison(perturbed, source_reference, floor), contract["tolerance"]
    )


def test_action_in_projection_binds_real_noise_and_independently_passes() -> None:
    out = ROOT / "capture_semantics_action_in_proj"
    result = load(out / "result.json")
    calibration = load(out / "calibration.json")
    dispatch = load(out / "dispatch_manifest.json")
    raw_receipt = load(ROOT / result["raw_gsim_receipt"])
    raw_spec_path = ROOT / raw_receipt["spec"]
    raw_stdout_path = ROOT / raw_receipt["stdout"]
    raw_stderr_path = ROOT / raw_receipt["stderr"]
    raw_spec = load(raw_spec_path)
    raw_result_page = json.loads(raw_stdout_path.read_text().strip().splitlines()[-1])
    contract = load(ROOT / "calibration_contract.json")
    partition = partition_by_id("atlas_p0243")
    capture = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"

    activation, weight, bias, source, reference_activation = _load_capture_values(
        partition, capture, PARTITIONS["action_in_proj"]
    )
    assert np.array_equal(reference_activation, activation)
    assert activation.shape == (50, 32)
    assert source["input"]["source_shape"] == [1, 50, 32]
    assert source["input"]["bridges"] == [
        {"op": "tensor.expand_shape", "region_id": "view_787"},
        {"op": "tensor.collapse_shape", "region_id": "view_787"},
    ]
    independently_bridged = bridge_inputs(activation, weight, bias, partition["geometry"])
    raw_output = (ROOT / result["device_output"]["path"]).read_bytes()
    device_output = (
        np.frombuffer(raw_output, dtype="<u2").astype(np.uint32) << 16
    ).view(np.float32).reshape(50, 720)
    output_scale = np.float32(independently_bridged["record"]["output_scale"])
    actual = device_output * output_scale
    source_reference = np.matmul(activation, weight, dtype=np.float32) + bias
    quantized_reference = (
        np.matmul(
            independently_bridged["decoded"]["A0"],
            independently_bridged["decoded"]["W"],
            dtype=np.float32,
        ) + independently_bridged["decoded"]["B_quant_domain"]
    ) * output_scale
    floor = float(contract["tolerance"]["max_relative_denominator_floor"])
    source_metrics = comparison(actual, source_reference, floor)
    quantized_metrics = comparison(actual, quantized_reference, floor)

    assert result["partition_id"] == "atlas_p0243"
    assert result["capture_regions"] == ["matmul_242", "add_197"]
    assert result["capture_semantics_executable_partitions"] == 2
    assert result["cycles"] == 573715
    assert result["image"]["instruction_words"] == 15175
    assert result["source_f32_comparison"] == source_metrics
    assert result["quantized_domain_reference_comparison"] == quantized_metrics
    assert passes_tolerance(source_metrics, contract["tolerance"])
    assert source_metrics["max_abs_error"] < 0.108
    assert source_metrics["cosine_similarity"] > 0.9993
    assert raw_receipt["assertion_clean"] is True
    assert raw_stderr_path.read_bytes() == b""
    assert raw_receipt["engine_binary_name"] == "atlas_gsim_sim_assert"
    assert raw_receipt["cycles"] == raw_result_page["cycles"] == result["cycles"]
    assert sha256_bytes(raw_output) == raw_receipt["raw_output_sha256"]
    assert raw_output == bytes.fromhex(raw_result_page["outputs"][0])
    assert "golden" not in raw_spec and "expected" not in raw_spec
    assert raw_receipt["spec_sha256"] == hashlib.sha256(raw_spec_path.read_bytes()).hexdigest()
    assert raw_receipt["stdout_sha256"] == hashlib.sha256(raw_stdout_path.read_bytes()).hexdigest()
    assert raw_receipt["stderr_sha256"] == hashlib.sha256(raw_stderr_path.read_bytes()).hexdigest()
    assert raw_receipt["kernel_binary_sha256"] == sha256_bytes(
        np.asarray(raw_spec["words"], dtype="<u4").tobytes()
    )
    assert calibration["schema"] == "atlas_capture_partition_calibration_v1"
    assert calibration["source_tensors"] == source
    assert dispatch["abi"] == partition["abi"]
    assert dispatch["lifetime"] == partition["lifetime"]
    assert dispatch["events"][-1] == {
        "capture_op_index": 8357,
        "phase": "release_Y0_after_frontier",
    }

    perturbed = actual.copy()
    perturbed[0, 0] += np.float32(1.0)
    assert not passes_tolerance(
        comparison(perturbed, source_reference, floor), contract["tolerance"]
    )

    # The requested 50x720x32 action output remains unbindable without its
    # preceding host prefix; do not substitute the model's final golden output.
    blocked = partition_by_id("atlas_p0390")
    assert blocked["geometry"] == {"M": 50, "K": 720, "N": 32}
    assert blocked["abi"]["inputs"][0]["origin"] == {
        "kind": "host_region",
        "region_id": "dtype_cast_471",
        "semantic": "dtype_cast",
        "bridges": [
            {"op": "tensor.expand_shape", "region_id": "view_1321"},
            {"op": "tensor.collapse_shape", "region_id": "view_1321"},
        ],
    }

    mutated = copy.deepcopy(partition)
    mutated["abi"]["inputs"][0]["origin"]["bridges"] = []
    try:
        _load_capture_values(mutated, capture, PARTITIONS["action_in_proj"])
    except ValueError as error:
        assert "activation view bridge changed" in str(error)
    else:
        raise AssertionError("missing activation view bridge was accepted")


def test_action_time_mlp_in_three_dispatches_independently_reconstruct_full_result() -> None:
    out = ROOT / "capture_semantics_action_time_mlp_in"
    result = load(out / "result.json")
    unsliced = load(ROOT / "capture_semantics_action_time_mlp_in_diagnostic/result.json")
    calibration = load(out / "calibration.json")
    dispatch = load(out / "dispatch_manifest.json")
    contract = load(ROOT / "calibration_contract.json")
    partition = partition_by_id("atlas_p0244")
    capture = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
    activation, weight, bias, source, reference_activation = _load_capture_values(
        partition, capture, PARTITIONS["action_time_mlp_in"]
    )
    assert source["input"]["kind"] == "qualified_partition_plus_explicit_host_preprocess"
    assert unsliced["candidate_outcome"] == "invalid_harness_dram_alias"
    assert unsliced["invalidity"]["overage_bytes"] == 133664
    assert unsliced["capture_semantics_executable_partitions"] == 2
    assert source["input"]["predecessor"]["partition_id"] == "atlas_p0243"
    assert source["input"]["host_bridge"]["shape"] == [50, 720]
    assert activation.shape == reference_activation.shape == (50, 1440)

    actual_tiles = []
    quantized_tiles = []
    word_tiles = []
    expected_ranges = [[0, 256], [256, 512], [512, 720]]
    assert [item["n_range"] for item in dispatch["device_dispatches"]] == expected_ranges
    assert len(result["raw_gsim_receipts"]) == 3
    assert calibration["measurements"]["kind"] == "per_n_slice_independent_calibration"
    for device_record, calibration_record, receipt_name in zip(
        dispatch["device_dispatches"],
        calibration["measurements"]["dispatches"],
        result["raw_gsim_receipts"],
    ):
        n0, n1 = device_record["n_range"]
        assert calibration_record["n_range"] == [n0, n1]
        bridged = bridge_inputs(
            activation,
            np.ascontiguousarray(weight[:, n0:n1]),
            np.ascontiguousarray(bias[n0:n1]),
            {"M": 50, "K": 1440, "N": n1 - n0},
            code_cap=16.0,
        )
        assert calibration_record["measurements"] == bridged["record"]
        receipt = load(ROOT / receipt_name)
        assert receipt_name == device_record["raw_gsim_receipt"]
        assert receipt["assertion_clean"] is True
        assert receipt["stderr_observation"] == "empty"
        assert (ROOT / receipt["stderr"]).read_bytes() == b""
        raw_spec = load(ROOT / receipt["spec"])
        assert "golden" not in raw_spec and "expected" not in raw_spec
        raw_page = json.loads((ROOT / receipt["stdout"]).read_text().strip().splitlines()[-1])
        raw = bytes.fromhex(raw_page["outputs"][0])
        assert sha256_bytes(raw) == receipt["raw_output_sha256"]
        assert raw == (ROOT / device_record["device_output"]["path"]).read_bytes()
        dram = validate_gsim_dram_window(load(ROOT / device_record["image"]["command_buffer"]))
        assert dram == device_record["dram_preflight"]
        assert dram["alias_free"] is True
        words = np.frombuffer(raw, dtype="<u2").reshape(50, n1 - n0)
        device = (words.astype(np.uint32) << 16).view(np.float32)
        scale = np.float32(bridged["record"]["output_scale"])
        actual_tiles.append(device * scale)
        quantized_tiles.append((
            np.matmul(bridged["decoded"]["A0"], bridged["decoded"]["W"], dtype=np.float32)
            + bridged["decoded"]["B_quant_domain"]
        ) * scale)
        word_tiles.append(words)

    actual = np.concatenate(actual_tiles, axis=1)
    quantized_reference = np.concatenate(quantized_tiles, axis=1)
    source_reference = np.matmul(reference_activation, weight, dtype=np.float32) + bias
    combined_words = np.concatenate(word_tiles, axis=1).astype("<u2", copy=False)
    combined_raw = (ROOT / result["device_output"]["path"]).read_bytes()
    assert combined_raw == combined_words.tobytes()
    assert sha256_bytes(combined_raw) == result["device_output"]["raw_sha256"]
    floor = float(contract["tolerance"]["max_relative_denominator_floor"])
    source_metrics = comparison(actual, source_reference, floor)
    quantized_metrics = comparison(actual, quantized_reference, floor)
    assert result["source_f32_comparison"] == source_metrics
    assert result["quantized_domain_reference_comparison"] == quantized_metrics
    assert result["partition_id"] == "atlas_p0244"
    assert result["capture_semantics_executable_partitions"] == 3
    assert result["image"]["dispatch_count"] == 3
    assert result["cycles"] == sum(item["cycles"] for item in dispatch["device_dispatches"])
    assert passes_tolerance(source_metrics, contract["tolerance"])
    assert source_metrics["max_abs_error"] < 0.067
    assert source_metrics["cosine_similarity"] > 0.999

    perturbed = actual.copy()
    perturbed[0, 0] += np.float32(1.0)
    assert not passes_tolerance(
        comparison(perturbed, source_reference, floor), contract["tolerance"]
    )
