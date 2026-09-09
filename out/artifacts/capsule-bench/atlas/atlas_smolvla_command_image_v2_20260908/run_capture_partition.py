#!/usr/bin/env python3
"""Qualify explicitly bound real SmolVLA capture partitions on Atlas RTL.

This is deliberately a bounded partition dispatcher.  It validates each planner
ABI, loads original capture tensors, applies the checked calibration contract,
executes the emitted image, and publishes the f32 value at the recorded graph
definition.  It does not execute the surrounding host graph.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(ROOT / "submission"))

from merlin.targetgen.program_oracle import run_program_verilator_oracle  # noqa: E402
from mlir_oot.capture_bridge import (  # noqa: E402
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
from mlir_oot.accelerator_semantics import (  # noqa: E402
    UnsupportedAcceleratorContract,
    _validate_batched_command_buffer,
)


PARTITIONS = {
    "text_layer0_attn_qk": {
        "partition_id": "atlas_p0102",
        "fqn": "",
        "capture_regions": ["matmul_101"],
        "operand_bundle": "capture_semantics_text_layer0_attn_qk/capture_operands.npz",
        "operand_receipt": "capture_semantics_text_layer0_attn_qk/capture_boundary.json",
        "output_dir": "capture_semantics_text_layer0_attn_qk",
        "qualified_total": 4,
        "max_cycles": 50_000_000,
        "fp8_code_cap": 16.0,
        "source_frontier_scale": 0.125,
    },
    "state_proj": {
        "partition_id": "atlas_p0098",
        "fqn": "model.state_proj",
        "capture_regions": ["matmul_97", "add_99"],
        "input_arg": 511,
        "input_name": "state",
        "input_archive_key": "in4",
        "input_source_shape": [1, 32],
        "input_bridges": [],
        "weight_arg": 490,
        "weight_name": "model.state_proj.weight",
        "weight_source_shape": [960, 32],
        "bias_arg": 491,
        "bias_name": "model.state_proj.bias",
        "bias_shape": [960],
        "output_dir": "capture_semantics_state_proj",
        "qualified_total": 1,
        "max_cycles": 1_000_000,
    },
    "action_in_proj": {
        "partition_id": "atlas_p0243",
        "fqn": "model.action_in_proj",
        "capture_regions": ["matmul_242", "add_197"],
        "input_arg": 512,
        "input_name": "noise",
        "input_archive_key": "in5",
        "input_source_shape": [1, 50, 32],
        "input_bridges": [
            {"op": "tensor.expand_shape", "region_id": "view_787"},
            {"op": "tensor.collapse_shape", "region_id": "view_787"},
        ],
        "weight_arg": 492,
        "weight_name": "model.action_in_proj.weight",
        "weight_source_shape": [720, 32],
        "bias_arg": 493,
        "bias_name": "model.action_in_proj.bias",
        "bias_shape": [720],
        "output_dir": "capture_semantics_action_in_proj",
        "qualified_total": 2,
        "max_cycles": 1_000_000,
    },
    "action_time_mlp_in": {
        "partition_id": "atlas_p0244",
        "fqn": "model.action_time_mlp_in",
        "capture_regions": ["matmul_243", "add_199"],
        "input_origin": {
            "kind": "host_region",
            "region_id": "cat_51",
            "semantic": "cat",
            "bridges": [
                {"op": "tensor.expand_shape", "region_id": "view_789"},
                {"op": "tensor.collapse_shape", "region_id": "view_789"},
            ],
        },
        "predecessor": "action_in_proj",
        "weight_arg": 496,
        "weight_name": "model.action_time_mlp_in.weight",
        "weight_source_shape": [720, 1440],
        "bias_arg": 497,
        "bias_name": "model.action_time_mlp_in.bias",
        "bias_shape": [720],
        "output_dir": "capture_semantics_action_time_mlp_in",
        "qualified_total": 3,
        "max_cycles": 50_000_000,
        "n_tiles": [256, 256, 208],
        "fp8_code_cap": 16.0,
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_embedding_720() -> tuple[np.ndarray, dict]:
    """Reproduce capture regions iota_38..expand_147 in their recorded dtypes."""
    index = np.arange(360, dtype=np.int64)
    exponent = np.where(
        index < np.int64(180),
        index.astype(np.float64) * np.float64("0.0027855153203342618"),
        np.float64(1.0)
        - (np.int64(359) - index).astype(np.float64)
        * np.float64("0.0027855153203342618"),
    )
    angle = (
        np.reciprocal(np.power(np.float64(1000.0), exponent) * np.float64(0.004))
        * np.float64(2.0)
        * np.float64("3.1415926535897931")
    )
    embedding = np.concatenate((np.sin(angle), np.cos(angle))).astype(np.float32)
    expanded = np.ascontiguousarray(np.broadcast_to(embedding, (50, 720)))
    return expanded, {
        "schema": "smolvla_action_time_embedding_host_bridge_v1",
        "capture_regions": [
            "iota_38", "compare_2", "dtype_cast_242", "mul_293", "add_198",
            "sub_33", "dtype_cast_243", "mul_294", "sub_34", "select_30",
            "pow_65", "mul_295", "elementwise_2", "mul_296", "mul_297",
            "mul_298", "unsqueeze_198", "unsqueeze_199", "mul_299", "sin_32",
            "cos_32", "cat_50", "dtype_cast_244", "unsqueeze_200", "expand_147",
        ],
        "equation": "f32(concat(sin(theta), cos(theta))); theta=2*pi/(0.004*1000**exponent), timestep=1",
        "source_dtype": "f64 transcendental intermediates, then f32 cast",
        "shape": [50, 720],
        "raw_sha256": sha256_bytes(expanded.astype("<f4", copy=False).tobytes()),
    }


def _load_qualified_predecessor(binding: dict) -> tuple[np.ndarray, dict]:
    predecessor = PARTITIONS[binding["predecessor"]]
    out = ROOT / predecessor["output_dir"]
    result = load_json(out / "result.json")
    calibration = load_json(out / "calibration.json")
    receipt = load_json(ROOT / result["raw_gsim_receipt"])
    raw_path = ROOT / result["device_output"]["path"]
    raw = raw_path.read_bytes()
    if result.get("partition_id") != predecessor["partition_id"]:
        raise ValueError("qualified predecessor identity changed")
    if not result.get("acceptance", {}).get("passed"):
        raise ValueError("qualified predecessor no longer passes its fixed gate")
    if not receipt.get("assertion_clean") or receipt.get("stderr_observation") != "empty":
        raise ValueError("qualified predecessor lacks assertion-clean RTL evidence")
    if sha256_bytes(raw) != result["device_output"]["raw_sha256"]:
        raise ValueError("qualified predecessor device output hash changed")
    shape = tuple(int(value) for value in result["device_output"]["shape"])
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != math.prod(shape):
        raise ValueError("qualified predecessor device output extent changed")
    decoded = (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    scale = np.float32(calibration["measurements"]["output_scale"])
    return decoded * scale, {
        "partition_id": predecessor["partition_id"],
        "result": f"{predecessor['output_dir']}/result.json",
        "result_sha256": sha256_file(out / "result.json"),
        "device_output": result["device_output"]["path"],
        "device_output_sha256": sha256_bytes(raw),
        "output_scale": float(scale),
        "raw_gsim_receipt": result["raw_gsim_receipt"],
        "raw_gsim_receipt_sha256": sha256_file(ROOT / result["raw_gsim_receipt"]),
    }


def _load_capture_values(
    partition: dict, capture: Path, binding: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict, np.ndarray]:
    if binding.get("operand_bundle"):
        receipt_path = ROOT / binding["operand_receipt"]
        receipt = load_json(receipt_path)
        archive_path = ROOT / binding["operand_bundle"]
        if (
            receipt.get("schema") != "atlas_real_capture_batched_boundary_v1"
            or receipt.get("partition_id") != binding["partition_id"]
            or receipt.get("capture_regions") != binding["capture_regions"]
            or receipt.get("geometry") != partition["geometry"]
            or receipt.get("operands", {}).get("archive_sha256") != sha256_file(archive_path)
            or not receipt.get("binding", {}).get("complete_graph_output_bit_exact")
            or receipt.get("source_frontier") != {
                "kind": "torch.ops.aten.mul.Tensor",
                "fx_node": "mul_22",
                "scalar": binding["source_frontier_scale"],
                "scalar_hex": float(binding["source_frontier_scale"]).hex(),
                "partition_plan_region": "mul_32",
                "partition_plan_op_index": 3063,
                "sole_immediate_consumer": True,
                "fold": "Y_frontier = Y_source * 0.125",
            }
        ):
            raise ValueError("captured batched operand receipt is absent, stale, or untrusted")
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {"A0", "W", "Y_source", "Y_frontier"}:
                raise ValueError("captured batched operand archive has unexpected tensors")
            activation = np.ascontiguousarray(archive["A0"], dtype=np.float32)
            captured_weight = np.ascontiguousarray(archive["W"], dtype=np.float32)
            source_matmul = np.ascontiguousarray(archive["Y_source"], dtype=np.float32)
            reference = np.ascontiguousarray(archive["Y_frontier"], dtype=np.float32)
        for name, value in (
            ("A0", activation),
            ("W", captured_weight),
            ("Y_source", source_matmul),
            ("Y_frontier", reference),
        ):
            record = receipt["operands"][name]
            if (
                record.get("shape") != list(value.shape)
                or record.get("dtype") != "f32"
                or record.get("raw_sha256")
                != sha256_bytes(value.astype("<f4", copy=False).tobytes())
            ):
                raise ValueError(f"captured batched tensor {name} differs from its receipt")
        geometry = partition["geometry"]
        if (
            activation.shape != (geometry["B"], geometry["M"], geometry["K"])
            or captured_weight.shape != (geometry["B"], geometry["K"], geometry["N"])
            or source_matmul.shape != (geometry["B"], geometry["M"], geometry["N"])
            or reference.shape != (geometry["B"], geometry["M"], geometry["N"])
        ):
            raise ValueError("captured batched tensor shapes differ from the partition ABI")
        expected_frontier = np.ascontiguousarray(
            source_matmul * np.float32(binding["source_frontier_scale"]), dtype=np.float32
        )
        if not np.array_equal(reference, expected_frontier):
            raise ValueError("captured p0102 frontier no longer equals its bound scalar fold")
        independent_source = np.matmul(activation, captured_weight, dtype=np.float32)
        independent_frontier = independent_source * np.float32(
            binding["source_frontier_scale"]
        )
        oracle = receipt.get("independent_oracle", {})
        source_oracle_error = float(np.max(np.abs(independent_source - source_matmul)))
        frontier_oracle_error = float(np.max(np.abs(independent_frontier - reference)))
        if (
            oracle.get("implementation") != "numpy.matmul(dtype=float32)"
            or oracle.get("derived_from_rtl") is not False
            or oracle.get("consumes_gsim_output") is not False
            or oracle.get("source_matmul_max_abs_error") != source_oracle_error
            or oracle.get("frontier_max_abs_error") != frontier_oracle_error
            or source_oracle_error > oracle.get("source_matmul_max_abs_limit", -math.inf)
            or frontier_oracle_error > oracle.get("frontier_max_abs_limit", -math.inf)
        ):
            raise ValueError("captured p0102 independent NumPy oracle check changed")
        source = {
            "kind": "capture_equivalent_exported_program_boundary",
            "receipt": binding["operand_receipt"],
            "receipt_sha256": sha256_file(receipt_path),
            "operand_bundle": binding["operand_bundle"],
            "operand_bundle_sha256": sha256_file(archive_path),
            "complete_graph_output_bit_exact": True,
            "complete_graph_output_sha256": receipt["binding"]["complete_graph_output_sha256"],
            "frontier": receipt["source_frontier"],
            "independent_oracle": oracle,
        }
        return activation, captured_weight, None, source, reference
    if binding.get("predecessor"):
        if partition["abi"]["inputs"][0]["origin"] != binding["input_origin"]:
            raise ValueError("partition host-chain activation origin changed")
        predecessor_actual, predecessor_record = _load_qualified_predecessor(binding)
        predecessor_partition = _select_partition(PARTITIONS[binding["predecessor"]])
        pred_activation, pred_weight, pred_bias, _, _ = _load_capture_values(
            predecessor_partition, capture, PARTITIONS[binding["predecessor"]]
        )
        predecessor_source = (
            np.matmul(pred_activation, pred_weight, dtype=np.float32) + pred_bias
        )
        time_embedding, host_bridge = _time_embedding_720()
        activation = np.ascontiguousarray(
            np.concatenate((predecessor_actual, time_embedding), axis=1)
        )
        reference_activation = np.ascontiguousarray(
            np.concatenate((predecessor_source, time_embedding), axis=1)
        )
        source_input = {
            "kind": "qualified_partition_plus_explicit_host_preprocess",
            "predecessor": predecessor_record,
            "host_bridge": host_bridge,
            "device_chain_shape": list(activation.shape),
            "device_chain_sha256": sha256_bytes(
                activation.astype("<f4", copy=False).tobytes()
            ),
            "source_reference_shape": list(reference_activation.shape),
            "source_reference_sha256": sha256_bytes(
                reference_activation.astype("<f4", copy=False).tobytes()
            ),
        }
    else:
        input_order = load_json(capture / "input_order.json")
        archive_index = int(binding["input_archive_key"][2:])
        if input_order.get(binding["input_name"]) != archive_index:
            raise ValueError("capture input_order no longer matches the partition binding")
        abi_inputs = partition["abi"]["inputs"]
        if [entry["origin"]["argument_index"] for entry in abi_inputs] != [
            binding["input_arg"], binding["weight_arg"], binding["bias_arg"]
        ]:
            raise ValueError("partition argument mapping changed")
        actual_input_bridges = abi_inputs[0]["origin"].get("bridges", [])
        if actual_input_bridges != binding["input_bridges"]:
            raise ValueError("partition activation view bridge changed")
        with np.load(capture / "inputs.npz", allow_pickle=False) as archive:
            key = binding["input_archive_key"]
            if key not in archive.files:
                raise ValueError(f"capture inputs.npz has no {key} tensor")
            source_activation = np.asarray(archive[key], dtype=np.float32).copy()
        if list(source_activation.shape) != binding["input_source_shape"]:
            raise ValueError("capture activation source shape changed")
        activation = np.ascontiguousarray(source_activation.reshape(
            partition["geometry"]["M"], partition["geometry"]["K"]
        ))
        reference_activation = activation
        source_input = {
            "kind": "capture_input",
            "argument_index": binding["input_arg"],
            "archive_key": binding["input_archive_key"],
            "capture_name": binding["input_name"],
            "source_shape": list(source_activation.shape),
            "device_view_shape": list(activation.shape),
            "bridges": binding["input_bridges"],
            "raw_sha256": sha256_bytes(source_activation.astype("<f4", copy=False).tobytes()),
        }

    weight_manifest = load_json(capture / "weights.safetensors.manifest.json")
    if weight_manifest.get(str(binding["weight_arg"])) != {
        "weight": binding["weight_name"],
        "kind": "param",
        "dtype": "float32",
        "shape": binding["weight_source_shape"],
    }:
        raise ValueError("capture weight argument no longer matches the partition binding")
    if weight_manifest.get(str(binding["bias_arg"])) != {
        "weight": binding["bias_name"],
        "kind": "param",
        "dtype": "float32",
        "shape": binding["bias_shape"],
    }:
        raise ValueError("capture bias argument no longer matches the partition binding")
    if [entry["origin"].get("argument_index") for entry in partition["abi"]["inputs"][1:]] != [
        binding["weight_arg"], binding["bias_arg"]
    ]:
        raise ValueError("partition weight/bias argument mapping changed")

    source_weight, weight_source = read_f32_safetensor(
        capture / "weights.safetensors", binding["weight_name"]
    )
    bias, bias_source = read_f32_safetensor(
        capture / "weights.safetensors", binding["bias_name"]
    )
    # This is the exact, planner-validated weight bridge for all bound projections.
    captured_weight = np.ascontiguousarray(source_weight.T)
    source = {
        "capture": (
            capture.relative_to(REPO).as_posix()
            if capture.is_relative_to(REPO) else "caller-provided capture directory"
        ),
        "input": source_input,
        "weight": weight_source,
        "weight_capture_bridge": "linalg.transpose permutation=[1,0]",
        "bias": bias_source,
    }
    return activation, captured_weight, bias, source, reference_activation


def _select_partition(binding: dict) -> dict:
    plan = load_json(ROOT / "whole_capture_plan/partition_plan.json")
    matches = [entry for entry in plan["partitions"]
               if entry.get("partition_id") == binding["partition_id"]
               and entry.get("fqn") == binding["fqn"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one bound capture partition")
    partition = matches[0]
    if partition.get("kind") == "matmul_batched":
        geometry = partition.get("geometry", {})
        inputs = partition.get("abi", {}).get("inputs", [])
        outputs = partition.get("abi", {}).get("outputs", [])
        if (
            partition.get("source_semantic") != "batch_matmul"
            or partition.get("bias_fused")
            or [row.get("name") for row in inputs] != ["A0", "W"]
            or [row.get("device_dtype") for row in inputs] != ["fp8_e4m3", "fp8_e4m3"]
            or len(outputs) != 1
            or outputs[0].get("name") != "Y0"
            or outputs[0].get("device_dtype") != "bf16"
            or any(int(geometry.get(key, 0)) <= 0 for key in ("B", "M", "K", "N"))
        ):
            raise ValueError("captured batched partition ABI changed")
    else:
        validate_partition_abi(partition)
    if partition.get("capture_regions") != binding["capture_regions"]:
        raise ValueError("bound capture-region identity changed")
    return partition


def _qualified_command_buffer(partition: dict, bridged: dict, run_dir: Path) -> tuple[dict, Path, dict]:
    kernel_id = partition["kernel_id"]
    command_path = ROOT / "whole_capture_plan/command_buffers" / f"{kernel_id}.json"
    planned_kernel = ROOT / "whole_capture_plan/kernels" / f"{kernel_id}.S.gz"
    kernel = run_dir / "kernel.S"
    cb = load_json(command_path)
    if not planned_kernel.is_file():
        raise ValueError("whole-capture planned kernel is absent")
    planned_source = gzip.decompress(planned_kernel.read_bytes())
    kernel.write_bytes(planned_source)
    if len(cb.get("commands", [])) != partition["image"]["command_count"]:
        raise ValueError("command-buffer count differs from the partition plan")
    input_names = [entry["name"] for entry in partition["abi"]["inputs"]]
    for name in input_names:
        tensor = cb.get("tensors", {}).get(name)
        raw = bridged["preloads"][name]
        abi = next(entry for entry in partition["abi"]["inputs"] if entry["name"] == name)
        if tensor is None or len(raw) != abi["device_bytes"]:
            raise ValueError(f"device preload {name} no longer matches the planned ABI")
        tensor["preload_b64"] = base64.b64encode(raw).decode("ascii")
    command_contract = None
    if partition.get("kind") == "matmul_batched":
        command_contract = _validate_batched_command_buffer(partition, cb)
    evidence = {
        "command_buffer": command_path.relative_to(ROOT).as_posix(),
        "command_buffer_sha256_without_preloads": sha256_file(command_path),
        "kernel": planned_kernel.relative_to(ROOT).as_posix(),
        "kernel_gzip_sha256": sha256_file(planned_kernel),
        "kernel_uncompressed_sha256": sha256_bytes(planned_source),
        "instruction_words": sum(
            line.lstrip().startswith(".word") for line in kernel.read_text().splitlines()
        ),
        "command_contract": command_contract,
    }
    return copy.deepcopy(cb), kernel, evidence


def _bridge_batched_inputs(
    activation: np.ndarray,
    weight: np.ndarray,
    geometry: dict,
    code_cap: float = FP8_MAX_FINITE,
) -> dict:
    expected_a = (geometry["B"], geometry["M"], geometry["K"])
    expected_w = (geometry["B"], geometry["K"], geometry["N"])
    activation = np.ascontiguousarray(activation, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    if activation.shape != expected_a or weight.shape != expected_w:
        raise ValueError("batched capture values do not match the planned geometry")
    qa = calibrate_e4m3("A0", activation, code_cap)
    qw = calibrate_e4m3("W", weight, code_cap)
    output_scale = qa["scale"] * qw["scale"]
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("invalid batched output scale")
    return {
        "preloads": {"A0": qa["codes"].tobytes(), "W": qw["codes"].tobytes()},
        "decoded": {"A0": qa["decoded"], "W": qw["decoded"]},
        "record": {
            "activation": qa["record"],
            "weight": qw["record"],
            "output_scale": output_scale,
            "output_scale_hex": float(output_scale).hex(),
        },
    }


def _batched_dispatch_manifest(partition: dict) -> dict:
    definition = partition["lifetime"]["definition_op_index"]
    release = partition["lifetime"]["last_frontier_use_op_index"]
    if release < definition:
        raise ValueError("batched partition output lifetime ends before its definition")
    return {
        "schema": "atlas_single_partition_dispatch_v1",
        "claim": "host/device skeleton for one partition; no whole-model dispatch claim",
        "partition_id": partition["partition_id"],
        "kernel_id": partition["kernel_id"],
        "capture_regions": partition["capture_regions"],
        "fqn": partition["fqn"],
        "abi": partition["abi"],
        "lifetime": partition["lifetime"],
        "events": [
            {"phase": "bind_capture_inputs", "capture_op_index": partition["capture_op_index"]},
            {"phase": "quantize_A0_W", "capture_op_index": partition["capture_op_index"]},
            {"phase": "launch_device_image", "capture_op_index": partition["capture_op_index"]},
            {"phase": "dequantize_and_publish_Y0", "capture_op_index": definition},
            {"phase": "release_Y0_after_frontier", "capture_op_index": release},
        ],
    }


def _batched_command_negative_controls(partition: dict, command: dict) -> dict:
    controls = {}
    mutations = {
        "raw_weight_consumer": lambda cb: cb["commands"][1]["operands"].update(rhs="W"),
        "missing_resident_evict": lambda cb: cb["commands"].pop(),
    }
    for name, mutate in mutations.items():
        malformed = copy.deepcopy(command)
        mutate(malformed)
        try:
            _validate_batched_command_buffer(partition, malformed)
        except UnsupportedAcceleratorContract as error:
            controls[name] = {"status": "rejected_fail_closed", "reason": str(error)}
        else:
            raise ValueError(f"batched command negative control {name} was accepted")
    return controls


def _tiled_interface(m: int, k: int, n: int) -> str:
    return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"}} {{
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{n}xf8E4M3FN>
  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{m}x{k}xf8E4M3FN>
  %B = merlin_iface.tensor {{name = "B", role = "bias"}} : tensor<{n}xbf16>
  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{k}x{n}xf8E4M3FN>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{m}x{k}xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = ["bias_add"], output_dtype = "bf16", bias = "B"}} : (!merlin_iface.acc<bf16>) -> tensor<{m}x{n}xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}}
'''


def _compile_tiled_image(geometry: dict, run_dir: Path) -> tuple[dict, Path, dict]:
    """Compile one deterministic N slice from the same verified Atlas dialect."""
    m, k, n = (int(geometry[key]) for key in ("M", "K", "N"))
    interface = run_dir / "interface.mlir"
    command_path = run_dir / "command_buffer.json"
    kernel = run_dir / "kernel.S"
    source = _tiled_interface(m, k, n)
    interface.write_text(source, encoding="utf-8")
    tool = ROOT / "submission/mlir_oot/atlas-opt"
    process = subprocess.run(
        [str(tool), f"--emit-command-buffer={command_path}",
         "--emit-target-artifact", str(interface)],
        capture_output=True, text=True, timeout=60,
    )
    if process.returncode:
        raise RuntimeError(f"Atlas N-tile compilation failed: {process.stderr[-500:]}")
    if process.stderr:
        raise RuntimeError("Atlas N-tile compilation produced unexpected stderr")
    kernel.write_text(process.stdout, encoding="utf-8")
    cb = load_json(command_path)
    evidence = {
        "kind": "compiler_generated_n_slice",
        "interface": interface.relative_to(ROOT).as_posix(),
        "interface_sha256": sha256_file(interface),
        "command_buffer": command_path.relative_to(ROOT).as_posix(),
        "command_buffer_sha256_without_preloads": sha256_file(command_path),
        "kernel": kernel.relative_to(ROOT).as_posix(),
        "kernel_uncompressed_sha256": sha256_file(kernel),
        "instruction_words": sum(
            line.lstrip().startswith(".word") for line in process.stdout.splitlines()
        ),
        "command_count": len(cb.get("commands", [])),
        "geometry": geometry,
    }
    return cb, kernel, evidence


def _compile_batched_image(partition: dict, run_dir: Path) -> tuple[dict, Path, dict]:
    """Compile the exact capture interface with the isolated assertion-clean fix."""
    kernel_id = partition["kernel_id"]
    interface_source = ROOT / "whole_capture_plan/interfaces" / f"{kernel_id}.mlir"
    interface = run_dir / "interface.mlir"
    command_path = run_dir / "command_buffer.json"
    kernel = run_dir / "kernel.S"
    interface.write_bytes(interface_source.read_bytes())
    tool = ROOT / "shape_batch_qualification_v1/backend_fixed/mlir_oot/atlas-opt"
    process = subprocess.run(
        [str(tool), f"--emit-command-buffer={command_path}", "--emit-target-artifact", str(interface)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode:
        raise RuntimeError(f"Atlas batched image compilation failed: {process.stderr[-500:]}")
    if process.stderr:
        raise RuntimeError("Atlas batched image compilation produced unexpected stderr")
    kernel.write_text(process.stdout, encoding="utf-8")
    cb = load_json(command_path)
    command_contract = _validate_batched_command_buffer(partition, cb)
    instruction_words = sum(
        line.lstrip().startswith(".word") for line in process.stdout.splitlines()
    )
    compile_receipt_path = (
        ROOT / "shape_batch_qualification_v1/evidence/receipts/compile" / f"{kernel_id}.json"
    )
    compile_receipt = load_json(compile_receipt_path)
    if (
        compile_receipt.get("qualified") is not True
        or compile_receipt.get("geometry") != partition["geometry"]
        or compile_receipt.get("assembly_sha256") != sha256_file(kernel)
        or compile_receipt.get("instruction_words") != instruction_words
        or compile_receipt.get("compiler_sha256") != sha256_file(tool)
    ):
        raise ValueError("fresh batched image differs from its fixed-backend compile receipt")
    evidence = {
        "kind": "fresh_exact_interface_fixed_backend_image",
        "interface": interface.relative_to(ROOT).as_posix(),
        "interface_sha256": sha256_file(interface),
        "command_buffer": command_path.relative_to(ROOT).as_posix(),
        "command_buffer_sha256_without_preloads": sha256_file(command_path),
        "kernel": kernel.relative_to(ROOT).as_posix(),
        "kernel_uncompressed_sha256": sha256_file(kernel),
        "instruction_words": instruction_words,
        "compiler": tool.relative_to(ROOT).as_posix(),
        "compiler_sha256": sha256_file(tool),
        "compile_receipt": compile_receipt_path.relative_to(ROOT).as_posix(),
        "compile_receipt_sha256": sha256_file(compile_receipt_path),
        "command_contract": command_contract,
    }
    return cb, kernel, evidence


def _public_oracle_record(oracle: dict) -> dict:
    provenance = oracle["oracle"].get("provenance", {})
    adoption = provenance.get("adoption_record", {})
    return {
        "kind": oracle["oracle"].get("kind"),
        "derived_from_rtl": oracle["oracle"].get("derived_from_rtl"),
        "fidelity": oracle["oracle"].get("fidelity"),
        "engine": oracle["oracle"].get("engine"),
        "provenance": {
            "engine": provenance.get("engine"),
            "wrapper": provenance.get("wrapper"),
            "binaries": provenance.get("binaries"),
            "adoption_record": {
                "name": Path(str(adoption.get("path", "provenance.json"))).name,
                "sha256": adoption.get("sha256"),
            },
        },
    }


def _run_raw_gsim(
    vsim_dir: Path, run_dir: Path, cb: dict, out: Path, max_cycles: int
) -> tuple[dict, bytes]:
    """Replay while retaining the exact no-golden GSIM transaction and result page."""
    kernel_binary = run_dir / "kernel.bin"
    if not kernel_binary.is_file():
        raise ValueError("oracle did not retain the assembled kernel binary")
    words = np.frombuffer(kernel_binary.read_bytes(), dtype="<u4").astype(np.uint32).tolist()
    preloads = []
    for name, tensor in cb["tensors"].items():
        if tensor.get("role") not in {"weight", "input", "bias"}:
            continue
        preload = tensor.get("preload_b64")
        if not isinstance(preload, str):
            raise ValueError(f"GSIM input tensor {name} has no bound preload")
        preloads.append([int(tensor["base"]), base64.b64decode(preload).hex()])
    output = cb["tensors"]["Y0"]
    output_bytes = int(np.prod(output["shape"])) * 2
    spec = {
        "words": words,
        "preload": preloads,
        "reads": [[int(output["base"]), output_bytes]],
        "max_cycles": max_cycles,
    }
    spec_path = out / "raw_gsim_spec.json"
    stdout_path = out / "raw_gsim_stdout.txt"
    stderr_path = out / "raw_gsim_stderr.txt"
    spec_path.write_text(json.dumps(spec, separators=(",", ":")) + "\n", encoding="utf-8")
    binary = vsim_dir / "atlas_gsim_sim_assert"
    process = subprocess.run(
        [str(binary), str(spec_path)], capture_output=True, text=True, timeout=120
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"GSIM raw replay failed rc={process.returncode}: {process.stderr[-500:]}")
    if "Assertion failed" in process.stdout or "Assertion failed" in process.stderr:
        raise RuntimeError("assertion-enabled GSIM replay reported a hardware contract violation")
    if process.stderr:
        raise RuntimeError("assertion-enabled GSIM replay produced unexpected stderr")
    line = next(
        (line for line in reversed(process.stdout.splitlines()) if line.strip().startswith("{")), None
    )
    if line is None:
        raise RuntimeError("GSIM raw replay produced no JSON result page")
    raw = json.loads(line)
    outputs = raw.get("outputs", [])
    if not raw.get("halted") or len(outputs) != 1:
        raise RuntimeError("GSIM raw replay did not halt with exactly one output")
    raw_output = bytes.fromhex(outputs[0])
    if len(raw_output) != output_bytes:
        raise RuntimeError("GSIM raw replay returned the wrong output extent")
    receipt = {
        "schema": "atlas_real_capture_raw_gsim_receipt_v1",
        "claim": "exact submitted words/preloads and raw device readback; no golden in spec",
        "engine_binary_name": binary.name,
        "engine_sha256": sha256_file(binary),
        "kernel_binary_sha256": sha256_file(kernel_binary),
        "spec": spec_path.relative_to(ROOT).as_posix(),
        "spec_sha256": sha256_file(spec_path),
        "stdout": stdout_path.relative_to(ROOT).as_posix(),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": stderr_path.relative_to(ROOT).as_posix(),
        "stderr_sha256": sha256_file(stderr_path),
        "stderr_observation": "empty",
        "assertion_clean": True,
        "returncode": process.returncode,
        "halted": bool(raw["halted"]),
        "halt_reason": raw.get("halt_reason"),
        "cycles": int(raw["cycles"]),
        "reads": raw.get("reads"),
        "writes": raw.get("writes"),
        "wrap_hits": raw.get("wrap_hits"),
        "alias_collisions": raw.get("alias_collisions"),
        "final_pc_available": False,
        "raw_output_sha256": sha256_bytes(raw_output),
        "raw_output_bytes": len(raw_output),
    }
    (out / "raw_gsim_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt, raw_output


def _execute_device_image(
    *, vsim_dir: Path, out: Path, run_dir: Path, cb: dict, kernel: Path,
    max_cycles: int,
) -> dict:
    """Execute one alias-free command image twice and retain an exact witness."""
    dram_preflight = validate_gsim_dram_window(cb)
    (out / "dram_preflight.json").write_text(
        json.dumps(dram_preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    oracle = run_program_verilator_oracle(
        "atlas",
        model_ext="npu_model",
        vsim_dir=vsim_dir,
        engine="gsim",
        cb=cb,
        kernel_s=kernel,
        inputs=[],
        workdir=run_dir,
        timeout=120,
        max_cycles=max_cycles,
    )
    oracle_device_output = np.asarray(oracle["outputs"]["Y0"], dtype=np.float32)
    oracle_words, roundtrip_device_output = f32_to_bf16_rne(oracle_device_output)
    raw_receipt, raw_device_output = _run_raw_gsim(
        vsim_dir, run_dir, cb, out, max_cycles
    )
    if raw_receipt["cycles"] != oracle["cycles"]:
        raise ValueError("raw GSIM replay cycle count differs from the oracle run")
    if raw_device_output != oracle_words.tobytes():
        raise ValueError("retained raw GSIM readback differs from oracle output")
    if not np.array_equal(oracle_device_output, roundtrip_device_output):
        raise ValueError("oracle returned values not exactly representable by its declared BF16 ABI")
    device_shape = tuple(int(value) for value in cb["tensors"]["Y0"]["shape"])
    device_words = np.frombuffer(raw_device_output, dtype="<u2").copy()
    device_output = (
        (device_words.astype(np.uint32) << 16).view(np.float32).reshape(device_shape)
    )
    output_path = out / "device_output.bf16.bin"
    output_path.write_bytes(device_words.tobytes())
    return {
        "oracle": oracle,
        "raw_receipt": raw_receipt,
        "device_output": device_output,
        "device_words": device_words,
        "device_output_path": output_path,
        "dram_preflight": dram_preflight,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=tuple(PARTITIONS), nargs="?", default="state_proj")
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument(
        "--vsim-dir",
        type=Path,
        default=Path(os.environ["MERLIN_ATLAS_GSIM_DIR"])
        if os.environ.get("MERLIN_ATLAS_GSIM_DIR") else None,
        help="Atlas GSIM directory (or set MERLIN_ATLAS_GSIM_DIR)",
    )
    args = parser.parse_args()
    if args.vsim_dir is None:
        parser.error("--vsim-dir or MERLIN_ATLAS_GSIM_DIR is required")
    capture = args.capture.resolve()
    vsim_dir = args.vsim_dir.resolve()
    binding = PARTITIONS[args.partition]
    out = ROOT / binding["output_dir"]
    contract = load_json(ROOT / "calibration_contract.json")
    partition = _select_partition(binding)
    is_batched = partition.get("kind") == "matmul_batched"
    dispatch = (
        _batched_dispatch_manifest(partition)
        if is_batched else build_dispatch_manifest(partition)
    )
    activation, weight, bias, source, reference_activation = _load_capture_values(
        partition, capture, binding
    )
    out.mkdir(exist_ok=True)
    code_cap = float(binding.get("fp8_code_cap", FP8_MAX_FINITE))
    dispatch_records = []
    calibration_records = []
    actual_tiles = []
    quantized_tiles = []
    word_tiles = []
    tile_widths = binding.get("n_tiles")
    if tile_widths:
        if sum(tile_widths) != partition["geometry"]["N"] or any(
            not isinstance(width, int) or width <= 0 for width in tile_widths
        ):
            raise ValueError("N-tile schedule does not exactly cover the planned output")
        n0 = 0
        for tile_index, width in enumerate(tile_widths):
            n1 = n0 + width
            tile_geometry = {
                "M": partition["geometry"]["M"],
                "K": partition["geometry"]["K"],
                "N": width,
            }
            bridged_tile = bridge_inputs(
                activation, np.ascontiguousarray(weight[:, n0:n1]),
                np.ascontiguousarray(bias[n0:n1]), tile_geometry, code_cap=code_cap,
            )
            tile_out = out / f"dispatch_{tile_index:03d}_n{n0:04d}_{n1:04d}"
            run_dir = tile_out / "gsim_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            cb, kernel, tile_image = _compile_tiled_image(tile_geometry, run_dir)
            for name in ("A0", "W", "B"):
                raw = bridged_tile["preloads"][name]
                expected = math.prod(cb["tensors"][name]["shape"]) * (
                    2 if cb["tensors"][name]["dtype"] == "bf16" else 1
                )
                if len(raw) != expected:
                    raise ValueError(f"N-tile preload {name} has wrong byte extent")
                cb["tensors"][name]["preload_b64"] = base64.b64encode(raw).decode("ascii")
            executed = _execute_device_image(
                vsim_dir=vsim_dir, out=tile_out, run_dir=run_dir, cb=cb,
                kernel=kernel, max_cycles=binding["max_cycles"],
            )
            output_scale = np.float32(bridged_tile["record"]["output_scale"])
            actual_tiles.append(executed["device_output"] * output_scale)
            quantized_tiles.append((
                np.matmul(
                    bridged_tile["decoded"]["A0"], bridged_tile["decoded"]["W"],
                    dtype=np.float32,
                ) + bridged_tile["decoded"]["B_quant_domain"]
            ) * output_scale)
            word_tiles.append(executed["device_words"].reshape(tile_geometry["M"], width))
            dispatch_records.append({
                "dispatch_index": tile_index,
                "n_range": [n0, n1],
                "cycles": int(executed["oracle"]["cycles"]),
                "engine": _public_oracle_record(executed["oracle"]),
                "image": tile_image,
                "dram_preflight": executed["dram_preflight"],
                "raw_gsim_receipt": (
                    tile_out / "raw_gsim_receipt.json"
                ).relative_to(ROOT).as_posix(),
                "device_output": {
                    "path": executed["device_output_path"].relative_to(ROOT).as_posix(),
                    "dtype": "bf16",
                    "shape": list(executed["device_output"].shape),
                    "raw_sha256": sha256_bytes(executed["device_words"].tobytes()),
                },
            })
            calibration_records.append({
                "dispatch_index": tile_index,
                "n_range": [n0, n1],
                "measurements": bridged_tile["record"],
            })
            n0 = n1
        actual = np.concatenate(actual_tiles, axis=1)
        quantized_reference = np.concatenate(quantized_tiles, axis=1)
        device_words = np.concatenate(word_tiles, axis=1).astype("<u2", copy=False)
        device_output_path = out / "device_output.bf16.bin"
        device_output_path.write_bytes(device_words.tobytes())
        device_output = actual
        cycles = sum(item["cycles"] for item in dispatch_records)
        oracle = None
        image = {
            "kind": "three compiler-generated N-slice images",
            "dispatch_count": len(dispatch_records),
            "instruction_words_total": sum(
                item["image"]["instruction_words"] for item in dispatch_records
            ),
            "n_ranges": [item["n_range"] for item in dispatch_records],
        }
        raw_receipt_field = {
            "raw_gsim_receipts": [item["raw_gsim_receipt"] for item in dispatch_records]
        }
        measurements = {
            "kind": "per_n_slice_independent_calibration",
            "dispatches": calibration_records,
        }
    else:
        bridged = (
            _bridge_batched_inputs(
                activation, weight, partition["geometry"], code_cap=code_cap
            )
            if is_batched else bridge_inputs(
                activation, weight, bias, partition["geometry"], code_cap=code_cap,
            )
        )
        run_dir = out / "gsim_run"
        run_dir.mkdir(exist_ok=True)
        if is_batched:
            cb, kernel, image = _compile_batched_image(partition, run_dir)
            for name in ("A0", "W"):
                raw = bridged["preloads"][name]
                abi = next(entry for entry in partition["abi"]["inputs"] if entry["name"] == name)
                if len(raw) != abi["device_bytes"]:
                    raise ValueError(f"batched device preload {name} differs from the planned ABI")
                cb["tensors"][name]["preload_b64"] = base64.b64encode(raw).decode("ascii")
        else:
            cb, kernel, image = _qualified_command_buffer(partition, bridged, run_dir)
        executed = _execute_device_image(
            vsim_dir=vsim_dir, out=out, run_dir=run_dir, cb=cb,
            kernel=kernel, max_cycles=binding["max_cycles"],
        )
        output_scale = np.float32(bridged["record"]["output_scale"])
        frontier_scale = np.float32(
            binding.get("source_frontier_scale", 1.0) if is_batched else 1.0
        )
        actual = executed["device_output"] * output_scale * frontier_scale
        quantized_reference = (
            np.matmul(bridged["decoded"]["A0"], bridged["decoded"]["W"], dtype=np.float32)
            + (0 if is_batched else bridged["decoded"]["B_quant_domain"])
        ) * output_scale * frontier_scale
        device_words = executed["device_words"]
        device_output = executed["device_output"]
        device_output_path = executed["device_output_path"]
        oracle = executed["oracle"]
        cycles = int(oracle["cycles"])
        raw_receipt_field = {
            "raw_gsim_receipt": f"{binding['output_dir']}/raw_gsim_receipt.json"
        }
        measurements = bridged["record"]
        if is_batched:
            measurements["source_frontier_scale"] = float(frontier_scale)
            measurements["source_frontier_scale_hex"] = float(frontier_scale).hex()
        if (
            not is_batched
            and image["instruction_words"] != partition["image"]["instruction_words"]
        ):
            raise ValueError("executed image word count differs from the partition plan")

    source_reference = (
        reference_activation if is_batched else
        np.matmul(reference_activation, weight, dtype=np.float32) + bias
    )
    floor = float(contract["tolerance"]["max_relative_denominator_floor"])
    source_comparison = comparison(actual, source_reference, floor)
    quantized_comparison = comparison(actual, quantized_reference, floor)
    raw_partition_output_diagnostic = None
    if is_batched:
        raw_partition_output_diagnostic = comparison(
            actual / frontier_scale,
            source_reference / frontier_scale,
            floor,
        )
    tolerance = contract["tolerance"]
    passed = passes_tolerance(source_comparison, tolerance)
    negative_controls = None
    if is_batched:
        negative_controls = _batched_command_negative_controls(partition, cb)
        perturbed_reference = source_reference.copy()
        perturbed_reference.flat[0] += np.float32(
            2.0 * float(tolerance["max_abs_error"]) + 1.0
        )
        perturbed_comparison = comparison(actual, perturbed_reference, floor)
        if passes_tolerance(perturbed_comparison, tolerance):
            raise ValueError("perturbed source reference passed the fixed numeric gate")
        negative_controls["perturbed_source_reference"] = {
            "status": "rejected_by_fixed_numeric_gate",
            "perturbed_element": 0,
            "perturbation": float(2.0 * float(tolerance["max_abs_error"]) + 1.0),
            "comparison": perturbed_comparison,
        }

    dispatch["calibration_contract"] = "calibration_contract.json"
    if dispatch_records:
        dispatch["device_dispatches"] = dispatch_records
        dispatch["device_dispatch_policy"] = (
            "contiguous N slices; each dispatch repeats A, binds W[:,n0:n1]/B[n0:n1], "
            "and publishes Y[:,n0:n1] before concatenation"
        )
    dispatch["qualified_capture_semantics"] = bool(passed)
    calibration = {
        "schema": "atlas_capture_partition_calibration_v1",
        "contract": contract,
        "partition_id": binding["partition_id"],
        "source_tensors": source,
        "measurements": measurements,
    }
    result = {
        "schema": "atlas_real_capture_partition_qualification_v1",
        "claim": (
            f"one real SmolVLA {args.partition} partition qualified on elaborated RTL"
            if passed else
            f"real SmolVLA {args.partition} diagnostic rejected by the fixed numeric gate"
        ),
        "not_claimed": [
            "whole-model dispatch",
            "whole-model numeric correctness",
            "source bit-exactness",
            "whole-model performance",
        ],
        "partition_id": binding["partition_id"],
        "fqn": binding["fqn"],
        "capture_regions": partition["capture_regions"],
        "qualified_boundary": (
            {
                "accelerator_partition": binding["partition_id"],
                "source_contraction": partition["capture_regions"],
                "immediate_host_frontier": source["frontier"],
                "claim": (
                    "physical p0102 contraction and BF16 publication through its exact sole "
                    "captured scalar consumer; no later mask, softmax, or graph execution"
                ),
            }
            if is_batched else None
        ),
        "capture_semantics_executable_partitions": (
            binding["qualified_total"] if passed else binding["qualified_total"] - 1
        ),
        "candidate_outcome": "qualified" if passed else "rejected_numeric",
        "structural_partitions_total": 391,
        "engine": (
            _public_oracle_record(oracle) if oracle is not None else
            {"kind": "assertion-enabled elaborated RTL GSIM", "dispatch_count": len(dispatch_records)}
        ),
        "cycles": cycles,
        "image": image,
        "device_output": {
            "path": device_output_path.relative_to(ROOT).as_posix(),
            "dtype": "bf16",
            "shape": list(device_output.shape),
            "raw_sha256": sha256_bytes(device_words.tobytes()),
        },
        **raw_receipt_field,
        "scale_equation": {
            "activation": "qA = E4M3FN_RNE(A / sA)",
            "weight": "qW_i = E4M3FN_RNE(W[:,n0:n1] / sW_i)",
            "bias": (
                "not_applicable_for_unbiased_batched_matmul" if is_batched else
                "qB_i = BF16_RNE(B[n0:n1] / (sA * sW_i))"
            ),
            "output": (
                "Y_frontier_f32 = f32(Y_bf16_device) * (sA * sW) * 0.125"
                if is_batched else
                "Y_f32[:,n0:n1] = f32(Y_bf16_device_i) * (sA * sW_i)"
            ),
        },
        "reference_oracles": {
            "source": (
                "capture-equivalent PyTorch FX mul_22 output, independently checked as "
                "NumPy float32 matmul(A0, W) * 0.125 after complete-graph bit-exact binding"
                if is_batched else
                "NumPy float32 matmul over independently loaded capture tensors"
            ),
            "quantized_domain": (
                "NumPy float32 batched matmul over independently decoded E4M3 operands; "
                "does not consume GSIM output"
                if is_batched else
                "NumPy float32 matmul over independently decoded device-domain operands"
            ),
        },
        "source_f32_comparison": source_comparison,
        "quantized_domain_reference_comparison": quantized_comparison,
        "acceptance": {
            "thresholds": tolerance,
            "passed": bool(passed),
        },
    }
    if raw_partition_output_diagnostic is not None:
        result["raw_partition_output_diagnostic"] = {
            "comparison": raw_partition_output_diagnostic,
            "acceptance_boundary": False,
            "reason": (
                "the captured graph's sole immediate consumer scales this value by 0.125; "
                "the qualified publication boundary is the recorded mul_32 frontier"
            ),
        }
    if negative_controls is not None:
        result["negative_controls"] = negative_controls
    (out / "dispatch_manifest.json").write_text(
        json.dumps(dispatch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
