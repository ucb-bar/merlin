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
    bridge_inputs,
    build_dispatch_manifest,
    comparison,
    f32_to_bf16_rne,
    passes_tolerance,
    read_f32_safetensor,
    sha256_bytes,
    validate_partition_abi,
)


PARTITIONS = {
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
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_capture_values(partition: dict, capture: Path, binding: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    input_order = load_json(capture / "input_order.json")
    weight_manifest = load_json(capture / "weights.safetensors.manifest.json")
    archive_index = int(binding["input_archive_key"][2:])
    if input_order.get(binding["input_name"]) != archive_index:
        raise ValueError("capture input_order no longer matches the partition binding")
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
    source_weight, weight_source = read_f32_safetensor(
        capture / "weights.safetensors", binding["weight_name"]
    )
    bias, bias_source = read_f32_safetensor(
        capture / "weights.safetensors", binding["bias_name"]
    )
    # This is the exact, planner-validated weight bridge for both bound projections.
    captured_weight = np.ascontiguousarray(source_weight.T)
    source = {
        "capture": (
            capture.relative_to(REPO).as_posix()
            if capture.is_relative_to(REPO) else "caller-provided capture directory"
        ),
        "input": {
            "argument_index": binding["input_arg"],
            "archive_key": binding["input_archive_key"],
            "capture_name": binding["input_name"],
            "source_shape": list(source_activation.shape),
            "device_view_shape": list(activation.shape),
            "bridges": binding["input_bridges"],
            "raw_sha256": sha256_bytes(source_activation.astype("<f4", copy=False).tobytes()),
        },
        "weight": weight_source,
        "weight_capture_bridge": "linalg.transpose permutation=[1,0]",
        "bias": bias_source,
    }
    return activation, captured_weight, bias, source


def _select_partition(binding: dict) -> dict:
    plan = load_json(ROOT / "whole_capture_plan/partition_plan.json")
    matches = [entry for entry in plan["partitions"]
               if entry.get("partition_id") == binding["partition_id"]
               and entry.get("fqn") == binding["fqn"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one bound capture partition")
    partition = matches[0]
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
    for name in ("A0", "W", "B"):
        tensor = cb.get("tensors", {}).get(name)
        raw = bridged["preloads"][name]
        abi = next(entry for entry in partition["abi"]["inputs"] if entry["name"] == name)
        if tensor is None or len(raw) != abi["device_bytes"]:
            raise ValueError(f"device preload {name} no longer matches the planned ABI")
        tensor["preload_b64"] = base64.b64encode(raw).decode("ascii")
    evidence = {
        "command_buffer": command_path.relative_to(ROOT).as_posix(),
        "command_buffer_sha256_without_preloads": sha256_file(command_path),
        "kernel": planned_kernel.relative_to(ROOT).as_posix(),
        "kernel_gzip_sha256": sha256_file(planned_kernel),
        "kernel_uncompressed_sha256": sha256_bytes(planned_source),
        "instruction_words": sum(
            line.lstrip().startswith(".word") for line in kernel.read_text().splitlines()
        ),
    }
    return copy.deepcopy(cb), kernel, evidence


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


def _run_raw_gsim(vsim_dir: Path, run_dir: Path, cb: dict, out: Path) -> tuple[dict, bytes]:
    """Replay while retaining the exact no-golden GSIM transaction and result page."""
    kernel_binary = run_dir / "kernel.bin"
    if not kernel_binary.is_file():
        raise ValueError("oracle did not retain the assembled kernel binary")
    words = np.frombuffer(kernel_binary.read_bytes(), dtype="<u4").astype(np.uint32).tolist()
    preloads = []
    for name in ("W", "A0", "B"):
        tensor = cb["tensors"][name]
        preloads.append([int(tensor["base"]), base64.b64decode(tensor["preload_b64"]).hex()])
    output = cb["tensors"]["Y0"]
    output_bytes = int(np.prod(output["shape"])) * 2
    spec = {
        "words": words,
        "preload": preloads,
        "reads": [[int(output["base"]), output_bytes]],
        "max_cycles": 1_000_000,
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
    dispatch = build_dispatch_manifest(partition)
    activation, weight, bias, source = _load_capture_values(partition, capture, binding)
    bridged = bridge_inputs(activation, weight, bias, partition["geometry"])

    out.mkdir(exist_ok=True)
    run_dir = out / "gsim_run"
    run_dir.mkdir(exist_ok=True)
    cb, kernel, image = _qualified_command_buffer(partition, bridged, run_dir)
    oracle = run_program_verilator_oracle(
        "atlas",
        model_ext="npu_model",
        vsim_dir=vsim_dir,
        engine="gsim",
        cb=cb,
        kernel_s=kernel,
        # Exact raw encodings are attached to cb tensors.  This bypasses the
        # legacy JSON value-list path without bypassing the RTL oracle.
        inputs=[],
        workdir=run_dir,
        timeout=120,
        max_cycles=1_000_000,
    )
    oracle_device_output = np.asarray(oracle["outputs"]["Y0"], dtype=np.float32)
    oracle_words, roundtrip_device_output = f32_to_bf16_rne(oracle_device_output)
    raw_receipt, raw_device_output = _run_raw_gsim(vsim_dir, run_dir, cb, out)
    if raw_receipt["cycles"] != oracle["cycles"]:
        raise ValueError("raw GSIM replay cycle count differs from the oracle run")
    if raw_device_output != oracle_words.tobytes():
        raise ValueError("retained raw GSIM readback differs from oracle output")
    device_words = np.frombuffer(raw_device_output, dtype="<u2").copy()
    device_shape = tuple(int(value) for value in cb["tensors"]["Y0"]["shape"])
    device_output = (
        (device_words.astype(np.uint32) << 16).view(np.float32).reshape(device_shape)
    )
    if not np.array_equal(oracle_device_output, roundtrip_device_output):
        raise ValueError("oracle returned values not exactly representable by its declared BF16 ABI")
    device_output_path = out / "device_output.bf16.bin"
    device_output_path.write_bytes(device_words.tobytes())
    output_scale = np.float32(bridged["record"]["output_scale"])
    actual = device_output * output_scale
    source_reference = np.matmul(activation, weight, dtype=np.float32) + bias
    quantized_reference = (
        np.matmul(bridged["decoded"]["A0"], bridged["decoded"]["W"], dtype=np.float32)
        + bridged["decoded"]["B_quant_domain"]
    ) * output_scale
    floor = float(contract["tolerance"]["max_relative_denominator_floor"])
    source_comparison = comparison(actual, source_reference, floor)
    quantized_comparison = comparison(actual, quantized_reference, floor)
    tolerance = contract["tolerance"]
    passed = passes_tolerance(source_comparison, tolerance)
    if image["instruction_words"] != partition["image"]["instruction_words"]:
        raise ValueError("executed image word count differs from the partition plan")

    dispatch["calibration_contract"] = "calibration_contract.json"
    dispatch["qualified_capture_semantics"] = bool(passed)
    calibration = {
        "schema": "atlas_capture_partition_calibration_v1",
        "contract": contract,
        "partition_id": binding["partition_id"],
        "source_tensors": source,
        "measurements": bridged["record"],
    }
    result = {
        "schema": "atlas_real_capture_partition_qualification_v1",
        "claim": f"one real SmolVLA {args.partition} partition executed on elaborated RTL",
        "not_claimed": [
            "whole-model dispatch",
            "whole-model numeric correctness",
            "source bit-exactness",
            "whole-model performance",
        ],
        "partition_id": binding["partition_id"],
        "fqn": binding["fqn"],
        "capture_regions": partition["capture_regions"],
        "capture_semantics_executable_partitions": binding["qualified_total"],
        "structural_partitions_total": 391,
        "engine": _public_oracle_record(oracle),
        "cycles": oracle["cycles"],
        "image": image,
        "device_output": {
            "path": device_output_path.relative_to(ROOT).as_posix(),
            "dtype": "bf16",
            "shape": list(device_output.shape),
            "raw_sha256": sha256_bytes(device_words.tobytes()),
        },
        "raw_gsim_receipt": f"{binding['output_dir']}/raw_gsim_receipt.json",
        "scale_equation": {
            "activation": "qA = E4M3FN_RNE(A / sA)",
            "weight": "qW = E4M3FN_RNE(W / sW)",
            "bias": "qB = BF16_RNE(B / (sA * sW))",
            "output": "Y_f32 = f32(Y_bf16_device) * (sA * sW)",
        },
        "source_f32_comparison": source_comparison,
        "quantized_domain_reference_comparison": quantized_comparison,
        "acceptance": {
            "thresholds": tolerance,
            "passed": bool(passed),
        },
    }
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
