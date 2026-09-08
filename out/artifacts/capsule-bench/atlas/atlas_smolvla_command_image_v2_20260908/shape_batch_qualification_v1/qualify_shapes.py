#!/usr/bin/env python3
"""Fresh, fail-closed qualification of the SmolVLA Atlas kernel-shape library.

Compilation and shape-level RTL tests are deliberately separate from physical capture
partition qualification.  A shape receipt is mapped to every occurrence of that shape,
but it never promotes an occurrence: only an independently saved, capture-bound numeric
receipt can do that.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
REPO = ARTIFACT.parents[4]
PLAN_PATH = ARTIFACT / "whole_capture_plan/partition_plan.json"
ISOLATED_BASELINE = HERE / "backend_baseline/mlir_oot"
ISOLATED_FIXED = HERE / "backend_fixed/mlir_oot"
ATLAS_OPT = ISOLATED_FIXED / "atlas-opt"
BASELINE_ATLAS_OPT = ISOLATED_BASELINE / "atlas-opt"
GSIM_ROOT = Path("/scratch/agustin/tmp/gsim-atlas-core")
IMEM_WORDS = 32768
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(HERE / "backend_fixed"))

from merlin.targetgen.fp8_codec import fp8_e4m3_encode  # noqa: E402
from mlir_oot import encoder as atlas_encoder  # noqa: E402


# These are intentionally small/sparse inputs on exact FP8/BF16 values.  The full
# emitted loop nests still execute, including the compact K/N/B loops and tails.
NUMERIC_CASES = (
    "matmul_1_32_960_bias",
    "matmul_50_32_720_bias",
    "matmul_50_720_32_bias",
    "matmul_batched_15_50_64_113",
)

# The fix is deliberately in the shared compact-loop row-store helper, so every
# compact-loop image must change.  This explicit allowlist makes a missing or
# newly introduced shape fail closed instead of silently accepting hash drift.
EXPECTED_ASSEMBLY_CHANGES = {
    "matmul_1024_3072_768_bias",
    "matmul_1024_768_3072_bias",
    "matmul_1024_768_768_bias",
    "matmul_113_2560_960",
    "matmul_113_320_320",
    "matmul_113_960_2560",
    "matmul_113_960_320",
    "matmul_113_960_960",
    "matmul_1_32_960_bias",
    "matmul_50_1440_720_bias",
    "matmul_50_2048_720",
    "matmul_50_32_720_bias",
    "matmul_50_720_2048",
    "matmul_50_720_320",
    "matmul_50_720_32_bias",
    "matmul_50_720_720_bias",
    "matmul_50_720_960",
    "matmul_50_960_720",
    "matmul_64_12288_960",
    "matmul_768_768_1024",
    "matmul_batched_12_1024_1024_64",
    "matmul_batched_12_1024_64_1024",
    "matmul_batched_15_113_113_64",
    "matmul_batched_15_113_64_113",
    "matmul_batched_15_50_113_64",
    "matmul_batched_15_50_163_64",
    "matmul_batched_15_50_64_113",
    "matmul_batched_15_50_64_163",
}

# These results are the only existing evidence allowed to promote a physical capture
# occurrence.  Shape tests below are never consulted by this table.
DIRECT_CAPTURE_RESULTS = (
    "capture_semantics_state_proj/result.json",
    "capture_semantics_action_in_proj/result.json",
    "capture_semantics_action_time_mlp_in/result.json",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_digest(path: Path) -> tuple[str, dict[str, str]]:
    files = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    }
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\0")
    return digest.hexdigest(), files


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _control_flow_check(words: list[int]) -> dict[str, int]:
    branches = jumps = backward_edges = 0
    for index, word in enumerate(words):
        opcode = word & 0x7F
        if opcode == 0x63:
            branches += 1
            value = (
                ((word >> 31) & 1) << 12
                | ((word >> 7) & 1) << 11
                | ((word >> 25) & 0x3F) << 5
                | ((word >> 8) & 0xF) << 1
            )
            immediate = _signed(value, 13)
        elif opcode == 0x6F:
            jumps += 1
            value = (
                ((word >> 31) & 1) << 20
                | ((word >> 12) & 0xFF) << 12
                | ((word >> 20) & 1) << 11
                | ((word >> 21) & 0x3FF) << 1
            )
            immediate = _signed(value, 21)
        else:
            continue
        if immediate % 2:
            raise ValueError(f"unaligned control immediate at word {index}")
        # Atlas's compact-loop encoder and its existing regression tests use the
        # RISC-V compressed-address unit here, hence immediate / 2.
        target = index + immediate // 2
        if not 0 <= target < len(words):
            raise ValueError(
                f"control target outside image: index={index} immediate={immediate} target={target}"
            )
        backward_edges += int(target < index)
    return {"branches": branches, "jumps": jumps, "backward_edges": backward_edges}


def _parse_words(assembly: str) -> list[int]:
    words = []
    for line in assembly.splitlines():
        fields = line.lstrip().split()
        if fields and fields[0] == ".word" and len(fields) >= 2:
            words.append(int(fields[1].rstrip(","), 0) & 0xFFFFFFFF)
    return words


def validate_plan(plan: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    if plan.get("schema") != "atlas_whole_capture_partition_plan_v1":
        raise ValueError("unexpected partition-plan schema")
    kernels = plan.get("kernel_library")
    partitions = plan.get("partitions")
    if not isinstance(kernels, list) or not isinstance(partitions, list):
        raise ValueError("partition plan lacks kernel_library/partitions arrays")
    if len(kernels) != 28 or len(partitions) != 391:
        raise ValueError(
            f"capture cardinality changed: kernels={len(kernels)} partitions={len(partitions)}"
        )
    kernel_by_id = {row.get("kernel_id"): row for row in kernels}
    partition_by_id = {row.get("partition_id"): row for row in partitions}
    if None in kernel_by_id or len(kernel_by_id) != len(kernels):
        raise ValueError("kernel IDs are missing or duplicated")
    if None in partition_by_id or len(partition_by_id) != len(partitions):
        raise ValueError("partition IDs are missing or duplicated")
    occurrences = Counter()
    for partition in partitions:
        kernel_id = partition.get("kernel_id")
        if kernel_id not in kernel_by_id:
            raise ValueError(f"partition refers to unknown kernel: {kernel_id}")
        occurrences[kernel_id] += 1
    for kernel_id, kernel in kernel_by_id.items():
        if occurrences[kernel_id] != kernel.get("partition_occurrences"):
            raise ValueError(f"occurrence count drift for {kernel_id}")
    return kernel_by_id, partition_by_id


def compile_kernel(
    kernel: dict[str, Any], *, atlas_opt: Path, receipt_dir: Path,
    enforce_expected_changes: bool = True,
) -> tuple[dict[str, Any], list[int], str, dict[str, Any] | None]:
    kernel_id = str(kernel["kernel_id"])
    change_expected = enforce_expected_changes and kernel_id in EXPECTED_ASSEMBLY_CHANGES
    interface = ARTIFACT / str(kernel["interface"])
    started = time.monotonic_ns()
    errors: list[str] = []
    words: list[int] = []
    assembly = ""
    command_buffer = None
    stderr = ""
    returncode = -1
    control_flow = None
    with tempfile.TemporaryDirectory(prefix="atlas-shape-compile-") as raw_tmp:
        command_path = Path(raw_tmp) / "command_buffer.json"
        try:
            if sha256_file(interface) != kernel.get("interface_sha256"):
                raise ValueError("interface hash differs from source plan")
            proc = subprocess.run(
                [
                    str(atlas_opt),
                    f"--emit-command-buffer={command_path}",
                    "--emit-target-artifact",
                    str(interface),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            returncode, stderr, assembly = proc.returncode, proc.stderr, proc.stdout
            if proc.returncode:
                errors.append(f"atlas-opt returned {proc.returncode}")
            words = _parse_words(assembly)
            if not words:
                errors.append("compiler emitted no instruction words")
            if len(words) > IMEM_WORDS:
                errors.append(f"image exceeds IMEM: {len(words)} > {IMEM_WORDS}")
            assembly_changed = sha256_bytes(assembly.encode()) != kernel.get("assembly_sha256")
            if len(words) != kernel.get("instruction_words") and not change_expected:
                errors.append("fresh instruction count differs from source plan")
            if assembly_changed and not change_expected:
                errors.append("fresh assembly hash differs from source plan")
            if change_expected and not assembly_changed:
                errors.append("expected hardware-bounds backend change did not affect image")
            if not command_path.is_file():
                errors.append("compiler emitted no command buffer")
            else:
                command_buffer = load_json(command_path)
                if command_buffer.get("target") != "atlas":
                    errors.append("command buffer target is not atlas")
                if len(command_buffer.get("commands", [])) != kernel.get("command_count"):
                    errors.append("fresh command count differs from source plan")
            if words:
                try:
                    control_flow = _control_flow_check(words)
                except ValueError as error:
                    errors.append(str(error))
                try:
                    atlas_encoder.validate_pair_banks(words)
                except ValueError as error:
                    errors.append(f"pair-bank validation failed: {error}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed_ns = time.monotonic_ns() - started
    receipt = {
        "schema": "atlas_shape_compile_receipt_v1",
        "kernel_id": kernel_id,
        "kind": kernel["kind"],
        "geometry": kernel["geometry"],
        "bias_fused": kernel["bias_fused"],
        "partition_occurrences": kernel["partition_occurrences"],
        "source_interface": kernel["interface"],
        "source_interface_sha256": kernel["interface_sha256"],
        "compiler": str(atlas_opt),
        "compiler_sha256": sha256_file(atlas_opt) if atlas_opt.is_file() else None,
        "returncode": returncode,
        "stderr": stderr,
        "instruction_words": len(words),
        "imem_words": IMEM_WORDS,
        "assembly_sha256": sha256_bytes(assembly.encode()),
        "baseline_assembly_sha256": kernel["assembly_sha256"],
        "baseline_assembly_match": sha256_bytes(assembly.encode()) == kernel["assembly_sha256"],
        "hardware_bounds_change_expected": change_expected,
        "command_buffer_sha256": (
            sha256_bytes((json.dumps(command_buffer, sort_keys=True) + "\n").encode())
            if command_buffer is not None
            else None
        ),
        "control_flow": control_flow,
        "pair_bank_validation": "passed" if words and not any("pair-bank" in e for e in errors) else "failed",
        "elapsed_ns": elapsed_ns,
        "errors": errors,
        "qualified": not errors,
        "claim_scope": "shape compilation and static image validation only; not numeric or physical-partition qualification",
    }
    receipt_path = receipt_dir / f"{kernel_id}.json"
    write_json(receipt_path, receipt)
    return receipt, words, assembly, command_buffer


def _bf16_bytes(values: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype("<u2").tobytes()


def _fp8_bytes(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    palette = {float(value): fp8_e4m3_encode(float(value)) for value in np.unique(flat)}
    return bytes(palette[float(value)] for value in flat)


def _pattern(rows: int, cols: int, offset: int) -> np.ndarray:
    r = np.arange(rows, dtype=np.int32)[:, None]
    c = np.arange(cols, dtype=np.int32)[None, :]
    return (((r * 3 + c + offset) % 5) - 2).astype(np.float32)


def numeric_fixture(kernel_id: str, cb: dict[str, Any]) -> tuple[list[tuple[int, bytes]], np.ndarray]:
    tensors = cb["tensors"]
    if kernel_id.startswith("matmul_batched_"):
        b, m, k = map(int, tensors["A0"]["shape"])
        _, _, n = map(int, tensors["W"]["shape"])
        activations = np.zeros((b, m, k), dtype=np.float32)
        weights = np.empty((b, k, n), dtype=np.float32)
        expected = np.empty((b, m, n), dtype=np.float32)
        for batch in range(b):
            weights[batch] = _pattern(k, n, batch)
            for row in range(m):
                selected = (row * 7 + batch * 11) % k
                if row == m - 1:
                    selected = k - 1
                activations[batch, row, selected] = 1.0
                expected[batch, row] = weights[batch, selected]
        return [
            (int(tensors["A0"]["base"]), _fp8_bytes(activations)),
            (int(tensors["W"]["base"]), _fp8_bytes(weights)),
        ], expected

    m, k = map(int, tensors["A0"]["shape"])
    _, n = map(int, tensors["W"]["shape"])
    activations = np.zeros((m, k), dtype=np.float32)
    weights = _pattern(k, n, 1)
    expected = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        selected = (row * 17) % k
        if row == m - 1:
            selected = k - 1
        activations[row, selected] = 1.0
        expected[row] = weights[selected]
    preload = [
        (int(tensors["W"]["base"]), _fp8_bytes(weights)),
        (int(tensors["A0"]["base"]), _fp8_bytes(activations)),
    ]
    if "B" in tensors:
        bias = ((np.arange(n, dtype=np.int32) % 3) - 1).astype(np.float32)
        expected += bias
        preload.append((int(tensors["B"]["base"]), _bf16_bytes(bias)))
    return preload, expected


def run_numeric_case(
    kernel_id: str,
    *,
    words: list[int],
    cb: dict[str, Any],
    compile_receipt_path: Path,
    gsim_binary: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_dir = output_root / kernel_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preload, expected = numeric_fixture(kernel_id, cb)
    output = cb["tensors"]["Y0"]
    output_bytes = math.prod(output["shape"]) * 2
    max_cycles = 60_000_000
    spec = {
        "words": words,
        "preload": [[base, raw.hex()] for base, raw in preload],
        "reads": [[int(output["base"]), output_bytes]],
        "max_cycles": max_cycles,
    }
    spec_raw = (json.dumps(spec, separators=(",", ":")) + "\n").encode()
    with gzip.GzipFile(filename="", mode="wb", fileobj=(case_dir / "raw_gsim_spec.json.gz").open("wb"), mtime=0) as stream:
        stream.write(spec_raw)
    started = time.monotonic_ns()
    errors: list[str] = []
    raw_output = b""
    raw_result: dict[str, Any] = {}
    returncode = -1
    stdout = stderr = ""
    with tempfile.TemporaryDirectory(prefix="atlas-shape-rtl-") as raw_tmp:
        spec_path = Path(raw_tmp) / "spec.json"
        spec_path.write_bytes(spec_raw)
        try:
            proc = subprocess.run(
                [str(gsim_binary), str(spec_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            if returncode:
                errors.append(f"GSIM returned {returncode}")
            if stderr:
                errors.append("GSIM stderr was non-empty")
            if "Assertion failed" in stdout or "Assertion failed" in stderr:
                errors.append("assertion-enabled GSIM reported an RTL assertion")
            line = next(
                (line for line in reversed(stdout.splitlines()) if line.strip().startswith("{")),
                None,
            )
            if line is None:
                errors.append("GSIM emitted no JSON result")
            else:
                raw_result = json.loads(line)
                if not raw_result.get("halted"):
                    errors.append("GSIM program did not halt")
                outputs = raw_result.get("outputs", [])
                if len(outputs) != 1:
                    errors.append(f"GSIM returned {len(outputs)} output regions")
                else:
                    raw_output = bytes.fromhex(outputs[0])
                    if len(raw_output) != output_bytes:
                        errors.append("GSIM output extent differs from command buffer")
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{type(error).__name__}: {error}")
    elapsed_ns = time.monotonic_ns() - started
    (case_dir / "raw_output.bf16.bin").write_bytes(raw_output)
    with gzip.GzipFile(filename="", mode="wb", fileobj=(case_dir / "raw_gsim_stdout.txt.gz").open("wb"), mtime=0) as stream:
        stream.write(stdout.encode())
    (case_dir / "raw_gsim_stderr.txt").write_text(stderr, encoding="utf-8")
    comparison = None
    if len(raw_output) == output_bytes:
        actual = (
            np.frombuffer(raw_output, dtype="<u2").astype(np.uint32) << 16
        ).view(np.float32).reshape(expected.shape)
        mismatches = int(np.count_nonzero(actual != expected))
        comparison = {
            "elements": int(expected.size),
            "mismatches": mismatches,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "expected_sha256": sha256_bytes(_bf16_bytes(expected)),
        }
        if mismatches:
            errors.append(f"{mismatches} numeric mismatches")
    else:
        errors.append("numeric comparison unavailable")
    receipt = {
        "schema": "atlas_shape_rtl_numeric_receipt_v1",
        "kernel_id": kernel_id,
        "claim_scope": "synthetic exact-value shape test on assertion-enabled elaborated RTL GSIM; not capture-partition qualification",
        "compile_receipt": compile_receipt_path.relative_to(HERE).as_posix(),
        "compile_receipt_sha256": sha256_file(compile_receipt_path),
        "engine_binary": str(gsim_binary),
        "engine_sha256": sha256_file(gsim_binary) if gsim_binary.is_file() else None,
        "engine_kind": "assertion-enabled elaborated RTL GSIM",
        "returncode": returncode,
        "halted": bool(raw_result.get("halted")),
        "halt_reason": raw_result.get("halt_reason"),
        "cycles": int(raw_result.get("cycles", 0)),
        "elapsed_ns": elapsed_ns,
        "cycles_per_second": (
            float(raw_result.get("cycles", 0)) / (elapsed_ns / 1_000_000_000)
            if elapsed_ns
            else None
        ),
        "assertion_clean": "Assertion failed" not in stdout + stderr,
        "stderr_observation": "empty" if not stderr else "nonempty",
        "spec": f"numeric/{kernel_id}/raw_gsim_spec.json.gz",
        "spec_sha256": sha256_bytes(spec_raw),
        "stdout_sha256": sha256_bytes(stdout.encode()),
        "stderr_sha256": sha256_bytes(stderr.encode()),
        "raw_output": f"numeric/{kernel_id}/raw_output.bf16.bin",
        "raw_output_sha256": sha256_bytes(raw_output),
        "comparison": comparison,
        "errors": errors,
        "qualified": not errors,
    }
    write_json(case_dir / "receipt.json", receipt)
    return receipt


def run_pre_fix_negative_control(
    *,
    words: list[int],
    cb: dict[str, Any],
    compile_receipt_path: Path,
    gsim_binary: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Prove the same unchunked image still trips the assertion used by the fix."""
    kernel_id = "matmul_batched_15_50_64_113"
    case_dir = output_root / "pre_fix_batched_15_50_64_113"
    case_dir.mkdir(parents=True, exist_ok=True)
    preload, _expected = numeric_fixture(kernel_id, cb)
    output = cb["tensors"]["Y0"]
    spec = {
        "words": words,
        "preload": [[base, raw.hex()] for base, raw in preload],
        "reads": [[int(output["base"]), math.prod(output["shape"]) * 2]],
        "max_cycles": 60_000_000,
    }
    spec_raw = (json.dumps(spec, separators=(",", ":")) + "\n").encode()
    with gzip.GzipFile(
        filename="", mode="wb",
        fileobj=(case_dir / "raw_gsim_spec.json.gz").open("wb"), mtime=0,
    ) as stream:
        stream.write(spec_raw)
    started = time.monotonic_ns()
    returncode = -1
    stdout = stderr = ""
    process_error = None
    with tempfile.TemporaryDirectory(prefix="atlas-shape-negative-") as raw_tmp:
        spec_path = Path(raw_tmp) / "spec.json"
        spec_path.write_bytes(spec_raw)
        try:
            proc = subprocess.run(
                [str(gsim_binary), str(spec_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            process_error = f"{type(error).__name__}: {error}"
    elapsed_ns = time.monotonic_ns() - started
    with gzip.GzipFile(
        filename="", mode="wb",
        fileobj=(case_dir / "raw_gsim_stdout.txt.gz").open("wb"), mtime=0,
    ) as stream:
        stream.write(stdout.encode())
    (case_dir / "raw_gsim_stderr.txt").write_text(stderr, encoding="utf-8")
    assertion = "DMA VMEM transfer range exceeds VMEM capacity"
    passed = (
        process_error is None
        and returncode != 0
        and "Assertion failed" in stderr
        and assertion in stderr
        and not stdout.strip()
    )
    receipt = {
        "schema": "atlas_shape_rtl_negative_control_v1",
        "control": "pre-fix unchunked batched image on identical 15x50x64x113 workload/stimulus",
        "claim_scope": "assertion-sensitivity control only; never positive qualification evidence",
        "kernel_id": kernel_id,
        "compile_receipt": compile_receipt_path.relative_to(HERE).as_posix(),
        "compile_receipt_sha256": sha256_file(compile_receipt_path),
        "engine_binary": str(gsim_binary),
        "engine_sha256": sha256_file(gsim_binary),
        "engine_kind": "assertion-enabled elaborated RTL GSIM",
        "returncode": returncode,
        "elapsed_ns": elapsed_ns,
        "expected_assertion": assertion,
        "stderr_observation": "expected_assertion" if passed else "unexpected",
        "spec": "negative/pre_fix_batched_15_50_64_113/raw_gsim_spec.json.gz",
        "spec_sha256": sha256_bytes(spec_raw),
        "stdout_sha256": sha256_bytes(stdout.encode()),
        "stderr_sha256": sha256_bytes(stderr.encode()),
        "process_error": process_error,
        "control_passed": passed,
    }
    write_json(case_dir / "receipt.json", receipt)
    return receipt


def _verify_raw_capture_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path)
    errors = []
    if receipt.get("schema") != "atlas_real_capture_raw_gsim_receipt_v1":
        errors.append("unexpected raw receipt schema")
    for key, wanted in (
        ("returncode", 0),
        ("halted", True),
        ("assertion_clean", True),
        ("stderr_observation", "empty"),
    ):
        if receipt.get(key) != wanted:
            errors.append(f"{key} != {wanted!r}")
    for path_key, hash_key in (("spec", "spec_sha256"), ("stdout", "stdout_sha256"), ("stderr", "stderr_sha256")):
        source = ARTIFACT / str(receipt.get(path_key, ""))
        if not source.is_file() or sha256_file(source) != receipt.get(hash_key):
            errors.append(f"{path_key} missing or hash mismatch")
    engine = GSIM_ROOT / str(receipt.get("engine_binary_name", ""))
    if not engine.is_file() or sha256_file(engine) != receipt.get("engine_sha256"):
        errors.append("engine binary missing or hash mismatch")
    return {
        "path": path.relative_to(ARTIFACT).as_posix(),
        "sha256": sha256_file(path),
        "cycles": receipt.get("cycles"),
        "engine_sha256": receipt.get("engine_sha256"),
        "errors": errors,
        "qualified": not errors,
    }


def direct_capture_qualifications(
    partition_by_id: dict[str, dict], compile_receipts: dict[str, dict]
) -> dict[str, dict[str, Any]]:
    qualifications: dict[str, dict[str, Any]] = {}
    for relative in DIRECT_CAPTURE_RESULTS:
        result_path = ARTIFACT / relative
        result = load_json(result_path)
        partition_id = str(result.get("partition_id"))
        errors: list[str] = []
        partition = partition_by_id.get(partition_id)
        if partition is None:
            errors.append("result refers to unknown partition")
            continue
        if result.get("schema") != "atlas_real_capture_partition_qualification_v1":
            errors.append("unexpected capture result schema")
        if result.get("fqn") != partition.get("fqn"):
            errors.append("FQN differs from plan")
        if result.get("capture_regions") != partition.get("capture_regions"):
            errors.append("capture regions differ from plan")
        if not result.get("acceptance", {}).get("passed"):
            errors.append("fixed numeric acceptance gate did not pass")
        comparison = result.get("source_f32_comparison", {})
        thresholds = result.get("acceptance", {}).get("thresholds", {})
        if comparison.get("cosine_similarity", -math.inf) < thresholds.get("cosine_min", math.inf):
            errors.append("source cosine is below its fixed threshold")
        if comparison.get("max_abs_error", math.inf) > thresholds.get("max_abs_error", -math.inf):
            errors.append("source max-absolute error exceeds its fixed threshold")
        device_output = ARTIFACT / str(result.get("device_output", {}).get("path", ""))
        if not device_output.is_file() or sha256_file(device_output) != result.get("device_output", {}).get("raw_sha256"):
            errors.append("device output missing or hash mismatch")
        receipt_paths = result.get("raw_gsim_receipts") or [result.get("raw_gsim_receipt")]
        raw_receipts = []
        for receipt_path in receipt_paths:
            if not receipt_path:
                errors.append("missing raw GSIM receipt path")
                continue
            verified = _verify_raw_capture_receipt(ARTIFACT / receipt_path)
            raw_receipts.append(verified)
            errors.extend(verified["errors"])
        kernel_id = str(partition["kernel_id"])
        if not compile_receipts.get(kernel_id, {}).get("qualified"):
            errors.append("fresh shape compilation did not qualify")
        qualifications[partition_id] = {
            "source_kind": "direct_capture_bound_rtl_numeric_receipt",
            "result": relative,
            "result_sha256": sha256_file(result_path),
            "kernel_id": kernel_id,
            "source_f32_comparison": comparison,
            "raw_gsim_receipts": raw_receipts,
            "errors": errors,
            "qualified": not errors,
        }
    return qualifications


def build_partition_map(
    plan: dict[str, Any],
    compile_receipts: dict[str, dict],
    numeric_receipts: dict[str, dict],
    direct_qualifications: dict[str, dict],
) -> dict[str, Any]:
    rows = []
    for partition in plan["partitions"]:
        partition_id = partition["partition_id"]
        kernel_id = partition["kernel_id"]
        compile_ok = bool(compile_receipts.get(kernel_id, {}).get("qualified"))
        shape_numeric = numeric_receipts.get(kernel_id)
        direct = direct_qualifications.get(partition_id)
        physical_ok = bool(compile_ok and direct and direct.get("qualified"))
        rows.append({
            "partition_id": partition_id,
            "capture_regions": partition["capture_regions"],
            "fqn": partition.get("fqn"),
            "kernel_id": kernel_id,
            "compile_receipt": f"receipts/compile/{kernel_id}.json",
            "compile_qualified": compile_ok,
            "shape_rtl_numeric_receipt": (
                f"numeric/{kernel_id}/receipt.json" if shape_numeric is not None else None
            ),
            "shape_rtl_numeric_qualified": bool(shape_numeric and shape_numeric.get("qualified")),
            "physical_partition_qualified": physical_ok,
            "physical_qualification_source": direct.get("result") if physical_ok else None,
            "physical_unqualified_reason": (
                None
                if physical_ok
                else (
                    "fresh shape compilation failed"
                    if not compile_ok
                    else "no passing direct capture-bound RTL numeric receipt; shape evidence is non-transitive"
                )
            ),
        })
    counts = {
        "capture_partitions_total": len(rows),
        "compile_qualified_partitions": sum(row["compile_qualified"] for row in rows),
        "shape_rtl_numeric_tested_partitions": sum(row["shape_rtl_numeric_receipt"] is not None for row in rows),
        "shape_rtl_numeric_qualified_partitions": sum(row["shape_rtl_numeric_qualified"] for row in rows),
        "physical_partitions_qualified": sum(row["physical_partition_qualified"] for row in rows),
        "physical_partitions_unqualified": sum(not row["physical_partition_qualified"] for row in rows),
    }
    return {
        "schema": "atlas_capture_partition_receipt_map_v1",
        "claim": "shape receipts are mapped for traceability only; physical qualification requires direct capture-bound RTL numeric evidence",
        "counts": counts,
        "partitions": rows,
    }


def run(*, atlas_opt: Path, gsim_root: Path, skip_rtl: bool = False) -> dict[str, Any]:
    overall_started = time.monotonic_ns()
    plan_raw = PLAN_PATH.read_bytes()
    plan_sha256 = sha256_bytes(plan_raw)
    plan = json.loads(plan_raw)
    kernel_by_id, partition_by_id = validate_plan(plan)
    if set(kernel_by_id) != EXPECTED_ASSEMBLY_CHANGES:
        raise ValueError("hardware-bounds image-change allowlist differs from shape library")
    output_root = HERE / "evidence"
    compile_dir = output_root / "receipts/compile"
    compile_receipts: dict[str, dict] = {}
    runtime_artifacts: dict[str, tuple[list[int], dict[str, Any]]] = {}
    for kernel_id in sorted(kernel_by_id):
        receipt, words, _assembly, cb = compile_kernel(
            kernel_by_id[kernel_id], atlas_opt=atlas_opt, receipt_dir=compile_dir
        )
        compile_receipts[kernel_id] = receipt
        if receipt["qualified"] and cb is not None:
            runtime_artifacts[kernel_id] = (words, cb)

    numeric_receipts: dict[str, dict] = {}
    negative_control = None
    gsim_binary = gsim_root / "atlas_gsim_sim_assert"
    if not skip_rtl:
        if not gsim_binary.is_file():
            raise FileNotFoundError(f"assertion-enabled GSIM is unavailable: {gsim_binary}")
        for kernel_id in NUMERIC_CASES:
            if kernel_id not in runtime_artifacts:
                numeric_receipts[kernel_id] = {
                    "qualified": False,
                    "errors": ["fresh compilation did not produce a runnable artifact"],
                }
                continue
            words, cb = runtime_artifacts[kernel_id]
            numeric_receipts[kernel_id] = run_numeric_case(
                kernel_id,
                words=words,
                cb=cb,
                compile_receipt_path=compile_dir / f"{kernel_id}.json",
                gsim_binary=gsim_binary,
                output_root=output_root / "numeric",
            )
        baseline_kernel = kernel_by_id["matmul_batched_15_50_64_113"]
        baseline_receipt_dir = output_root / "negative/pre_fix_batched_15_50_64_113"
        baseline_receipt, baseline_words, _assembly, baseline_cb = compile_kernel(
            baseline_kernel,
            atlas_opt=BASELINE_ATLAS_OPT,
            receipt_dir=baseline_receipt_dir,
            enforce_expected_changes=False,
        )
        if not baseline_receipt["qualified"] or baseline_cb is None:
            raise RuntimeError("isolated pre-fix backend no longer reproduces the planned image")
        negative_control = run_pre_fix_negative_control(
            words=baseline_words,
            cb=baseline_cb,
            compile_receipt_path=(
                baseline_receipt_dir / "matmul_batched_15_50_64_113.json"
            ),
            gsim_binary=gsim_binary,
            output_root=output_root / "negative",
        )

    direct = direct_capture_qualifications(partition_by_id, compile_receipts)
    partition_map = build_partition_map(plan, compile_receipts, numeric_receipts, direct)
    write_json(output_root / "partition_receipt_map.json", partition_map)
    if sha256_file(PLAN_PATH) != plan_sha256:
        raise RuntimeError("partition plan changed during qualification; refusing mixed-provenance output")
    elapsed_ns = time.monotonic_ns() - overall_started
    compile_elapsed = sum(int(row["elapsed_ns"]) for row in compile_receipts.values())
    numeric_elapsed = sum(int(row.get("elapsed_ns", 0)) for row in numeric_receipts.values())
    negative_elapsed = int((negative_control or {}).get("elapsed_ns", 0))
    baseline_tree_sha256, baseline_files = tree_digest(ISOLATED_BASELINE)
    fixed_tree_sha256, fixed_files = tree_digest(ISOLATED_FIXED)
    changed_backend_files = sorted(
        name for name in set(baseline_files) | set(fixed_files)
        if baseline_files.get(name) != fixed_files.get(name)
    )
    summary = {
        "schema": "atlas_smolvla_shape_batch_qualification_v1",
        "status": (
            "bounded_progress_fail_closed"
            if all(row["qualified"] for row in compile_receipts.values())
            and all(row.get("qualified") for row in numeric_receipts.values())
            and negative_control is not None
            and negative_control.get("control_passed")
            else (
                "bounded_progress_with_detected_rtl_failure"
                if all(row["qualified"] for row in compile_receipts.values())
                and numeric_receipts
                else "qualification_failed"
            )
        ),
        "claim": "fresh compilation of 28 unique FP8 contraction shapes plus representative RTL numerics; not whole-model E2E qualification",
        "source_plan": PLAN_PATH.relative_to(ARTIFACT).as_posix(),
        "source_plan_sha256": plan_sha256,
        "counts": {
            "unique_shapes_total": len(kernel_by_id),
            "unique_shapes_compile_tested": len(compile_receipts),
            "unique_shapes_compile_qualified": sum(row["qualified"] for row in compile_receipts.values()),
            "unique_shapes_compile_unqualified": sum(not row["qualified"] for row in compile_receipts.values()),
            "unique_shapes_rtl_numeric_tested": len(numeric_receipts),
            "unique_shapes_rtl_numeric_qualified": sum(row.get("qualified", False) for row in numeric_receipts.values()),
            "unique_shapes_rtl_numeric_unqualified": len(numeric_receipts) - sum(row.get("qualified", False) for row in numeric_receipts.values()),
            "rtl_negative_controls_tested": int(negative_control is not None),
            "rtl_negative_controls_passed": int(bool(negative_control and negative_control.get("control_passed"))),
            **partition_map["counts"],
        },
        "timing": {
            "overall_elapsed_ns": elapsed_ns,
            "compile_subprocess_elapsed_ns": compile_elapsed,
            "rtl_numeric_subprocess_elapsed_ns": numeric_elapsed,
            "rtl_negative_control_elapsed_ns": negative_elapsed,
            "per_shape_compile_elapsed_ns": {
                kernel_id: row["elapsed_ns"] for kernel_id, row in sorted(compile_receipts.items())
            },
            "per_shape_rtl_numeric_elapsed_ns": {
                kernel_id: row.get("elapsed_ns") for kernel_id, row in sorted(numeric_receipts.items())
            },
        },
        "compile_receipts": {
            kernel_id: f"receipts/compile/{kernel_id}.json" for kernel_id in sorted(compile_receipts)
        },
        "rtl_numeric_receipts": {
            kernel_id: f"numeric/{kernel_id}/receipt.json" for kernel_id in sorted(numeric_receipts)
        },
        "direct_physical_qualifications": direct,
        "rtl_negative_control": {
            "receipt": "negative/pre_fix_batched_15_50_64_113/receipt.json",
            "control_passed": bool(negative_control and negative_control.get("control_passed")),
        },
        "isolated_backend": {
            "baseline_tree_sha256": baseline_tree_sha256,
            "fixed_tree_sha256": fixed_tree_sha256,
            "changed_files": changed_backend_files,
            "expected_changed_files": ["codegen.py"],
            "source_copy_scope": "complete mlir_oot package: 13 Python modules plus atlas-opt",
            "hardware_bounds": {
                "dma_beat_bytes": 32,
                "vmem_dma_line_capacity": 49152,
                "source": "AtlasCore24 DMA launch assertions: size>>5 and final line < 0xc000",
            },
        },
        "partition_receipt_map": "partition_receipt_map.json",
        "fail_closed": {
            "shape_compile_does_not_imply_shape_numeric": True,
            "shape_numeric_does_not_imply_physical_partition_numeric": True,
            "whole_model_runnable": False,
            "whole_model_numerically_qualified": False,
        },
    }
    write_json(output_root / "qualification.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-opt", type=Path, default=ATLAS_OPT)
    parser.add_argument("--gsim-root", type=Path, default=GSIM_ROOT)
    parser.add_argument("--skip-rtl", action="store_true", help="compile-only diagnostic; fails the complete gate")
    args = parser.parse_args()
    summary = run(atlas_opt=args.atlas_opt, gsim_root=args.gsim_root, skip_rtl=args.skip_rtl)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "timing": summary["timing"]}, indent=2, sort_keys=True))
    counts = summary["counts"]
    return 0 if (
        counts["unique_shapes_compile_qualified"] == counts["unique_shapes_total"]
        and counts["unique_shapes_rtl_numeric_qualified"] == len(NUMERIC_CASES)
        and counts["rtl_negative_controls_passed"] == 1
        and counts["physical_partitions_qualified"] == 3
        and counts["physical_partitions_unqualified"] == 388
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
