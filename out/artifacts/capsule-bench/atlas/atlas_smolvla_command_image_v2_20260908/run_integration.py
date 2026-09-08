#!/usr/bin/env python3
"""Compile and execute deterministic multi-kernel Atlas command images."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(ROOT / "submission"))

from merlin.targetgen.program_oracle import (  # noqa: E402
    run_program_functional_oracle,
    run_program_verilator_oracle,
)
from mlir_oot.cmdbuf import build_command_buffer  # noqa: E402
from mlir_oot.codegen import emit_program  # noqa: E402
from mlir_oot.frontend import parse_verified  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_rows(rows: int, cols: int) -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(cols)] for row in range(rows)]


def weight(rows: int, cols: int, offset: int) -> list[list[float]]:
    # Small integers are exact in FP8, BF16, and the Python reference.
    return [[float(((row * 3 + col + offset) % 5) - 2) for col in range(cols)]
            for row in range(rows)]


def fixture(case: str, cb: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    tensors = cb["tensors"]
    if case in ("bf16_movement_single", "bf16_movements"):
        x0 = weight(4, 8, 3)
        if case == "bf16_movement_single":
            return [
                {"name": "X0", "base": tensors["X0"]["base"], "dtype": "bf16", "values": x0},
            ], {"Y0": np.asarray(x0, dtype=np.float32)}
        x1 = weight(3, 7, 4)
        inputs = [
            {"name": "X0", "base": tensors["X0"]["base"], "dtype": "bf16", "values": x0},
            {"name": "X1", "base": tensors["X1"]["base"], "dtype": "bf16", "values": x1},
        ]
        return inputs, {
            "Y0": np.asarray(x0, dtype=np.float32),
            "Y1": np.asarray(x1, dtype=np.float32),
        }
    if case == "independent":
        a0, a1 = identity_rows(4, 32), identity_rows(3, 32)
        w0, w1 = weight(32, 8, 0), weight(32, 7, 1)
        inputs = [
            {"name": "W0", "base": tensors["W0"]["base"], "dtype": "fp8_e4m3", "values": w0},
            {"name": "A0", "base": tensors["A0"]["base"], "dtype": "fp8_e4m3", "values": a0},
            {"name": "W1", "base": tensors["W1"]["base"], "dtype": "fp8_e4m3", "values": w1},
            {"name": "A1", "base": tensors["A1"]["base"], "dtype": "fp8_e4m3", "values": a1},
        ]
        return inputs, {
            "Y0": np.asarray(w0[:4], dtype=np.float32),
            "Y1": np.asarray(w1[:3], dtype=np.float32),
        }
    if case == "smolvla_tail_50_720_32":
        # This is an exact SmolVLA contraction shape.  One nonzero per A row
        # makes the reference insensitive to FP8 accumulation-order rounding,
        # while still executing all 720 K iterations and the K=16 tail.
        w = weight(720, 32, 0)
        # Preload only bytes which are nonzero/observed.  The RTL harness's
        # DRAM starts at zero; splitting the identity into 50 one-byte regions
        # also avoids the legacy oracle's 128-KiB single-argv limit without
        # changing the kernel or its 50x720 memory access pattern.
        inputs = [{
            "name": "W",
            "base": tensors["W"]["base"],
            "dtype": "fp8_e4m3",
            "values": w[:50],
        }]
        inputs += [{
            "name": f"A0_identity_{row}",
            "base": tensors["A0"]["base"] + row * 720 + row,
            "dtype": "fp8_e4m3",
            "values": [1.0],
        } for row in range(50)]
        return inputs, {"Y0": np.asarray(w[:50], dtype=np.float32)}

    a0 = identity_rows(4, 32)
    w0 = [[0.0] * 8 for _ in range(32)]
    for row in range(4):
        w0[row][row] = 1.0
    w1 = weight(8, 5, 2)
    inputs = [
        {"name": "W0", "base": tensors["W0"]["base"], "dtype": "fp8_e4m3", "values": w0},
        {"name": "A0", "base": tensors["A0"]["base"], "dtype": "fp8_e4m3", "values": a0},
        {"name": "W1", "base": tensors["W1"]["base"], "dtype": "bf16", "values": w1},
    ]
    return inputs, {
        "Y0": np.asarray(identity_rows(4, 8), dtype=np.float32),
        "Y1": np.asarray(w1[:4], dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=(
            "bf16_movements",
            "bf16_movement_single",
            "independent",
            "chained",
            "smolvla_tail_50_720_32",
        ),
    )
    parser.add_argument("--engine", choices=("functional", "gsim"), default="gsim")
    parser.add_argument("--max-cycles", type=int, default=200000)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    case_dir = ROOT / "cases" / args.case
    interface = case_dir / "two_matmuls.mlir"
    workload = parse_verified(interface.read_text(encoding="utf-8"))
    cb = build_command_buffer(workload)
    kernel = case_dir / "kernel.S"
    command_buffer = case_dir / "command_buffer.json"
    kernel.write_text(emit_program(workload), encoding="utf-8")
    command_buffer.write_text(json.dumps(cb, indent=2) + "\n", encoding="utf-8")
    inputs, expected = fixture(args.case, cb)

    run_dir = case_dir / f"{args.engine}_run"
    run_dir.mkdir(exist_ok=True)
    if args.engine == "functional":
        result = run_program_functional_oracle(
            "atlas",
            model_ext="npu_model",
            cb=cb,
            kernel_s=kernel,
            inputs=inputs,
            workdir=run_dir,
            timeout=args.timeout,
            max_cycles=args.max_cycles,
        )
    else:
        result = run_program_verilator_oracle(
            "atlas",
            model_ext="npu_model",
            vsim_dir=Path("/scratch/agustin/tmp/gsim-atlas-core"),
            engine="gsim",
            cb=cb,
            kernel_s=kernel,
            inputs=inputs,
            workdir=run_dir,
            timeout=args.timeout,
            max_cycles=args.max_cycles,
        )
    comparisons = {}
    all_exact = True
    for name, reference in expected.items():
        actual = np.asarray(result["outputs"][name], dtype=np.float32)
        shape_ok = actual.shape == reference.shape
        mismatches = None if not shape_ok else int(np.count_nonzero(actual != reference))
        max_abs_error = None if not shape_ok else float(np.max(np.abs(actual - reference)))
        comparisons[name] = {
            "shape": list(actual.shape),
            "expected_shape": list(reference.shape),
            "mismatches": mismatches,
            "max_abs_error": max_abs_error,
        }
        all_exact &= shape_ok and mismatches == 0

    record = {
        "schema": "atlas_multi_kernel_execution_v2",
        "case": args.case,
        "claim": (
            "elaborated-RTL GSIM execution of one command image"
            if args.engine == "gsim"
            else "functional-model execution of one command image; not RTL cycle evidence"
        ),
        "engine": args.engine,
        "halted": True,
        "cycles": result["cycles"],
        "oracle": result["oracle"],
        "instruction_words": sum(
            line.lstrip().startswith(".word") for line in kernel.read_text().splitlines()
        ),
        "kernel_sha256": sha256(kernel),
        "command_buffer_sha256": sha256(command_buffer),
        "commands": len(cb["commands"]),
        "kernel_outputs": list(cb["kernel_abi"]["outputs"]),
        "comparisons": comparisons,
        "all_outputs_bit_exact": all_exact,
    }
    (case_dir / f"{args.engine}_result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
