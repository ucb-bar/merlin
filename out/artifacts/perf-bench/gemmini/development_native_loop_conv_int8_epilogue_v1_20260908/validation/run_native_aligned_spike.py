#!/usr/bin/env python3
"""Exact local Spike check for i8 conv + i32 LOAD3 bias + scale + ReLU."""
from __future__ import annotations

import io
import json
import os
import sys
from math import prod
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[6]
ARTIFACT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ARTIFACT / "compiler"), str(ROOT / "merlin" / "python")]
# Bind compilation to the copied target adapter containing the i32-bias native-interface
# harness.  This makes the proof self-contained and cannot alter the curated/92-of-96 target.
os.environ["MERLIN_TARGET_PATH"] = str(ARTIFACT / "runtime_target")

from xdsl.printer import Printer
from mlir_oot.codegen import llvm_emit
from mlir_oot.gemmini_opt import Pipeline
from merlin.runtime.backends import base
from merlin.targetgen.contract.compile import compile_lowered_to_elf


def nested(values: list[int], shape: list[int]):
    if len(shape) == 1:
        return list(values)
    chunk = prod(shape[1:])
    return [nested(values[i * chunk:(i + 1) * chunk], shape[1:])
            for i in range(shape[0])]


def flattened(value):
    if isinstance(value, list):
        return [item for part in value for item in flattened(part)]
    return [value]


def reference(ifm: list[int], weight: list[int], bias: list[int]) -> list[int]:
    result = []
    for oy in range(3):
        for ox in range(4):
            for oc in range(16):
                total = bias[oc]
                for ky in range(3):
                    for kx in range(3):
                        iy = oy * 2 + ky - 1
                        ix = ox * 2 + kx - 1
                        if 0 <= iy < 5 and 0 <= ix < 7:
                            for ic in range(3):
                                total += ifm[(iy * 7 + ix) * 3 + ic] * (
                                    weight[((ky * 3 + kx) * 3 + ic) * 16 + oc])
                # The declared native contract: accumulator bias, positive scale,
                # RNE, saturation, ReLU.  0.125 is binary-exact.
                scaled = int(np.rint(np.float32(max(0, total)) * np.float32(0.125)))
                result.append(max(-128, min(127, scaled)))
    return result


def main() -> int:
    fixture = ARTIFACT / "tests/fixtures/native_conv_i8_bias_scale_relu.mlir"
    pipe = Pipeline(fixture.read_text()).run()
    if pipe.declined is not None:
        raise RuntimeError(pipe.declined)
    module = llvm_emit.emit(pipe.plan, pipe.instrs, pipe.staging)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    lowered = stream.getvalue() + "\n"

    shapes = {"IFM": [1, 5, 7, 3], "W": [27, 16], "B": [16], "Y0": [12, 16]}
    ifm = [((i * 5 + 3) % 7) - 3 for i in range(prod(shapes["IFM"]))]
    weight = [((i * 3 + 1) % 5) - 2 for i in range(prod(shapes["W"]))]
    bias = [((i * 37 + 11) % 101) - 50 for i in range(16)]
    expected = reference(ifm, weight, bias)
    work = ARTIFACT / "validation/native_aligned_spike"
    elf = compile_lowered_to_elf(
        pipe.plan.command_buffer, lowered, work, target="gemmini",
        inputs={"IFM": nested(ifm, shapes["IFM"]),
                "W": nested(weight, shapes["W"]), "B": bias})
    console = base.get_backend("gemmini").run_elf(elf, simulator="spike", timeout=180)
    outputs, metrics = base.get_backend("gemmini").parse_output(console)
    actual = [int(value) for value in flattened(outputs["Y0"])]
    status = "passed" if actual == expected else "failed"
    work.mkdir(parents=True, exist_ok=True)
    (work / "lowered.mlir").write_text(lowered)
    (work / "spike.log").write_text(console)
    (work / "command_buffer.json").write_text(
        json.dumps(pipe.plan.command_buffer, sort_keys=True, indent=2) + "\n")
    receipt = {
        "schema": "merlin_native_aligned_loop_conv_spike_v1",
        "status": status,
        "qualification": "local_spike_functional_not_firesim_or_fpga",
        "arithmetic": "i8xi8_to_i32_plus_i32_bias_then_f32_scale_rne_saturate_relu",
        "scale_f32": 0.125,
        "outputs": len(expected),
        "mismatches": sum(a != b for a, b in zip(actual, expected)),
        "expected": expected,
        "actual": actual,
        "metrics": metrics,
        "native_loop_conv_descriptors": sum(
            ins.kind == "loop_conv_ws" for ins in pipe.instrs),
        "load3_bias_descriptors": sum(
            ins.kind == "loop_conv_ws" and not ins.attrs["no_bias"]
            for ins in pipe.instrs),
        "elf": str(elf),
    }
    (work / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "qualification", "outputs", "mismatches", "metrics",
        "native_loop_conv_descriptors", "load3_bias_descriptors")}, sort_keys=True))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
