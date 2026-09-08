#!/usr/bin/env python3
"""Generate a two-descriptor compute-only LOOP_CONV regression kernel."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from xdsl.printer import Printer

from mlir_oot.codegen import llvm_emit
from mlir_oot.lowering.plan import Buffer, Plan
from mlir_oot.lowering.schedule import Scheduler
from mlir_oot.lowering.source_conv_plan import Convolution
from mlir_oot.tables import isa


HERE = Path(__file__).resolve().parent


def main() -> None:
    ifm_encoding = {
        "schema": "grouped_axes_storage_v1",
        "logical_shape": [1, 3, 4, 4],
        "physical_shape": [1, 3, 4, 4],
        "axis_groups": [[0], [1], [2], [3]],
        "dtype": "i8",
        "strides_elements": [48, 16, 4, 1],
        "storage_elements": 48,
        "offset_elements": 0,
    }
    weight_encoding = {
        "schema": "grouped_axes_storage_v1",
        "logical_shape": [32, 3, 1, 1],
        "physical_shape": [3, 32],
        "axis_groups": [[2, 3, 1], [0]],
        "dtype": "i8",
        "strides_elements": [32, 1],
        "storage_elements": 96,
        "offset_elements": 0,
    }
    dst_encoding = {
        "schema": "permuted_axes_storage_v1",
        "logical_shape": [1, 32, 4, 4],
        "physical_shape": [1, 4, 4, 32],
        "permutation": [0, 2, 3, 1],
        "dtype": "i32",
        "strides_elements": [512, 128, 32, 1],
        "storage_elements": 512,
        "offset_elements": 0,
    }
    buffers = {
        "ifm": Buffer("ifm", [1, 3, 4, 4], "i8", "input", ifm_encoding),
        "weight": Buffer("weight", [3, 32], "i8", "weight", weight_encoding),
        # The micro-harness consumes the physical buffer directly; keeping this an
        # intermediate exercises the same no-ABI-escape contract as canonical t2.
        "dst": Buffer("dst", [1, 32, 4, 4], "i32", "intermediate", dst_encoding),
    }
    conv = Convolution(
        "ifm", "weight", "dst", 1, 3, 4, 4, 32, 1, 1, 4, 4,
        1, 1, 1, 1, 0, 0, 0, 0,
        input_layout="NCHW_batch1", weight_layout="HWIO",
        output_layout="NHWC_accumulator", output_dtype="i32",
        compute_only_native=True)
    plan = Plan("gemmini", buffers, [conv], {"params": {}}, ["ifm", "weight", "dst"])
    schedule = Scheduler(plan)
    # Exactly two output-channel descriptors: the second is the regression
    # witness for stale C0 accumulation after LoopConv's private D-row rotates.
    with patch("mlir_oot.lowering.schedule.loop_conv.auto_tile",
               return_value=(1, 4, 4, 16, 1, 1, 3)):
        instructions = schedule.run()
    loops = [ins for ins in instructions if ins.kind == "loop_conv_ws"]
    assert len(loops) == 2
    assert all(ins.attrs["no_bias"] and not ins.attrs["write_output"] for ins in loops)
    clear_configs = [ins for ins in instructions
                     if ins.kind == "config_ld" and ins.attrs.get("load_id") == 2]
    clears = [ins for ins in instructions
              if ins.kind == "mvin" and ins.attrs.get("load_id") == 2]
    assert len(clear_configs) == 2
    assert all(ins.attrs["stride"] == 0 and ins.attrs["block_stride"] == 16
               and ins.attrs["pixel_repeats"] == 1 for ins in clear_configs)
    assert len(clears) == 8
    expected_rows = [0, 4, 8, 12] * 2
    assert [ins.attrs["local"] & 0x3FFF for ins in clears] == expected_rows
    assert all(ins.attrs["local"] & isa.ACC_ADDR_BIT for ins in clears)
    assert all(not (ins.attrs["local"] & isa.ACC_ACCUMULATE_BIT) for ins in clears)
    assert all(ins.attrs["rows"] == 4 and ins.attrs["cols"] == 16 for ins in clears)
    module = llvm_emit.emit(plan, instructions, schedule.staging)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    (HERE / "target.mlir").write_text(stream.getvalue() + "\n")


if __name__ == "__main__":
    main()
