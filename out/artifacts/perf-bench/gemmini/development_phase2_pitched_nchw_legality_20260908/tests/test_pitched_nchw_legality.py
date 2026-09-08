#!/usr/bin/env python3
"""Regression gates for batch-one pitched-NCHW LOOP_CONV addressing."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
GEMMINI = ROOT.parent
COMPILER = GEMMINI / "development_phase2_second_native_conv_bridge_20260908" / "compiler"
SEALED = GEMMINI / "development_phase2_compute_only_loop_conv_20260908"
sys.path.insert(0, str(COMPILER))

from mlir_oot.lowering.plan import Buffer, Plan, row_pitch
from mlir_oot.lowering.schedule import Scheduler
from mlir_oot.lowering.source_conv_plan import Convolution


def ordinary_nchw(name: str, *, channels: int = 64, height: int = 56,
                  width: int = 56, pitch: int = 64) -> Buffer:
    shape = [1, channels, height, width]
    return Buffer(name, shape, "i8", "intermediate", {
        "schema": "grouped_axes_storage_v1",
        "logical_shape": shape,
        "physical_shape": shape,
        "axis_groups": [[0], [1], [2], [3]],
        "dtype": "i8",
        "strides_elements": [channels * height * pitch, height * pitch, pitch, 1],
        "storage_elements": channels * height * pitch,
        "offset_elements": 0,
    })


def encoded_dst(*, channels: int = 64, height: int = 56, width: int = 56) -> Buffer:
    pitch = row_pitch(channels)
    return Buffer("dst", [1, channels, height, width], "i32", "intermediate", {
        "schema": "permuted_axes_storage_v1",
        "logical_shape": [1, channels, height, width],
        "physical_shape": [1, height, width, channels],
        "permutation": [0, 2, 3, 1],
        "dtype": "i32",
        "strides_elements": [height * width * pitch, width * pitch, pitch, 1],
        "storage_elements": height * width * pitch,
        "offset_elements": 0,
    })


def second_conv_plan() -> tuple[Plan, Convolution]:
    buffers = {
        "ifm": ordinary_nchw("ifm"),
        "weight": Buffer("weight", [64, 64], "i8", "weight"),
        "dst": encoded_dst(),
    }
    conv = Convolution(
        "ifm", "weight", "dst", 1, 64, 56, 56, 64, 1, 1, 56, 56,
        1, 1, 1, 1, 0, 0, 0, 0,
        input_layout="NCHW_batch1", weight_layout="HWIO",
        output_layout="NHWC_accumulator", output_dtype="i32",
        compute_only_native=True, diagnostic_transposed_compute_only=True)
    return Plan("gemmini", buffers, [conv], {"params": {}}, list(buffers)), conv


class PitchedNchwLegalityTest(unittest.TestCase):
    def test_second_resnet_conv_is_addressable_without_an_input_copy(self):
        plan, conv = second_conv_plan()
        selected, reason = Scheduler(plan).native_convolution_eligibility(conv)
        self.assertTrue(selected, reason)

        schedule = Scheduler(plan)
        with patch("mlir_oot.lowering.schedule.loop_conv.auto_tile",
                   return_value=(1, 2, 3, 16, 1, 1, 16)):
            schedule.convolution(conv)
        loops = [ins for ins in schedule.instrs if ins.kind == "loop_conv_ws"]
        self.assertTrue(loops)
        self.assertTrue(all(ins.attrs["in_col_dim"] == 64 for ins in loops))
        self.assertTrue(all(ins.attrs["out_col_dim"] == 56 for ins in loops))
        witness = next(ins for ins in loops
                       if ins.attrs["input_offset"] == 16 * 56 * 64 + 2 * 64 + 3)
        self.assertEqual(witness.attrs["input_offset"], 57475)

    def test_noncanonical_stride_vector_fails_closed(self):
        plan, conv = second_conv_plan()
        plan.buffers["ifm"].storage_encoding["strides_elements"][1] += 1
        selected, reason = Scheduler(plan).native_convolution_eligibility(conv)
        self.assertFalse(selected)
        self.assertEqual(reason, "trans_input_3120_pitched_nchw_encoding_is_not_proven")

    def test_all_later_resnet_convs_are_semantically_representable(self):
        command_buffer = json.loads(
            (SEALED / "validation/canonical_resnet50/command_buffer.json").read_text())
        encodings = command_buffer["params"]["storage_encodings"]
        convs = [command for command in command_buffer["commands"]
                 if command["opcode"] == "CONV2D"]
        pitches: dict[tuple[int, int], int] = {}
        representable = 0
        for command in convs:
            attrs = command["attributes"]
            encoding = encodings[command["operands"]["ifm"]]
            dst = command_buffer["tensors"][command["operands"]["dst"]]
            n, ci, hi, wi = encoding["logical_shape"]
            pitch = encoding["strides_elements"][-2]
            kh, kw, kci, co = attrs["kernel"]
            stride, dilation, padding = attrs["stride"], attrs["dilation"], attrs["padding"]
            canonical = (
                n == 1 and kci == ci and kh == kw
                and stride[0] == stride[1] and dilation[0] == dilation[1]
                and len(set(padding)) == 1 and 0 <= padding[0] < kh
                and pitch >= wi
                and encoding["strides_elements"] ==
                    [ci * hi * pitch, hi * pitch, pitch, 1]
                and max(n, ci, hi, pitch, co, kh, kw) < (1 << 16)
                and stride[0] < (1 << 8) and dilation[0] < (1 << 10)
                and kh * kw * ci * 128 * 128 <= (1 << 31) - 1)
            self.assertEqual((dst["role"], dst["dtype"]), ("intermediate", "i32"))
            representable += int(canonical)
            pitches[(wi, pitch)] = pitches.get((wi, pitch), 0) + 1
        self.assertEqual(representable, 53)
        self.assertEqual(pitches, {
            (224, 224): 1, (56, 64): 13, (28, 32): 13,
            (14, 16): 19, (7, 16): 7,
        })


if __name__ == "__main__":
    unittest.main()
