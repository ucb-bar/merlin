#!/usr/bin/env python3
"""Exact structural gates for full-width compute-only LOOP_CONV."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "compiler"))

from mlir_oot.codegen import llvm_emit
from mlir_oot.lowering.plan import Buffer, Plan, row_pitch
from mlir_oot.lowering.schedule import Scheduler
from mlir_oot.lowering.source_conv_plan import Convolution
from mlir_oot.tables import isa, loop_conv, rtl_facts


def encoded_dst(co=17, ho=16, wo=16):
    pitch = row_pitch(co)
    return Buffer("dst", [1, co, ho, wo], "i32", "intermediate", {
        "schema": "permuted_axes_storage_v1",
        "logical_shape": [1, co, ho, wo],
        "physical_shape": [1, ho, wo, co],
        "permutation": [0, 2, 3, 1],
        "dtype": "i32",
        "strides_elements": [ho * wo * pitch, wo * pitch, pitch, 1],
        "storage_elements": ho * wo * pitch,
        "offset_elements": 0,
    })


class ComputeOnlyLoopConvTest(unittest.TestCase):
    def make(self):
        buffers = {
            "ifm": Buffer("ifm", [1, 3, 16, 16], "i8", "input"),
            "weight": Buffer("weight", [3, 17], "i8", "weight"),
            "dst": encoded_dst(),
        }
        conv = Convolution(
            "ifm", "weight", "dst", 1, 3, 16, 16, 17, 1, 1, 16, 16,
            1, 1, 1, 1, 0, 0, 0, 0,
            input_layout="NCHW_batch1", weight_layout="HWIO",
            output_layout="NHWC_accumulator", output_dtype="i32",
            compute_only_native=True)
        plan = Plan("gemmini", buffers, [conv], {"params": {}}, list(buffers))
        return plan, conv

    def test_compute_only_deletes_im2col_and_never_uses_narrow_store(self):
        plan, conv = self.make()
        schedule = Scheduler(plan)
        schedule.convolution(conv)
        kinds = [ins.kind for ins in schedule.instrs]
        self.assertIn("loop_conv_ws", kinds)
        self.assertIn("mvout", kinds)
        self.assertNotIn("im2col_row", kinds)
        self.assertNotIn("loop_ws_block", kinds)
        loops = [ins for ins in schedule.instrs if ins.kind == "loop_conv_ws"]
        self.assertTrue(all(not ins.attrs["write_output"] for ins in loops))
        self.assertTrue(all(ins.attrs["trans_input_3120"] for ins in loops))
        self.assertTrue(all(ins.attrs["max_pixels_per_row"] == 1 for ins in loops))
        launch = loop_conv.static_descriptor(**{
            key: value for key, value in loops[0].attrs.items()
            if key in loop_conv.static_descriptor.__annotations__
            and key not in {"return"}
        })[-1]
        self.assertEqual((launch[1] >> 8) & 0xff, 1)
        ex = next(ins for ins in schedule.instrs if ins.kind == "config_ex")
        self.assertTrue(ex.attrs["a_transpose"])
        _, rs1, _ = isa.config_ex(
            dataflow=ex.attrs["dataflow"], sys_act=ex.attrs["act"],
            acc_scale=ex.attrs["acc_scale"], a_stride=ex.attrs["a_stride"],
            c_stride=ex.attrs["c_stride"], a_transpose=ex.attrs["a_transpose"],
            b_transpose=ex.attrs["b_transpose"])
        self.assertEqual((rs1 >> 8) & 1, 1)
        moves = [ins for ins in schedule.instrs if ins.kind == "mvout"]
        self.assertTrue(all(ins.attrs["local"] & isa.ACC_FULL_ROW_BIT for ins in moves))

    def test_each_descriptor_exactly_clears_all_owned_accumulator_rows(self):
        plan, conv = self.make()
        schedule = Scheduler(plan)
        # Force several output tiles so the test observes actual accumulator reuse.
        with patch("mlir_oot.lowering.schedule.loop_conv.auto_tile",
                   return_value=(1, 2, 2, 16, 1, 1, 3)):
            schedule.convolution(conv)
        kinds = [ins.kind for ins in schedule.instrs]
        zero = kinds.index("host_zero_i32")
        self.assertEqual(kinds[zero + 1], "fence")
        loops = [i for i, kind in enumerate(kinds) if kind == "loop_conv_ws"]
        self.assertEqual(len(loops), 8 * 8 * 2)
        emitted = [schedule.instrs[i] for i in loops]
        self.assertTrue(all(ins.attrs["no_bias"] for ins in emitted))
        left = 0
        for descriptor_index, loop_index in enumerate(loops):
            cols = 16 if descriptor_index % 2 == 0 else 1
            descriptor_expected = [(0, 2, cols), (2, 2, cols)]
            prefix = schedule.instrs[left:loop_index]
            configs = [ins for ins in prefix
                       if ins.kind == "config_ld" and ins.attrs.get("load_id") == 2]
            clears = [ins for ins in prefix
                      if ins.kind == "mvin" and ins.attrs.get("load_id") == 2]
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].attrs["stride"], 0)
            self.assertEqual(configs[0].attrs["block_stride"], 4)
            self.assertEqual(configs[0].attrs["pixel_repeats"], 1)
            self.assertEqual(
                [(ins.attrs["local"] & 0x3FFF, ins.attrs["rows"], ins.attrs["cols"])
                 for ins in clears], descriptor_expected)
            self.assertTrue(all(ins.bufs == ["__stage_0_loop_conv_zero_bias"]
                                for ins in clears))
            self.assertTrue(all(ins.attrs["local"] & isa.ACC_ADDR_BIT for ins in clears))
            self.assertTrue(all(not (ins.attrs["local"] & isa.ACC_ACCUMULATE_BIT)
                                for ins in clears))
            left = loop_index + 1

    def test_full_width_mvout_chunks_every_output_width_to_dim(self):
        for width in (16, 17, 31, 32, 33, 64):
            with self.subTest(width=width):
                ifm_encoding = {
                    "schema": "grouped_axes_storage_v1",
                    "logical_shape": [1, 3, 1, width],
                    "physical_shape": [1, 3, 1, width],
                    "axis_groups": [[0], [1], [2], [3]],
                    "dtype": "i8",
                    "strides_elements": [3 * width, width, width, 1],
                    "storage_elements": 3 * width,
                    "offset_elements": 0,
                }
                weight_encoding = {
                    "schema": "grouped_axes_storage_v1",
                    "logical_shape": [17, 3, 1, 1],
                    "physical_shape": [3, 17],
                    "axis_groups": [[2, 3, 1], [0]],
                    "dtype": "i8",
                    "strides_elements": [17, 1],
                    "storage_elements": 51,
                    "offset_elements": 0,
                }
                buffers = {
                    "ifm": Buffer("ifm", [1, 3, 1, width], "i8", "input",
                                  ifm_encoding),
                    "weight": Buffer("weight", [3, 17], "i8", "weight",
                                     weight_encoding),
                    "dst": encoded_dst(co=17, ho=1, wo=width),
                }
                conv = Convolution(
                    "ifm", "weight", "dst", 1, 3, 1, width, 17, 1, 1, 1, width,
                    1, 1, 1, 1, 0, 0, 0, 0,
                    input_layout="NCHW_batch1", weight_layout="HWIO",
                    output_layout="NHWC_accumulator", output_dtype="i32",
                    compute_only_native=True)
                plan = Plan("gemmini", buffers, [conv], {"params": {}}, list(buffers))
                schedule = Scheduler(plan)
                with patch("mlir_oot.lowering.schedule.loop_conv.auto_tile",
                           return_value=(1, 1, width, 16, 1, 1, 3)):
                    schedule.convolution(conv)
                moves = [ins for ins in schedule.instrs if ins.kind == "mvout"]
                chunks = [16] * (width // 16)
                if width % 16:
                    chunks.append(width % 16)
                self.assertEqual([ins.attrs["rows"] for ins in moves], chunks * 2)
                self.assertTrue(all(1 <= ins.attrs["rows"] <= 16 for ins in moves))
                self.assertEqual(
                    [ins.attrs["local"] & 0x3FFF for ins in moves],
                    [i for i in range(0, width, 16)] * 2)

    def test_undef_stage_is_explicitly_initialized_in_generated_llvm(self):
        plan, _ = self.make()
        schedule = Scheduler(plan)
        instrs = schedule.run()
        self.assertEqual(sum(ins.kind == "host_zero_i32" for ins in instrs), 1)
        module = llvm_emit.emit(plan, instrs, schedule.staging)
        # The initializer is executable IR, not a presumption about LLVM global contents.
        stores = [op for op in module.walk() if op.name == "llvm.store"]
        self.assertTrue(stores)
        self.assertEqual(
            sum(op.name == "llvm.inline_asm" and op.asm_string.data == "fence"
                for op in module.walk()),
            sum(ins.kind == "fence" for ins in instrs))

    def test_batch_greater_than_one_fails_closed(self):
        plan, conv = self.make()
        conv.batch = 2
        plan.buffers["ifm"].shape = [2, 3, 16, 16]
        selected, reason = Scheduler(plan).native_convolution_eligibility(conv)
        self.assertFalse(selected)
        self.assertEqual(reason, "nchw_trans_input_3120_requires_batch_one")

    def test_transposed_input_tile_uses_transposed_capacity_formula(self):
        tile = loop_conv.auto_tile(
            batch=1, ho=112, wo=112, co=64, kh=7, kw=7, ci=3,
            stride=2, kernel_dilation=1, trans_input_3120=True)
        common = dict(
            stride=2, kernel_dilation=1, trans_input_3120=True,
            batches=tile[0], porows=tile[1], pocols=tile[2], pochs=tile[3],
            krows=tile[4], kcols=tile[5], kchs=tile[6])
        self.assertLessEqual(
            loop_conv.total_rows(accumulator=False, **common),
            rtl_facts.SPAD_ROWS // 2)
        self.assertLessEqual(
            loop_conv.total_rows(accumulator=True, **common),
            rtl_facts.ACC_ROWS // 2)


if __name__ == "__main__":
    unittest.main()
