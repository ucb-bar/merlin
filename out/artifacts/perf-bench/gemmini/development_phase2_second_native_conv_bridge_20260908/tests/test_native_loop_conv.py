#!/usr/bin/env python3
"""Behavioral checks for capability-selected native convolution lowering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "compiler"))

from mlir_oot.lowering.plan import Buffer, Plan
from mlir_oot.lowering.schedule import Scheduler
from mlir_oot.lowering.source_conv_plan import Convolution
from mlir_oot.lowering.native_conv_selector import select_native_loop_conv
from mlir_oot.tables import isa, loop_conv, rtl_facts as F
from mlir_oot.codegen import gemmini_module, llvm_emit
from mlir_oot.gemmini_opt import Pipeline


def scheduler(buffers: list[Buffer]) -> Scheduler:
    table = {buf.name: buf for buf in buffers}
    return Scheduler(Plan("gemmini", table, [], {}, list(table)))


class NativeLoopConvTest(unittest.TestCase):
    def test_native_epilogue_requires_bias_then_scale_then_activation_order(self) -> None:
        activation = Buffer("ifm", [1, 5, 7, 3], "i8", "input")
        weight = Buffer("weight", [3, 3, 3, 16], "i8", "weight")
        dst = Buffer("dst", [1, 3, 4, 16], "i8", "output")
        bias = Buffer("bias", [16], "i32", "bias")
        conv = Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1,
            input_layout="NHWC", weight_layout="HWIO", output_layout="NHWC",
            output_dtype="i8", bias="bias", acc_scale=0.5, relu=True)
        selected, reason = select_native_loop_conv(
            conv, activation, weight, dst,
            epilogue_stages=("bias", "acc_scale", "relu"),
            bias="bias", bias_buffer=bias, acc_scale=0.5)
        self.assertTrue(selected, reason)
        selected, reason = select_native_loop_conv(
            conv, activation, weight, dst,
            epilogue_stages=("acc_scale", "bias", "relu"),
            bias="bias", bias_buffer=bias, acc_scale=0.5)
        self.assertFalse(selected)
        self.assertEqual(
            reason, "native_loop_conv_epilogue_order_must_be_bias_scale_activation")

    def test_capacity_selected_tile_fits_double_buffered_memories(self) -> None:
        tile = loop_conv.auto_tile(
            batch=1, ho=112, wo=112, co=64, kh=7, kw=7, ci=3,
            stride=2, kernel_dilation=1)
        common = dict(
            stride=2, kernel_dilation=1, batches=tile[0], porows=tile[1],
            pocols=tile[2], pochs=tile[3], krows=tile[4], kcols=tile[5],
            kchs=tile[6])
        self.assertLessEqual(
            loop_conv.total_rows(accumulator=False, **common), F.SPAD_ROWS // 2)
        self.assertLessEqual(
            loop_conv.total_rows(accumulator=True, **common), F.ACC_ROWS // 2)

    def test_descriptor_matches_the_public_gemmini_macro(self) -> None:
        records = loop_conv.static_descriptor(
            batch_size=1, in_row_dim=5, in_col_dim=7, in_channels=3,
            out_channels=16, out_row_dim=3, out_col_dim=4,
            pool_out_row_dim=3, pool_out_col_dim=4,
            stride=2, padding=1, kernel_dim=3, kernel_dilation=1,
            pool_size=1, pool_stride=1, pool_padding=0,
            batches=1, porows=2, pocols=4, pochs=16,
            krows=3, kcols=3, kchs=3,
            lpad=1, rpad=0, upad=1, dpad=0,
            plpad=0, prpad=0, pupad=0, pdpad=0,
            orows=2, ocols=4, in_stride=3, weight_stride=16,
            out_stride=16, no_bias=True, no_pool=True,
            downsample=False, wrot180=False, input_dilated=False,
            activation=isa.NO_ACTIVATION, trans_output_1203=False,
            trans_weight_1203=False, trans_weight_0132=False,
            trans_input_3120=False, max_pixels_per_row=1, dw=False,
            a_spad_id=0, b_spad_id=0)

        # Literal expansion of Jack's pinned gemmini_loop_conv_ws macro.
        expected = [
            (16, (16 << 48) | (3 << 32) | (5 << 16) | 1,
             (1 << 56) | (2 << 48) | (4 << 32) | (3 << 16) | 3),
            (17, (3 << 48) | (4 << 32) | (1 << 16) | (1 << 8),
             (1 << 48) | (2 << 32) | (4 << 16) | 16),
            (18, (3 << 48) | (3 << 32) | (3 << 16) | 1,
             (1 << 32) | 7),
            (19, (2 << 48) | 1,
             (3 << 48) | (16 << 32) | (16 << 16) | 4),
            (15, (1 << 8) | 1, 1),
        ]
        self.assertEqual(records, expected)
        self.assertTrue(F.HAS_LOOP_CONV)
        self.assertFalse(F.HAS_IM2COL)

    def test_native_nhwc_i8_convolution_emits_no_host_im2col(self) -> None:
        s = scheduler([
            Buffer("ifm", [1, 5, 7, 3], "i8", "input"),
            Buffer("weight", [3, 3, 3, 16], "i8", "weight"),
            Buffer("dst", [1, 3, 4, 16], "i8", "output"),
        ])
        s.convolution(Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1,
            input_layout="NHWC", weight_layout="HWIO", output_layout="NHWC",
            output_dtype="i8"))

        kinds = [ins.kind for ins in s.instrs]
        self.assertIn("loop_conv_ws", kinds)
        self.assertNotIn("im2col_row", kinds)
        self.assertNotIn("loop_ws_block", kinds)
        ex = next(ins for ins in s.instrs if ins.kind == "config_ex")
        # gemmini_extended3_config_ex names C_stride before A_stride: convolution puts the
        # input spatial stride in A_stride and leaves accumulator rows contiguous.
        self.assertEqual((ex.attrs["a_stride"], ex.attrs["c_stride"]), (2, 1))

    def test_public_i8_conv2d_reaches_capability_selected_native_path(self) -> None:
        text = (BUNDLE / "tests" / "fixtures" / "native_conv_i8.mlir").read_text()
        pipe = Pipeline(text).run()

        self.assertIsNone(pipe.declined)
        kinds = [ins.kind for ins in pipe.instrs]
        self.assertEqual(kinds.count("loop_conv_ws"), 1)
        self.assertNotIn("im2col_row", kinds)
        self.assertNotIn("loop_ws_block", kinds)
        self.assertNotIn("im2col_recipes", pipe.plan.command_buffer.get("params", {}))
        receipt = pipe.plan.command_buffer["params"]["convolution_lowering"]
        self.assertEqual(receipt["native_loop_conv_count"], 1)
        self.assertEqual(receipt["fallback_count"], 0)

    def test_public_full_width_conv2d_preserves_the_existing_fallback(self) -> None:
        text = (BUNDLE / "tests" / "fixtures" / "native_conv_i8.mlir").read_text()
        text = text.replace('output_dtype = "i8"', 'output_dtype = "i32"')
        text = text.replace('-> tensor<12x16xi8>', '-> tensor<12x16xi32>')
        pipe = Pipeline(text).run()

        self.assertIsNone(pipe.declined)
        kinds = [ins.kind for ins in pipe.instrs]
        self.assertNotIn("loop_conv_ws", kinds)
        self.assertIn("loop_ws_block", kinds)
        self.assertIn("im2col_recipes", pipe.plan.command_buffer["params"])

    def test_public_i8_conv2d_falls_back_when_capability_is_absent(self) -> None:
        text = (BUNDLE / "tests" / "fixtures" / "native_conv_i8.mlir").read_text()
        with patch.object(F, "HAS_LOOP_CONV", False):
            pipe = Pipeline(text).run()

        self.assertIsNone(pipe.declined)
        kinds = [ins.kind for ins in pipe.instrs]
        self.assertNotIn("loop_conv_ws", kinds)
        self.assertIn("im2col_recipes", pipe.plan.command_buffer["params"])

    def test_native_reduction_tiling_keeps_output_until_the_final_k_slice(self) -> None:
        text = (BUNDLE / "tests" / "fixtures" /
                "native_conv_i8_reduction_tiled.mlir").read_text()
        pipe = Pipeline(text).run()

        convs = [ins for ins in pipe.instrs if ins.kind == "loop_conv_ws"]
        self.assertGreater(len(convs), 1)
        self.assertTrue(any(ins.attrs["kchs"] < 1024 for ins in convs))
        self.assertEqual(sum(bool(ins.attrs["write_output"]) for ins in convs), 1)
        self.assertTrue(convs[-1].attrs["write_output"])

    def test_full_width_source_convolution_falls_back_without_semantic_change(self) -> None:
        s = scheduler([
            Buffer("ifm", [1, 3, 5, 7], "i8", "input"),
            Buffer("weight", [16, 27], "i8", "weight"),
            Buffer("dst", [1, 16, 3, 4], "i32", "output"),
        ])
        s.convolution(Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1))

        kinds = [ins.kind for ins in s.instrs]
        self.assertNotIn("loop_conv_ws", kinds)
        self.assertIn("im2col_row", kinds)
        self.assertIn("loop_ws_block", kinds)

    def test_exact_narrow_source_convolution_uses_narrow_loop_ws_without_host_boundary(self) -> None:
        # The target-neutral source epilogue formation produces this NCHW task.  The current target
        # cannot use native LOOP_CONV for NCHW, but it must still delete the i32 boundary by using a
        # reduction-resident narrow LOOP_WS readout rather than quietly restoring an i32 result.
        s = scheduler([
            Buffer("ifm", [1, 3, 5, 7], "i8", "input"),
            Buffer("weight", [16, 27], "i8", "weight"),
            Buffer("dst", [1, 16, 3, 4], "i8", "output"),
        ])
        s.convolution(Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1,
            output_dtype="i8", relu=True))

        kinds = [ins.kind for ins in s.instrs]
        self.assertNotIn("loop_conv_ws", kinds)
        self.assertIn("im2col_row", kinds)
        # The three fences synchronize host-filled row slabs with Gemmini; they are internal im2col
        # dependencies, not the eliminated device-to-host epilogue boundary.
        self.assertEqual(kinds.count("fence"), 3)
        loops = [ins for ins in s.instrs if ins.kind == "loop_ws_block"]
        self.assertTrue(loops)
        self.assertTrue(all(ins.attrs["full_c"] is False for ins in loops))
        self.assertTrue(all(ins.attrs["c_dtype"] == "i8" for ins in loops))
        self.assertTrue(all(ins.attrs["acc_act"] == isa.RELU for ins in loops))

    def test_native_descriptor_reaches_verified_target_ir_and_llvm(self) -> None:
        buffers = {
            "ifm": Buffer("ifm", [1, 5, 7, 3], "i8", "input"),
            "weight": Buffer("weight", [3, 3, 3, 16], "i8", "weight"),
            "dst": Buffer("dst", [1, 3, 4, 16], "i8", "output"),
        }
        conv = Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1,
            input_layout="NHWC", weight_layout="HWIO", output_layout="NHWC",
            output_dtype="i8")
        plan = Plan("gemmini", buffers, [conv], {}, ["ifm", "weight", "dst"])
        s = Scheduler(plan)
        instrs = s.run()

        target = gemmini_module.build(plan, instrs, s.staging)
        self.assertEqual(sum(op.name == "gemmini.loop_conv_ws" for op in target.walk()), 1)
        llvm = llvm_emit.emit(plan, instrs, s.staging)
        asm = [op for op in llvm.walk() if op.name == "llvm.inline_asm"]
        functs = [op.asm_string.data for op in asm]
        for funct in range(15, 22):
            self.assertEqual(functs.count(isa.asm_string(funct)), 1)

    def test_selection_is_exposed_in_the_compiler_receipt(self) -> None:
        buffers = {
            "ifm": Buffer("ifm", [1, 5, 7, 3], "i8", "input"),
            "weight": Buffer("weight", [3, 3, 3, 16], "i8", "weight"),
            "dst": Buffer("dst", [1, 3, 4, 16], "i8", "output"),
        }
        conv = Convolution(
            "ifm", "weight", "dst", 1, 3, 5, 7, 16, 3, 3, 3, 4,
            2, 2, 1, 1, 1, 1, 1, 1,
            input_layout="NHWC", weight_layout="HWIO", output_layout="NHWC",
            output_dtype="i8")
        cb = {"params": {"convolution_lowering": {}}}
        plan = Plan("gemmini", buffers, [conv], cb, list(buffers))
        Scheduler(plan).run()

        receipt = cb["params"]["convolution_lowering"]
        self.assertEqual(receipt["native_loop_conv_count"], 1)
        self.assertEqual(receipt["fallback_count"], 0)
        self.assertEqual(receipt["selections"][0]["reason"], "selected")

    def test_boolean_mask_storage_is_byte_addressable(self) -> None:
        self.assertEqual(Buffer("mask", [3, 5], "i1", "scratch").nbytes, 48)


if __name__ == "__main__":
    unittest.main()
