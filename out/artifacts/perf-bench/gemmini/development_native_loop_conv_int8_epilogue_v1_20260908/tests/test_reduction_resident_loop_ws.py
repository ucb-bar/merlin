#!/usr/bin/env python3
"""Focused static regression checks for dependency-safe LOOP_WS tiling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "compiler"))

from mlir_oot.lowering.plan import Buffer, Contraction, Plan
from mlir_oot.lowering.schedule import Instr, Scheduler
from mlir_oot.lowering.source_conv_plan import Convolution
from mlir_oot.codegen.llvm_emit import Emitter
from mlir_oot.tables import isa, loop_ws


def scheduler(buffers: list[Buffer]) -> Scheduler:
    table = {buf.name: buf for buf in buffers}
    return Scheduler(Plan("gemmini", table, [], {}, list(table)))


class ReductionResidentLoopWsTest(unittest.TestCase):
    def test_each_overlappable_descriptor_programs_its_own_slot(self) -> None:
        buffers = {
            "a": Buffer("a", [32, 32], "i8", "input"),
            "b": Buffer("b", [32, 32], "i8", "weight"),
            "c": Buffer("c", [32, 32], "i32", "output"),
        }
        emitter = Emitter(Plan("gemmini", buffers, [], {}, ["a", "b", "c"]), [], {})
        emitted_functs: list[int] = []
        emitter.rocc = lambda funct, rs1, rs2: emitted_functs.append(funct)
        attrs = {
            "rows": 16, "cols": 16, "depth": 16,
            "a_stride": 32, "b_stride": 32, "c_stride": 32,
            "a_offset": 0, "b_offset": 0, "c_offset": 0, "d_offset": 0,
            "full_c": True, "accumulate": False,
        }
        first = Instr("loop_ws_block", dict(attrs), ["a", "b", "c"])
        second_attrs = dict(attrs)
        second_attrs.update(a_offset=16, b_offset=16, c_offset=64)
        second = Instr("loop_ws_block", second_attrs, ["a", "b", "c"])
        state: dict[str, tuple[int, int, int]] = {}

        emitter._emit_loop_ws_block_cached(first, state)
        first_end = len(emitted_functs)
        emitter._emit_loop_ws_block_cached(second, state)

        descriptor = [
            loop_ws.opcode("k_LOOP_WS_CONFIG_BOUNDS"),
            loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_AB"),
            loop_ws.opcode("k_LOOP_WS_CONFIG_ADDRS_DC"),
            loop_ws.opcode("k_LOOP_WS_CONFIG_STRIDES_AB"),
            loop_ws.opcode("k_LOOP_WS_CONFIG_STRIDES_DC"),
            loop_ws.opcode("k_LOOP_WS"),
        ]
        self.assertEqual(emitted_functs[:first_end], [isa.K_CONFIG] * 5 + descriptor)
        self.assertEqual(emitted_functs[first_end:], descriptor)
        self.assertEqual(set(state), {"ex", "st", "ld0", "ld1", "ld2"})

    def test_resnet_projection_conv_keeps_all_k_tiles_in_each_descriptor(self) -> None:
        # Source op 323: [512,256] @ [256,28], repeated for 28 output rows.
        self.assertEqual(loop_ws.block_shape(32, 2, 16), (16, 2, 16))
        s = scheduler([
            Buffer("ifm", [1, 256, 56, 56], "i8", "input"),
            Buffer("weight", [512, 256], "i8", "weight"),
            Buffer("dst", [1, 512, 28, 28], "i32", "output"),
        ])
        s.convolution(Convolution(
            "ifm", "weight", "dst", 1, 256, 56, 56, 512, 1, 1, 28, 28,
            2, 2, 1, 1, 0, 0, 0, 0, True))
        loops = [ins for ins in s.instrs if ins.kind == "loop_ws_block"]
        self.assertEqual(len(loops), 56)
        self.assertTrue(all(ins.attrs["depth"] == 256 for ins in loops))
        self.assertTrue(all(not ins.attrs["accumulate"] for ins in loops))
        self.assertFalse(any(ins.kind == "fence" for ins in s.instrs))

    def test_resnet_fc_keeps_all_k_tiles_in_each_descriptor(self) -> None:
        # Source op 1230: [1,2048] @ [2048,1000].
        self.assertEqual(loop_ws.block_shape(1, 63, 128), (1, 3, 128))
        s = scheduler([
            Buffer("lhs", [1, 2048], "i8", "input"),
            Buffer("rhs", [2048, 1000], "i8", "weight"),
            Buffer("dst", [1, 1000], "i32", "output"),
        ])
        s.contraction(Contraction("lhs", "rhs", "dst", 1, 2048, 1000, 2048, 1000))
        loops = [ins for ins in s.instrs if ins.kind == "loop_ws_block"]
        self.assertEqual(len(loops), 21)
        self.assertTrue(all(ins.attrs["depth"] == 2048 for ins in loops))
        self.assertTrue(all(not ins.attrs["accumulate"] for ins in loops))
        self.assertFalse(any(ins.kind == "fence" for ins in s.instrs))

    def test_contraction_falls_back_when_full_reduction_cannot_fit(self) -> None:
        self.assertIsNone(loop_ws.reduction_resident_block_shape(1, 1, 257))
        s = scheduler([
            Buffer("lhs", [16, 4112], "i8", "input"),
            Buffer("rhs", [4112, 16], "i8", "weight"),
            Buffer("dst", [16, 16], "i32", "output"),
        ])
        s.contraction(Contraction("lhs", "rhs", "dst", 16, 4112, 16, 4112, 16))
        self.assertFalse(any(ins.kind == "loop_ws_block" for ins in s.instrs))
        self.assertTrue(any(ins.kind == "compute" for ins in s.instrs))

    def test_conv_serializes_only_a_required_k_spill(self) -> None:
        s = scheduler([
            Buffer("ifm", [1, 4112, 1, 16], "i8", "input"),
            Buffer("weight", [16, 4112], "i8", "weight"),
            Buffer("dst", [1, 16, 1, 16], "i32", "output"),
        ])
        s.convolution(Convolution(
            "ifm", "weight", "dst", 1, 4112, 1, 16, 16, 1, 1, 1, 16,
            1, 1, 1, 1, 0, 0, 0, 0, True))
        kinds = [ins.kind for ins in s.instrs]
        self.assertEqual(kinds, ["loop_ws_block", "fence", "loop_ws_block"])
        loops = [ins for ins in s.instrs if ins.kind == "loop_ws_block"]
        self.assertEqual([ins.attrs["accumulate"] for ins in loops], [False, True])
        self.assertEqual(loops[0].attrs["c_offset"], loops[1].attrs["d_offset"])


if __name__ == "__main__":
    unittest.main()
