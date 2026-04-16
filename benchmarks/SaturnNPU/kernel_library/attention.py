from __future__ import annotations

from typing import Any

from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction

try:
    from .manifest_loader import load_imm
except ImportError:
    from manifest_loader import load_imm


def test() -> list[Instruction[Any]]:
    """Single-tile materialized attention: softmax((Q @ K) * scale) @ V.

    Q, K, and V are fp8 32x32 tiles. The output is stored as two bf16 32x16
    halves at DRAM 0x1000 and 0x1400.
    """
    insts: list[Instruction[Any]] = []

    # VMEM registers: Q, K, V, scale, out left, out right.
    for rd, value in (
        (1, 0x2000),
        (2, 0x2400),
        (3, 0x2800),
        (4, 0x2C00),
        (5, 0x3000),
        (6, 0x3400),
    ):
        insts.extend(load_imm(rd, value))

    # DRAM registers: Q, K, V, scale, output halves, transfer size.
    for rd, value in (
        (7, 0x0000),
        (8, 0x0400),
        (9, 0x0800),
        (10, 0x0C00),
        (11, 0x1000),
        (12, 0x1400),
        (13, 1024),
    ):
        insts.extend(load_imm(rd, value))

    insts.extend(
        [
            Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
            Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=7, rs2=13, channel=0)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=8, rs2=13, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=9, rs2=13, channel=0)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=10, rs2=13, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
            Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)),
            Instruction("vload", VectorArgs(vd=2, rs1=3, imm12=0)),
            Instruction("vload", VectorArgs(vd=3, rs1=4, imm12=0)),
        ]
    )

    # scores = Q @ K, popped as bf16 halves m4/m5.
    insts.extend(
        [
            Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=1)),
            Instruction("delay", ScalarArgs(imm=17)),
            Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
            Instruction("delay", ScalarArgs(imm=33)),
            Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4, vs1=0)),
            Instruction("vmul.bf16", VectorArgs(vd=6, vs1=4, vs2=3)),
            Instruction("vmul.bf16", VectorArgs(vd=7, vs1=5, vs2=3)),
            Instruction("delay", ScalarArgs(imm=2)),
        ]
    )

    # Stable row softmax over the two score halves.
    insts.extend(
        [
            Instruction("vredmax.row.bf16", VectorArgs(vd=8, vs1=6)),
            Instruction("vredmax.row.bf16", VectorArgs(vd=9, vs1=7)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vmaximum.bf16", VectorArgs(vd=10, vs1=8, vs2=9)),
            Instruction("delay", ScalarArgs(imm=2)),
            Instruction("vsub.bf16", VectorArgs(vd=11, vs1=6, vs2=10)),
            Instruction("vsub.bf16", VectorArgs(vd=12, vs1=7, vs2=10)),
            Instruction("delay", ScalarArgs(imm=2)),
            Instruction("vexp.bf16", VectorArgs(vd=13, vs1=11)),
            Instruction("vexp.bf16", VectorArgs(vd=14, vs1=12)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vredsum.row.bf16", VectorArgs(vd=15, vs1=13)),
            Instruction("vredsum.row.bf16", VectorArgs(vd=16, vs1=14)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vadd.bf16", VectorArgs(vd=17, vs1=15, vs2=16)),
            Instruction("delay", ScalarArgs(imm=2)),
            Instruction("vrecip.bf16", VectorArgs(vd=18, vs1=17)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vmul.bf16", VectorArgs(vd=19, vs1=13, vs2=18)),
            Instruction("vmul.bf16", VectorArgs(vd=20, vs1=14, vs2=18)),
            Instruction("delay", ScalarArgs(imm=2)),
        ]
    )

    # Pack probability halves to fp8, then value matmul.
    insts.extend(
        [
            Instruction("seli", ScalarArgs(rd=0, imm=1)),
            Instruction("vpack.bf16.fp8", VectorArgs(vd=21, vs1=19, es1=0)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=2)),
            Instruction("delay", ScalarArgs(imm=17)),
            Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=21, vs2=0)),
            Instruction("delay", ScalarArgs(imm=33)),
            Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=22, vs1=0)),
            Instruction("vstore", VectorArgs(vd=22, rs1=5, imm12=0)),
            Instruction("vstore", VectorArgs(vd=23, rs1=6, imm12=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5, rs2=13, channel=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=6, rs2=13, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        ]
    )
    return insts
