from __future__ import annotations

from typing import Any

from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction


def _gelu_half(insts: list[Instruction[Any]], src: int, dst: int) -> None:
    # m6 = x^2, m7 = x^3, m8 = coeff*x^3, m9 = x + coeff*x^3
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=6, vs1=src, vs2=src)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=7, vs1=6, vs2=src)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=8, vs1=7, vs2=3)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vadd.bf16", VectorArgs(vd=9, vs1=src, vs2=8)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    # m10 = k*(x + coeff*x^3), m11 = tanh(m10)
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=10, vs1=9, vs2=4)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vtanh.bf16", VectorArgs(vd=11, vs1=10)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))
    # m12 = 1 + tanh(...), m13 = 0.5*x, dst = m13*m12
    insts.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=5, vs2=11)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=13, vs1=src, vs2=2)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=dst, vs1=13, vs2=12)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))


def test() -> list[Instruction[Any]]:
    """GELU tanh approximation for a split bf16 32x32 tile."""
    insts: list[Instruction[Any]] = []

    # VMEM: input halves, constants half/coeff/k, output halves.
    insts.extend(
        [
            Instruction("lui", ScalarArgs(rd=1, imm=0x2)),
            Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),
            Instruction("addi", ScalarArgs(rd=3, rs1=2, imm=1024)),
            Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),
            Instruction("lui", ScalarArgs(rd=5, imm=0x3)),
            Instruction("addi", ScalarArgs(rd=6, rs1=5, imm=1024)),
            Instruction("addi", ScalarArgs(rd=7, rs1=6, imm=1024)),
            Instruction("addi", ScalarArgs(rd=8, rs1=0, imm=0)),
            Instruction("addi", ScalarArgs(rd=9, rs1=0, imm=1024)),
            Instruction("addi", ScalarArgs(rd=10, rs1=9, imm=1024)),
            Instruction("addi", ScalarArgs(rd=11, rs1=10, imm=1024)),
            Instruction("lui", ScalarArgs(rd=12, imm=0x1)),
            Instruction("addi", ScalarArgs(rd=12, rs1=12, imm=0)),
            Instruction("lui", ScalarArgs(rd=13, imm=0x1)),
            Instruction("addi", ScalarArgs(rd=13, rs1=13, imm=1024)),
            Instruction("lui", ScalarArgs(rd=14, imm=0x1)),
            Instruction("addi", ScalarArgs(rd=14, rs1=14, imm=1024)),
            Instruction("addi", ScalarArgs(rd=15, rs1=14, imm=1024)),
            Instruction("addi", ScalarArgs(rd=16, rs1=0, imm=1024)),
        ]
    )

    insts.extend(
        [
            Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
            Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=8, rs2=16, channel=0)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=9, rs2=16, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=10, rs2=16, channel=0)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=11, rs2=16, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=5, rs1=12, rs2=16, channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        ]
    )

    insts.extend(
        [
            Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
            Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)),
            Instruction("vload", VectorArgs(vd=2, rs1=3, imm12=0)),
            Instruction("vload", VectorArgs(vd=3, rs1=4, imm12=0)),
            Instruction("vload", VectorArgs(vd=4, rs1=5, imm12=0)),
            Instruction("vli.all", VectorArgs(vd=5, imm=1)),
        ]
    )

    _gelu_half(insts, src=0, dst=14)
    _gelu_half(insts, src=1, dst=15)
    insts.extend(
        [
            Instruction("vstore", VectorArgs(vd=14, rs1=6, imm12=0)),
            Instruction("vstore", VectorArgs(vd=15, rs1=7, imm12=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=14, rs1=6, rs2=16, channel=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=15, rs1=7, rs2=16, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        ]
    )
    return insts
