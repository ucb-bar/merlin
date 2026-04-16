from __future__ import annotations

from typing import Any

from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction


def test() -> list[Instruction[Any]]:
    """Elementwise bf16 division for split 32x32 tiles: C = A / B."""
    insts: list[Instruction[Any]] = []
    insts.extend(
        [
            Instruction("lui", ScalarArgs(rd=1, imm=0x2)),
            Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),
            Instruction("addi", ScalarArgs(rd=3, rs1=2, imm=1024)),
            Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),
            Instruction("lui", ScalarArgs(rd=5, imm=0x3)),
            Instruction("addi", ScalarArgs(rd=6, rs1=5, imm=1024)),
            Instruction("addi", ScalarArgs(rd=7, rs1=0, imm=0)),
            Instruction("addi", ScalarArgs(rd=8, rs1=0, imm=1024)),
            Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),
            Instruction("addi", ScalarArgs(rd=10, rs1=9, imm=1024)),
            Instruction("lui", ScalarArgs(rd=11, imm=0x1)),
            Instruction("addi", ScalarArgs(rd=11, rs1=11, imm=-1024)),
            Instruction("addi", ScalarArgs(rd=12, rs1=11, imm=1024)),
            Instruction("addi", ScalarArgs(rd=13, rs1=0, imm=1024)),
        ]
    )

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
        ]
    )

    insts.extend(
        [
            Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
            Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)),
            Instruction("vload", VectorArgs(vd=2, rs1=3, imm12=0)),
            Instruction("vload", VectorArgs(vd=3, rs1=4, imm12=0)),
            Instruction("vrecip.bf16", VectorArgs(vd=4, vs1=2)),
            Instruction("vrecip.bf16", VectorArgs(vd=5, vs1=3)),
            Instruction("delay", ScalarArgs(imm=8)),
            Instruction("vmul.bf16", VectorArgs(vd=6, vs1=0, vs2=4)),
            Instruction("vmul.bf16", VectorArgs(vd=7, vs1=1, vs2=5)),
            Instruction("delay", ScalarArgs(imm=2)),
            Instruction("vstore", VectorArgs(vd=6, rs1=5, imm12=0)),
            Instruction("vstore", VectorArgs(vd=7, rs1=6, imm12=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5, rs2=13, channel=0)),
            Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=6, rs2=13, channel=1)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
            Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        ]
    )
    return insts
