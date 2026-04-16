from __future__ import annotations

from typing import Any

from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction


def test() -> list[Instruction[Any]]:
    """Pack split bf16 halves into one fp8 32x32 tile with unit ERF scale."""
    insts: list[Instruction[Any]] = []
    insts.append(Instruction("lui", ScalarArgs(rd=1, imm=0x2)))
    insts.append(Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)))
    insts.append(Instruction("lui", ScalarArgs(rd=3, imm=0x3)))
    insts.append(Instruction("addi", ScalarArgs(rd=4, rs1=0, imm=0)))
    insts.append(Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=1024)))
    insts.append(Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=2047)))
    insts.append(Instruction("addi", ScalarArgs(rd=6, rs1=6, imm=769)))
    insts.append(Instruction("addi", ScalarArgs(rd=7, rs1=0, imm=1024)))
    insts.append(Instruction("seli", ScalarArgs(rd=5, imm=1)))
    insts.append(Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)))
    insts.append(Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)))
    insts.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=7, channel=0)))
    insts.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=5, rs2=7, channel=1)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
    insts.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
    insts.append(Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)))
    insts.append(Instruction("vpack.bf16.fp8", VectorArgs(vd=2, vs1=0, es1=5)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))
    insts.append(Instruction("vstore", VectorArgs(vd=2, rs1=3, imm12=0)))
    insts.append(Instruction("dma.store.ch<N>", DmaArgs(rd=6, rs1=3, rs2=7, channel=0)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    return insts
