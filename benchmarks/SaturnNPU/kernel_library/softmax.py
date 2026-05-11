from __future__ import annotations

from typing import Any

from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction


def test() -> list[Instruction[Any]]:
    """Stable row softmax for a split bf16 32x32 tile."""
    insts: list[Instruction[Any]] = []

    insts.append(Instruction("lui", ScalarArgs(rd=1, imm=0x2)))
    insts.append(Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)))
    insts.append(Instruction("lui", ScalarArgs(rd=3, imm=0x3)))
    insts.append(Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)))
    insts.append(Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=0)))
    insts.append(Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=1024)))
    insts.append(Instruction("addi", ScalarArgs(rd=7, rs1=0, imm=2047)))
    insts.append(Instruction("addi", ScalarArgs(rd=7, rs1=7, imm=769)))
    insts.append(Instruction("addi", ScalarArgs(rd=8, rs1=7, imm=1024)))
    insts.append(Instruction("addi", ScalarArgs(rd=9, rs1=0, imm=1024)))

    insts.append(Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)))
    insts.append(Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)))
    insts.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=9, channel=0)))
    insts.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=9, channel=1)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insts.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
    insts.append(Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)))

    insts.append(Instruction("vredmax.row.bf16", VectorArgs(vd=2, vs1=0)))
    insts.append(Instruction("vredmax.row.bf16", VectorArgs(vd=3, vs1=1)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))
    insts.append(Instruction("vmaximum.bf16", VectorArgs(vd=4, vs1=2, vs2=3)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))

    insts.append(Instruction("vsub.bf16", VectorArgs(vd=5, vs1=0, vs2=4)))
    insts.append(Instruction("vsub.bf16", VectorArgs(vd=6, vs1=1, vs2=4)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))

    insts.append(Instruction("vexp.bf16", VectorArgs(vd=7, vs1=5)))
    insts.append(Instruction("vexp.bf16", VectorArgs(vd=8, vs1=6)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))

    insts.append(Instruction("vredsum.row.bf16", VectorArgs(vd=9, vs1=7)))
    insts.append(Instruction("vredsum.row.bf16", VectorArgs(vd=10, vs1=8)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))
    insts.append(Instruction("vadd.bf16", VectorArgs(vd=11, vs1=9, vs2=10)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))

    insts.append(Instruction("vrecip.bf16", VectorArgs(vd=12, vs1=11)))
    insts.append(Instruction("delay", ScalarArgs(imm=8)))

    insts.append(Instruction("vmul.bf16", VectorArgs(vd=13, vs1=7, vs2=12)))
    insts.append(Instruction("vmul.bf16", VectorArgs(vd=14, vs1=8, vs2=12)))
    insts.append(Instruction("delay", ScalarArgs(imm=2)))

    insts.append(Instruction("vstore", VectorArgs(vd=13, rs1=3, imm12=0)))
    insts.append(Instruction("vstore", VectorArgs(vd=14, rs1=4, imm12=0)))
    insts.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=9, channel=0)))
    insts.append(Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=4, rs2=9, channel=1)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insts.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
    return insts
