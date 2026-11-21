# Scalar VMEM load/store roundtrip at a nonzero address.
#
# Scalar LSU addresses are bytes, while the DMA VMEM operand is a 32-bit word address.  Therefore
# scalar byte address 0x4000 and DMA word address 0x1000 name the same location.  Keeping the address
# nonzero makes a missing x4 conversion observable; a base-zero test cannot distinguish the units.
#
# @EXPECT_MEMORY 0x90000000 dd42341200000000000000000000000000000000000000000000000000000000

    ADDI  x28, x0, 1
    LI    x8, 0x00004000

    LI    x10, 0xAABBCCDD
    SW    x10, x8, 0
    LW    x9, x8, 0
    DELAY 2
    LI    x30, 0xAABBCCDD
    BNE   x9, x30, fail
    NOP

    ADDI  x11, x0, 0x42
    SB    x11, x8, 1
    LBU   x9, x8, 1
    DELAY 2
    ADDI  x30, x0, 0x42
    BNE   x9, x30, fail
    NOP

    LI    x12, 0x1234
    SH    x12, x8, 2
    LHU   x9, x8, 2
    DELAY 2
    LI    x30, 0x1234
    BNE   x9, x30, fail
    NOP

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x3, 0x90000000
    LI    x6, 0x00001000
    LI    x2, 32
    DMA.STORE x3, x6, x2, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
