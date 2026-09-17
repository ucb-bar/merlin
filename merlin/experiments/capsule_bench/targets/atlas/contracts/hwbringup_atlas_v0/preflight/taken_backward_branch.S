# Observable, definitely-taken backward-branch probe.
#
# Atlas assembly uses RISC-V byte displacements, while the RTL core keeps its PC as an instruction-word
# index.  Correct execution divides the encoded displacement by four.  The backward branch below is at
# word 5 and targets word 3; treating the -8-byte displacement as -4 words instead restarts at word 1,
# reinitializes both counters, and never halts.  Thus this checks the real control-flow semantics rather
# than merely checking that BNE decodes.
#
# On a correct target the loop executes exactly three times.  Its trip count is written into VMEM using
# the scalar byte address and copied to DRAM using the DMA word address for target-neutral readback.
#
# @EXPECT_MEMORY 0x90000000 0300000000000000000000000000000000000000000000000000000000000000

    ADDI  x28, x0, 1
    ADDI  x16, x0, 3
    ADDI  x17, x0, 0

loop:
    ADDI  x17, x17, 1
    ADDI  x16, x16, -1
    BNE   x16, x0, loop
    NOP

    LI    x8, 0x00004000
    SW    x17, x8, 0

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
