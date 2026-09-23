/* Bare-metal gemmini matmul kernel baked into the ChipTop BootROM (reset vector
 * 0x10000) for the GSIM gemmini M3 full-SoC run.  It issues a DRAM-free gemmini
 * RoCC command stream (config_ex + preload + compute) straight from M-mode at
 * boot -- no TSI program-load, no DRAM, no TLB.  This replaces the stock chipyard
 * bootrom (which wfi-spins at 0x10034 waiting for a TSI IPI that never arrives
 * once SerialRAM/SimTSI are pruned, GSIM_ENGINE.md 5).
 *
 * RoCC custom-3 (opcode 0x7b) instruction fields (rocket-chip RoCCInstruction):
 *   funct7[31:25] = gemmini funct  (config=0, compute_flip=4, compute_stay=5, preload=6)
 *   rs2[24:20], rs1[19:15]         = integer source regs (a1, a0)
 *   funct3[14:12] = {xd,xs1,xs2}   = 0b011 = 3  (no writeback, two source operands)
 * Operand words (rs1/rs2) are the SAME DRAM-free scratchpad/accumulator encodings
 * the M2 harness drove (gsim_gemmini_cmd_encode.py 4a).
 */
.section .text
.globl _start
_start:
    /* config_ex: WS dataflow, act=0, shift=0, acc_scale=1.0f */
    li a0, 0x3f80000000010004
    li a1, 0x0001000000000000
    .insn r 0x7b, 0x3, 0x0, x0, a0, a1
    /* preload: BD = scratchpad row16 -> C = accumulator row0 (16x16) */
    li a0, 0x0010001000000010
    li a1, 0x0010001080000000
    .insn r 0x7b, 0x3, 0x6, x0, a0, a1
    /* compute_preloaded: A = spad row0, BD = spad row16 (funct=4) */
    li a0, 0x0010001000000000
    li a1, 0x0010001000000010
    .insn r 0x7b, 0x3, 0x4, x0, a0, a1
    /* 12x compute_and_stay (funct=5) accumulate passes -> extend the busy window */
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    .insn r 0x7b, 0x3, 0x5, x0, a0, a1
    /* park: spin in ROM so the PC stays live and observable */
1:  j 1b
