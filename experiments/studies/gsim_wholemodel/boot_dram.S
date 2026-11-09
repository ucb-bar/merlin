# Boot ROM for a DRAM-resident image under GSIM: hand control to 0x8000_0000.
#
# NOT `lui t0, 0x80000`: on RV64 lui sign-extends its 32-bit result, so that lands t0 at
# 0xffff_ffff_8000_0000 and the jump goes to a sign-extended address that is not the image.
# Building the constant by shift is unambiguous at any XLEN.
    .section .text
    .globl _start
_start:
    li   t0, 1
    slli t0, t0, 31         # t0 = 0x0000_0000_8000_0000
    jr   t0
