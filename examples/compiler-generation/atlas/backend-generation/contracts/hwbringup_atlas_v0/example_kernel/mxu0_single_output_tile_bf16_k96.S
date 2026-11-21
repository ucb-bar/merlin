# mxu0_single_output_tile_bf16_k96.S — C_tile = A_tile @ B_tile + bias (K=96)
#                                       weight double-buffer with overlap
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu0_single_output_tile_bf16_k96.py 42 1
#
# K=96 (3 tiles) diagnostic variant: loop body executes exactly ONCE.
# Isolates whether a single loop-body iteration is sufficient to fail.
#
# Overlap patterns demonstrated:
#   [a] weight push + acc push — prologue pushes bias and first weights
#       concurrently using Slot A (port 0) and Slot B (port 1)
#   [b] compute + weight push — main loop pre-loads one tile ahead
#       and overlaps MATMUL.ACC with the next weight push
#
# TRF bank allocation:
#   0      activations
#   4 → 5  even weight tiles (raw → transposed)
#   6 → 7  odd weight tiles (raw → transposed)
#   8      bias (FP8)
#   2, 3   BF16 result (pop output)
#
# DRAM layout (matches gen_mxu0_single_output_tile_bf16_k96.py):
#   0x00000  activations     (3 tiles × 1024B = 3KB)
#   0x00C00  weights         (3 tiles × 1024B = 3KB)
#   0x01800  bias FP8        (1024B)
#   0x01C00  result bank 2   (1024B BF16, lanes 0-15)
#   0x02000  result bank 3   (1024B BF16, lanes 16-31)

    LI    x10, 0
    LI    x11, 3             # total K tiles (K=96 / 32 = 3)
    LI    x12, 1024          # bytes per tile
    LI    x16, 0x90000000    # act DRAM offset (FPGA base + 0)
    LI    x17, 0x90000C00    # wgt DRAM offset (FPGA base + 0x00C00)
    LI    x18, 0x90001800    # bias DRAM offset (FPGA base + 0x01800)
    LI    x23, 0x90001C00    # result DRAM offset (FPGA base + 0x01C00)
    LI    x24, 1024          # result bytes per bank
    LI    x30, 1             # const 1
    LI    x5, 0x00000000
    DMA.CONFIG  x5, 0

    # Bias -> TRF 8
    LI    x6, 512
    DMA.LOAD  x6, x18, x12, 3
    DMA.WAIT  3
    LI    x6, 512
    VLOAD 8, x6, 0
    DELAY 33

    # Weights[0] -> TRF 4 -> 5
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 33
    VTRPOSE.XLU 5, 4
    DELAY 64

    # Overlap-safe prologue push pair
    VMATPUSH.W.MXU0       0, 5
    VMATPUSH.ACC.FP8.MXU0 0, 8
    DELAY 31

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

    # Preload weights[1] -> TRF 6 -> 7
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 33
    VTRPOSE.XLU 7, 6
    DELAY 64

    ADD   x17, x17, x12
    ADDI  x10, x10, 1

    # First activation tile
    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 33
    ADD   x16, x16, x12

    # Overlap-safe first compute/push
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 7
    DELAY 40

    LI    x29, 1

tile_loop:
    BEQ   x10, x11, epilogue
    NOP

    BEQ   x29, x0, preload_for_buf1
    NOP

preload_for_buf0:
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 33
    VTRPOSE.XLU 5, 4
    DELAY 64
    BEQ   x0, x0, do_matmul
    NOP

preload_for_buf1:
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 33
    VTRPOSE.XLU 7, 6
    DELAY 64

do_matmul:
    ADD   x17, x17, x12
    ADDI  x10, x10, 1

    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 33
    ADD   x16, x16, x12

    BEQ   x29, x0, matmul_buf0
    NOP

matmul_buf1:
    VMATMUL.ACC.MXU0  0, 0, 1
    VMATPUSH.W.MXU0   0, 5
    DELAY 40
    BEQ   x0, x0, advance
    NOP

matmul_buf0:
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 7
    DELAY 40

advance:
    XOR   x29, x29, x30
    BEQ   x0, x0, tile_loop
    NOP

epilogue:
    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 33

    BEQ   x29, x0, last_buf0
    NOP

last_buf1:
    VMATMUL.ACC.MXU0  0, 0, 1
    BEQ   x0, x0, do_pop
    NOP

last_buf0:
    VMATMUL.ACC.MXU0  0, 0, 0

do_pop:
    DELAY 95
    VMATPOP.BF16.MXU0  2, 0
    DELAY 31

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33

    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 33

    LI    x8, 512
    DMA.STORE x23, x8, x24, 2
    DMA.WAIT  2

    LI    x9, 768
    ADD   x25, x23, x24
    DMA.STORE x25, x9, x24, 3
    DMA.WAIT  3

    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    ECALL
