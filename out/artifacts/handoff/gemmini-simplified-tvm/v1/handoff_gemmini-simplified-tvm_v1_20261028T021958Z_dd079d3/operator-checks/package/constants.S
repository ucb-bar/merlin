.section .rodata.model_constants,"a",@progbits
.balign 64
.globl model_constants
model_constants:
.incbin "/scratch/grammy/merlin-tvm-firesim-ready/out/artifacts/handoff/gemmini-simplified-tvm/v1/handoff_gemmini-simplified-tvm_v1_20261028T021958Z_dd079d3/operator-checks/package/constants.bin"
.section .note.GNU-stack,"",@progbits
