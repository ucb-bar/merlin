.section .rodata.dse,"a",@progbits
.balign 64
.globl dse_inputs
dse_inputs:
.incbin "/scratch/grammy/merlin-tvm-firesim-ready/out/build/firesim-preparation/elf-smolvla-action-input/guest/inputs.bin"
.balign 64
.globl dse_expected
dse_expected:
.incbin "/scratch/grammy/merlin-tvm-firesim-ready/out/build/firesim-preparation/elf-smolvla-action-input/guest/expected.bin"
.section .note.GNU-stack,"",@progbits
