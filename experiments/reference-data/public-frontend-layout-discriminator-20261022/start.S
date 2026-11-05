.section .text.init
.globl _start
_start:
  la sp, stack_top
  li t0, 0x6000
  csrs mstatus, t0
  call main
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sd a0, 0(t0)
1: j 1b
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
.globl fromhost
tohost: .dword 0
fromhost: .dword 0
.section .bss
.balign 64
.skip 4096
stack_top:
