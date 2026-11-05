.section .text.init
.globl _start
_start:
  csrr a0, mhartid
  bnez a0, park
  la t0, scratch
  li t1, 0x1122334455667788
  sd t1, 0(t0)
  li t2, 0xaa
  sb t2, 3(t0)
  li t2, 0xbbcc
  sh t2, 6(t0)
  fence rw,rw
  ld t3, 0(t0)
  li t4, 0xbbcc3344aa667788
  bne t3, t4, fail
  li t5, 64
loop:
  sd t1, 0(t0)
  addi t0, t0, 64
  addi t1, t1, 1
  addi t5, t5, -1
  bnez t5, loop
  fence rw,rw
  la t0, scratch
  li t1, 0x1122334455667788
  li t5, 64
check:
  ld t2, 0(t0)
  bne t1, t2, fail
  addi t0, t0, 64
  addi t1, t1, 1
  addi t5, t5, -1
  bnez t5, check
  li a0, 1
  j exit
fail:
  li a0, 3
exit:
  la t0, tohost
  sd a0, 0(t0)
park:
  j park
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost: .dword 0
.balign 64
.globl fromhost
fromhost: .dword 0
.section .bss
.balign 64
scratch: .skip 4096
