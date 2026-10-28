/* Single-hart RV64GC diagnostic startup, machine mode, Spike HTIF. */
.section .text.init
.globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, _stack_top
  la t0, trap
  csrw mtvec, t0
  li t0, (1 << 13)
  csrs mstatus, t0
  csrw fcsr, zero
  la t0, _bss_start
  la t1, _bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  la t0, _stack_bottom
  la t1, _stack_top
  li t2, 0x55aa55aa55aa55aa
4:
  bgeu t0, t1, 5f
  sd t2, 0(t0)
  addi t0, t0, 8
  j 4b
5:
  call main
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sd a0, 0(t0)
3:
  j 3b
  .balign 4
trap:
  csrr t1, mcause
  slli t1, t1, 1
  ori t1, t1, 1
  li t2, 0x1000
  or t1, t1, t2
  la t0, tohost
  sd t1, 0(t0)
  j 3b
