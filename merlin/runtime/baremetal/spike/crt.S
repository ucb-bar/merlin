# Merlin bare-metal startup for spike (machine mode, SMP).
#
# - installs a fail-fast trap handler (exits with 0x1000|mcause<<1|1 so bugs
#   surface as spike "*** FAILED ***" instead of silent hangs),
# - enables the vector unit (mstatus.VS = Initial),
# - gives every hart its own stack,
# - hart 0 clears .bss and releases the other harts via boot_ready,
# - all harts then call main(mhartid).
  .section .text.init
  .globl _start
_start:
  la    t0, _merlin_trap
  csrw  mtvec, t0
  # enable FP (mstatus.FS = Initial, bits 13:14) and the vector unit
  # (mstatus.VS = Initial, bits 9:10). The model uses f32/f64; RVV uses VS.
  li    t0, (1 << 13) | (1 << 9)
  csrs  mstatus, t0
  csrr  t0, mhartid
  la    sp, _stack_top
  li    t1, 16384             # MERLIN_STACK_PER_HART
  mul   t2, t0, t1
  sub   sp, sp, t2
  bnez  t0, 2f
  la    t3, _bss_start        # hart 0 clears .bss
1:
  la    t4, _bss_end
  bgeu  t3, t4, 3f
  sd    zero, 0(t3)
  addi  t3, t3, 8
  j     1b
3:
  fence rw, rw
  la    t3, boot_ready
  li    t4, 1
  sw    t4, 0(t3)
2:
  la    t3, boot_ready        # every hart waits until .bss is cleared
4:
  lw    t4, 0(t3)
  beqz  t4, 4b
  fence rw, rw
  csrr  a0, mhartid
  call  main
5:
  j     5b

_merlin_trap:                 # fail fast: exit(0x1000 | (mcause << 1) | 1)
  csrr  t0, mcause
  slli  t0, t0, 1
  ori   t0, t0, 1
  li    t1, 0x1000
  or    t0, t0, t1
  la    t1, tohost
  sd    t0, 0(t1)
6:
  j     6b

  .data
  .globl boot_ready
  .align 3
boot_ready:
  .word 0
