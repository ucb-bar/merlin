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

# Fail fast on any trap, THROUGH the console ABI rather than by writing tohost directly.
# htif_exit(0x800 | mcause) reproduces the previous HTIF word bit for bit -- that backend encodes a
# code as (code << 1) | 1, so 0x800|mcause becomes 0x1000 | (mcause << 1) | 1 as before -- while on a
# board with no host the UART backend PRINTS the cause instead. That is the difference between a
# reported fault and an unexplained hang on hardware we cannot attach a debugger to.
# Assumes sp is valid, which it is for any trap after startup (sp is set a few instructions in).
#
# Before exiting it reports WHERE. `mcause` alone says a store faulted; it does not say which store,
# and on a wild-store bug that is the entire question. `mepc` (the faulting instruction) maps straight
# to a symbol through `objdump`, and `mtval` is the address that was actually accessed -- so a fault
# that used to need a whole-program memref trace to localise becomes two numbers. Printed before the
# exit word, and each field is emitted separately so a console that dies partway still carries the
# fields that made it out.
_merlin_trap:
  csrr  s0, mcause            # saved in callee-saved regs: the calls below clobber t0-t2/a0
  csrr  s1, mepc
  csrr  s2, mtval
  la    a0, _trap_msg
  call  htif_puts
  mv    a0, s0
  call  htif_putd
  la    a0, _trap_at
  call  htif_puts
  mv    a0, s1
  call  _merlin_put_hex
  la    a0, _trap_val
  call  htif_puts
  mv    a0, s2
  call  _merlin_put_hex
  la    a0, _trap_end
  call  htif_puts
  mv    a0, s0
  li    t0, 0x800             # not an `ori` immediate (I-type is -2048..2047)
  or    a0, a0, t0
  call  htif_exit
6:
  j     6b

# 64-bit hex through the console ABI, which has no unsigned printer (htif_putd is signed decimal, and
# an address with the top bit set comes back negative through it).
_merlin_put_hex:
  addi  sp, sp, -32
  sd    ra, 24(sp)
  sd    s3, 16(sp)
  sd    s4, 8(sp)
  mv    s3, a0
  li    s4, 60
7:
  srl   t0, s3, s4
  andi  t0, t0, 0xf
  la    t1, _trap_digits
  add   t1, t1, t0
  lbu   a0, 0(t1)
  call  htif_putc
  addi  s4, s4, -4
  bgez  s4, 7b
  ld    ra, 24(sp)
  ld    s3, 16(sp)
  ld    s4, 8(sp)
  addi  sp, sp, 32
  ret

  .section .rodata
_trap_msg:
  .asciz "\nTRAP mcause="
_trap_at:
  .asciz " mepc=0x"
_trap_val:
  .asciz " mtval=0x"
_trap_end:
  .asciz "\n"
_trap_digits:
  .asciz "0123456789abcdef"
  .section .text.init

  .data
  .globl boot_ready
  .align 3
boot_ready:
  .word 0
