
.section .text.init,"ax",@progbits
.balign 4
.globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, atlas_block_trap
  csrw mtvec,t0
/* ATLAS_SKIP_BSS_CLEAR drops the BSS clear. Valid only when the loader zero-fills memory (the Chipyard SimDRAM LOADMEM
   path); never define it for QEMU or any loader that leaves BSS uninitialised. */
#ifndef ATLAS_SKIP_BSS_CLEAR
  la t0, __bss_start
  la t1, __bss_end
1: bgeu t0,t1,2f
  sd zero,0(t0)
  addi t0,t0,8
  j 1b
2:
#endif
  li t0,0x6000
  csrs mstatus,t0
  csrwi fcsr,0
  call atlas_workload_main
3: wfi
  j 3b
.balign 4
atlas_block_trap:
  csrr a0,mcause
  csrr a1,mepc
  csrr a2,mtval
  call atlas_block_trap_report
  j 3b
