.section .text.init,"ax",@progbits
.global _start
_start:
  .option push
  .option norelax
  la gp,__global_pointer$
  .option pop
  la sp,__stack_top
  la t0,trap_entry
  csrw mtvec,t0
  csrw mie,zero
  csrr t0,mstatus
  li t1,0x6000
  or t0,t0,t1
  andi t0,t0,-9
  csrw mstatus,t0
  csrw fcsr,zero
  la t0,__bss_start
  la t1,__bss_end
1: bgeu t0,t1,2f
  sd zero,0(t0)
  addi t0,t0,8
  j 1b
2: call main
3: j 3b
.balign 4
trap_entry:
  la t0,atlas_test_mailbox
  csrr t1,mcause
  sd t1,56(t0)
  csrr t1,mepc
  sd t1,64(t0)
  csrr t1,mtval
  sd t1,72(t0)
  li t1,4
  sd t1,8(t0)
  fence rw,rw
  la t0,tohost
  li t1,7
  sd t1,0(t0)
4: j 4b

/* The Tiny C harness supplies tohost; HTIF loaders also require fromhost. */
.section .tohost,"aw",@progbits
.balign 64
.global fromhost
.type fromhost,@object
.size fromhost,8
fromhost:
  .dword 0
