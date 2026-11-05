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
  la t0,tiny_full_mailbox
  csrr t1,mcause
  sd t1,56(t0)
  csrr t1,mepc
  sd t1,64(t0)
  csrr t1,mtval
  sd t1,72(t0)
  li t1,4
  sd t1,8(t0)
  fence rw,rw
  csrr a0,mcause
  csrr a1,mepc
  csrr a2,mtval
  call tiny_full_trap_report
  la t0,tohost
  li t1,7
  sd t1,0(t0)
.global tiny_full_trap_stop
.type tiny_full_trap_stop,@function
tiny_full_trap_stop:
4: j 4b
.size tiny_full_trap_stop,.-tiny_full_trap_stop
.size trap_entry,.-trap_entry

.section .tohost,"aw",@progbits
.balign 64
.global tohost
.type tohost,@object
tohost:
  .quad 0
.size tohost,8
.balign 64
.global fromhost
.type fromhost,@object
fromhost:
  .quad 0
.size fromhost,8
