.section .text.frontend_probe,"ax",@progbits
.option push
.option norvc
.option norelax
.balign 64
.global layout0
.type layout0,@function
layout0:
.global frontend_0_begin
frontend_0_begin:
csrr t0,mcycle
.insn r 0x7b,3,6,x0,a0,a1
.insn r 0x7b,3,4,x0,a2,a3
nop
fence
.rept 45
nop
.endr
.global frontend_0_end
frontend_0_end:
csrr t1,mcycle
.rept 11
nop
.endr
.insn r 0x7b,3,3,x0,a4,a5
fence
sub a0,t1,t0
ret
.size layout0,.-layout0
.balign 64
.rept 11
nop
.endr
.global layout44
.type layout44,@function
layout44:
.global frontend_44_begin
frontend_44_begin:
csrr t0,mcycle
.insn r 0x7b,3,6,x0,a0,a1
.insn r 0x7b,3,4,x0,a2,a3
nop
fence
.rept 45
nop
.endr
.global frontend_44_end
frontend_44_end:
csrr t1,mcycle
.rept 11
nop
.endr
.insn r 0x7b,3,3,x0,a4,a5
fence
sub a0,t1,t0
ret
.size layout44,.-layout44
.option pop
