	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_buddyext1p0"
	.file	"matmul_dispatch_0"
	.section	.text.matmul_dispatch_0_matmul_4x4x4_f32,"ax",@progbits
	.p2align	1
	.type	matmul_dispatch_0_matmul_4x4x4_f32,@function
matmul_dispatch_0_matmul_4x4x4_f32:
.Lfunc_begin0:
	.file	1 "/Users/sparshsingh/work/merlin/compiler/plugins/target/Gemmini/test/Output/gemmini_matmul_lowering.mlir.tmp.dir" "configured_module_matmul_dispatch_0.mlir"
	.loc	1 1 0
	.cfi_startproc
	addi	sp, sp, -80
	.cfi_def_cfa_offset 80
	sd	ra, 72(sp)
	sd	s0, 64(sp)
	sd	s1, 56(sp)
	sd	s2, 48(sp)
	sd	s3, 40(sp)
	sd	s4, 32(sp)
	sd	s5, 24(sp)
	sd	s6, 16(sp)
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	addi	s0, sp, 80
	.cfi_def_cfa s0, 0
.Ltmp0:
	.loc	1 10 8 prologue_end
	ld	a1, 32(a1)
	li	a2, 64
	.loc	1 16 8
	ld	s2, 8(a0)
	ld	s1, 16(a0)
	ld	s3, 24(a0)
	.loc	1 10 8
	ld	s4, 0(a1)
	.loc	1 11 8
	ld	s5, 8(a1)
	.loc	1 12 8
	ld	s6, 16(a1)
	.loc	1 16 8
	sd	a2, -72(s0)
	ld	a0, 8(s1)
	ld	a2, 8(s3)
	addi	a1, s0, -80
	li	a3, 0
	jalr	s2
	sext.w	a1, a0
	bnez	a1, .LBB0_6
	ld	a0, -80(s0)
	li	a2, 4
	mv	a3, a0
.LBB0_2:
	.loc	1 0 8 is_stmt 0
	li	a4, 16
	mv	a5, a3
.LBB0_3:
	.loc	1 16 8 is_stmt 1
	sw	zero, 0(a5)
	addi	a4, a4, -4
	addi	a5, a5, 4
	bnez	a4, .LBB0_3
	addi	a1, a1, 1
	addi	a3, a3, 16
	bne	a1, a2, .LBB0_2
	.loc	1 0 8 is_stmt 0
	li	a1, 1
	li	a2, 127
	slli	a3, a1, 48
	slli	a4, a2, 39
	addi	a4, a4, 1
	slli	a4, a4, 16
	addi	a4, a4, 4
	.loc	1 16 8
	config_ex	a4, a3
	li	a3, 2
	slli	a4, a2, 55
	addi	a4, a4, 4
	config_ex	a3, a4
	li	a3, 4
	slli	a2, a2, 35
	addi	a2, a2, 1
	slli	a2, a2, 20
	addi	a4, a2, 257
	config_ex	a4, a3
	addi	a4, a2, 265
	config_ex	a4, a3
	li	a4, 16
	addi	a2, a2, 273
	config_ex	a2, a4
	lui	a2, 65537
	lui	a4, 196611
	slli	a2, a2, 4
	slli	a4, a4, 6
	addi	a2, a2, 1
	addi	a4, a4, 12
	loop_ws_config_bounds	a4, a2
	loop_ws_config_addrs_ab	s4, s5
	loop_ws_config_addrs_dc	a0, s6
	loop_ws_config_strides_ab	a3, a3
	loop_ws_config_strides_dc	a3, a3
	loop_ws	a1, zero
	flush	zero
	mv	a2, sp
	addi	a1, a2, -16
	mv	sp, a1
	sd	a0, -16(a2)
	ld	a0, 0(s1)
	ld	a2, 0(s3)
	li	a3, 0
	jalr	s2
.LBB0_6:
	.loc	1 0 0 epilogue_begin
	addi	sp, s0, -80
	.cfi_def_cfa sp, 80
	ld	ra, 72(sp)
	ld	s0, 64(sp)
	ld	s1, 56(sp)
	ld	s2, 48(sp)
	ld	s3, 40(sp)
	ld	s4, 32(sp)
	ld	s5, 24(sp)
	ld	s6, 16(sp)
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	addi	sp, sp, 80
	.cfi_def_cfa_offset 0
	ret
.Ltmp1:
.Lfunc_end0:
	.size	matmul_dispatch_0_matmul_4x4x4_f32, .Lfunc_end0-matmul_dispatch_0_matmul_4x4x4_f32
	.cfi_endproc

	.section	.text.iree_hal_executable_library_query,"ax",@progbits
	.globl	iree_hal_executable_library_query
	.p2align	1
	.type	iree_hal_executable_library_query,@function
iree_hal_executable_library_query:
.Liree_hal_executable_library_query$local:
	.type	.Liree_hal_executable_library_query$local,@function
.Lfunc_begin1:
	.cfi_startproc
	addiw	a0, a0, -6
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(iree_hal_executable_library_query_v0)
	snez	a0, a0
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
	addi	a0, a0, -1
	and	a0, a0, a1
	ret
.Lfunc_end1:
	.size	iree_hal_executable_library_query, .Lfunc_end1-iree_hal_executable_library_query
	.size	.Liree_hal_executable_library_query$local, .Lfunc_end1-iree_hal_executable_library_query
	.cfi_endproc

	.section	.text.iree_h2f_ieee,"ax",@progbits
	.p2align	1
	.type	iree_h2f_ieee,@function
iree_h2f_ieee:
.Lfunc_begin2:
	.cfi_startproc
	li	a3, 31
	andi	a1, a0, 1023
	lui	a4, 8
	slli	a3, a3, 10
	and	a2, a0, a3
	and	a0, a0, a4
	slli	a0, a0, 16
	beqz	a2, .LBB2_4
	bne	a2, a3, .LBB2_5
	beqz	a1, .LBB2_6
	lui	a1, 523264
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB2_4:
	lui	a2, 210944
	or	a0, a0, a2
	fcvt.s.wu	fa5, a1
	fmv.w.x	fa4, a0
	fmul.s	fa0, fa5, fa4
	ret
.LBB2_5:
	add	a1, a1, a2
	lui	a2, 28
	add	a1, a1, a2
	slli	a1, a1, 13
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB2_6:
	lui	a1, 522240
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.Lfunc_end2:
	.size	iree_h2f_ieee, .Lfunc_end2-iree_h2f_ieee
	.cfi_endproc

	.section	.text.iree_f2h_ieee,"ax",@progbits
	.p2align	1
	.type	iree_f2h_ieee,@function
iree_f2h_ieee:
.Lfunc_begin3:
	.cfi_startproc
	fmv.x.w	a2, fa0
	lui	a4, 522240
	and	a1, a2, a4
	srliw	a0, a2, 16
	beqz	a1, .LBB3_6
	slli	a3, a2, 41
	srli	a3, a3, 41
	bne	a1, a4, .LBB3_4
	beqz	a3, .LBB3_5
	lui	a1, 8
	addi	a1, a1, -1
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB3_4:
	lui	a4, 290816
	bgeu	a4, a1, .LBB3_7
.LBB3_5:
	li	a1, 31
	slli	a1, a1, 10
.LBB3_6:
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB3_7:
	srli	a1, a1, 23
	li	a4, 113
	bgeu	a1, a4, .LBB3_9
	lui	a2, 8
	and	a0, a0, a2
	mv	a0, a0
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB3_9:
	lui	a4, 2
	and	a2, a2, a4
	lui	a4, 1
	seqz	a2, a2
	sub	a3, a3, a2
	li	a2, 15
	add	a3, a3, a4
	srliw	a4, a3, 23
	srliw	a3, a3, 13
	add	a1, a1, a4
	snez	a4, a4
	addi	a1, a1, -127
	addi	a4, a4, -1
	slli	a1, a1, 10
	and	a3, a3, a4
	add	a1, a1, a3
	slli	a2, a2, 10
	add	a1, a1, a2
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.Lfunc_end3:
	.size	iree_f2h_ieee, .Lfunc_end3-iree_f2h_ieee
	.cfi_endproc

	.section	.text.__gnu_h2f_ieee,"ax",@progbits
	.p2align	1
	.type	__gnu_h2f_ieee,@function
__gnu_h2f_ieee:
.Lfunc_begin4:
	.cfi_startproc
	li	a3, 31
	andi	a1, a0, 1023
	lui	a4, 8
	slli	a3, a3, 10
	and	a2, a0, a3
	and	a0, a0, a4
	slli	a0, a0, 16
	beqz	a2, .LBB4_4
	bne	a2, a3, .LBB4_5
	beqz	a1, .LBB4_6
	lui	a1, 523264
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB4_4:
	lui	a2, 210944
	or	a0, a0, a2
	fcvt.s.wu	fa5, a1
	fmv.w.x	fa4, a0
	fmul.s	fa0, fa5, fa4
	ret
.LBB4_5:
	add	a1, a1, a2
	lui	a2, 28
	add	a1, a1, a2
	slli	a1, a1, 13
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB4_6:
	lui	a1, 522240
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.Lfunc_end4:
	.size	__gnu_h2f_ieee, .Lfunc_end4-__gnu_h2f_ieee
	.cfi_endproc

	.section	.text.__extendhfsf2,"ax",@progbits
	.p2align	1
	.type	__extendhfsf2,@function
__extendhfsf2:
.Lfunc_begin5:
	.cfi_startproc
	fmv.x.w	a1, fa0
	li	a3, 31
	slli	a3, a3, 10
	andi	a2, a1, 1023
	slli	a0, a1, 16
	and	a4, a1, a3
	lui	a5, 524288
	and	a0, a0, a5
	beqz	a4, .LBB5_4
	bne	a4, a3, .LBB5_5
	beqz	a2, .LBB5_6
	lui	a1, 523264
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB5_4:
	lui	a1, 210944
	or	a0, a0, a1
	fcvt.s.wu	fa5, a2
	fmv.w.x	fa4, a0
	fmul.s	fa0, fa5, fa4
	ret
.LBB5_5:
	slli	a1, a1, 49
	srli	a1, a1, 49
	lui	a2, 28
	add	a1, a1, a2
	slli	a1, a1, 13
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB5_6:
	lui	a1, 522240
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.Lfunc_end5:
	.size	__extendhfsf2, .Lfunc_end5-__extendhfsf2
	.cfi_endproc

	.section	.text.__gnu_f2h_ieee,"ax",@progbits
	.p2align	1
	.type	__gnu_f2h_ieee,@function
__gnu_f2h_ieee:
.Lfunc_begin6:
	.cfi_startproc
	fmv.x.w	a2, fa0
	lui	a4, 522240
	and	a1, a2, a4
	srliw	a0, a2, 16
	beqz	a1, .LBB6_6
	slli	a3, a2, 41
	srli	a3, a3, 41
	bne	a1, a4, .LBB6_4
	beqz	a3, .LBB6_5
	lui	a1, 8
	addi	a1, a1, -1
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB6_4:
	lui	a4, 290816
	bgeu	a4, a1, .LBB6_7
.LBB6_5:
	li	a1, 31
	slli	a1, a1, 10
.LBB6_6:
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB6_7:
	srli	a1, a1, 23
	li	a4, 113
	bgeu	a1, a4, .LBB6_9
	lui	a2, 8
	and	a0, a0, a2
	mv	a0, a0
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.LBB6_9:
	lui	a4, 2
	and	a2, a2, a4
	lui	a4, 1
	seqz	a2, a2
	sub	a3, a3, a2
	li	a2, 15
	add	a3, a3, a4
	srliw	a4, a3, 23
	srliw	a3, a3, 13
	add	a1, a1, a4
	snez	a4, a4
	addi	a1, a1, -127
	addi	a4, a4, -1
	slli	a1, a1, 10
	and	a3, a3, a4
	add	a1, a1, a3
	slli	a2, a2, 10
	add	a1, a1, a2
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	slli	a0, a0, 48
	srai	a0, a0, 48
	ret
.Lfunc_end6:
	.size	__gnu_f2h_ieee, .Lfunc_end6-__gnu_f2h_ieee
	.cfi_endproc

	.section	.text.__truncsfhf2,"ax",@progbits
	.p2align	1
	.type	__truncsfhf2,@function
__truncsfhf2:
.Lfunc_begin7:
	.cfi_startproc
	fmv.x.w	a2, fa0
	lui	a4, 522240
	and	a1, a2, a4
	srliw	a0, a2, 16
	beqz	a1, .LBB7_9
	slli	a3, a2, 41
	srli	a3, a3, 41
	bne	a1, a4, .LBB7_4
	beqz	a3, .LBB7_5
	lui	a1, 8
	addi	a1, a1, -1
	or	a0, a0, a1
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sh	a0, 12(sp)
	flw	fa0, 12(sp)
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB7_4:
	lui	a4, 290816
	bgeu	a4, a1, .LBB7_6
.LBB7_5:
	li	a1, 31
	slli	a1, a1, 10
	j	.LBB7_9
.LBB7_6:
	srli	a1, a1, 23
	li	a4, 113
	bgeu	a1, a4, .LBB7_8
	li	a1, 0
	j	.LBB7_9
.LBB7_8:
	lui	a4, 2
	and	a2, a2, a4
	lui	a4, 1
	seqz	a2, a2
	sub	a3, a3, a2
	li	a2, 15
	add	a3, a3, a4
	srliw	a4, a3, 23
	srliw	a3, a3, 13
	add	a1, a1, a4
	snez	a4, a4
	addi	a1, a1, -127
	addi	a4, a4, -1
	slli	a1, a1, 10
	and	a3, a3, a4
	add	a1, a1, a3
	slli	a2, a2, 10
	add	a1, a1, a2
.LBB7_9:
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sh	a0, 12(sp)
	flw	fa0, 12(sp)
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end7:
	.size	__truncsfhf2, .Lfunc_end7-__truncsfhf2
	.cfi_endproc

	.section	.text.__extendhfdf2,"ax",@progbits
	.p2align	1
	.type	__extendhfdf2,@function
__extendhfdf2:
.Lfunc_begin8:
	.cfi_startproc
	fmv.x.w	a1, fa0
	li	a3, 31
	slli	a3, a3, 10
	andi	a2, a1, 1023
	slli	a0, a1, 16
	and	a4, a1, a3
	lui	a5, 524288
	and	a0, a0, a5
	beqz	a4, .LBB8_4
	bne	a4, a3, .LBB8_5
	beqz	a2, .LBB8_6
	lui	a1, 523264
	or	a0, a0, a1
	fmv.w.x	fa5, a0
	fcvt.d.s	fa0, fa5
	ret
.LBB8_4:
	lui	a1, 210944
	or	a0, a0, a1
	fcvt.s.wu	fa5, a2
	fmv.w.x	fa4, a0
	fmul.s	fa5, fa5, fa4
	fcvt.d.s	fa0, fa5
	ret
.LBB8_5:
	slli	a1, a1, 49
	srli	a1, a1, 49
	lui	a2, 28
	add	a1, a1, a2
	slli	a1, a1, 13
	or	a0, a0, a1
	fmv.w.x	fa5, a0
	fcvt.d.s	fa0, fa5
	ret
.LBB8_6:
	lui	a1, 522240
	or	a0, a0, a1
	fmv.w.x	fa5, a0
	fcvt.d.s	fa0, fa5
	ret
.Lfunc_end8:
	.size	__extendhfdf2, .Lfunc_end8-__extendhfdf2
	.cfi_endproc

	.section	.text.__truncdfhf2,"ax",@progbits
	.p2align	1
	.type	__truncdfhf2,@function
__truncdfhf2:
.Lfunc_begin9:
	.cfi_startproc
	fcvt.s.d	fa5, fa0
	fmv.x.w	a2, fa5
	lui	a4, 522240
	and	a1, a2, a4
	srliw	a0, a2, 16
	beqz	a1, .LBB9_9
	slli	a3, a2, 41
	srli	a3, a3, 41
	bne	a1, a4, .LBB9_4
	beqz	a3, .LBB9_5
	lui	a1, 8
	addi	a1, a1, -1
	or	a0, a0, a1
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sh	a0, 12(sp)
	flw	fa0, 12(sp)
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB9_4:
	lui	a4, 290816
	bgeu	a4, a1, .LBB9_6
.LBB9_5:
	li	a1, 31
	slli	a1, a1, 10
	j	.LBB9_9
.LBB9_6:
	srli	a1, a1, 23
	li	a4, 113
	bgeu	a1, a4, .LBB9_8
	li	a1, 0
	j	.LBB9_9
.LBB9_8:
	lui	a4, 2
	and	a2, a2, a4
	lui	a4, 1
	seqz	a2, a2
	sub	a3, a3, a2
	li	a2, 15
	add	a3, a3, a4
	srliw	a4, a3, 23
	srliw	a3, a3, 13
	add	a1, a1, a4
	snez	a4, a4
	addi	a1, a1, -127
	addi	a4, a4, -1
	slli	a1, a1, 10
	and	a3, a3, a4
	add	a1, a1, a3
	slli	a2, a2, 10
	add	a1, a1, a2
.LBB9_9:
	lui	a2, 8
	and	a0, a0, a2
	or	a0, a0, a1
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sh	a0, 12(sp)
	flw	fa0, 12(sp)
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end9:
	.size	__truncdfhf2, .Lfunc_end9-__truncdfhf2
	.cfi_endproc

	.section	.text.fma,"ax",@progbits
	.p2align	1
	.type	fma,@function
fma:
.Lfunc_begin10:
	.cfi_startproc
	fmadd.d	fa0, fa0, fa1, fa2
	ret
.Lfunc_end10:
	.size	fma, .Lfunc_end10-fma
	.cfi_endproc

	.section	.text.__math_invalidf,"ax",@progbits
	.p2align	1
	.type	__math_invalidf,@function
__math_invalidf:
.Lfunc_begin11:
	.cfi_startproc
	fsub.s	fa5, fa0, fa0
	fdiv.s	fa0, fa5, fa5
	ret
.Lfunc_end11:
	.size	__math_invalidf, .Lfunc_end11-__math_invalidf
	.cfi_endproc

	.section	.text.__math_oflowf,"ax",@progbits
	.p2align	1
	.type	__math_oflowf,@function
__math_oflowf:
.Lfunc_begin12:
	.cfi_startproc
	sext.w	a0, a0
	lui	a1, 458752
	fmv.w.x	fa5, a1
	fmv.s	fa4, fa5
	beqz	a0, .LBB12_2
	lui	a0, 983040
	fmv.w.x	fa4, a0
.LBB12_2:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	fsw	fa4, 12(sp)
	flw	fa4, 12(sp)
	fmul.s	fa0, fa4, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end12:
	.size	__math_oflowf, .Lfunc_end12-__math_oflowf
	.cfi_endproc

	.section	.text.__math_xflowf,"ax",@progbits
	.p2align	1
	.type	__math_xflowf,@function
__math_xflowf:
.Lfunc_begin13:
	.cfi_startproc
	sext.w	a0, a0
	fmv.s	fa5, fa0
	beqz	a0, .LBB13_2
	fneg.s	fa5, fa0
.LBB13_2:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	fsw	fa5, 12(sp)
	flw	fa5, 12(sp)
	fmul.s	fa0, fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end13:
	.size	__math_xflowf, .Lfunc_end13-__math_xflowf
	.cfi_endproc

	.section	.text.__math_uflowf,"ax",@progbits
	.p2align	1
	.type	__math_uflowf,@function
__math_uflowf:
.Lfunc_begin14:
	.cfi_startproc
	sext.w	a0, a0
	lui	a1, 65536
	fmv.w.x	fa5, a1
	fmv.s	fa4, fa5
	beqz	a0, .LBB14_2
	lui	a0, 589824
	fmv.w.x	fa4, a0
.LBB14_2:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	fsw	fa4, 12(sp)
	flw	fa4, 12(sp)
	fmul.s	fa0, fa4, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end14:
	.size	__math_uflowf, .Lfunc_end14-__math_uflowf
	.cfi_endproc

	.section	.text.ceilf,"ax",@progbits
	.p2align	1
	.type	ceilf,@function
ceilf:
.Lfunc_begin15:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	fmv.x.w	a0, fa0
	slli	a1, a0, 33
	srli	a1, a1, 56
	li	a2, 149
	bltu	a2, a1, .LBB15_9
	li	a2, 127
	bltu	a1, a2, .LBB15_4
	addi	a1, a1, -127
	lui	a2, 2048
	addi	a2, a2, -1
	srlw	a2, a2, a1
	and	a3, a2, a0
	beqz	a3, .LBB15_9
	lui	a3, 505856
	srli	a4, a0, 63
	fmv.w.x	fa5, a3
	lui	a3, 1046528
	sraw	a1, a3, a1
	addi	a4, a4, -1
	fadd.s	fa5, fa0, fa5
	and	a2, a2, a4
	fsw	fa5, 8(sp)
	add	a0, a0, a2
	and	a0, a0, a1
	j	.LBB15_8
.LBB15_4:
	lui	a1, 505856
	fmv.w.x	fa5, a1
	fadd.s	fa5, fa0, fa5
	fsw	fa5, 12(sp)
	bltz	a0, .LBB15_7
	beqz	a0, .LBB15_9
	lui	a0, 260096
	j	.LBB15_8
.LBB15_7:
	lui	a0, 524288
.LBB15_8:
	fmv.w.x	fa0, a0
.LBB15_9:
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end15:
	.size	ceilf, .Lfunc_end15-ceilf
	.cfi_endproc

	.section	.text.expf,"ax",@progbits
	.p2align	1
	.type	expf,@function
expf:
.Lfunc_begin16:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	.cfi_remember_state
	fmv.x.w	a0, fa0
	slli	a0, a0, 33
	srli	a0, a0, 53
	li	a1, 1067
	bgeu	a0, a1, .LBB16_2
.LBB16_1:
	fcvt.d.s	fa5, fa0
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(.promoted_doubles.expf)
	lui	a1, 2151
	li	a2, -1945
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi1)
	fld	fa4, 0(a0)
	fld	fa3, 8(a0)
	fld	fa2, 16(a0)
	fld	fa1, 24(a0)
.Lpcrel_hi2:
	auipc	a0, %pcrel_hi(__exp2f_data)
	slli	a1, a1, 39
	fmv.d.x	fa0, a1
	li	a1, 1023
	slli	a2, a2, 51
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi2)
	slli	a1, a1, 52
	fmul.d	fa5, fa5, fa4
	fmv.d.x	fa4, a2
	fadd.d	fa0, fa5, fa0
	fmv.x.d	a2, fa0
	fadd.d	fa4, fa0, fa4
	fmv.d.x	fa0, a1
	fsub.d	fa5, fa5, fa4
	andi	a1, a2, 31
	slli	a2, a2, 47
	slli	a1, a1, 3
	add	a0, a0, a1
	ld	a0, 0(a0)
	fmadd.d	fa4, fa5, fa3, fa2
	fmul.d	fa3, fa5, fa5
	fmadd.d	fa5, fa5, fa1, fa0
	add	a0, a0, a2
	fmv.d.x	fa2, a0
	fmadd.d	fa5, fa4, fa3, fa5
	fmul.d	fa5, fa5, fa2
	fcvt.s.d	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB16_2:
	.cfi_restore_state
	.cfi_remember_state
	lui	a1, 1046528
	fmv.w.x	fa5, a1
	feq.s	a1, fa0, fa5
	beqz	a1, .LBB16_4
	fmv.w.x	fa0, zero
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB16_4:
	.cfi_restore_state
	.cfi_remember_state
	li	a1, 2040
	bgeu	a0, a1, .LBB16_7
	lui	a0, 273175
	addi	a0, a0, 535
	fmv.w.x	fa5, a0
	flt.s	a0, fa5, fa0
	beqz	a0, .LBB16_8
	lui	a0, 458752
	fmv.w.x	fa5, a0
	fsw	fa5, 8(sp)
	flw	fa4, 8(sp)
	fmul.s	fa0, fa4, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB16_7:
	.cfi_restore_state
	.cfi_remember_state
	fadd.s	fa0, fa0, fa0
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB16_8:
	.cfi_restore_state
	lui	a0, 797951
	addi	a0, a0, 436
	fmv.w.x	fa5, a0
	flt.s	a0, fa0, fa5
	beqz	a0, .LBB16_1
	lui	a0, 65536
	fmv.w.x	fa5, a0
	fsw	fa5, 12(sp)
	flw	fa4, 12(sp)
	fmul.s	fa0, fa4, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end16:
	.size	expf, .Lfunc_end16-expf
	.cfi_endproc

	.section	.text.feclearexcept,"ax",@progbits
	.p2align	1
	.type	feclearexcept,@function
feclearexcept:
.Lfunc_begin17:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end17:
	.size	feclearexcept, .Lfunc_end17-feclearexcept
	.cfi_endproc

	.section	.text.feraiseexcept,"ax",@progbits
	.p2align	1
	.type	feraiseexcept,@function
feraiseexcept:
.Lfunc_begin18:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end18:
	.size	feraiseexcept, .Lfunc_end18-feraiseexcept
	.cfi_endproc

	.section	.text.fetestexcept,"ax",@progbits
	.p2align	1
	.type	fetestexcept,@function
fetestexcept:
.Lfunc_begin19:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end19:
	.size	fetestexcept, .Lfunc_end19-fetestexcept
	.cfi_endproc

	.section	.text.fegetround,"ax",@progbits
	.p2align	1
	.type	fegetround,@function
fegetround:
.Lfunc_begin20:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end20:
	.size	fegetround, .Lfunc_end20-fegetround
	.cfi_endproc

	.section	.text.__fesetround,"ax",@progbits
	.p2align	1
	.type	__fesetround,@function
__fesetround:
.Lfunc_begin21:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end21:
	.size	__fesetround, .Lfunc_end21-__fesetround
	.cfi_endproc

	.section	.text.fegetenv,"ax",@progbits
	.p2align	1
	.type	fegetenv,@function
fegetenv:
.Lfunc_begin22:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end22:
	.size	fegetenv, .Lfunc_end22-fegetenv
	.cfi_endproc

	.section	.text.fesetenv,"ax",@progbits
	.p2align	1
	.type	fesetenv,@function
fesetenv:
.Lfunc_begin23:
	.cfi_startproc
	li	a0, 0
	ret
.Lfunc_end23:
	.size	fesetenv, .Lfunc_end23-fesetenv
	.cfi_endproc

	.section	.text.floorf,"ax",@progbits
	.p2align	1
	.type	floorf,@function
floorf:
.Lfunc_begin24:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	.cfi_remember_state
	fmv.x.w	a0, fa0
	slli	a1, a0, 33
	srli	a1, a1, 56
	li	a2, 149
	bltu	a2, a1, .LBB24_8
	li	a2, 127
	bltu	a1, a2, .LBB24_5
	addi	a1, a1, -127
	lui	a2, 2048
	addi	a2, a2, -1
	srlw	a2, a2, a1
	and	a3, a2, a0
	beqz	a3, .LBB24_8
	lui	a3, 505856
	fmv.w.x	fa5, a3
	lui	a3, 1046528
	sraw	a1, a3, a1
	srli	a3, a0, 31
	and	a2, a2, a3
	fadd.s	fa5, fa0, fa5
	add	a0, a0, a2
	fsw	fa5, 8(sp)
	and	a0, a0, a1
.LBB24_4:
	fmv.w.x	fa0, a0
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB24_5:
	.cfi_restore_state
	.cfi_remember_state
	lui	a1, 505856
	fmv.w.x	fa5, a1
	fadd.s	fa5, fa0, fa5
	fsw	fa5, 12(sp)
	bltz	a0, .LBB24_7
	fmv.w.x	fa0, zero
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB24_7:
	.cfi_restore_state
	.cfi_remember_state
	fmv.w.x	fa5, zero
	feq.s	a0, fa0, fa5
	beqz	a0, .LBB24_9
.LBB24_8:
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB24_9:
	.cfi_restore_state
	lui	a0, 784384
	j	.LBB24_4
.Lfunc_end24:
	.size	floorf, .Lfunc_end24-floorf
	.cfi_endproc

	.section	.text.fmaf,"ax",@progbits
	.p2align	1
	.type	fmaf,@function
fmaf:
.Lfunc_begin25:
	.cfi_startproc
	fcvt.d.s	fa5, fa0
	fcvt.d.s	fa3, fa1
	fcvt.d.s	fa4, fa2
	fmul.d	fa3, fa5, fa3
	fadd.d	fa5, fa3, fa4
	fmv.x.d	a0, fa5
	slli	a1, a0, 35
	srli	a1, a1, 35
	lui	a2, 65536
	bne	a1, a2, .LBB25_4
	li	a1, 2047
	slli	a1, a1, 52
	and	a2, a0, a1
	beq	a2, a1, .LBB25_4
	fsub.d	fa2, fa5, fa3
	feq.d	a1, fa2, fa4
	beqz	a1, .LBB25_5
	fsub.d	fa2, fa5, fa4
	feq.d	a1, fa2, fa3
	beqz	a1, .LBB25_5
.LBB25_4:
	fcvt.s.d	fa0, fa5
	ret
.LBB25_5:
	srli	a1, a0, 63
	flt.d	a2, fa3, fa4
	xori	a2, a2, 1
	bne	a2, a1, .LBB25_8
	fsub.d	fa5, fa4, fa5
	fadd.d	fa5, fa3, fa5
	fmv.d.x	fa4, zero
	flt.d	a2, fa5, fa4
	xori	a2, a2, 1
	beq	a1, a2, .LBB25_9
.LBB25_7:
	addi	a0, a0, 1
	fmv.d.x	fa5, a0
	fcvt.s.d	fa0, fa5
	ret
.LBB25_8:
	fsub.d	fa5, fa3, fa5
	fadd.d	fa5, fa5, fa4
	fmv.d.x	fa4, zero
	flt.d	a2, fa5, fa4
	xori	a2, a2, 1
	bne	a1, a2, .LBB25_7
.LBB25_9:
	addi	a0, a0, -1
	fmv.d.x	fa5, a0
	fcvt.s.d	fa0, fa5
	ret
.Lfunc_end25:
	.size	fmaf, .Lfunc_end25-fmaf
	.cfi_endproc

	.section	.text.fmodf,"ax",@progbits
	.p2align	1
	.type	fmodf,@function
fmodf:
.Lfunc_begin26:
	.cfi_startproc
	fmv.x.w	a4, fa1
	slliw	a2, a4, 1
	beqz	a2, .LBB26_8
	slli	a3, a4, 33
	srli	a0, a3, 33
	lui	a1, 522240
	bltu	a1, a0, .LBB26_8
	fmv.x.w	a6, fa0
	slli	a1, a6, 33
	srli	a1, a1, 56
	li	a5, 255
	beq	a1, a5, .LBB26_8
	slliw	a5, a6, 1
	bgeu	a2, a5, .LBB26_9
	srli	a3, a3, 56
	lui	a2, 2048
	addi	a5, a2, -1
	beqz	a1, .LBB26_11
	and	a0, a6, a5
	or	a2, a2, a0
	beqz	a3, .LBB26_14
.LBB26_6:
	and	a4, a4, a5
	lui	a0, 2048
	or	a4, a4, a0
	blt	a3, a1, .LBB26_18
.LBB26_7:
	subw	a3, a2, a4
	bgez	a3, .LBB26_21
	j	.LBB26_22
.LBB26_8:
	fmul.s	fa5, fa0, fa1
	fdiv.s	fa0, fa5, fa5
	ret
.LBB26_9:
	beq	a5, a2, .LBB26_26
	ret
.LBB26_11:
	li	a1, 0
	slliw	a2, a6, 9
	bltz	a2, .LBB26_13
.LBB26_12:
	slliw	a2, a2, 1
	addiw	a1, a1, -1
	bgez	a2, .LBB26_12
.LBB26_13:
	li	a2, 1
	sub	a2, a2, a1
	sllw	a2, a6, a2
	bnez	a3, .LBB26_6
.LBB26_14:
	li	a3, 0
	slliw	a5, a4, 9
	bltz	a5, .LBB26_16
.LBB26_15:
	slliw	a5, a5, 1
	addiw	a3, a3, -1
	bgez	a5, .LBB26_15
.LBB26_16:
	li	a0, 1
	sub	a0, a0, a3
	sllw	a4, a4, a0
	blt	a3, a1, .LBB26_18
	j	.LBB26_7
.LBB26_17:
	addiw	a1, a1, -1
	slli	a2, a2, 1
	bge	a3, a1, .LBB26_20
.LBB26_18:
	subw	a5, a2, a4
	bltz	a5, .LBB26_17
	mv	a2, a5
	bnez	a5, .LBB26_17
	j	.LBB26_26
.LBB26_20:
	mv	a1, a3
	subw	a3, a2, a4
	bltz	a3, .LBB26_22
.LBB26_21:
	mv	a2, a3
	beqz	a3, .LBB26_26
.LBB26_22:
	srliw	a0, a2, 23
	lui	a3, 524288
	bnez	a0, .LBB26_24
.LBB26_23:
	srliw	a0, a2, 22
	slli	a2, a2, 1
	addiw	a1, a1, -1
	beqz	a0, .LBB26_23
.LBB26_24:
	and	a0, a6, a3
	blez	a1, .LBB26_27
	lui	a3, 1046528
	add	a2, a2, a3
	slli	a1, a1, 23
	or	a1, a1, a2
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.LBB26_26:
	fmv.w.x	fa5, zero
	fmul.s	fa0, fa0, fa5
	ret
.LBB26_27:
	li	a3, 1
	sub	a3, a3, a1
	srlw	a1, a2, a3
	or	a0, a0, a1
	fmv.w.x	fa0, a0
	ret
.Lfunc_end26:
	.size	fmodf, .Lfunc_end26-fmodf
	.cfi_endproc

	.section	.text.frexpf,"ax",@progbits
	.p2align	1
	.type	frexpf,@function
frexpf:
.Lfunc_begin27:
	.cfi_startproc
	fmv.x.w	a1, fa0
	srliw	a2, a1, 23
	zext.b	a2, a2
	li	a3, 255
	beq	a2, a3, .LBB27_5
	bnez	a2, .LBB27_4
	fmv.w.x	fa5, zero
	feq.s	a1, fa0, fa5
	bnez	a1, .LBB27_6
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sd	ra, 8(sp)
	sd	s0, 0(sp)
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	lui	a1, 391168
	fmv.w.x	fa5, a1
	fmul.s	fa0, fa0, fa5
	mv	s0, a0
	call	frexpf
	mv	a0, s0
	lw	a1, 0(s0)
	addi	a1, a1, -64
	ld	ra, 8(sp)
	ld	s0, 0(sp)
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	sw	a1, 0(a0)
	ret
.LBB27_4:
	addi	a2, a2, -126
	sw	a2, 0(a0)
	lui	a0, 526336
	addi	a0, a0, -1
	and	a0, a0, a1
	lui	a1, 258048
	or	a0, a0, a1
	fmv.w.x	fa0, a0
.LBB27_5:
	ret
.LBB27_6:
	sw	zero, 0(a0)
	ret
.Lfunc_end27:
	.size	frexpf, .Lfunc_end27-frexpf
	.cfi_endproc

	.section	.text.ldexpf,"ax",@progbits
	.p2align	1
	.type	ldexpf,@function
ldexpf:
.Lfunc_begin28:
	.cfi_startproc
	sext.w	a1, a0
	li	a2, 128
	blt	a1, a2, .LBB28_6
	lui	a2, 520192
	sext.w	a1, a0
	fmv.w.x	fa5, a2
	li	a2, 255
	fmul.s	fa0, fa0, fa5
	bltu	a1, a2, .LBB28_11
	li	a0, 381
	bltu	a1, a0, .LBB28_4
	li	a1, 381
.LBB28_4:
	fmul.s	fa0, fa0, fa5
	addi	a0, a1, -254
.LBB28_5:
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB28_6:
	li	a2, -127
	blt	a2, a1, .LBB28_5
	lui	a2, 51200
	fmv.w.x	fa5, a2
	li	a2, -229
	fmul.s	fa0, fa0, fa5
	bltu	a2, a1, .LBB28_12
	li	a0, -330
	bltu	a0, a1, .LBB28_10
	li	a1, -330
.LBB28_10:
	fmul.s	fa0, fa0, fa5
	addi	a0, a1, 204
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB28_11:
	addi	a0, a0, -127
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB28_12:
	addi	a0, a0, 102
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.Lfunc_end28:
	.size	ldexpf, .Lfunc_end28-ldexpf
	.cfi_endproc

	.section	.text.scalbnf,"ax",@progbits
	.p2align	1
	.type	scalbnf,@function
scalbnf:
.Lfunc_begin29:
	.cfi_startproc
	sext.w	a1, a0
	li	a2, 128
	blt	a1, a2, .LBB29_6
	lui	a2, 520192
	sext.w	a1, a0
	fmv.w.x	fa5, a2
	li	a2, 255
	fmul.s	fa0, fa0, fa5
	bltu	a1, a2, .LBB29_11
	li	a0, 381
	bltu	a1, a0, .LBB29_4
	li	a1, 381
.LBB29_4:
	fmul.s	fa0, fa0, fa5
	addi	a0, a1, -254
.LBB29_5:
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB29_6:
	li	a2, -127
	blt	a2, a1, .LBB29_5
	lui	a2, 51200
	fmv.w.x	fa5, a2
	li	a2, -229
	fmul.s	fa0, fa0, fa5
	bltu	a2, a1, .LBB29_12
	li	a0, -330
	bltu	a0, a1, .LBB29_10
	li	a1, -330
.LBB29_10:
	fmul.s	fa0, fa0, fa5
	addi	a0, a1, 204
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB29_11:
	addi	a0, a0, -127
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.LBB29_12:
	addi	a0, a0, 102
	slli	a0, a0, 23
	lui	a1, 260096
	add	a0, a0, a1
	fmv.w.x	fa5, a0
	fmul.s	fa0, fa0, fa5
	ret
.Lfunc_end29:
	.size	scalbnf, .Lfunc_end29-scalbnf
	.cfi_endproc

	.section	.text.powf,"ax",@progbits
	.p2align	1
	.type	powf,@function
powf:
.Lfunc_begin30:
	.cfi_startproc
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	.cfi_remember_state
	fmv.x.w	a1, fa0
	fmv.x.w	a0, fa1
	lui	a2, 526336
	add	a2, a2, a1
	srliw	a3, a2, 24
	li	a4, 129
	slliw	a2, a0, 1
	bltu	a3, a4, .LBB30_6
	lui	a3, 4096
	addw	a4, a2, a3
	bgeu	a3, a4, .LBB30_6
	li	a6, 0
.LBB30_3:
.Lpcrel_hi3:
	auipc	a2, %pcrel_hi(.promoted_doubles.powf)
	lui	a3, 789712
	lui	a4, 1046528
	li	a5, -1025
	slli	a5, a5, 52
	fmv.d.x	fa5, a5
.Lpcrel_hi4:
	auipc	a5, %pcrel_hi(__powf_log2_data)
	fcvt.d.s	fa4, fa1
	add	a3, a3, a1
	addi	a5, a5, %pcrel_lo(.Lpcrel_hi4)
	and	a4, a4, a3
	sub	a1, a1, a4
	srli	a4, a3, 15
	andi	a4, a4, 240
	add	a4, a4, a5
	lui	a5, 65535
	fmv.w.x	fa3, a1
	lui	a0, 16479
	addi	a1, a2, %pcrel_lo(.Lpcrel_hi3)
	slli	a5, a5, 35
	slli	a0, a0, 36
	fld	fa2, 32(a1)
	fld	fa1, 0(a1)
	fld	fa0, 8(a1)
	fld	ft0, 16(a1)
	fld	ft1, 24(a1)
	fld	ft2, 0(a4)
	sraiw	a2, a3, 23
	fld	ft3, 8(a4)
	fcvt.d.s	fa3, fa3
	fmadd.d	fa5, fa3, ft2, fa5
	fcvt.d.w	fa3, a2
	fadd.d	fa3, ft3, fa3
	fmul.d	ft2, fa5, fa5
	fmadd.d	fa1, fa5, fa1, fa0
	fmadd.d	fa0, fa5, ft0, ft1
	fmadd.d	fa5, fa5, fa2, fa3
	fmul.d	fa3, ft2, ft2
	fmadd.d	fa5, fa0, ft2, fa5
	fmadd.d	fa5, fa1, fa3, fa5
	fmul.d	fa5, fa5, fa4
	fmv.x.d	a2, fa5
	and	a2, a2, a5
	addi	a0, a0, 1
	bgeu	a2, a0, .LBB30_11
.LBB30_4:
	fld	fa4, 48(a1)
	fld	fa3, 56(a1)
	fld	fa2, 64(a1)
	lui	a0, 2141
	li	a1, -1955
.Lpcrel_hi5:
	auipc	a2, %pcrel_hi(__exp2f_data)
	slli	a0, a0, 39
	fmv.d.x	fa1, a0
	li	a0, 1023
	slli	a1, a1, 51
	addi	a2, a2, %pcrel_lo(.Lpcrel_hi5)
	slli	a0, a0, 52
	fmv.d.x	fa0, a1
	fadd.d	fa1, fa5, fa1
	fmv.x.d	a1, fa1
	fadd.d	fa1, fa1, fa0
	fmv.d.x	fa0, a0
	fsub.d	fa5, fa5, fa1
	andi	a0, a1, 31
	add	a1, a1, a6
	slli	a0, a0, 3
	slli	a1, a1, 47
	add	a0, a0, a2
	ld	a0, 0(a0)
	fmadd.d	fa4, fa5, fa4, fa3
	fmul.d	fa3, fa5, fa5
	fmadd.d	fa5, fa5, fa2, fa0
	add	a0, a0, a1
	fmv.d.x	fa2, a0
	fmadd.d	fa5, fa4, fa3, fa5
	fmul.d	fa5, fa5, fa2
	fcvt.s.d	fa5, fa5
.LBB30_5:
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_6:
	.cfi_restore_state
	.cfi_remember_state
	addiw	a4, a2, -1
	lui	a3, 1044480
	addi	a3, a3, -1
	bgeu	a4, a3, .LBB30_26
	slli	a2, a1, 1
	addiw	a2, a2, -1
	bgeu	a2, a3, .LBB30_32
	bltz	a1, .LBB30_15
	li	a6, 0
	srliw	a2, a1, 23
	bnez	a2, .LBB30_3
.LBB30_10:
	lui	a1, 307200
	fmv.w.x	fa5, a1
	fmul.s	fa5, fa0, fa5
	fmv.x.w	a1, fa5
	slli	a1, a1, 33
	srli	a1, a1, 33
	lui	a2, 1001472
	add	a1, a1, a2
	j	.LBB30_3
.LBB30_11:
	fld	fa4, 40(a1)
	flt.d	a0, fa4, fa5
	beqz	a0, .LBB30_19
	lui	a0, 458752
	fmv.w.x	fa5, a0
	fmv.s	fa4, fa5
	beqz	a6, .LBB30_14
	lui	a0, 983040
	fmv.w.x	fa4, a0
.LBB30_14:
	fsw	fa4, 8(sp)
	flw	fa4, 8(sp)
	fmul.s	fa5, fa4, fa5
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_15:
	.cfi_restore_state
	.cfi_remember_state
	slli	a2, a0, 33
	srli	a2, a2, 56
	li	a3, 127
	bltu	a2, a3, .LBB30_24
	li	a3, 150
	bgeu	a3, a2, .LBB30_23
.LBB30_17:
	li	a6, 0
.LBB30_18:
	slli	a1, a1, 33
	srli	a1, a1, 33
	srliw	a2, a1, 23
	bnez	a2, .LBB30_3
	j	.LBB30_10
.LBB30_19:
	lui	a0, 983435
	slli	a0, a0, 34
	fmv.d.x	fa4, a0
	fle.d	a0, fa5, fa4
	beqz	a0, .LBB30_4
	lui	a0, 65536
	fmv.w.x	fa5, a0
	fmv.s	fa4, fa5
	beqz	a6, .LBB30_22
	lui	a0, 589824
	fmv.w.x	fa4, a0
.LBB30_22:
	fsw	fa4, 12(sp)
	flw	fa4, 12(sp)
	fmul.s	fa5, fa4, fa5
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_23:
	.cfi_restore_state
	.cfi_remember_state
	sub	a3, a3, a2
	li	a2, 1
	sllw	a2, a2, a3
	addiw	a3, a2, -1
	and	a3, a3, a0
	beqz	a3, .LBB30_25
.LBB30_24:
	fsub.s	fa5, fa0, fa0
	fdiv.s	fa5, fa5, fa5
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_25:
	.cfi_restore_state
	.cfi_remember_state
	and	a2, a2, a0
	lui	a6, 16
	bnez	a2, .LBB30_18
	j	.LBB30_17
.LBB30_26:
	lui	a3, 260096
	fmv.w.x	fa5, a3
	beq	a1, a3, .LBB30_5
	beqz	a2, .LBB30_5
	slliw	a1, a1, 1
	lui	a3, 1044480
	bltu	a3, a1, .LBB30_38
	addi	a3, a3, 1
	bgeu	a2, a3, .LBB30_38
	lui	a2, 520192
	bne	a1, a2, .LBB30_39
	lui	a0, 260096
	fmv.w.x	fa5, a0
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_32:
	.cfi_restore_state
	.cfi_remember_state
	fmul.s	fa5, fa0, fa0
	bgez	a1, .LBB30_36
	slli	a1, a0, 33
	srli	a1, a1, 56
	addi	a2, a1, -151
	li	a3, -24
	bltu	a2, a3, .LBB30_36
	li	a2, 150
	sub	a2, a2, a1
	li	a1, 1
	sllw	a1, a1, a2
	addiw	a2, a1, -1
	and	a1, a1, a0
	and	a2, a2, a0
	snez	a2, a2
	seqz	a1, a1
	or	a1, a1, a2
	bnez	a1, .LBB30_36
	fneg.s	fa5, fa5
.LBB30_36:
	bgez	a0, .LBB30_5
	lui	a0, 260096
	fmv.w.x	fa4, a0
	fdiv.s	fa5, fa4, fa5
	fsw	fa5, 4(sp)
	flw	fa5, 4(sp)
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_38:
	.cfi_restore_state
	.cfi_remember_state
	fadd.s	fa5, fa0, fa1
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_39:
	.cfi_restore_state
	.cfi_remember_state
	srliw	a1, a1, 24
	sltiu	a1, a1, 127
	srli	a0, a0, 63
	bne	a1, a0, .LBB30_41
	fmul.s	fa5, fa1, fa1
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB30_41:
	.cfi_restore_state
	fmv.w.x	fa5, zero
	fmv.s	fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end30:
	.size	powf, .Lfunc_end30-powf
	.cfi_endproc

	.section	.text.rintf,"ax",@progbits
	.p2align	1
	.type	rintf,@function
rintf:
.Lfunc_begin31:
	.cfi_startproc
	fmv.x.w	a0, fa0
	lui	a1, 520192
	and	a1, a1, a0
	lui	a2, 305152
	bltu	a2, a1, .LBB31_3
	lui	a1, 831488
	fmv.w.x	fa5, a1
	lui	a1, 307200
	fmv.w.x	fa4, a1
	bgez	a0, .LBB31_4
	fadd.s	fa5, fa0, fa5
	fadd.s	fa0, fa5, fa4
	fmv.w.x	fa5, zero
	feq.s	a1, fa0, fa5
	bnez	a1, .LBB31_5
.LBB31_3:
	ret
.LBB31_4:
	fadd.s	fa4, fa0, fa4
	fadd.s	fa0, fa4, fa5
	fmv.w.x	fa5, zero
	feq.s	a1, fa0, fa5
	beqz	a1, .LBB31_3
.LBB31_5:
	bgez	a0, .LBB31_7
	lui	a0, 524288
	fmv.w.x	fa5, a0
.LBB31_7:
	fmv.s	fa0, fa5
	ret
.Lfunc_end31:
	.size	rintf, .Lfunc_end31-rintf
	.cfi_endproc

	.section	.text.roundf,"ax",@progbits
	.p2align	1
	.type	roundf,@function
roundf:
.Lfunc_begin32:
	.cfi_startproc
	fmv.x.w	a0, fa0
	slli	a1, a0, 33
	srli	a1, a1, 56
	li	a2, 149
	bgeu	a2, a1, .LBB32_2
.LBB32_1:
	ret
.LBB32_2:
	fabs.s	fa5, fa0
	lui	a2, 307200
	fmv.w.x	fa4, a2
	li	a2, 125
	fadd.s	fa4, fa5, fa4
	bltu	a2, a1, .LBB32_4
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	fsw	fa4, 12(sp)
	fmv.w.x	fa5, zero
	fmul.s	fa0, fa0, fa5
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB32_4:
	lui	a1, 831488
	fmv.w.x	fa3, a1
	lui	a1, 258048
	fadd.s	fa4, fa4, fa3
	fsub.s	fa4, fa4, fa5
	fmv.w.x	fa3, a1
	flt.s	a1, fa3, fa4
	beqz	a1, .LBB32_6
	fadd.s	fa5, fa5, fa4
	lui	a1, 784384
	fmv.w.x	fa4, a1
	fadd.s	fa0, fa5, fa4
	j	.LBB32_8
.LBB32_6:
	lui	a1, 782336
	fmv.w.x	fa3, a1
	fle.s	a1, fa4, fa3
	fadd.s	fa0, fa5, fa4
	beqz	a1, .LBB32_8
	lui	a1, 260096
	fmv.w.x	fa5, a1
	fadd.s	fa0, fa0, fa5
.LBB32_8:
	bgez	a0, .LBB32_1
	fneg.s	fa0, fa0
	ret
.Lfunc_end32:
	.size	roundf, .Lfunc_end32-roundf
	.cfi_endproc

	.type	__unnamed_1,@object
	.section	.rodata.__unnamed_1,"a",@progbits
__unnamed_1:
	.asciz	"matmul_dispatch_0"
	.size	__unnamed_1, 18

	.type	iree_hal_executable_library_query_v0_header,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_header,"aw",@progbits
	.p2align	4, 0x0
iree_hal_executable_library_query_v0_header:
	.word	6
	.zero	4
	.quad	__unnamed_1
	.word	0
	.word	0
	.size	iree_hal_executable_library_query_v0_header, 24

	.type	__unnamed_2,@object
	.section	.rodata.__unnamed_2,"a",@progbits
__unnamed_2:
	.asciz	"free"
	.size	__unnamed_2, 5

	.type	__unnamed_3,@object
	.section	.rodata.__unnamed_3,"a",@progbits
__unnamed_3:
	.asciz	"malloc"
	.size	__unnamed_3, 7

	.type	iree_hal_executable_library_query_v0_import_names,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_import_names,"aw",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_import_names:
	.quad	__unnamed_2
	.quad	__unnamed_3
	.size	iree_hal_executable_library_query_v0_import_names, 16

	.type	iree_hal_executable_library_query_v0_funcs,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_funcs,"aw",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_funcs:
	.quad	matmul_dispatch_0_matmul_4x4x4_f32
	.size	iree_hal_executable_library_query_v0_funcs, 8

	.type	iree_hal_executable_library_query_v0_attrs,@object
	.section	.rodata.iree_hal_executable_library_query_v0_attrs,"a",@progbits
	.p2align	4, 0x0
iree_hal_executable_library_query_v0_attrs:
	.quad	0
	.half	0
	.byte	0
	.byte	3
	.word	1
	.word	1
	.half	1
	.half	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.size	iree_hal_executable_library_query_v0_attrs, 64

	.type	__unnamed_4,@object
	.section	.rodata.__unnamed_4,"a",@progbits
__unnamed_4:
	.asciz	"matmul_dispatch_0_matmul_4x4x4_f32"
	.size	__unnamed_4, 35

	.type	iree_hal_executable_library_query_v0_names,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_names,"aw",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_names:
	.quad	__unnamed_4
	.size	iree_hal_executable_library_query_v0_names, 8

	.type	__unnamed_5,@object
	.section	.rodata.__unnamed_5,"a",@progbits
__unnamed_5:
	.asciz	"/Users/sparshsingh/work/merlin/compiler/plugins/target/Gemmini/test/Output/gemmini_matmul_lowering.mlir.tmp.dir/configured_module_matmul_dispatch_0.mlir"
	.size	__unnamed_5, 153

	.type	iree_hal_executable_library_query_v0_source_locations,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_source_locations,"aw",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_source_locations:
	.word	3
	.word	152
	.quad	__unnamed_5
	.size	iree_hal_executable_library_query_v0_source_locations, 16

	.type	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names,@object
	.section	.rodata.iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names,"a",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names:
	.size	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names, 0

	.type	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations,@object
	.section	.rodata.iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations,"a",@progbits
	.p2align	3, 0x0
iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations:
	.size	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations, 0

	.type	iree_hal_executable_library_query_v0_stage_location_tables,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0_stage_location_tables,"aw",@progbits
	.p2align	4, 0x0
iree_hal_executable_library_query_v0_stage_location_tables:
	.word	0
	.zero	4
	.quad	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names
	.quad	iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations
	.size	iree_hal_executable_library_query_v0_stage_location_tables, 24

	.type	iree_hal_executable_library_query_v0,@object
	.section	.data.rel.ro.iree_hal_executable_library_query_v0,"aw",@progbits
	.p2align	4, 0x0
iree_hal_executable_library_query_v0:
	.quad	iree_hal_executable_library_query_v0_header
	.word	2
	.zero	4
	.quad	iree_hal_executable_library_query_v0_import_names
	.word	1
	.zero	4
	.quad	iree_hal_executable_library_query_v0_funcs
	.quad	iree_hal_executable_library_query_v0_attrs
	.quad	0
	.quad	0
	.quad	iree_hal_executable_library_query_v0_names
	.quad	0
	.quad	0
	.quad	iree_hal_executable_library_query_v0_source_locations
	.quad	iree_hal_executable_library_query_v0_stage_location_tables
	.zero	4
	.zero	4
	.zero	16
	.size	iree_hal_executable_library_query_v0, 128

	.type	__exp2f_data,@object
	.section	.rodata.__exp2f_data,"a",@progbits
	.p2align	3, 0x0
__exp2f_data:
	.quad	4607182418800017408
	.quad	4607140297302181236
	.quad	4607100335213349135
	.quad	4607062579818421073
	.quad	4607027079437701499
	.quad	4606993883449571754
	.quad	4606963042313658936
	.quad	4606934607594512097
	.quad	4606908631985796885
	.quad	4606885169335019979
	.quad	4606864274668794914
	.quad	4606846004218661165
	.quad	4606830415447468583
	.quad	4606817567076339586
	.quad	4606807519112221737
	.quad	4606800332876043653
	.quad	4606796071031487437
	.quad	4606794797614391156
	.quad	4606796578062795143
	.quad	4606801479247646227
	.quad	4606809569504174299
	.quad	4606820918663955941
	.quad	4606835598087680144
	.quad	4606853680698631517
	.quad	4606875241016906669
	.quad	4606900355194379847
	.quad	4606929101050434204
	.quad	4606961558108475497
	.quad	4606997807633245319
	.quad	4607037932668951391
	.quad	4607082018078232794
	.quad	4607130150581978432
	.quad	0x42e8000000000000
	.quad	0x3fac6af84b912394
	.quad	0x3fcebfce50fac4f3
	.quad	0x3fe62e42ff0c52d6
	.quad	0x4338000000000000
	.quad	0x40471547652b82fe
	.quad	0x3ebc6af84b912394
	.quad	0x3f2ebfce50fac4f3
	.quad	0x3f962e42ff0c52d6
	.size	__exp2f_data, 328

	.type	__powf_log2_data,@object
	.section	.rodata.__powf_log2_data,"a",@progbits
	.p2align	3, 0x0
__powf_log2_data:
	.quad	0x3ff661ec79f8f3be
	.quad	0xbfdefec65b963019
	.quad	0x3ff571ed4aaf883d
	.quad	0xbfdb0b6832d4fca4
	.quad	0x3ff49539f0f010b0
	.quad	0xbfd7418b0a1fb77b
	.quad	0x3ff3c995b0b80385
	.quad	0xbfd39de91a6dcf7b
	.quad	0x3ff30d190c8864a5
	.quad	0xbfd01d9bf3f2b631
	.quad	0x3ff25e227b0b8ea0
	.quad	0xbfc97c1d1b3b7af0
	.quad	0x3ff1bb4a4a1a343f
	.quad	0xbfc2f9e393af3c9f
	.quad	0x3ff12358f08ae5ba
	.quad	0xbfb960cbbf788d5c
	.quad	0x3ff0953f419900a7
	.quad	0xbfaa6f9db6475fce
	.quad	0x3ff0000000000000
	.quad	0x0000000000000000
	.quad	0x3fee608cfd9a47ac
	.quad	0x3fb338ca9f24f53d
	.quad	0x3feca4b31f026aa0
	.quad	0x3fc476a9543891ba
	.quad	0x3feb2036576afce6
	.quad	0x3fce840b4ac4e4d2
	.quad	0x3fe9c2d163a1aa2d
	.quad	0x3fd40645f0c6651c
	.quad	0x3fe886e6037841ed
	.quad	0x3fd88e9c2c1b9ff8
	.quad	0x3fe767dcf5534862
	.quad	0x3fdce0a44eb17bcc
	.quad	0x3fd27616c9496e0b
	.quad	0xbfd71969a075c67a
	.quad	0x3fdec70a6ca7badd
	.quad	0xbfe7154748bef6c8
	.quad	0x3ff71547652ab82b
	.size	__powf_log2_data, 296

	.type	.promoted_doubles.expf,@object
	.section	.rodata..promoted_doubles.expf,"a",@progbits
	.p2align	4, 0x0
.promoted_doubles.expf:
	.quad	0x40471547652b82fe
	.quad	0x3ebc6af84b912394
	.quad	0x3f2ebfce50fac4f3
	.quad	0x3f962e42ff0c52d6
	.size	.promoted_doubles.expf, 32

	.type	.promoted_doubles.powf,@object
	.section	.rodata..promoted_doubles.powf,"a",@progbits
	.p2align	4, 0x0
.promoted_doubles.powf:
	.quad	0x3fd27616c9496e0b
	.quad	0xbfd71969a075c67a
	.quad	0x3fdec70a6ca7badd
	.quad	0xbfe7154748bef6c8
	.quad	0x3ff71547652ab82b
	.quad	0x405fffffffd1d571
	.quad	0x3fac6af84b912394
	.quad	0x3fcebfce50fac4f3
	.quad	0x3fe62e42ff0c52d6
	.size	.promoted_doubles.powf, 72

	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.ascii	"\264B"
	.byte	25
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	2
	.byte	46
	.byte	0
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	110
	.byte	14
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	3
	.byte	36
	.byte	0
	.byte	3
	.byte	14
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.half	4
	.word	.debug_abbrev
	.byte	8
	.byte	1
	.word	.Linfo_string0
	.half	44
	.word	.Linfo_string1
	.word	.Lline_table_start0
	.word	.Linfo_string2

	.quad	.Lfunc_begin0
	.word	.Lfunc_end0-.Lfunc_begin0
	.byte	2
	.quad	.Lfunc_begin0
	.word	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	88
	.word	.Linfo_string3
	.word	.Linfo_string3
	.byte	1
	.byte	1
	.word	71

	.byte	3
	.word	.Linfo_string4
	.byte	5
	.byte	4
	.byte	0
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"IREE"
.Linfo_string1:
	.asciz	"configured_module_matmul_dispatch_0.mlir"
.Linfo_string2:
	.asciz	"/Users/sparshsingh/work/merlin/compiler/plugins/target/Gemmini/test/Output/gemmini_matmul_lowering.mlir.tmp.dir"
.Linfo_string3:
	.asciz	"matmul_dispatch_0_matmul_4x4x4_f32"
.Linfo_string4:
	.asciz	"int"
	.section	.debug_pubnames,"",@progbits
	.word	.LpubNames_end0-.LpubNames_start0
.LpubNames_start0:
	.half	2
	.word	.Lcu_begin0
	.word	79
	.word	42
	.asciz	"matmul_dispatch_0_matmul_4x4x4_f32"
	.word	0
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.word	.LpubTypes_end0-.LpubTypes_start0
.LpubTypes_start0:
	.half	2
	.word	.Lcu_begin0
	.word	79
	.word	71
	.asciz	"int"
	.word	0
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
