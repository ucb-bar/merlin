; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_4_operator_conv2d1(ptr noalias %y, ptr noalias %x, ptr noalias %conv2d_cnhw) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %y, metadata !12, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata ptr %conv2d_cnhw, metadata !14, metadata !DIExpression()), !dbg !15
  %pad_temp9 = alloca [12 x float], align 16, !dbg !15
  call void @llvm.dbg.declare(metadata ptr %pad_temp9, metadata !16, metadata !DIExpression()), !dbg !15
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %x, i64 48, ptr %x, i64 48, ptr %conv2d_cnhw, i64 32), !dbg !15
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !17, metadata !DIExpression()), !dbg !15
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !15
  br i1 %.not, label %if_else, label %if_then, !dbg !15

common.ret:                                       ; preds = %if_then3, %if_else5, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %if_else5 ], [ -1, %if_then3 ]
  ret i32 %common.ret.op, !dbg !15

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !15
  br label %common.ret

if_else:                                          ; preds = %entry
  %gemmini_admission2 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %y, i64 48, ptr %y, i64 48, ptr %conv2d_cnhw, i64 32), !dbg !15
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission2, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission2, metadata !17, metadata !DIExpression()), !dbg !15
  %.not10 = icmp eq i32 %gemmini_admission2, 0, !dbg !15
  br i1 %.not10, label %if_else5, label %if_then3, !dbg !15

if_then3:                                         ; preds = %if_else
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !15
  br label %common.ret

if_else5:                                         ; preds = %if_else
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  tail call void @tvm_gemmini_dma_acc_write(ptr %y, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %pad_temp9, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  %0 = getelementptr inbounds float, ptr %y, i64 1, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %0, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %1 = getelementptr inbounds float, ptr %pad_temp9, i64 1, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %1, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  %2 = getelementptr inbounds float, ptr %y, i64 2, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %2, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %3 = getelementptr inbounds float, ptr %pad_temp9, i64 2, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %3, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !20, metadata !DIExpression()), !dbg !15
  %4 = getelementptr inbounds float, ptr %y, i64 3, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %4, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %5 = getelementptr inbounds float, ptr %pad_temp9, i64 3, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %5, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !20, metadata !DIExpression()), !dbg !15
  %6 = getelementptr inbounds float, ptr %y, i64 4, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %6, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %7 = getelementptr inbounds float, ptr %pad_temp9, i64 4, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %7, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !20, metadata !DIExpression()), !dbg !15
  %8 = getelementptr inbounds float, ptr %y, i64 5, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %8, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %9 = getelementptr inbounds float, ptr %pad_temp9, i64 5, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %9, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !20, metadata !DIExpression()), !dbg !15
  %10 = getelementptr inbounds float, ptr %y, i64 6, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %10, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %11 = getelementptr inbounds float, ptr %pad_temp9, i64 6, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %11, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !20, metadata !DIExpression()), !dbg !15
  %12 = getelementptr inbounds float, ptr %y, i64 7, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %12, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %13 = getelementptr inbounds float, ptr %pad_temp9, i64 7, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %13, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !20, metadata !DIExpression()), !dbg !15
  %14 = getelementptr inbounds float, ptr %y, i64 8, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %14, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %15 = getelementptr inbounds float, ptr %pad_temp9, i64 8, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %15, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !20, metadata !DIExpression()), !dbg !15
  %16 = getelementptr inbounds float, ptr %y, i64 9, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %16, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %17 = getelementptr inbounds float, ptr %pad_temp9, i64 9, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %17, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !20, metadata !DIExpression()), !dbg !15
  %18 = getelementptr inbounds float, ptr %y, i64 10, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %18, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %19 = getelementptr inbounds float, ptr %pad_temp9, i64 10, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %19, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !20, metadata !DIExpression()), !dbg !15
  %20 = getelementptr inbounds float, ptr %y, i64 11, !dbg !15
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %20, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  %21 = getelementptr inbounds float, ptr %pad_temp9, i64 11, !dbg !15
  call void @tvm_gemmini_dma_out(ptr nonnull %21, i32 0, i32 1, i32 1, i64 4, i32 0), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !15
  call void @tvm_gemmini_end(), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %22 = load float, ptr %pad_temp9, align 16, !dbg !15, !tbaa !27
  %23 = load float, ptr %x, align 4, !dbg !15, !tbaa !30
  %24 = call float @llvm.fmuladd.f32(float %22, float %23, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %25 = load float, ptr %1, align 4, !dbg !15, !tbaa !27
  %26 = getelementptr inbounds float, ptr %x, i64 1, !dbg !15
  %27 = load float, ptr %26, align 4, !dbg !15, !tbaa !30
  %28 = call float @llvm.fmuladd.f32(float %25, float %27, float %24), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %29 = load float, ptr %3, align 8, !dbg !15, !tbaa !27
  %30 = getelementptr inbounds float, ptr %x, i64 2, !dbg !15
  %31 = load float, ptr %30, align 4, !dbg !15, !tbaa !30
  %32 = call float @llvm.fmuladd.f32(float %29, float %31, float %28), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %33 = load float, ptr %7, align 16, !dbg !15, !tbaa !27
  %34 = getelementptr inbounds float, ptr %x, i64 3, !dbg !15
  %35 = load float, ptr %34, align 4, !dbg !15, !tbaa !30
  %36 = call float @llvm.fmuladd.f32(float %33, float %35, float %32), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %37 = load float, ptr %9, align 4, !dbg !15, !tbaa !27
  %38 = getelementptr inbounds float, ptr %x, i64 4, !dbg !15
  %39 = load float, ptr %38, align 4, !dbg !15, !tbaa !30
  %40 = call float @llvm.fmuladd.f32(float %37, float %39, float %36), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !15
  %41 = load float, ptr %11, align 8, !dbg !15, !tbaa !27
  %42 = getelementptr inbounds float, ptr %x, i64 5, !dbg !15
  %43 = load float, ptr %42, align 4, !dbg !15, !tbaa !30
  %44 = call float @llvm.fmuladd.f32(float %41, float %43, float %40), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %44, ptr %conv2d_cnhw, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  %45 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 1
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %46 = call float @llvm.fmuladd.f32(float %25, float %23, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %47 = call float @llvm.fmuladd.f32(float %29, float %27, float %46), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %48 = load float, ptr %5, align 4, !dbg !15, !tbaa !27
  %49 = call float @llvm.fmuladd.f32(float %48, float %31, float %47), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %50 = call float @llvm.fmuladd.f32(float %37, float %35, float %49), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %51 = call float @llvm.fmuladd.f32(float %41, float %39, float %50), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !15
  %52 = load float, ptr %13, align 4, !dbg !15, !tbaa !27
  %53 = call float @llvm.fmuladd.f32(float %52, float %43, float %51), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %53, ptr %45, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !15
  %54 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 2
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %55 = call float @llvm.fmuladd.f32(float %33, float %23, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %56 = call float @llvm.fmuladd.f32(float %37, float %27, float %55), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %57 = call float @llvm.fmuladd.f32(float %41, float %31, float %56), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %58 = load float, ptr %15, align 16, !dbg !15, !tbaa !27
  %59 = call float @llvm.fmuladd.f32(float %58, float %35, float %57), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %60 = load float, ptr %17, align 4, !dbg !15, !tbaa !27
  %61 = call float @llvm.fmuladd.f32(float %60, float %39, float %59), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !15
  %62 = load float, ptr %19, align 8, !dbg !15, !tbaa !27
  %63 = call float @llvm.fmuladd.f32(float %62, float %43, float %61), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %63, ptr %54, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  %64 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 3
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %65 = call float @llvm.fmuladd.f32(float %37, float %23, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %66 = call float @llvm.fmuladd.f32(float %41, float %27, float %65), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %67 = call float @llvm.fmuladd.f32(float %52, float %31, float %66), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %68 = call float @llvm.fmuladd.f32(float %60, float %35, float %67), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %69 = call float @llvm.fmuladd.f32(float %62, float %39, float %68), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !15
  %70 = load float, ptr %21, align 4, !dbg !15, !tbaa !27
  %71 = call float @llvm.fmuladd.f32(float %70, float %43, float %69), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %71, ptr %64, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !15
  %72 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 4
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %73 = getelementptr inbounds float, ptr %x, i64 6, !dbg !15
  %74 = load float, ptr %73, align 4, !dbg !15, !tbaa !30
  %75 = call float @llvm.fmuladd.f32(float %22, float %74, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %76 = getelementptr inbounds float, ptr %x, i64 7, !dbg !15
  %77 = load float, ptr %76, align 4, !dbg !15, !tbaa !30
  %78 = call float @llvm.fmuladd.f32(float %25, float %77, float %75), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %79 = getelementptr inbounds float, ptr %x, i64 8, !dbg !15
  %80 = load float, ptr %79, align 4, !dbg !15, !tbaa !30
  %81 = call float @llvm.fmuladd.f32(float %29, float %80, float %78), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %82 = getelementptr inbounds float, ptr %x, i64 9, !dbg !15
  %83 = load float, ptr %82, align 4, !dbg !15, !tbaa !30
  %84 = call float @llvm.fmuladd.f32(float %33, float %83, float %81), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %85 = getelementptr inbounds float, ptr %x, i64 10, !dbg !15
  %86 = load float, ptr %85, align 4, !dbg !15, !tbaa !30
  %87 = call float @llvm.fmuladd.f32(float %37, float %86, float %84), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !15
  %88 = getelementptr inbounds float, ptr %x, i64 11, !dbg !15
  %89 = load float, ptr %88, align 4, !dbg !15, !tbaa !30
  %90 = call float @llvm.fmuladd.f32(float %41, float %89, float %87), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %90, ptr %72, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  %91 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 5
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %92 = call float @llvm.fmuladd.f32(float %25, float %74, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %93 = call float @llvm.fmuladd.f32(float %29, float %77, float %92), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %94 = call float @llvm.fmuladd.f32(float %48, float %80, float %93), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %95 = call float @llvm.fmuladd.f32(float %37, float %83, float %94), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %96 = call float @llvm.fmuladd.f32(float %41, float %86, float %95), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !15
  %97 = call float @llvm.fmuladd.f32(float %52, float %89, float %96), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %97, ptr %91, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !15
  %98 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 6
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %99 = call float @llvm.fmuladd.f32(float %33, float %74, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %100 = call float @llvm.fmuladd.f32(float %37, float %77, float %99), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %101 = call float @llvm.fmuladd.f32(float %41, float %80, float %100), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %102 = call float @llvm.fmuladd.f32(float %58, float %83, float %101), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %103 = call float @llvm.fmuladd.f32(float %60, float %86, float %102), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !15
  %104 = call float @llvm.fmuladd.f32(float %62, float %89, float %103), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %104, ptr %98, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !23, metadata !DIExpression()), !dbg !15
  %105 = getelementptr inbounds float, ptr %conv2d_cnhw, i64 7
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %106 = call float @llvm.fmuladd.f32(float %37, float %74, float 0.000000e+00), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %107 = call float @llvm.fmuladd.f32(float %41, float %77, float %106), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %108 = call float @llvm.fmuladd.f32(float %52, float %80, float %107), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %109 = call float @llvm.fmuladd.f32(float %60, float %83, float %108), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %110 = call float @llvm.fmuladd.f32(float %62, float %86, float %109), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !15
  %111 = call float @llvm.fmuladd.f32(float %70, float %89, float %110), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !15
  store float %111, ptr %105, align 4, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !23, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @tvm_gemmini_end(), !dbg !15
  br label %common.ret
}

define dllexport noundef i32 @stage_4_operator_conv2d_transpose(ptr noalias %x, ptr noalias %w, ptr noalias %compute) local_unnamed_addr #0 !dbg !34 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !36, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr %w, metadata !37, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr %compute, metadata !38, metadata !DIExpression()), !dbg !39
  %scalar = alloca float, align 8, !dbg !39
  %kernel_transform34 = alloca [8 x float], align 16, !dbg !39
  %data_pad35 = alloca [40 x float], align 16, !dbg !39
  %data_dilate36 = alloca [12 x float], align 16, !dbg !39
  call void @llvm.dbg.declare(metadata ptr %data_dilate36, metadata !40, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %data_pad35, metadata !41, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %kernel_transform34, metadata !42, metadata !DIExpression()), !dbg !39
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %w, i64 32, ptr %w, i64 32, ptr %compute, i64 48), !dbg !39
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !43, metadata !DIExpression()), !dbg !39
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !39
  br i1 %.not, label %if_else, label %if_then, !dbg !39

common.ret:                                       ; preds = %if_then3, %for_end_i2_013.1, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %for_end_i2_013.1 ], [ -1, %if_then3 ]
  ret i32 %common.ret.op, !dbg !39

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !39
  br label %common.ret

if_else:                                          ; preds = %entry
  %gemmini_admission2 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %x, i64 48, ptr %x, i64 48, ptr %compute, i64 48), !dbg !39
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission2, metadata !43, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission2, metadata !43, metadata !DIExpression()), !dbg !39
  %.not37 = icmp eq i32 %gemmini_admission2, 0, !dbg !39
  br i1 %.not37, label %if_else5, label %if_then3, !dbg !39

if_then3:                                         ; preds = %if_else
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !39
  br label %common.ret

if_else5:                                         ; preds = %if_else
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !47, metadata !DIExpression()), !dbg !39
  tail call void @tvm_gemmini_dma_acc_write(ptr %x, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %data_dilate36, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !47, metadata !DIExpression()), !dbg !39
  %0 = getelementptr inbounds float, ptr %x, i64 1, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %0, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %1 = getelementptr inbounds float, ptr %data_dilate36, i64 1, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %1, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !47, metadata !DIExpression()), !dbg !39
  %2 = getelementptr inbounds float, ptr %x, i64 2, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %2, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %3 = getelementptr inbounds float, ptr %data_dilate36, i64 2, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %3, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !47, metadata !DIExpression()), !dbg !39
  %4 = getelementptr inbounds float, ptr %x, i64 3, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %4, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %5 = getelementptr inbounds float, ptr %data_dilate36, i64 3, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %5, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !47, metadata !DIExpression()), !dbg !39
  %6 = getelementptr inbounds float, ptr %x, i64 4, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %6, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %7 = getelementptr inbounds float, ptr %data_dilate36, i64 4, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %7, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !47, metadata !DIExpression()), !dbg !39
  %8 = getelementptr inbounds float, ptr %x, i64 5, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %8, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %9 = getelementptr inbounds float, ptr %data_dilate36, i64 5, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %9, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !47, metadata !DIExpression()), !dbg !39
  %10 = getelementptr inbounds float, ptr %x, i64 6, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %10, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %11 = getelementptr inbounds float, ptr %data_dilate36, i64 6, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %11, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !47, metadata !DIExpression()), !dbg !39
  %12 = getelementptr inbounds float, ptr %x, i64 7, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %12, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %13 = getelementptr inbounds float, ptr %data_dilate36, i64 7, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %13, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !47, metadata !DIExpression()), !dbg !39
  %14 = getelementptr inbounds float, ptr %x, i64 8, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %14, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %15 = getelementptr inbounds float, ptr %data_dilate36, i64 8, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %15, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !47, metadata !DIExpression()), !dbg !39
  %16 = getelementptr inbounds float, ptr %x, i64 9, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %16, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %17 = getelementptr inbounds float, ptr %data_dilate36, i64 9, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %17, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !47, metadata !DIExpression()), !dbg !39
  %18 = getelementptr inbounds float, ptr %x, i64 10, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %18, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %19 = getelementptr inbounds float, ptr %data_dilate36, i64 10, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %19, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !47, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !47, metadata !DIExpression()), !dbg !39
  %20 = getelementptr inbounds float, ptr %x, i64 11, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %20, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %21 = getelementptr inbounds float, ptr %data_dilate36, i64 11, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %21, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  %invariant.gep112 = getelementptr inbounds float, ptr %data_pad35, i64 1, !dbg !39
  %invariant.gep113 = getelementptr inbounds float, ptr %data_pad35, i64 2, !dbg !39
  %invariant.gep115 = getelementptr inbounds float, ptr %data_pad35, i64 3, !dbg !39
  %invariant.gep117 = getelementptr inbounds float, ptr %data_pad35, i64 4, !dbg !39
  br label %for_begin_i3_015.preheader, !dbg !39

for_begin_i3_015.preheader:                       ; preds = %if_else5, %for_begin_i3_015.preheader
  %indvars.iv = phi i64 [ 0, %if_else5 ], [ %indvars.iv.next, %for_begin_i3_015.preheader ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  %22 = mul nuw nsw i64 %indvars.iv, 5
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %23 = getelementptr inbounds float, ptr %data_pad35, i64 %22, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %23, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep = getelementptr float, ptr %invariant.gep112, i64 %22, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep114 = getelementptr float, ptr %invariant.gep113, i64 %22, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep114, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep116 = getelementptr float, ptr %invariant.gep115, i64 %22, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep116, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep118 = getelementptr float, ptr %invariant.gep117, i64 %22, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep118, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !46, metadata !DIExpression()), !dbg !39
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !39
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next, metadata !45, metadata !DIExpression()), !dbg !39
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4, !dbg !39
  br i1 %exitcond.not, label %for_end_i2_013, label %for_begin_i3_015.preheader, !dbg !39, !prof !60

for_end_i2_013:                                   ; preds = %for_begin_i3_015.preheader
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  %invariant.gep119 = getelementptr inbounds float, ptr %data_pad35, i64 20, !dbg !39
  %invariant.gep121 = getelementptr inbounds float, ptr %data_pad35, i64 21, !dbg !39
  %invariant.gep123 = getelementptr inbounds float, ptr %data_pad35, i64 22, !dbg !39
  %invariant.gep125 = getelementptr inbounds float, ptr %data_pad35, i64 23, !dbg !39
  %invariant.gep127 = getelementptr inbounds float, ptr %data_pad35, i64 24, !dbg !39
  br label %for_begin_i3_015.preheader.1, !dbg !39

for_begin_i3_015.preheader.1:                     ; preds = %for_begin_i3_015.preheader.1, %for_end_i2_013
  %indvars.iv.1 = phi i64 [ 0, %for_end_i2_013 ], [ %indvars.iv.next.1, %for_begin_i3_015.preheader.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  %24 = mul nuw nsw i64 %indvars.iv.1, 5
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep120 = getelementptr float, ptr %invariant.gep119, i64 %24, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep120, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep122 = getelementptr float, ptr %invariant.gep121, i64 %24, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep122, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep124 = getelementptr float, ptr %invariant.gep123, i64 %24, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep124, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep126 = getelementptr float, ptr %invariant.gep125, i64 %24, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep126, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !46, metadata !DIExpression()), !dbg !39
  tail call void @llvm.dbg.value(metadata ptr null, metadata !48, metadata !DIExpression()), !dbg !39
  store ptr null, ptr %scalar, align 8, !dbg !39, !tbaa !49
  tail call void @llvm.dbg.value(metadata ptr %scalar, metadata !48, metadata !DIExpression(DW_OP_deref)), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %scalar, i32 0, i32 1, i32 1, i64 0, i32 1), !dbg !39
  %gep128 = getelementptr float, ptr %invariant.gep127, i64 %24, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep128, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !46, metadata !DIExpression()), !dbg !39
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv.1, 1, !dbg !39
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.1, metadata !45, metadata !DIExpression()), !dbg !39
  %exitcond.1.not = icmp eq i64 %indvars.iv.next.1, 4, !dbg !39
  br i1 %exitcond.1.not, label %for_end_i2_013.1, label %for_begin_i3_015.preheader.1, !dbg !39, !prof !60

for_end_i2_013.1:                                 ; preds = %for_begin_i3_015.preheader.1
  call void @llvm.dbg.declare(metadata i64 2, metadata !44, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_end(), !dbg !39
  %invariant.gep = getelementptr inbounds float, ptr %data_pad35, i64 6
  call void @llvm.dbg.declare(metadata i32 0, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %data_dilate36, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %invariant.gep, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %1, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.1 = getelementptr inbounds float, ptr %data_pad35, i64 7, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %3, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.2 = getelementptr inbounds float, ptr %data_pad35, i64 8, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.2, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %5, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.169 = getelementptr inbounds float, ptr %data_pad35, i64 11, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.169, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %7, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.1.1 = getelementptr inbounds float, ptr %data_pad35, i64 12, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.1.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %9, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.2.1 = getelementptr inbounds float, ptr %data_pad35, i64 13, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.2.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %11, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.172 = getelementptr inbounds float, ptr %data_pad35, i64 26, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.172, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %13, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.1.173 = getelementptr inbounds float, ptr %data_pad35, i64 27, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.1.173, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %15, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.2.174 = getelementptr inbounds float, ptr %data_pad35, i64 28, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.2.174, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %17, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.169.1 = getelementptr inbounds float, ptr %data_pad35, i64 31, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.169.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %19, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.1.1.1 = getelementptr inbounds float, ptr %data_pad35, i64 32, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.1.1.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !46, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %21, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !39
  %gep.2.1.1 = getelementptr inbounds float, ptr %data_pad35, i64 33, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %gep.2.1.1, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !46, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !45, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !44, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !61, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  %25 = getelementptr inbounds float, ptr %w, i64 3, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %25, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %kernel_transform34, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  %26 = getelementptr inbounds float, ptr %w, i64 2, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %26, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %27 = getelementptr inbounds float, ptr %kernel_transform34, i64 1, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %27, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  %28 = getelementptr inbounds float, ptr %w, i64 1, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %28, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %29 = getelementptr inbounds float, ptr %kernel_transform34, i64 2, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %29, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr %w, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %30 = getelementptr inbounds float, ptr %kernel_transform34, i64 3, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %30, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !61, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !61, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  %31 = getelementptr inbounds float, ptr %w, i64 7, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %31, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %32 = getelementptr inbounds float, ptr %kernel_transform34, i64 4, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %32, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !65, metadata !DIExpression()), !dbg !39
  %33 = getelementptr inbounds float, ptr %w, i64 6, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %33, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %34 = getelementptr inbounds float, ptr %kernel_transform34, i64 5, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %34, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  %35 = getelementptr inbounds float, ptr %w, i64 5, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %35, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %36 = getelementptr inbounds float, ptr %kernel_transform34, i64 6, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %36, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !64, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !65, metadata !DIExpression()), !dbg !39
  %37 = getelementptr inbounds float, ptr %w, i64 4, !dbg !39
  call void @tvm_gemmini_dma_acc_write(ptr nonnull %37, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  %38 = getelementptr inbounds float, ptr %kernel_transform34, i64 7, !dbg !39
  call void @tvm_gemmini_dma_out(ptr nonnull %38, i32 0, i32 1, i32 1, i64 2, i32 0), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !63, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !62, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !61, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_end(), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !66, metadata !DIExpression()), !dbg !39
  %39 = load float, ptr %34, align 4, !tbaa !67
  %invariant.gep179 = getelementptr inbounds float, ptr %data_pad35, i64 24, !dbg !39
  %40 = load float, ptr %32, align 16, !tbaa !67
  %invariant.gep177 = getelementptr inbounds float, ptr %data_pad35, i64 23, !dbg !39
  %41 = load float, ptr %30, align 4, !tbaa !67
  %42 = load float, ptr %29, align 8, !tbaa !67
  %43 = load float, ptr %27, align 4, !tbaa !67
  %invariant.gep171 = getelementptr inbounds float, ptr %data_pad35, i64 4, !dbg !39
  %44 = load float, ptr %kernel_transform34, align 16, !tbaa !67
  %invariant.gep169 = getelementptr inbounds float, ptr %data_pad35, i64 3, !dbg !39
  %45 = load float, ptr %38, align 4, !tbaa !67
  %46 = load float, ptr %36, align 8, !tbaa !67
  %invariant.gep163 = getelementptr inbounds float, ptr %data_pad35, i64 23, !dbg !39
  %invariant.gep161 = getelementptr inbounds float, ptr %data_pad35, i64 22, !dbg !39
  %invariant.gep155 = getelementptr inbounds float, ptr %data_pad35, i64 3, !dbg !39
  %invariant.gep153 = getelementptr inbounds float, ptr %data_pad35, i64 2, !dbg !39
  %invariant.gep147 = getelementptr inbounds float, ptr %data_pad35, i64 22, !dbg !39
  %invariant.gep145 = getelementptr inbounds float, ptr %data_pad35, i64 21, !dbg !39
  %invariant.gep141 = getelementptr inbounds float, ptr %data_pad35, i64 2, !dbg !39
  %invariant.gep135 = getelementptr inbounds float, ptr %data_pad35, i64 21, !dbg !39
  %invariant.gep133 = getelementptr inbounds float, ptr %data_pad35, i64 20, !dbg !39
  %invariant.gep129 = getelementptr inbounds float, ptr %data_pad35, i64 1, !dbg !39
  %.pre111 = load float, ptr %38, align 4, !dbg !39, !tbaa !67
  %.pre110 = load float, ptr %36, align 8, !dbg !39, !tbaa !67
  %.pre109 = load float, ptr %34, align 4, !dbg !39, !tbaa !67
  %.pre108 = load float, ptr %32, align 16, !dbg !39, !tbaa !67
  %.pre107 = load float, ptr %30, align 4, !dbg !39, !tbaa !67
  %.pre106 = load float, ptr %29, align 8, !dbg !39, !tbaa !67
  %.pre105 = load float, ptr %27, align 4, !dbg !39, !tbaa !67
  %.pre = load float, ptr %kernel_transform34, align 16, !dbg !39, !tbaa !67
  call void @llvm.dbg.declare(metadata i64 0, metadata !66, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %47 = load float, ptr %data_pad35, align 16, !dbg !39, !tbaa !74
  %48 = call float @llvm.fmuladd.f32(float %47, float %.pre, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %49 = load float, ptr %invariant.gep129, align 4, !dbg !39, !tbaa !74
  %50 = call float @llvm.fmuladd.f32(float %49, float %.pre105, float %48), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %51 = getelementptr inbounds float, ptr %data_pad35, i64 5, !dbg !39
  %52 = load float, ptr %51, align 4, !dbg !39, !tbaa !74
  %53 = call float @llvm.fmuladd.f32(float %52, float %.pre106, float %50), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %gep132 = getelementptr inbounds float, ptr %data_pad35, i64 6, !dbg !39
  %54 = load float, ptr %gep132, align 8, !dbg !39, !tbaa !74
  %55 = call float @llvm.fmuladd.f32(float %54, float %.pre107, float %53), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %56 = load float, ptr %invariant.gep133, align 16, !dbg !39, !tbaa !74
  %57 = call float @llvm.fmuladd.f32(float %56, float %.pre108, float %55), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %58 = load float, ptr %invariant.gep135, align 4, !dbg !39, !tbaa !74
  %59 = call float @llvm.fmuladd.f32(float %58, float %.pre109, float %57), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %gep138 = getelementptr inbounds float, ptr %data_pad35, i64 25, !dbg !39
  %60 = load float, ptr %gep138, align 4, !dbg !39, !tbaa !74
  %61 = call float @llvm.fmuladd.f32(float %60, float %.pre110, float %59), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !73, metadata !DIExpression()), !dbg !39
  %gep140 = getelementptr inbounds float, ptr %data_pad35, i64 26, !dbg !39
  %62 = load float, ptr %gep140, align 8, !dbg !39, !tbaa !74
  %63 = call float @llvm.fmuladd.f32(float %62, float %.pre111, float %61), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %63, ptr %compute, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  %64 = getelementptr inbounds float, ptr %compute, i64 1
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %65 = call float @llvm.fmuladd.f32(float %49, float %.pre, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %66 = load float, ptr %invariant.gep141, align 8, !dbg !39, !tbaa !74
  %67 = call float @llvm.fmuladd.f32(float %66, float %.pre105, float %65), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %68 = call float @llvm.fmuladd.f32(float %54, float %.pre106, float %67), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %gep144 = getelementptr inbounds float, ptr %data_pad35, i64 7, !dbg !39
  %69 = load float, ptr %gep144, align 4, !dbg !39, !tbaa !74
  %70 = call float @llvm.fmuladd.f32(float %69, float %.pre107, float %68), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %71 = load float, ptr %invariant.gep145, align 4, !dbg !39, !tbaa !74
  %72 = call float @llvm.fmuladd.f32(float %71, float %.pre108, float %70), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %73 = load float, ptr %invariant.gep147, align 8, !dbg !39, !tbaa !74
  %74 = call float @llvm.fmuladd.f32(float %73, float %.pre109, float %72), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %gep150 = getelementptr inbounds float, ptr %data_pad35, i64 26, !dbg !39
  %75 = load float, ptr %gep150, align 8, !dbg !39, !tbaa !74
  %76 = call float @llvm.fmuladd.f32(float %75, float %.pre110, float %74), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !73, metadata !DIExpression()), !dbg !39
  %gep152 = getelementptr inbounds float, ptr %data_pad35, i64 27, !dbg !39
  %77 = load float, ptr %gep152, align 4, !dbg !39, !tbaa !74
  %78 = call float @llvm.fmuladd.f32(float %77, float %.pre111, float %76), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %78, ptr %64, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  %79 = getelementptr inbounds float, ptr %compute, i64 2
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %80 = load float, ptr %invariant.gep153, align 8, !dbg !39, !tbaa !74
  %81 = call float @llvm.fmuladd.f32(float %80, float %.pre, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %82 = load float, ptr %invariant.gep155, align 4, !dbg !39, !tbaa !74
  %83 = call float @llvm.fmuladd.f32(float %82, float %.pre105, float %81), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %gep158 = getelementptr inbounds float, ptr %data_pad35, i64 7, !dbg !39
  %84 = load float, ptr %gep158, align 4, !dbg !39, !tbaa !74
  %85 = call float @llvm.fmuladd.f32(float %84, float %.pre106, float %83), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %gep160 = getelementptr inbounds float, ptr %data_pad35, i64 8, !dbg !39
  %86 = load float, ptr %gep160, align 16, !dbg !39, !tbaa !74
  %87 = call float @llvm.fmuladd.f32(float %86, float %.pre107, float %85), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %88 = load float, ptr %invariant.gep161, align 8, !dbg !39, !tbaa !74
  %89 = call float @llvm.fmuladd.f32(float %88, float %.pre108, float %87), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %90 = load float, ptr %invariant.gep163, align 4, !dbg !39, !tbaa !74
  %91 = call float @llvm.fmuladd.f32(float %90, float %.pre109, float %89), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %gep166 = getelementptr inbounds float, ptr %data_pad35, i64 27, !dbg !39
  %92 = load float, ptr %gep166, align 4, !dbg !39, !tbaa !74
  %93 = call float @llvm.fmuladd.f32(float %92, float %46, float %91), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !73, metadata !DIExpression()), !dbg !39
  %gep168 = getelementptr inbounds float, ptr %data_pad35, i64 28, !dbg !39
  %94 = load float, ptr %gep168, align 16, !dbg !39, !tbaa !74
  %95 = call float @llvm.fmuladd.f32(float %94, float %45, float %93), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %95, ptr %79, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  %96 = getelementptr inbounds float, ptr %compute, i64 3
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %97 = load float, ptr %invariant.gep169, align 4, !dbg !39, !tbaa !74
  %98 = call float @llvm.fmuladd.f32(float %97, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %99 = load float, ptr %invariant.gep171, align 16, !dbg !39, !tbaa !74
  %100 = call float @llvm.fmuladd.f32(float %99, float %43, float %98), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %gep174 = getelementptr inbounds float, ptr %data_pad35, i64 8, !dbg !39
  %101 = load float, ptr %gep174, align 16, !dbg !39, !tbaa !74
  %102 = call float @llvm.fmuladd.f32(float %101, float %42, float %100), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %gep176 = getelementptr inbounds float, ptr %data_pad35, i64 9, !dbg !39
  %103 = load float, ptr %gep176, align 4, !dbg !39, !tbaa !74
  %104 = call float @llvm.fmuladd.f32(float %103, float %41, float %102), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %105 = load float, ptr %invariant.gep177, align 4, !dbg !39, !tbaa !74
  %106 = call float @llvm.fmuladd.f32(float %105, float %40, float %104), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %107 = load float, ptr %invariant.gep179, align 16, !dbg !39, !tbaa !74
  %108 = call float @llvm.fmuladd.f32(float %107, float %39, float %106), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %gep182 = getelementptr inbounds float, ptr %data_pad35, i64 28, !dbg !39
  %109 = load float, ptr %gep182, align 16, !dbg !39, !tbaa !74
  %110 = call float @llvm.fmuladd.f32(float %109, float %46, float %108), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !73, metadata !DIExpression()), !dbg !39
  %gep184 = getelementptr inbounds float, ptr %data_pad35, i64 29, !dbg !39
  %111 = load float, ptr %gep184, align 4, !dbg !39, !tbaa !74
  %112 = call float @llvm.fmuladd.f32(float %111, float %45, float %110), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %112, ptr %96, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 4, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !66, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !66, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !69, metadata !DIExpression()), !dbg !39
  %113 = getelementptr inbounds float, ptr %compute, i64 4
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %114 = getelementptr inbounds float, ptr %data_pad35, i64 5, !dbg !39
  %115 = load float, ptr %114, align 4, !dbg !39, !tbaa !74
  %116 = call float @llvm.fmuladd.f32(float %115, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep130.1 = getelementptr inbounds float, ptr %data_pad35, i64 6, !dbg !39
  %117 = load float, ptr %gep130.1, align 8, !dbg !39, !tbaa !74
  %118 = call float @llvm.fmuladd.f32(float %117, float %43, float %116), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %119 = getelementptr inbounds float, ptr %data_pad35, i64 10, !dbg !39
  %120 = load float, ptr %119, align 8, !dbg !39, !tbaa !74
  %121 = call float @llvm.fmuladd.f32(float %120, float %42, float %118), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep132.1 = getelementptr inbounds float, ptr %data_pad35, i64 11, !dbg !39
  %122 = load float, ptr %gep132.1, align 4, !dbg !39, !tbaa !74
  %123 = call float @llvm.fmuladd.f32(float %122, float %41, float %121), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep134.1 = getelementptr inbounds float, ptr %data_pad35, i64 25, !dbg !39
  %124 = load float, ptr %gep134.1, align 4, !dbg !39, !tbaa !74
  %125 = call float @llvm.fmuladd.f32(float %124, float %40, float %123), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep136.1 = getelementptr inbounds float, ptr %data_pad35, i64 26, !dbg !39
  %126 = load float, ptr %gep136.1, align 8, !dbg !39, !tbaa !74
  %127 = call float @llvm.fmuladd.f32(float %126, float %39, float %125), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep138.1 = getelementptr inbounds float, ptr %data_pad35, i64 30, !dbg !39
  %128 = load float, ptr %gep138.1, align 8, !dbg !39, !tbaa !74
  %129 = call float @llvm.fmuladd.f32(float %128, float %46, float %127), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 4, metadata !73, metadata !DIExpression()), !dbg !39
  %gep140.1 = getelementptr inbounds float, ptr %data_pad35, i64 31, !dbg !39
  %130 = load float, ptr %gep140.1, align 4, !dbg !39, !tbaa !74
  %131 = call float @llvm.fmuladd.f32(float %130, float %45, float %129), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %131, ptr %113, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  %132 = getelementptr inbounds float, ptr %compute, i64 5
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %133 = call float @llvm.fmuladd.f32(float %117, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep142.1 = getelementptr inbounds float, ptr %data_pad35, i64 7, !dbg !39
  %134 = load float, ptr %gep142.1, align 4, !dbg !39, !tbaa !74
  %135 = call float @llvm.fmuladd.f32(float %134, float %43, float %133), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %136 = call float @llvm.fmuladd.f32(float %122, float %42, float %135), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep144.1 = getelementptr inbounds float, ptr %data_pad35, i64 12, !dbg !39
  %137 = load float, ptr %gep144.1, align 16, !dbg !39, !tbaa !74
  %138 = call float @llvm.fmuladd.f32(float %137, float %41, float %136), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep146.1 = getelementptr inbounds float, ptr %data_pad35, i64 26, !dbg !39
  %139 = load float, ptr %gep146.1, align 8, !dbg !39, !tbaa !74
  %140 = call float @llvm.fmuladd.f32(float %139, float %40, float %138), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep148.1 = getelementptr inbounds float, ptr %data_pad35, i64 27, !dbg !39
  %141 = load float, ptr %gep148.1, align 4, !dbg !39, !tbaa !74
  %142 = call float @llvm.fmuladd.f32(float %141, float %39, float %140), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep150.1 = getelementptr inbounds float, ptr %data_pad35, i64 31, !dbg !39
  %143 = load float, ptr %gep150.1, align 4, !dbg !39, !tbaa !74
  %144 = call float @llvm.fmuladd.f32(float %143, float %46, float %142), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 5, metadata !73, metadata !DIExpression()), !dbg !39
  %gep152.1 = getelementptr inbounds float, ptr %data_pad35, i64 32, !dbg !39
  %145 = load float, ptr %gep152.1, align 16, !dbg !39, !tbaa !74
  %146 = call float @llvm.fmuladd.f32(float %145, float %45, float %144), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %146, ptr %132, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  %147 = getelementptr inbounds float, ptr %compute, i64 6
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep154.1 = getelementptr inbounds float, ptr %data_pad35, i64 7, !dbg !39
  %148 = load float, ptr %gep154.1, align 4, !dbg !39, !tbaa !74
  %149 = call float @llvm.fmuladd.f32(float %148, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep156.1 = getelementptr inbounds float, ptr %data_pad35, i64 8, !dbg !39
  %150 = load float, ptr %gep156.1, align 16, !dbg !39, !tbaa !74
  %151 = call float @llvm.fmuladd.f32(float %150, float %43, float %149), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep158.1 = getelementptr inbounds float, ptr %data_pad35, i64 12, !dbg !39
  %152 = load float, ptr %gep158.1, align 16, !dbg !39, !tbaa !74
  %153 = call float @llvm.fmuladd.f32(float %152, float %42, float %151), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep160.1 = getelementptr inbounds float, ptr %data_pad35, i64 13, !dbg !39
  %154 = load float, ptr %gep160.1, align 4, !dbg !39, !tbaa !74
  %155 = call float @llvm.fmuladd.f32(float %154, float %41, float %153), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep162.1 = getelementptr inbounds float, ptr %data_pad35, i64 27, !dbg !39
  %156 = load float, ptr %gep162.1, align 4, !dbg !39, !tbaa !74
  %157 = call float @llvm.fmuladd.f32(float %156, float %40, float %155), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep164.1 = getelementptr inbounds float, ptr %data_pad35, i64 28, !dbg !39
  %158 = load float, ptr %gep164.1, align 16, !dbg !39, !tbaa !74
  %159 = call float @llvm.fmuladd.f32(float %158, float %39, float %157), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep166.1 = getelementptr inbounds float, ptr %data_pad35, i64 32, !dbg !39
  %160 = load float, ptr %gep166.1, align 16, !dbg !39, !tbaa !74
  %161 = call float @llvm.fmuladd.f32(float %160, float %46, float %159), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 6, metadata !73, metadata !DIExpression()), !dbg !39
  %gep168.1 = getelementptr inbounds float, ptr %data_pad35, i64 33, !dbg !39
  %162 = load float, ptr %gep168.1, align 4, !dbg !39, !tbaa !74
  %163 = call float @llvm.fmuladd.f32(float %162, float %45, float %161), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %163, ptr %147, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  %164 = getelementptr inbounds float, ptr %compute, i64 7
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep170.1 = getelementptr inbounds float, ptr %data_pad35, i64 8, !dbg !39
  %165 = load float, ptr %gep170.1, align 16, !dbg !39, !tbaa !74
  %166 = call float @llvm.fmuladd.f32(float %165, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep172.1 = getelementptr inbounds float, ptr %data_pad35, i64 9, !dbg !39
  %167 = load float, ptr %gep172.1, align 4, !dbg !39, !tbaa !74
  %168 = call float @llvm.fmuladd.f32(float %167, float %43, float %166), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep174.1 = getelementptr inbounds float, ptr %data_pad35, i64 13, !dbg !39
  %169 = load float, ptr %gep174.1, align 4, !dbg !39, !tbaa !74
  %170 = call float @llvm.fmuladd.f32(float %169, float %42, float %168), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep176.1 = getelementptr inbounds float, ptr %data_pad35, i64 14, !dbg !39
  %171 = load float, ptr %gep176.1, align 8, !dbg !39, !tbaa !74
  %172 = call float @llvm.fmuladd.f32(float %171, float %41, float %170), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep178.1 = getelementptr inbounds float, ptr %data_pad35, i64 28, !dbg !39
  %173 = load float, ptr %gep178.1, align 16, !dbg !39, !tbaa !74
  %174 = call float @llvm.fmuladd.f32(float %173, float %40, float %172), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep180.1 = getelementptr inbounds float, ptr %data_pad35, i64 29, !dbg !39
  %175 = load float, ptr %gep180.1, align 4, !dbg !39, !tbaa !74
  %176 = call float @llvm.fmuladd.f32(float %175, float %39, float %174), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep182.1 = getelementptr inbounds float, ptr %data_pad35, i64 33, !dbg !39
  %177 = load float, ptr %gep182.1, align 4, !dbg !39, !tbaa !74
  %178 = call float @llvm.fmuladd.f32(float %177, float %46, float %176), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 7, metadata !73, metadata !DIExpression()), !dbg !39
  %gep184.1 = getelementptr inbounds float, ptr %data_pad35, i64 34, !dbg !39
  %179 = load float, ptr %gep184.1, align 8, !dbg !39, !tbaa !74
  %180 = call float @llvm.fmuladd.f32(float %179, float %45, float %178), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %180, ptr %164, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 4, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !66, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !66, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !69, metadata !DIExpression()), !dbg !39
  %181 = getelementptr inbounds float, ptr %compute, i64 8
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %182 = getelementptr inbounds float, ptr %data_pad35, i64 10, !dbg !39
  %183 = load float, ptr %182, align 8, !dbg !39, !tbaa !74
  %184 = call float @llvm.fmuladd.f32(float %183, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep130.2 = getelementptr inbounds float, ptr %data_pad35, i64 11, !dbg !39
  %185 = load float, ptr %gep130.2, align 4, !dbg !39, !tbaa !74
  %186 = call float @llvm.fmuladd.f32(float %185, float %43, float %184), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %187 = getelementptr inbounds float, ptr %data_pad35, i64 15, !dbg !39
  %188 = load float, ptr %187, align 4, !dbg !39, !tbaa !74
  %189 = call float @llvm.fmuladd.f32(float %188, float %42, float %186), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep132.2 = getelementptr inbounds float, ptr %data_pad35, i64 16, !dbg !39
  %190 = load float, ptr %gep132.2, align 16, !dbg !39, !tbaa !74
  %191 = call float @llvm.fmuladd.f32(float %190, float %41, float %189), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep134.2 = getelementptr inbounds float, ptr %data_pad35, i64 30, !dbg !39
  %192 = load float, ptr %gep134.2, align 8, !dbg !39, !tbaa !74
  %193 = call float @llvm.fmuladd.f32(float %192, float %40, float %191), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep136.2 = getelementptr inbounds float, ptr %data_pad35, i64 31, !dbg !39
  %194 = load float, ptr %gep136.2, align 4, !dbg !39, !tbaa !74
  %195 = call float @llvm.fmuladd.f32(float %194, float %39, float %193), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep138.2 = getelementptr inbounds float, ptr %data_pad35, i64 35, !dbg !39
  %196 = load float, ptr %gep138.2, align 4, !dbg !39, !tbaa !74
  %197 = call float @llvm.fmuladd.f32(float %196, float %46, float %195), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 8, metadata !73, metadata !DIExpression()), !dbg !39
  %gep140.2 = getelementptr inbounds float, ptr %data_pad35, i64 36, !dbg !39
  %198 = load float, ptr %gep140.2, align 16, !dbg !39, !tbaa !74
  %199 = call float @llvm.fmuladd.f32(float %198, float %45, float %197), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %199, ptr %181, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !69, metadata !DIExpression()), !dbg !39
  %200 = getelementptr inbounds float, ptr %compute, i64 9
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %201 = call float @llvm.fmuladd.f32(float %185, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep142.2 = getelementptr inbounds float, ptr %data_pad35, i64 12, !dbg !39
  %202 = load float, ptr %gep142.2, align 16, !dbg !39, !tbaa !74
  %203 = call float @llvm.fmuladd.f32(float %202, float %43, float %201), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %204 = call float @llvm.fmuladd.f32(float %190, float %42, float %203), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep144.2 = getelementptr inbounds float, ptr %data_pad35, i64 17, !dbg !39
  %205 = load float, ptr %gep144.2, align 4, !dbg !39, !tbaa !74
  %206 = call float @llvm.fmuladd.f32(float %205, float %41, float %204), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep146.2 = getelementptr inbounds float, ptr %data_pad35, i64 31, !dbg !39
  %207 = load float, ptr %gep146.2, align 4, !dbg !39, !tbaa !74
  %208 = call float @llvm.fmuladd.f32(float %207, float %40, float %206), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep148.2 = getelementptr inbounds float, ptr %data_pad35, i64 32, !dbg !39
  %209 = load float, ptr %gep148.2, align 16, !dbg !39, !tbaa !74
  %210 = call float @llvm.fmuladd.f32(float %209, float %39, float %208), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep150.2 = getelementptr inbounds float, ptr %data_pad35, i64 36, !dbg !39
  %211 = load float, ptr %gep150.2, align 16, !dbg !39, !tbaa !74
  %212 = call float @llvm.fmuladd.f32(float %211, float %46, float %210), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 9, metadata !73, metadata !DIExpression()), !dbg !39
  %gep152.2 = getelementptr inbounds float, ptr %data_pad35, i64 37, !dbg !39
  %213 = load float, ptr %gep152.2, align 4, !dbg !39, !tbaa !74
  %214 = call float @llvm.fmuladd.f32(float %213, float %45, float %212), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %214, ptr %200, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !69, metadata !DIExpression()), !dbg !39
  %215 = getelementptr inbounds float, ptr %compute, i64 10
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep154.2 = getelementptr inbounds float, ptr %data_pad35, i64 12, !dbg !39
  %216 = load float, ptr %gep154.2, align 16, !dbg !39, !tbaa !74
  %217 = call float @llvm.fmuladd.f32(float %216, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep156.2 = getelementptr inbounds float, ptr %data_pad35, i64 13, !dbg !39
  %218 = load float, ptr %gep156.2, align 4, !dbg !39, !tbaa !74
  %219 = call float @llvm.fmuladd.f32(float %218, float %43, float %217), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep158.2 = getelementptr inbounds float, ptr %data_pad35, i64 17, !dbg !39
  %220 = load float, ptr %gep158.2, align 4, !dbg !39, !tbaa !74
  %221 = call float @llvm.fmuladd.f32(float %220, float %42, float %219), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep160.2 = getelementptr inbounds float, ptr %data_pad35, i64 18, !dbg !39
  %222 = load float, ptr %gep160.2, align 8, !dbg !39, !tbaa !74
  %223 = call float @llvm.fmuladd.f32(float %222, float %41, float %221), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep162.2 = getelementptr inbounds float, ptr %data_pad35, i64 32, !dbg !39
  %224 = load float, ptr %gep162.2, align 16, !dbg !39, !tbaa !74
  %225 = call float @llvm.fmuladd.f32(float %224, float %40, float %223), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep164.2 = getelementptr inbounds float, ptr %data_pad35, i64 33, !dbg !39
  %226 = load float, ptr %gep164.2, align 4, !dbg !39, !tbaa !74
  %227 = call float @llvm.fmuladd.f32(float %226, float %39, float %225), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep166.2 = getelementptr inbounds float, ptr %data_pad35, i64 37, !dbg !39
  %228 = load float, ptr %gep166.2, align 4, !dbg !39, !tbaa !74
  %229 = call float @llvm.fmuladd.f32(float %228, float %46, float %227), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 10, metadata !73, metadata !DIExpression()), !dbg !39
  %gep168.2 = getelementptr inbounds float, ptr %data_pad35, i64 38, !dbg !39
  %230 = load float, ptr %gep168.2, align 8, !dbg !39, !tbaa !74
  %231 = call float @llvm.fmuladd.f32(float %230, float %45, float %229), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %231, ptr %215, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !69, metadata !DIExpression()), !dbg !39
  %232 = getelementptr inbounds float, ptr %compute, i64 11
  call void @llvm.dbg.declare(metadata i32 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep170.2 = getelementptr inbounds float, ptr %data_pad35, i64 13, !dbg !39
  %233 = load float, ptr %gep170.2, align 4, !dbg !39, !tbaa !74
  %234 = call float @llvm.fmuladd.f32(float %233, float %44, float 0.000000e+00), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep172.2 = getelementptr inbounds float, ptr %data_pad35, i64 14, !dbg !39
  %235 = load float, ptr %gep172.2, align 8, !dbg !39, !tbaa !74
  %236 = call float @llvm.fmuladd.f32(float %235, float %43, float %234), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep174.2 = getelementptr inbounds float, ptr %data_pad35, i64 18, !dbg !39
  %237 = load float, ptr %gep174.2, align 8, !dbg !39, !tbaa !74
  %238 = call float @llvm.fmuladd.f32(float %237, float %42, float %236), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep176.2 = getelementptr inbounds float, ptr %data_pad35, i64 19, !dbg !39
  %239 = load float, ptr %gep176.2, align 4, !dbg !39, !tbaa !74
  %240 = call float @llvm.fmuladd.f32(float %239, float %41, float %238), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !70, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep178.2 = getelementptr inbounds float, ptr %data_pad35, i64 33, !dbg !39
  %241 = load float, ptr %gep178.2, align 4, !dbg !39, !tbaa !74
  %242 = call float @llvm.fmuladd.f32(float %241, float %40, float %240), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep180.2 = getelementptr inbounds float, ptr %data_pad35, i64 34, !dbg !39
  %243 = load float, ptr %gep180.2, align 8, !dbg !39, !tbaa !74
  %244 = call float @llvm.fmuladd.f32(float %243, float %39, float %242), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i32 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 0, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep182.2 = getelementptr inbounds float, ptr %data_pad35, i64 38, !dbg !39
  %245 = load float, ptr %gep182.2, align 8, !dbg !39, !tbaa !74
  %246 = call float @llvm.fmuladd.f32(float %245, float %46, float %244), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 1, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 11, metadata !73, metadata !DIExpression()), !dbg !39
  %gep184.2 = getelementptr inbounds float, ptr %data_pad35, i64 39, !dbg !39
  %247 = load float, ptr %gep184.2, align 4, !dbg !39, !tbaa !74
  %248 = call float @llvm.fmuladd.f32(float %247, float %45, float %246), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !72, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !71, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 2, metadata !70, metadata !DIExpression()), !dbg !39
  store float %248, ptr %232, align 4, !tbaa !76
  call void @llvm.dbg.declare(metadata i64 4, metadata !69, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata i64 3, metadata !66, metadata !DIExpression()), !dbg !39
  call void @tvm_gemmini_end(), !dbg !39
  br label %common.ret
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dllexport noundef i32 @stage_4_operator_sum(ptr noalias nocapture readonly %x, ptr noalias nocapture writeonly %x_red) local_unnamed_addr #1 !dbg !78 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !82, metadata !DIExpression()), !dbg !84
  tail call void @llvm.dbg.value(metadata ptr %x_red, metadata !83, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !87, metadata !DIExpression()), !dbg !84
  %0 = load float, ptr %x, align 4, !dbg !84, !tbaa !88
  %1 = fadd float %0, 0.000000e+00, !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  %2 = getelementptr inbounds float, ptr %x, i64 1, !dbg !84
  %3 = load float, ptr %2, align 4, !dbg !84, !tbaa !88
  %4 = fadd float %3, %1, !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  %5 = getelementptr inbounds float, ptr %x, i64 2, !dbg !84
  %6 = load float, ptr %5, align 4, !dbg !84, !tbaa !88
  %7 = fadd float %6, %4, !dbg !84
  call void @llvm.dbg.declare(metadata i64 3, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !87, metadata !DIExpression()), !dbg !84
  %8 = getelementptr inbounds float, ptr %x, i64 3, !dbg !84
  %9 = load float, ptr %8, align 4, !dbg !84, !tbaa !88
  %10 = fadd float %9, %7, !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  %11 = getelementptr inbounds float, ptr %x, i64 4, !dbg !84
  %12 = load float, ptr %11, align 4, !dbg !84, !tbaa !88
  %13 = fadd float %12, %10, !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  %14 = getelementptr inbounds float, ptr %x, i64 5, !dbg !84
  %15 = load float, ptr %14, align 4, !dbg !84, !tbaa !88
  %16 = fadd float %15, %13, !dbg !84
  call void @llvm.dbg.declare(metadata i64 3, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !86, metadata !DIExpression()), !dbg !84
  store float %16, ptr %x_red, align 4, !tbaa !90
  call void @llvm.dbg.declare(metadata i64 1, metadata !85, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !85, metadata !DIExpression()), !dbg !84
  %17 = getelementptr inbounds float, ptr %x_red, i64 1
  call void @llvm.dbg.declare(metadata i32 0, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !87, metadata !DIExpression()), !dbg !84
  %18 = getelementptr inbounds float, ptr %x, i64 6, !dbg !84
  %19 = load float, ptr %18, align 4, !dbg !84, !tbaa !88
  %20 = fadd float %19, 0.000000e+00, !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  %21 = getelementptr inbounds float, ptr %x, i64 7, !dbg !84
  %22 = load float, ptr %21, align 4, !dbg !84, !tbaa !88
  %23 = fadd float %22, %20, !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  %24 = getelementptr inbounds float, ptr %x, i64 8, !dbg !84
  %25 = load float, ptr %24, align 4, !dbg !84, !tbaa !88
  %26 = fadd float %25, %23, !dbg !84
  call void @llvm.dbg.declare(metadata i64 3, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !86, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i32 0, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 0, metadata !87, metadata !DIExpression()), !dbg !84
  %27 = getelementptr inbounds float, ptr %x, i64 9, !dbg !84
  %28 = load float, ptr %27, align 4, !dbg !84, !tbaa !88
  %29 = fadd float %28, %26, !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 1, metadata !87, metadata !DIExpression()), !dbg !84
  %30 = getelementptr inbounds float, ptr %x, i64 10, !dbg !84
  %31 = load float, ptr %30, align 4, !dbg !84, !tbaa !88
  %32 = fadd float %31, %29, !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !87, metadata !DIExpression()), !dbg !84
  %33 = getelementptr inbounds float, ptr %x, i64 11, !dbg !84
  %34 = load float, ptr %33, align 4, !dbg !84, !tbaa !88
  %35 = fadd float %34, %32, !dbg !84
  call void @llvm.dbg.declare(metadata i64 3, metadata !87, metadata !DIExpression()), !dbg !84
  call void @llvm.dbg.declare(metadata i64 2, metadata !86, metadata !DIExpression()), !dbg !84
  store float %35, ptr %17, align 4, !tbaa !90
  call void @llvm.dbg.declare(metadata i64 2, metadata !85, metadata !DIExpression()), !dbg !84
  ret i32 0, !dbg !84
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare i32 @tvm_gemmini_validate_tensor_buffers(ptr, i64, ptr, i64, ptr, i64) local_unnamed_addr

declare void @TVMAPISetLastError(ptr) local_unnamed_addr

declare void @tvm_gemmini_begin(i64, i64, i64) local_unnamed_addr

declare void @tvm_gemmini_dma_acc_write(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_out(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_end() local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #3 section ".text.tvm.fp16.conv" {
b0:
  %v0 = bitcast float %a0 to i32
  %0 = tail call float @llvm.fabs.f32(float %a0)
  %v1 = bitcast float %0 to i32
  %v2 = add nsw i32 %v1, -947912704
  %v3 = add nsw i32 %v1, -1199570944
  %v4 = icmp ult i32 %v2, %v3
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = lshr i32 %v0, 13
  %v6 = and i32 %v5, 65535
  %v7 = add nuw nsw i32 %v6, -114688
  %v8 = and i32 %v0, 8191
  %v9 = icmp ugt i32 %v8, 4096
  br i1 %v9, label %b2, label %b3

b2:                                               ; preds = %b1
  %v10 = add nuw nsw i32 %v6, -114687
  br label %b13

b3:                                               ; preds = %b1
  %v11 = icmp eq i32 %v8, 4096
  br i1 %v11, label %b4, label %b13

b4:                                               ; preds = %b3
  %v12 = and i32 %v7, 65535
  %v13 = and i32 %v5, 1
  %v14 = add nuw nsw i32 %v12, %v13
  br label %b13

b5:                                               ; preds = %b0
  %v15 = icmp ugt i32 %v1, 2139095040
  br i1 %v15, label %b6, label %b7

b6:                                               ; preds = %b5
  %v16 = lshr i32 %v0, 13
  %v17 = and i32 %v16, 511
  %v18 = or disjoint i32 %v17, 32256
  br label %b13

b7:                                               ; preds = %b5
  %v19 = icmp ugt i32 %v1, 1199570943
  br i1 %v19, label %b13, label %b8

b8:                                               ; preds = %b7
  %v20 = icmp ult i32 %v1, 754974720
  br i1 %v20, label %b13, label %b9

b9:                                               ; preds = %b8
  %v21 = lshr i32 %v1, 23
  %v22 = sub nsw i32 113, %v21
  %v23 = and i32 %v0, 8388607
  %v24 = or disjoint i32 %v23, 8388608
  %v25 = add nsw i32 %v21, -81
  %v26 = shl i32 %v24, %v25
  %v27 = icmp ne i32 %v26, 0
  %v28 = lshr i32 %v24, %v22
  %v29 = zext i1 %v27 to i32
  %v30 = lshr i32 %v28, 13
  %v31 = and i32 %v28, 8191
  %v32 = or i32 %v31, %v29
  %v33 = icmp ugt i32 %v32, 4096
  br i1 %v33, label %b10, label %b11

b10:                                              ; preds = %b9
  %v34 = add nuw nsw i32 %v30, 1
  br label %b13

b11:                                              ; preds = %b9
  %v35 = icmp eq i32 %v32, 4096
  br i1 %v35, label %b12, label %b13

b12:                                              ; preds = %b11
  %v36 = and i32 %v30, 1
  %v37 = add nuw nsw i32 %v36, %v30
  br label %b13

b13:                                              ; preds = %b12, %b11, %b10, %b8, %b7, %b6, %b4, %b3, %b2
  %v38 = phi i32 [ %v18, %b6 ], [ %v10, %b2 ], [ %v14, %b4 ], [ %v7, %b3 ], [ 31744, %b7 ], [ 0, %b8 ], [ %v34, %b10 ], [ %v37, %b12 ], [ %v30, %b11 ]
  %v39 = lshr i32 %v0, 16
  %v40 = and i32 %v39, 32768
  %v41 = or i32 %v38, %v40
  %vlast = trunc i32 %v41 to i16
  ret i16 %vlast
}

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #3 section ".text.tvm.fp16.conv" {
b0:
  %v1 = and i16 %a0, 32767
  %v2 = zext nneg i16 %v1 to i32
  %v3 = add nsw i16 %v1, -1024
  %v4 = icmp ult i16 %v3, 30720
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  %v5 = shl nuw nsw i32 %v2, 13
  %v6 = add nuw nsw i32 %v5, 939524096
  br label %b6

b2:                                               ; preds = %b0
  %v7 = icmp ugt i16 %v1, 31743
  br i1 %v7, label %b3, label %b4

b3:                                               ; preds = %b2
  %v8 = shl nuw nsw i32 %v2, 13
  %v9 = or i32 %v8, 2139095040
  br label %b6

b4:                                               ; preds = %b2
  %v10 = icmp eq i16 %v1, 0
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b4
  %v11 = icmp ult i16 %v1, 256
  %v12 = lshr i32 %v2, 8
  %v13 = select i1 %v11, i32 %v2, i32 %v12
  %v14 = select i1 %v11, i32 32, i32 24
  %v15 = icmp ult i32 %v13, 16
  %v16 = lshr i32 %v13, 4
  %v17 = add nsw i32 %v14, -4
  %v18 = select i1 %v15, i32 %v13, i32 %v16
  %v19 = select i1 %v15, i32 %v14, i32 %v17
  %v20 = icmp ult i32 %v18, 4
  %v21 = lshr i32 %v18, 2
  %v22 = add nsw i32 %v19, -2
  %v23 = select i1 %v20, i32 %v18, i32 %v21
  %v24 = select i1 %v20, i32 %v19, i32 %v22
  %v25 = icmp ult i32 %v23, 2
  %v26 = sub nsw i32 0, %v23
  %v27 = select i1 %v25, i32 %v26, i32 -2
  %v28 = add nsw i32 %v27, %v24
  %v29 = add nsw i32 %v28, -8
  %v30 = shl i32 %v2, %v29
  %v31 = xor i32 %v30, 8388608
  %v32 = shl i32 %v28, 23
  %v33 = sub i32 1124073472, %v32
  %v34 = or i32 %v31, %v33
  br label %b6

b6:                                               ; preds = %b5, %b4, %b3, %b1
  %v35 = phi i32 [ %v6, %b1 ], [ %v9, %b3 ], [ %v34, %b5 ], [ 0, %b4 ]
  %v36 = and i16 %a0, -32768
  %v37 = zext i16 %v36 to i32
  %v38 = shl nuw i32 %v37, 16
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  ret float %v40
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #4

attributes #0 = { "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nosync nounwind memory(none) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d -fp-contract=off"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 4}
!5 = distinct !DISubprogram(name: "stage_4_operator_conv2d1", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "y", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "x", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocalVariable(name: "conv2d_cnhw", arg: 3, scope: !5, file: !1, type: !9)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocalVariable(name: "pad_temp", scope: !5, file: !1, type: !9)
!17 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!18 = !DILocalVariable(name: "i2_0", scope: !5, file: !1, type: !8)
!19 = !DILocalVariable(name: "i3_0", scope: !5, file: !1, type: !8)
!20 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!21 = !DILocalVariable(name: "ff", scope: !5, file: !1, type: !8)
!22 = !DILocalVariable(name: "yy", scope: !5, file: !1, type: !8)
!23 = !DILocalVariable(name: "xx", scope: !5, file: !1, type: !8)
!24 = !DILocalVariable(name: "ry", scope: !5, file: !1, type: !8)
!25 = !DILocalVariable(name: "rx", scope: !5, file: !1, type: !8)
!26 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!27 = !{!28, !28, i64 0}
!28 = !{!"0x40bb3ef0", !29, i64 0}
!29 = !{!"tvm-tbaa"}
!30 = !{!31, !31, i64 0}
!31 = !{!"0x40bf2160", !29, i64 0}
!32 = !{!33, !33, i64 0}
!33 = !{!"0x40bd2cc0", !29, i64 0}
!34 = distinct !DISubprogram(name: "stage_4_operator_conv2d_transpose", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !35)
!35 = !{!36, !37, !38}
!36 = !DILocalVariable(name: "x", arg: 1, scope: !34, file: !1, type: !9)
!37 = !DILocalVariable(name: "w", arg: 2, scope: !34, file: !1, type: !9)
!38 = !DILocalVariable(name: "compute", arg: 3, scope: !34, file: !1, type: !9)
!39 = !DILocation(line: 0, scope: !34)
!40 = !DILocalVariable(name: "data_dilate", scope: !34, file: !1, type: !9)
!41 = !DILocalVariable(name: "data_pad", scope: !34, file: !1, type: !9)
!42 = !DILocalVariable(name: "kernel_transform", scope: !34, file: !1, type: !9)
!43 = !DILocalVariable(name: "gemmini_admission", scope: !34, file: !1, type: !8)
!44 = !DILocalVariable(name: "i1", scope: !34, file: !1, type: !8)
!45 = !DILocalVariable(name: "i2_0", scope: !34, file: !1, type: !8)
!46 = !DILocalVariable(name: "i3_0", scope: !34, file: !1, type: !8)
!47 = !DILocalVariable(name: "cse_var_1", scope: !34, file: !1, type: !8)
!48 = !DILocalVariable(name: "scalar", scope: !34, file: !1, type: !9)
!49 = !{!50, !50, i64 0}
!50 = !{!"0x40c46850.w4.b0", !51, i64 0}
!51 = !{!"0x40c46850.w8.b0", !52, i64 0}
!52 = !{!"0x40c46850.w16.b0", !53, i64 0}
!53 = !{!"0x40c46850.w32.b0", !54, i64 0}
!54 = !{!"0x40c46850.w64.b0", !55, i64 0}
!55 = !{!"0x40c46850.w128.b0", !56, i64 0}
!56 = !{!"0x40c46850.w256.b0", !57, i64 0}
!57 = !{!"0x40c46850.w512.b0", !58, i64 0}
!58 = !{!"0x40c46850.w1024.b0", !59, i64 0}
!59 = !{!"0x40c46850", !29, i64 0}
!60 = !{!"branch_weights", i32 1, i32 1048575}
!61 = !DILocalVariable(name: "i", scope: !34, file: !1, type: !8)
!62 = !DILocalVariable(name: "h_0", scope: !34, file: !1, type: !8)
!63 = !DILocalVariable(name: "w_0", scope: !34, file: !1, type: !8)
!64 = !DILocalVariable(name: "cse_var_3", scope: !34, file: !1, type: !8)
!65 = !DILocalVariable(name: "cse_var_2", scope: !34, file: !1, type: !8)
!66 = !DILocalVariable(name: "h", scope: !34, file: !1, type: !8)
!67 = !{!68, !68, i64 0}
!68 = !{!"0x40befd40", !29, i64 0}
!69 = !DILocalVariable(name: "w", scope: !34, file: !1, type: !8)
!70 = !DILocalVariable(name: "dc", scope: !34, file: !1, type: !8)
!71 = !DILocalVariable(name: "dh", scope: !34, file: !1, type: !8)
!72 = !DILocalVariable(name: "dw", scope: !34, file: !1, type: !8)
!73 = !DILocalVariable(name: "cse_var_4", scope: !34, file: !1, type: !8)
!74 = !{!75, !75, i64 0}
!75 = !{!"0x40bf1cb0", !29, i64 0}
!76 = !{!77, !77, i64 0}
!77 = !{!"0x40c14c90", !29, i64 0}
!78 = distinct !DISubprogram(name: "stage_4_operator_sum", scope: !1, file: !1, type: !79, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !81)
!79 = !DISubroutineType(types: !80)
!80 = !{!8, !9, !9}
!81 = !{!82, !83}
!82 = !DILocalVariable(name: "x", arg: 1, scope: !78, file: !1, type: !9)
!83 = !DILocalVariable(name: "x_red", arg: 2, scope: !78, file: !1, type: !9)
!84 = !DILocation(line: 0, scope: !78)
!85 = !DILocalVariable(name: "ax0", scope: !78, file: !1, type: !8)
!86 = !DILocalVariable(name: "k2", scope: !78, file: !1, type: !8)
!87 = !DILocalVariable(name: "k3", scope: !78, file: !1, type: !8)
!88 = !{!89, !89, i64 0}
!89 = !{!"0x40c03810", !29, i64 0}
!90 = !{!91, !91, i64 0}
!91 = !{!"0x40bafb00", !29, i64 0}
