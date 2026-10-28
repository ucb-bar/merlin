; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_3_operator_fused_conv2d_cast_add_cast1(ptr noalias %x, ptr noalias %y, ptr noalias %lv1, ptr noalias %compute) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %y, metadata !13, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %lv1, metadata !14, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %compute, metadata !15, metadata !DIExpression()), !dbg !16
  %pad_temp_reindex38 = alloca [24 x i8], align 16, !dbg !16
  %conv2d_nchw_reindex42 = alloca [12 x i32], align 16, !dbg !16
  call void @llvm.dbg.declare(metadata ptr %conv2d_nchw_reindex42, metadata !17, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %pad_temp_reindex38, metadata !19, metadata !DIExpression()), !dbg !16
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %lv1, i64 2, ptr %lv1, i64 2, ptr %compute, i64 12), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !20, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !20, metadata !DIExpression()), !dbg !16
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !16
  br i1 %.not, label %if_else, label %if_then, !dbg !16

common.ret:                                       ; preds = %if_then10, %if_then4, %if_else12, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %if_else12 ], [ -1, %if_then4 ], [ -1, %if_then10 ]
  ret i32 %common.ret.op, !dbg !16

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !16
  br label %common.ret

if_else:                                          ; preds = %entry
  %gemmini_admission3 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %y, i64 8, ptr %y, i64 8, ptr %compute, i64 12), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission3, metadata !20, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission3, metadata !20, metadata !DIExpression()), !dbg !16
  %.not43 = icmp eq i32 %gemmini_admission3, 0, !dbg !16
  br i1 %.not43, label %if_else6, label %if_then4, !dbg !16

if_then4:                                         ; preds = %if_else
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !16
  br label %common.ret

if_else6:                                         ; preds = %if_else
  %gemmini_admission9 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %x, i64 12, ptr %x, i64 12, ptr %compute, i64 12), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission9, metadata !20, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission9, metadata !20, metadata !DIExpression()), !dbg !16
  %.not44 = icmp eq i32 %gemmini_admission9, 0, !dbg !16
  br i1 %.not44, label %if_else12, label %if_then10, !dbg !16

if_then10:                                        ; preds = %if_else6
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !16
  br label %common.ret

if_else12:                                        ; preds = %if_else6
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !16
  tail call void @tvm_gemmini_zero_acc(i32 0, i32 6, i32 2), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %0 = load i8, ptr %x, align 1, !dbg !16, !tbaa !23
  store i8 %0, ptr %pad_temp_reindex38, align 16, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %1 = getelementptr inbounds i8, ptr %x, i64 1, !dbg !16
  %2 = load i8, ptr %1, align 1, !dbg !16, !tbaa !23
  %3 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 1, !dbg !16
  store i8 %2, ptr %3, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %4 = getelementptr inbounds i8, ptr %x, i64 4, !dbg !16
  %5 = load i8, ptr %4, align 1, !dbg !16, !tbaa !23
  %6 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 2, !dbg !16
  store i8 %5, ptr %6, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %7 = getelementptr inbounds i8, ptr %x, i64 5, !dbg !16
  %8 = load i8, ptr %7, align 1, !dbg !16, !tbaa !23
  %9 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 3, !dbg !16
  store i8 %8, ptr %9, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %10 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 4, !dbg !16
  store i8 %2, ptr %10, align 4, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %11 = getelementptr inbounds i8, ptr %x, i64 2, !dbg !16
  %12 = load i8, ptr %11, align 1, !dbg !16, !tbaa !23
  %13 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 5, !dbg !16
  store i8 %12, ptr %13, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %14 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 6, !dbg !16
  store i8 %8, ptr %14, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %15 = getelementptr inbounds i8, ptr %x, i64 6, !dbg !16
  %16 = load i8, ptr %15, align 1, !dbg !16, !tbaa !23
  %17 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 7, !dbg !16
  store i8 %16, ptr %17, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %18 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 8, !dbg !16
  store i8 %12, ptr %18, align 8, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %19 = getelementptr inbounds i8, ptr %x, i64 3, !dbg !16
  %20 = load i8, ptr %19, align 1, !dbg !16, !tbaa !23
  %21 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 9, !dbg !16
  store i8 %20, ptr %21, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %22 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 10, !dbg !16
  store i8 %16, ptr %22, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %23 = getelementptr inbounds i8, ptr %x, i64 7, !dbg !16
  %24 = load i8, ptr %23, align 1, !dbg !16, !tbaa !23
  %25 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 11, !dbg !16
  store i8 %24, ptr %25, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %26 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 12, !dbg !16
  store i8 %5, ptr %26, align 4, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %27 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 13, !dbg !16
  store i8 %8, ptr %27, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %28 = getelementptr inbounds i8, ptr %x, i64 8, !dbg !16
  %29 = load i8, ptr %28, align 1, !dbg !16, !tbaa !23
  %30 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 14, !dbg !16
  store i8 %29, ptr %30, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %31 = getelementptr inbounds i8, ptr %x, i64 9, !dbg !16
  %32 = load i8, ptr %31, align 1, !dbg !16, !tbaa !23
  %33 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 15, !dbg !16
  store i8 %32, ptr %33, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %34 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 16, !dbg !16
  store i8 %8, ptr %34, align 16, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %35 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 17, !dbg !16
  store i8 %16, ptr %35, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %36 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 18, !dbg !16
  store i8 %32, ptr %36, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %37 = getelementptr inbounds i8, ptr %x, i64 10, !dbg !16
  %38 = load i8, ptr %37, align 1, !dbg !16, !tbaa !23
  %39 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 19, !dbg !16
  store i8 %38, ptr %39, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %40 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 20, !dbg !16
  store i8 %16, ptr %40, align 4, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %41 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 21, !dbg !16
  store i8 %24, ptr %41, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %42 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 22, !dbg !16
  store i8 %38, ptr %42, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %43 = getelementptr inbounds i8, ptr %x, i64 11, !dbg !16
  %44 = load i8, ptr %43, align 1, !dbg !16, !tbaa !23
  %45 = getelementptr inbounds i8, ptr %pad_temp_reindex38, i64 23, !dbg !16
  store i8 %44, ptr %45, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !21, metadata !DIExpression()), !dbg !16
  call void @tvm_gemmini_dma_a(ptr nonnull %pad_temp_reindex38, i32 0, i32 6, i32 4, i64 4, i32 1), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %46 = load i8, ptr %y, align 1, !dbg !16, !tbaa !28
  store i8 %46, ptr %pad_temp_reindex38, align 16, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %47 = getelementptr inbounds i8, ptr %y, i64 1, !dbg !16
  %48 = load i8, ptr %47, align 1, !dbg !16, !tbaa !28
  store i8 %48, ptr %6, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %49 = getelementptr inbounds i8, ptr %y, i64 2, !dbg !16
  %50 = load i8, ptr %49, align 1, !dbg !16, !tbaa !28
  store i8 %50, ptr %10, align 4, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %51 = getelementptr inbounds i8, ptr %y, i64 3, !dbg !16
  %52 = load i8, ptr %51, align 1, !dbg !16, !tbaa !28
  store i8 %52, ptr %14, align 2, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %53 = getelementptr inbounds i8, ptr %y, i64 4, !dbg !16
  %54 = load i8, ptr %53, align 1, !dbg !16, !tbaa !28
  store i8 %54, ptr %3, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %55 = getelementptr inbounds i8, ptr %y, i64 5, !dbg !16
  %56 = load i8, ptr %55, align 1, !dbg !16, !tbaa !28
  store i8 %56, ptr %9, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  %57 = getelementptr inbounds i8, ptr %y, i64 6, !dbg !16
  %58 = load i8, ptr %57, align 1, !dbg !16, !tbaa !28
  store i8 %58, ptr %13, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !16
  %59 = getelementptr inbounds i8, ptr %y, i64 7, !dbg !16
  %60 = load i8, ptr %59, align 1, !dbg !16, !tbaa !28
  store i8 %60, ptr %17, align 1, !dbg !16, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
  call void @tvm_gemmini_dma_b(ptr nonnull %pad_temp_reindex38, i32 16, i32 4, i32 2, i64 2, i32 1), !dbg !16
  call void @tvm_gemmini_compute(i32 0, i32 16, i32 0, i32 6, i32 2, i32 4, i32 1), !dbg !16
  call void @tvm_gemmini_dma_out(ptr nonnull %conv2d_nchw_reindex42, i32 0, i32 6, i32 2, i64 2, i32 0), !dbg !16
  call void @tvm_gemmini_end(), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %61 = load i32, ptr %conv2d_nchw_reindex42, align 16, !dbg !16, !tbaa !30
  tail call void @llvm.dbg.value(metadata i32 %61, metadata !32, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %62 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 1, !dbg !16
  %63 = load i32, ptr %62, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %64 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 2, !dbg !16
  %65 = load i32, ptr %64, align 8, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %66 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 3, !dbg !16
  %67 = load i32, ptr %66, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %68 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 4, !dbg !16
  %69 = load i32, ptr %68, align 16, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %70 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 5, !dbg !16
  %71 = load i32, ptr %70, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %72 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 6, !dbg !16
  %73 = load i32, ptr %72, align 8, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %74 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 7, !dbg !16
  %75 = load i32, ptr %74, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %76 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 8, !dbg !16
  %77 = load i32, ptr %76, align 16, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %78 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 9, !dbg !16
  %79 = load i32, ptr %78, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  %80 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 10, !dbg !16
  %81 = load i32, ptr %80, align 8, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  %82 = getelementptr inbounds i32, ptr %conv2d_nchw_reindex42, i64 11, !dbg !16
  %83 = load i32, ptr %82, align 4, !dbg !16, !tbaa !30
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !33, metadata !DIExpression()), !dbg !16
  %84 = load i8, ptr %lv1, align 1, !dbg !16, !tbaa !34
  tail call void @llvm.dbg.value(metadata i8 %84, metadata !36, metadata !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !33, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !33, metadata !DIExpression()), !dbg !16
  %85 = getelementptr inbounds i8, ptr %lv1, i64 1, !dbg !16
  %86 = load i8, ptr %85, align 1, !dbg !16, !tbaa !34
  call void @llvm.dbg.declare(metadata i64 2, metadata !33, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !39, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata !DIArgList(i32 %61, i8 %84), metadata !40, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_plus, DW_OP_stack_value)), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 7, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 7, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 8, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 8, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 9, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 9, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 10, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 10, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 11, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 11, metadata !39, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !38, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !37, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !33, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !43, metadata !DIExpression()), !dbg !16
  %87 = trunc i32 %61 to i8, !dbg !16
  %88 = add i8 %84, %87, !dbg !16
  store i8 %88, ptr %compute, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !43, metadata !DIExpression()), !dbg !16
  %89 = trunc i32 %65 to i8, !dbg !16
  %90 = add i8 %84, %89, !dbg !16
  %91 = getelementptr inbounds i8, ptr %compute, i64 1, !dbg !16
  store i8 %90, ptr %91, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !43, metadata !DIExpression()), !dbg !16
  %92 = trunc i32 %69 to i8, !dbg !16
  %93 = add i8 %84, %92, !dbg !16
  %94 = getelementptr inbounds i8, ptr %compute, i64 2, !dbg !16
  store i8 %93, ptr %94, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 3, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 3, metadata !43, metadata !DIExpression()), !dbg !16
  %95 = trunc i32 %73 to i8, !dbg !16
  %96 = add i8 %84, %95, !dbg !16
  %97 = getelementptr inbounds i8, ptr %compute, i64 3, !dbg !16
  store i8 %96, ptr %97, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 4, metadata !43, metadata !DIExpression()), !dbg !16
  %98 = trunc i32 %77 to i8, !dbg !16
  %99 = add i8 %84, %98, !dbg !16
  %100 = getelementptr inbounds i8, ptr %compute, i64 4, !dbg !16
  store i8 %99, ptr %100, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 5, metadata !43, metadata !DIExpression()), !dbg !16
  %101 = trunc i32 %81 to i8, !dbg !16
  %102 = add i8 %84, %101, !dbg !16
  %103 = getelementptr inbounds i8, ptr %compute, i64 5, !dbg !16
  store i8 %102, ptr %103, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 3, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !33, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !33, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 6, metadata !43, metadata !DIExpression()), !dbg !16
  %104 = trunc i32 %63 to i8, !dbg !16
  %105 = add i8 %86, %104, !dbg !16
  %106 = getelementptr inbounds i8, ptr %compute, i64 6, !dbg !16
  store i8 %105, ptr %106, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 7, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 7, metadata !43, metadata !DIExpression()), !dbg !16
  %107 = trunc i32 %67 to i8, !dbg !16
  %108 = add i8 %86, %107, !dbg !16
  %109 = getelementptr inbounds i8, ptr %compute, i64 7, !dbg !16
  store i8 %108, ptr %109, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 8, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 8, metadata !43, metadata !DIExpression()), !dbg !16
  %110 = trunc i32 %71 to i8, !dbg !16
  %111 = add i8 %86, %110, !dbg !16
  %112 = getelementptr inbounds i8, ptr %compute, i64 8, !dbg !16
  store i8 %111, ptr %112, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 3, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 9, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 9, metadata !43, metadata !DIExpression()), !dbg !16
  %113 = trunc i32 %75 to i8, !dbg !16
  %114 = add i8 %86, %113, !dbg !16
  %115 = getelementptr inbounds i8, ptr %compute, i64 9, !dbg !16
  store i8 %114, ptr %115, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 10, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 10, metadata !43, metadata !DIExpression()), !dbg !16
  %116 = trunc i32 %79 to i8, !dbg !16
  %117 = add i8 %86, %116, !dbg !16
  %118 = getelementptr inbounds i8, ptr %compute, i64 10, !dbg !16
  store i8 %117, ptr %118, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 11, metadata !43, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 11, metadata !43, metadata !DIExpression()), !dbg !16
  %119 = trunc i32 %83 to i8, !dbg !16
  %120 = add i8 %86, %119, !dbg !16
  %121 = getelementptr inbounds i8, ptr %compute, i64 11, !dbg !16
  store i8 %120, ptr %121, align 1, !dbg !16, !tbaa !44
  call void @llvm.dbg.declare(metadata i64 3, metadata !42, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !41, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 2, metadata !33, metadata !DIExpression()), !dbg !16
  call void @tvm_gemmini_end(), !dbg !16
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @tvm_gemmini_validate_tensor_buffers(ptr, i64, ptr, i64, ptr, i64) local_unnamed_addr

declare void @TVMAPISetLastError(ptr) local_unnamed_addr

declare void @tvm_gemmini_begin(i64, i64, i64) local_unnamed_addr

declare void @tvm_gemmini_zero_acc(i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_a(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_b(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_compute(i32, i32, i32, i32, i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_out(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_end() local_unnamed_addr

; Function Attrs: nofree nosync nounwind memory(none)
define weak dso_local i16 @__truncsfhf2(float %a0) local_unnamed_addr #2 section ".text.tvm.fp16.conv" {
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
define weak dso_local float @__extendhfsf2(i16 %a0) local_unnamed_addr #2 section ".text.tvm.fp16.conv" {
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
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #3

attributes #0 = { "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nosync nounwind memory(none) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d -fp-contract=off"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 4}
!5 = distinct !DISubprogram(name: "stage_3_operator_fused_conv2d_cast_add_cast1", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocalVariable(name: "lv1", arg: 3, scope: !5, file: !1, type: !9)
!15 = !DILocalVariable(name: "compute", arg: 4, scope: !5, file: !1, type: !9)
!16 = !DILocation(line: 0, scope: !5)
!17 = !DILocalVariable(name: "conv2d_nchw_reindex", scope: !5, file: !1, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!19 = !DILocalVariable(name: "pad_temp_reindex", scope: !5, file: !1, type: !9)
!20 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!21 = !DILocalVariable(name: "ax0", scope: !5, file: !1, type: !8)
!22 = !DILocalVariable(name: "ax1", scope: !5, file: !1, type: !8)
!23 = !{!24, !24, i64 0}
!24 = !{!"0x40bbdbb0", !25, i64 0}
!25 = !{!"tvm-tbaa"}
!26 = !{!27, !27, i64 0}
!27 = !{!"0x40b74c50", !25, i64 0}
!28 = !{!29, !29, i64 0}
!29 = !{!"0x40bbdb70", !25, i64 0}
!30 = !{!31, !31, i64 0}
!31 = !{!"0x40b73450", !25, i64 0}
!32 = !DILocalVariable(name: "conv2d_nchw", scope: !5, file: !1, type: !18)
!33 = !DILocalVariable(name: "i1", scope: !5, file: !1, type: !8)
!34 = !{!35, !35, i64 0}
!35 = !{!"0x40bbcce0", !25, i64 0}
!36 = !DILocalVariable(name: "compute", scope: !5, file: !1, type: !18)
!37 = !DILocalVariable(name: "ax2", scope: !5, file: !1, type: !8)
!38 = !DILocalVariable(name: "ax3", scope: !5, file: !1, type: !8)
!39 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!40 = !DILocalVariable(name: "T_add", scope: !5, file: !1, type: !18)
!41 = !DILocalVariable(name: "i2", scope: !5, file: !1, type: !8)
!42 = !DILocalVariable(name: "i3", scope: !5, file: !1, type: !8)
!43 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!44 = !{!45, !45, i64 0}
!45 = !{!"0x40bbcfa0", !25, i64 0}
