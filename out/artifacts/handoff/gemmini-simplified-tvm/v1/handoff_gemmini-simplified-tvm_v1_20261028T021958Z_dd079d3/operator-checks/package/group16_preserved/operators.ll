; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_70_operator_kernel(ptr noalias %a, ptr noalias %b, ptr noalias %scales, ptr noalias %compute) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %a, metadata !14, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata ptr %b, metadata !15, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata ptr %scales, metadata !16, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata ptr %compute, metadata !17, metadata !DIExpression()), !dbg !18
  %compute114 = alloca [99 x i32], align 4, !dbg !18
  call void @llvm.dbg.declare(metadata ptr %compute114, metadata !19, metadata !DIExpression()), !dbg !18
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %scales, i64 132, ptr %scales, i64 132, ptr %compute, i64 396), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !21, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !21, metadata !DIExpression()), !dbg !18
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !18
  br i1 %.not, label %if_else, label %if_then, !dbg !18

common.ret:                                       ; preds = %if_then10, %if_then4, %for_end_j.2, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %for_end_j.2 ], [ -1, %if_then4 ], [ -1, %if_then10 ]
  ret i32 %common.ret.op, !dbg !18

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !18
  br label %common.ret

if_else:                                          ; preds = %entry
  %gemmini_admission3 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %b, i64 165, ptr %b, i64 165, ptr %compute, i64 396), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission3, metadata !21, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission3, metadata !21, metadata !DIExpression()), !dbg !18
  %.not15 = icmp eq i32 %gemmini_admission3, 0, !dbg !18
  br i1 %.not15, label %if_else6, label %if_then4, !dbg !18

if_then4:                                         ; preds = %if_else
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !18
  br label %common.ret

if_else6:                                         ; preds = %if_else
  %gemmini_admission9 = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %a, i64 15, ptr %a, i64 15, ptr %compute, i64 396), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission9, metadata !21, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission9, metadata !21, metadata !DIExpression()), !dbg !18
  %.not16 = icmp eq i32 %gemmini_admission9, 0, !dbg !18
  br i1 %.not16, label %if_else12, label %if_then10, !dbg !18

if_then10:                                        ; preds = %if_else6
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !18
  br label %common.ret

if_else12:                                        ; preds = %if_else6
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !18
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !23, metadata !DIExpression()), !dbg !18
  tail call void @tvm_gemmini_zero_acc(i32 0, i32 3, i32 11), !dbg !18
  tail call void @tvm_gemmini_dma_a(ptr %a, i32 0, i32 3, i32 5, i32 5, i32 0), !dbg !18
  tail call void @tvm_gemmini_dma_b(ptr %b, i32 16, i32 5, i32 11, i32 33, i32 0), !dbg !18
  tail call void @tvm_gemmini_compute(i32 0, i32 16, i32 0, i32 3, i32 11, i32 5, i32 1), !dbg !18
  call void @tvm_gemmini_dma_out(ptr nonnull %compute114, i32 0, i32 3, i32 11, i32 33, i32 0), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 11, metadata !23, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 11, metadata !23, metadata !DIExpression()), !dbg !18
  call void @tvm_gemmini_zero_acc(i32 0, i32 3, i32 11), !dbg !18
  call void @tvm_gemmini_dma_a(ptr %a, i32 0, i32 3, i32 5, i32 5, i32 0), !dbg !18
  %0 = getelementptr inbounds i8, ptr %b, i64 11, !dbg !18
  call void @tvm_gemmini_dma_b(ptr nonnull %0, i32 16, i32 5, i32 11, i32 33, i32 0), !dbg !18
  call void @tvm_gemmini_compute(i32 0, i32 16, i32 0, i32 3, i32 11, i32 5, i32 1), !dbg !18
  %1 = getelementptr inbounds i32, ptr %compute114, i64 11, !dbg !18
  call void @tvm_gemmini_dma_out(ptr nonnull %1, i32 0, i32 3, i32 11, i32 33, i32 0), !dbg !18
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 22, metadata !23, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 22, metadata !23, metadata !DIExpression()), !dbg !18
  call void @tvm_gemmini_zero_acc(i32 0, i32 3, i32 11), !dbg !18
  call void @tvm_gemmini_dma_a(ptr %a, i32 0, i32 3, i32 5, i32 5, i32 0), !dbg !18
  %2 = getelementptr inbounds i8, ptr %b, i64 22, !dbg !18
  call void @tvm_gemmini_dma_b(ptr nonnull %2, i32 16, i32 5, i32 11, i32 33, i32 0), !dbg !18
  call void @tvm_gemmini_compute(i32 0, i32 16, i32 0, i32 3, i32 11, i32 5, i32 1), !dbg !18
  %3 = getelementptr inbounds i32, ptr %compute114, i64 22, !dbg !18
  call void @tvm_gemmini_dma_out(ptr nonnull %3, i32 0, i32 3, i32 11, i32 33, i32 0), !dbg !18
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !18
  call void @tvm_gemmini_end(), !dbg !18
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 0, metadata !26, metadata !DIExpression()), !dbg !18
  %4 = load float, ptr %scales, align 4, !dbg !18, !tbaa !27
  %5 = load i32, ptr %compute114, align 4, !dbg !18, !tbaa !30
  %6 = sitofp i32 %5 to float, !dbg !18
  %7 = fmul float %4, %6, !dbg !18
  store float %7, ptr %compute, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !26, metadata !DIExpression()), !dbg !18
  %8 = getelementptr inbounds float, ptr %scales, i64 1, !dbg !18
  %9 = load float, ptr %8, align 4, !dbg !18, !tbaa !27
  %10 = getelementptr inbounds i32, ptr %compute114, i64 1, !dbg !18
  %11 = load i32, ptr %10, align 4, !dbg !18, !tbaa !30
  %12 = sitofp i32 %11 to float, !dbg !18
  %13 = fmul float %9, %12, !dbg !18
  %14 = getelementptr inbounds float, ptr %compute, i64 1, !dbg !18
  store float %13, ptr %14, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 2, metadata !26, metadata !DIExpression()), !dbg !18
  %15 = getelementptr inbounds float, ptr %scales, i64 2, !dbg !18
  %16 = load float, ptr %15, align 4, !dbg !18, !tbaa !27
  %17 = getelementptr inbounds i32, ptr %compute114, i64 2, !dbg !18
  %18 = load i32, ptr %17, align 4, !dbg !18, !tbaa !30
  %19 = sitofp i32 %18 to float, !dbg !18
  %20 = fmul float %16, %19, !dbg !18
  %21 = getelementptr inbounds float, ptr %compute, i64 2, !dbg !18
  store float %20, ptr %21, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 3, metadata !26, metadata !DIExpression()), !dbg !18
  %22 = getelementptr inbounds float, ptr %scales, i64 3, !dbg !18
  %23 = load float, ptr %22, align 4, !dbg !18, !tbaa !27
  %24 = getelementptr inbounds i32, ptr %compute114, i64 3, !dbg !18
  %25 = load i32, ptr %24, align 4, !dbg !18, !tbaa !30
  %26 = sitofp i32 %25 to float, !dbg !18
  %27 = fmul float %23, %26, !dbg !18
  %28 = getelementptr inbounds float, ptr %compute, i64 3, !dbg !18
  store float %27, ptr %28, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 4, metadata !26, metadata !DIExpression()), !dbg !18
  %29 = getelementptr inbounds float, ptr %scales, i64 4, !dbg !18
  %30 = load float, ptr %29, align 4, !dbg !18, !tbaa !27
  %31 = getelementptr inbounds i32, ptr %compute114, i64 4, !dbg !18
  %32 = load i32, ptr %31, align 4, !dbg !18, !tbaa !30
  %33 = sitofp i32 %32 to float, !dbg !18
  %34 = fmul float %30, %33, !dbg !18
  %35 = getelementptr inbounds float, ptr %compute, i64 4, !dbg !18
  store float %34, ptr %35, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 5, metadata !26, metadata !DIExpression()), !dbg !18
  %36 = getelementptr inbounds float, ptr %scales, i64 5, !dbg !18
  %37 = load float, ptr %36, align 4, !dbg !18, !tbaa !27
  %38 = getelementptr inbounds i32, ptr %compute114, i64 5, !dbg !18
  %39 = load i32, ptr %38, align 4, !dbg !18, !tbaa !30
  %40 = sitofp i32 %39 to float, !dbg !18
  %41 = fmul float %37, %40, !dbg !18
  %42 = getelementptr inbounds float, ptr %compute, i64 5, !dbg !18
  store float %41, ptr %42, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 6, metadata !26, metadata !DIExpression()), !dbg !18
  %43 = getelementptr inbounds float, ptr %scales, i64 6, !dbg !18
  %44 = load float, ptr %43, align 4, !dbg !18, !tbaa !27
  %45 = getelementptr inbounds i32, ptr %compute114, i64 6, !dbg !18
  %46 = load i32, ptr %45, align 4, !dbg !18, !tbaa !30
  %47 = sitofp i32 %46 to float, !dbg !18
  %48 = fmul float %44, %47, !dbg !18
  %49 = getelementptr inbounds float, ptr %compute, i64 6, !dbg !18
  store float %48, ptr %49, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 7, metadata !26, metadata !DIExpression()), !dbg !18
  %50 = getelementptr inbounds float, ptr %scales, i64 7, !dbg !18
  %51 = load float, ptr %50, align 4, !dbg !18, !tbaa !27
  %52 = getelementptr inbounds i32, ptr %compute114, i64 7, !dbg !18
  %53 = load i32, ptr %52, align 4, !dbg !18, !tbaa !30
  %54 = sitofp i32 %53 to float, !dbg !18
  %55 = fmul float %51, %54, !dbg !18
  %56 = getelementptr inbounds float, ptr %compute, i64 7, !dbg !18
  store float %55, ptr %56, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 8, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 8, metadata !26, metadata !DIExpression()), !dbg !18
  %57 = getelementptr inbounds float, ptr %scales, i64 8, !dbg !18
  %58 = load float, ptr %57, align 4, !dbg !18, !tbaa !27
  %59 = getelementptr inbounds i32, ptr %compute114, i64 8, !dbg !18
  %60 = load i32, ptr %59, align 4, !dbg !18, !tbaa !30
  %61 = sitofp i32 %60 to float, !dbg !18
  %62 = fmul float %58, %61, !dbg !18
  %63 = getelementptr inbounds float, ptr %compute, i64 8, !dbg !18
  store float %62, ptr %63, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 9, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 9, metadata !26, metadata !DIExpression()), !dbg !18
  %64 = getelementptr inbounds float, ptr %scales, i64 9, !dbg !18
  %65 = load float, ptr %64, align 4, !dbg !18, !tbaa !27
  %66 = getelementptr inbounds i32, ptr %compute114, i64 9, !dbg !18
  %67 = load i32, ptr %66, align 4, !dbg !18, !tbaa !30
  %68 = sitofp i32 %67 to float, !dbg !18
  %69 = fmul float %65, %68, !dbg !18
  %70 = getelementptr inbounds float, ptr %compute, i64 9, !dbg !18
  store float %69, ptr %70, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 10, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 10, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 10, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 10, metadata !26, metadata !DIExpression()), !dbg !18
  %71 = getelementptr inbounds float, ptr %scales, i64 10, !dbg !18
  %72 = load float, ptr %71, align 4, !dbg !18, !tbaa !27
  %73 = getelementptr inbounds i32, ptr %compute114, i64 10, !dbg !18
  %74 = load i32, ptr %73, align 4, !dbg !18, !tbaa !30
  %75 = sitofp i32 %74 to float, !dbg !18
  %76 = fmul float %72, %75, !dbg !18
  %77 = getelementptr inbounds float, ptr %compute, i64 10, !dbg !18
  store float %76, ptr %77, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 11, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 11, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 11, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 11, metadata !26, metadata !DIExpression()), !dbg !18
  %78 = getelementptr inbounds float, ptr %scales, i64 11, !dbg !18
  %79 = load float, ptr %78, align 4, !dbg !18, !tbaa !27
  %80 = load i32, ptr %1, align 4, !dbg !18, !tbaa !30
  %81 = sitofp i32 %80 to float, !dbg !18
  %82 = fmul float %79, %81, !dbg !18
  %83 = getelementptr inbounds float, ptr %compute, i64 11, !dbg !18
  store float %82, ptr %83, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 12, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 12, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 12, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 12, metadata !26, metadata !DIExpression()), !dbg !18
  %84 = getelementptr inbounds float, ptr %scales, i64 12, !dbg !18
  %85 = load float, ptr %84, align 4, !dbg !18, !tbaa !27
  %86 = getelementptr inbounds i32, ptr %compute114, i64 12, !dbg !18
  %87 = load i32, ptr %86, align 4, !dbg !18, !tbaa !30
  %88 = sitofp i32 %87 to float, !dbg !18
  %89 = fmul float %85, %88, !dbg !18
  %90 = getelementptr inbounds float, ptr %compute, i64 12, !dbg !18
  store float %89, ptr %90, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 13, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 13, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 13, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 13, metadata !26, metadata !DIExpression()), !dbg !18
  %91 = getelementptr inbounds float, ptr %scales, i64 13, !dbg !18
  %92 = load float, ptr %91, align 4, !dbg !18, !tbaa !27
  %93 = getelementptr inbounds i32, ptr %compute114, i64 13, !dbg !18
  %94 = load i32, ptr %93, align 4, !dbg !18, !tbaa !30
  %95 = sitofp i32 %94 to float, !dbg !18
  %96 = fmul float %92, %95, !dbg !18
  %97 = getelementptr inbounds float, ptr %compute, i64 13, !dbg !18
  store float %96, ptr %97, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 14, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 14, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 14, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 14, metadata !26, metadata !DIExpression()), !dbg !18
  %98 = getelementptr inbounds float, ptr %scales, i64 14, !dbg !18
  %99 = load float, ptr %98, align 4, !dbg !18, !tbaa !27
  %100 = getelementptr inbounds i32, ptr %compute114, i64 14, !dbg !18
  %101 = load i32, ptr %100, align 4, !dbg !18, !tbaa !30
  %102 = sitofp i32 %101 to float, !dbg !18
  %103 = fmul float %99, %102, !dbg !18
  %104 = getelementptr inbounds float, ptr %compute, i64 14, !dbg !18
  store float %103, ptr %104, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 15, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 15, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 15, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 15, metadata !26, metadata !DIExpression()), !dbg !18
  %105 = getelementptr inbounds float, ptr %scales, i64 15, !dbg !18
  %106 = load float, ptr %105, align 4, !dbg !18, !tbaa !27
  %107 = getelementptr inbounds i32, ptr %compute114, i64 15, !dbg !18
  %108 = load i32, ptr %107, align 4, !dbg !18, !tbaa !30
  %109 = sitofp i32 %108 to float, !dbg !18
  %110 = fmul float %106, %109, !dbg !18
  %111 = getelementptr inbounds float, ptr %compute, i64 15, !dbg !18
  store float %110, ptr %111, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 16, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 16, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 16, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 16, metadata !26, metadata !DIExpression()), !dbg !18
  %112 = getelementptr inbounds float, ptr %scales, i64 16, !dbg !18
  %113 = load float, ptr %112, align 4, !dbg !18, !tbaa !27
  %114 = getelementptr inbounds i32, ptr %compute114, i64 16, !dbg !18
  %115 = load i32, ptr %114, align 4, !dbg !18, !tbaa !30
  %116 = sitofp i32 %115 to float, !dbg !18
  %117 = fmul float %113, %116, !dbg !18
  %118 = getelementptr inbounds float, ptr %compute, i64 16, !dbg !18
  store float %117, ptr %118, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 17, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 17, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 17, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 17, metadata !26, metadata !DIExpression()), !dbg !18
  %119 = getelementptr inbounds float, ptr %scales, i64 17, !dbg !18
  %120 = load float, ptr %119, align 4, !dbg !18, !tbaa !27
  %121 = getelementptr inbounds i32, ptr %compute114, i64 17, !dbg !18
  %122 = load i32, ptr %121, align 4, !dbg !18, !tbaa !30
  %123 = sitofp i32 %122 to float, !dbg !18
  %124 = fmul float %120, %123, !dbg !18
  %125 = getelementptr inbounds float, ptr %compute, i64 17, !dbg !18
  store float %124, ptr %125, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 18, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 18, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 18, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 18, metadata !26, metadata !DIExpression()), !dbg !18
  %126 = getelementptr inbounds float, ptr %scales, i64 18, !dbg !18
  %127 = load float, ptr %126, align 4, !dbg !18, !tbaa !27
  %128 = getelementptr inbounds i32, ptr %compute114, i64 18, !dbg !18
  %129 = load i32, ptr %128, align 4, !dbg !18, !tbaa !30
  %130 = sitofp i32 %129 to float, !dbg !18
  %131 = fmul float %127, %130, !dbg !18
  %132 = getelementptr inbounds float, ptr %compute, i64 18, !dbg !18
  store float %131, ptr %132, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 19, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 19, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 19, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 19, metadata !26, metadata !DIExpression()), !dbg !18
  %133 = getelementptr inbounds float, ptr %scales, i64 19, !dbg !18
  %134 = load float, ptr %133, align 4, !dbg !18, !tbaa !27
  %135 = getelementptr inbounds i32, ptr %compute114, i64 19, !dbg !18
  %136 = load i32, ptr %135, align 4, !dbg !18, !tbaa !30
  %137 = sitofp i32 %136 to float, !dbg !18
  %138 = fmul float %134, %137, !dbg !18
  %139 = getelementptr inbounds float, ptr %compute, i64 19, !dbg !18
  store float %138, ptr %139, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 20, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 20, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 20, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 20, metadata !26, metadata !DIExpression()), !dbg !18
  %140 = getelementptr inbounds float, ptr %scales, i64 20, !dbg !18
  %141 = load float, ptr %140, align 4, !dbg !18, !tbaa !27
  %142 = getelementptr inbounds i32, ptr %compute114, i64 20, !dbg !18
  %143 = load i32, ptr %142, align 4, !dbg !18, !tbaa !30
  %144 = sitofp i32 %143 to float, !dbg !18
  %145 = fmul float %141, %144, !dbg !18
  %146 = getelementptr inbounds float, ptr %compute, i64 20, !dbg !18
  store float %145, ptr %146, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 21, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 21, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 21, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 21, metadata !26, metadata !DIExpression()), !dbg !18
  %147 = getelementptr inbounds float, ptr %scales, i64 21, !dbg !18
  %148 = load float, ptr %147, align 4, !dbg !18, !tbaa !27
  %149 = getelementptr inbounds i32, ptr %compute114, i64 21, !dbg !18
  %150 = load i32, ptr %149, align 4, !dbg !18, !tbaa !30
  %151 = sitofp i32 %150 to float, !dbg !18
  %152 = fmul float %148, %151, !dbg !18
  %153 = getelementptr inbounds float, ptr %compute, i64 21, !dbg !18
  store float %152, ptr %153, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 22, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 22, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 22, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 22, metadata !26, metadata !DIExpression()), !dbg !18
  %154 = getelementptr inbounds float, ptr %scales, i64 22, !dbg !18
  %155 = load float, ptr %154, align 4, !dbg !18, !tbaa !27
  %156 = load i32, ptr %3, align 4, !dbg !18, !tbaa !30
  %157 = sitofp i32 %156 to float, !dbg !18
  %158 = fmul float %155, %157, !dbg !18
  %159 = getelementptr inbounds float, ptr %compute, i64 22, !dbg !18
  store float %158, ptr %159, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 23, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 23, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 23, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 23, metadata !26, metadata !DIExpression()), !dbg !18
  %160 = getelementptr inbounds float, ptr %scales, i64 23, !dbg !18
  %161 = load float, ptr %160, align 4, !dbg !18, !tbaa !27
  %162 = getelementptr inbounds i32, ptr %compute114, i64 23, !dbg !18
  %163 = load i32, ptr %162, align 4, !dbg !18, !tbaa !30
  %164 = sitofp i32 %163 to float, !dbg !18
  %165 = fmul float %161, %164, !dbg !18
  %166 = getelementptr inbounds float, ptr %compute, i64 23, !dbg !18
  store float %165, ptr %166, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 24, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 24, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 24, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 24, metadata !26, metadata !DIExpression()), !dbg !18
  %167 = getelementptr inbounds float, ptr %scales, i64 24, !dbg !18
  %168 = load float, ptr %167, align 4, !dbg !18, !tbaa !27
  %169 = getelementptr inbounds i32, ptr %compute114, i64 24, !dbg !18
  %170 = load i32, ptr %169, align 4, !dbg !18, !tbaa !30
  %171 = sitofp i32 %170 to float, !dbg !18
  %172 = fmul float %168, %171, !dbg !18
  %173 = getelementptr inbounds float, ptr %compute, i64 24, !dbg !18
  store float %172, ptr %173, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 25, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 25, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 25, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 25, metadata !26, metadata !DIExpression()), !dbg !18
  %174 = getelementptr inbounds float, ptr %scales, i64 25, !dbg !18
  %175 = load float, ptr %174, align 4, !dbg !18, !tbaa !27
  %176 = getelementptr inbounds i32, ptr %compute114, i64 25, !dbg !18
  %177 = load i32, ptr %176, align 4, !dbg !18, !tbaa !30
  %178 = sitofp i32 %177 to float, !dbg !18
  %179 = fmul float %175, %178, !dbg !18
  %180 = getelementptr inbounds float, ptr %compute, i64 25, !dbg !18
  store float %179, ptr %180, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 26, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 26, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 26, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 26, metadata !26, metadata !DIExpression()), !dbg !18
  %181 = getelementptr inbounds float, ptr %scales, i64 26, !dbg !18
  %182 = load float, ptr %181, align 4, !dbg !18, !tbaa !27
  %183 = getelementptr inbounds i32, ptr %compute114, i64 26, !dbg !18
  %184 = load i32, ptr %183, align 4, !dbg !18, !tbaa !30
  %185 = sitofp i32 %184 to float, !dbg !18
  %186 = fmul float %182, %185, !dbg !18
  %187 = getelementptr inbounds float, ptr %compute, i64 26, !dbg !18
  store float %186, ptr %187, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 27, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 27, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 27, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 27, metadata !26, metadata !DIExpression()), !dbg !18
  %188 = getelementptr inbounds float, ptr %scales, i64 27, !dbg !18
  %189 = load float, ptr %188, align 4, !dbg !18, !tbaa !27
  %190 = getelementptr inbounds i32, ptr %compute114, i64 27, !dbg !18
  %191 = load i32, ptr %190, align 4, !dbg !18, !tbaa !30
  %192 = sitofp i32 %191 to float, !dbg !18
  %193 = fmul float %189, %192, !dbg !18
  %194 = getelementptr inbounds float, ptr %compute, i64 27, !dbg !18
  store float %193, ptr %194, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 28, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 28, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 28, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 28, metadata !26, metadata !DIExpression()), !dbg !18
  %195 = getelementptr inbounds float, ptr %scales, i64 28, !dbg !18
  %196 = load float, ptr %195, align 4, !dbg !18, !tbaa !27
  %197 = getelementptr inbounds i32, ptr %compute114, i64 28, !dbg !18
  %198 = load i32, ptr %197, align 4, !dbg !18, !tbaa !30
  %199 = sitofp i32 %198 to float, !dbg !18
  %200 = fmul float %196, %199, !dbg !18
  %201 = getelementptr inbounds float, ptr %compute, i64 28, !dbg !18
  store float %200, ptr %201, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 29, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 29, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 29, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 29, metadata !26, metadata !DIExpression()), !dbg !18
  %202 = getelementptr inbounds float, ptr %scales, i64 29, !dbg !18
  %203 = load float, ptr %202, align 4, !dbg !18, !tbaa !27
  %204 = getelementptr inbounds i32, ptr %compute114, i64 29, !dbg !18
  %205 = load i32, ptr %204, align 4, !dbg !18, !tbaa !30
  %206 = sitofp i32 %205 to float, !dbg !18
  %207 = fmul float %203, %206, !dbg !18
  %208 = getelementptr inbounds float, ptr %compute, i64 29, !dbg !18
  store float %207, ptr %208, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 30, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 30, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 30, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 30, metadata !26, metadata !DIExpression()), !dbg !18
  %209 = getelementptr inbounds float, ptr %scales, i64 30, !dbg !18
  %210 = load float, ptr %209, align 4, !dbg !18, !tbaa !27
  %211 = getelementptr inbounds i32, ptr %compute114, i64 30, !dbg !18
  %212 = load i32, ptr %211, align 4, !dbg !18, !tbaa !30
  %213 = sitofp i32 %212 to float, !dbg !18
  %214 = fmul float %210, %213, !dbg !18
  %215 = getelementptr inbounds float, ptr %compute, i64 30, !dbg !18
  store float %214, ptr %215, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 31, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 31, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 31, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 31, metadata !26, metadata !DIExpression()), !dbg !18
  %216 = getelementptr inbounds float, ptr %scales, i64 31, !dbg !18
  %217 = load float, ptr %216, align 4, !dbg !18, !tbaa !27
  %218 = getelementptr inbounds i32, ptr %compute114, i64 31, !dbg !18
  %219 = load i32, ptr %218, align 4, !dbg !18, !tbaa !30
  %220 = sitofp i32 %219 to float, !dbg !18
  %221 = fmul float %217, %220, !dbg !18
  %222 = getelementptr inbounds float, ptr %compute, i64 31, !dbg !18
  store float %221, ptr %222, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 32, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 32, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 32, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 32, metadata !26, metadata !DIExpression()), !dbg !18
  %223 = getelementptr inbounds float, ptr %scales, i64 32, !dbg !18
  %224 = load float, ptr %223, align 4, !dbg !18, !tbaa !27
  %225 = getelementptr inbounds i32, ptr %compute114, i64 32, !dbg !18
  %226 = load i32, ptr %225, align 4, !dbg !18, !tbaa !30
  %227 = sitofp i32 %226 to float, !dbg !18
  %228 = fmul float %224, %227, !dbg !18
  %229 = getelementptr inbounds float, ptr %compute, i64 32, !dbg !18
  store float %228, ptr %229, align 4, !dbg !18, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 33, metadata !25, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !18
  br label %for_body_j.1, !dbg !18

for_body_j.1:                                     ; preds = %for_body_j.1, %if_else12
  %indvars.iv.1 = phi i64 [ 0, %if_else12 ], [ %indvars.iv.next.1, %for_body_j.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.1, metadata !25, metadata !DIExpression()), !dbg !18
  %230 = add nuw nsw i64 %indvars.iv.1, 33, !dbg !18
  call void @llvm.dbg.declare(metadata i64 %230, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 %230, metadata !26, metadata !DIExpression()), !dbg !18
  %231 = getelementptr inbounds float, ptr %scales, i64 %indvars.iv.1, !dbg !18
  %232 = load float, ptr %231, align 4, !dbg !18, !tbaa !27
  %233 = getelementptr inbounds i32, ptr %compute114, i64 %230, !dbg !18
  %234 = load i32, ptr %233, align 4, !dbg !18, !tbaa !30
  %235 = sitofp i32 %234 to float, !dbg !18
  %236 = fmul float %232, %235, !dbg !18
  %237 = getelementptr inbounds float, ptr %compute, i64 %230, !dbg !18
  store float %236, ptr %237, align 4, !dbg !18, !tbaa !32
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv.1, 1, !dbg !18
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.1, metadata !25, metadata !DIExpression()), !dbg !18
  %exitcond.1.not = icmp eq i64 %indvars.iv.next.1, 33, !dbg !18
  br i1 %exitcond.1.not, label %for_body_j.2, label %for_body_j.1, !dbg !18, !prof !34

for_body_j.2:                                     ; preds = %for_body_j.1, %for_body_j.2
  %indvars.iv.2 = phi i64 [ %indvars.iv.next.2, %for_body_j.2 ], [ 0, %for_body_j.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.2, metadata !25, metadata !DIExpression()), !dbg !18
  %238 = add nuw nsw i64 %indvars.iv.2, 66, !dbg !18
  call void @llvm.dbg.declare(metadata i64 %238, metadata !26, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64 %238, metadata !26, metadata !DIExpression()), !dbg !18
  %239 = getelementptr inbounds float, ptr %scales, i64 %indvars.iv.2, !dbg !18
  %240 = load float, ptr %239, align 4, !dbg !18, !tbaa !27
  %241 = getelementptr inbounds i32, ptr %compute114, i64 %238, !dbg !18
  %242 = load i32, ptr %241, align 4, !dbg !18, !tbaa !30
  %243 = sitofp i32 %242 to float, !dbg !18
  %244 = fmul float %240, %243, !dbg !18
  %245 = getelementptr inbounds float, ptr %compute, i64 %238, !dbg !18
  store float %244, ptr %245, align 4, !dbg !18, !tbaa !32
  %indvars.iv.next.2 = add nuw nsw i64 %indvars.iv.2, 1, !dbg !18
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.2, metadata !25, metadata !DIExpression()), !dbg !18
  %exitcond.2.not = icmp eq i64 %indvars.iv.next.2, 33, !dbg !18
  br i1 %exitcond.2.not, label %for_end_j.2, label %for_body_j.2, !dbg !18, !prof !34

for_end_j.2:                                      ; preds = %for_body_j.2
  call void @llvm.dbg.declare(metadata i64 3, metadata !24, metadata !DIExpression()), !dbg !18
  call void @tvm_gemmini_end(), !dbg !18
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @tvm_gemmini_validate_tensor_buffers(ptr, i64, ptr, i64, ptr, i64) local_unnamed_addr

declare void @TVMAPISetLastError(ptr) local_unnamed_addr

declare void @tvm_gemmini_begin(i64, i64, i64) local_unnamed_addr

declare void @tvm_gemmini_zero_acc(i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_a(ptr, i32, i32, i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_b(ptr, i32, i32, i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_compute(i32, i32, i32, i32, i32, i32, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_out(ptr, i32, i32, i32, i32, i32) local_unnamed_addr

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
!5 = distinct !DISubprogram(name: "stage_70_operator_kernel", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !11, !11}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12)
!12 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!13 = !{!14, !15, !16, !17}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !5, file: !1, type: !9)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "scales", arg: 3, scope: !5, file: !1, type: !11)
!17 = !DILocalVariable(name: "compute", arg: 4, scope: !5, file: !1, type: !11)
!18 = !DILocation(line: 0, scope: !5)
!19 = !DILocalVariable(name: "compute", scope: !5, file: !1, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!21 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!22 = !DILocalVariable(name: "j_0_0", scope: !5, file: !1, type: !8)
!23 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!24 = !DILocalVariable(name: "i", scope: !5, file: !1, type: !8)
!25 = !DILocalVariable(name: "j", scope: !5, file: !1, type: !8)
!26 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!27 = !{!28, !28, i64 0}
!28 = !{!"0x40a59790", !29, i64 0}
!29 = !{!"tvm-tbaa"}
!30 = !{!31, !31, i64 0}
!31 = !{!"0x409431c0", !29, i64 0}
!32 = !{!33, !33, i64 0}
!33 = !{!"0x409440f0", !29, i64 0}
!34 = !{!"branch_weights", i32 1, i32 1048575}
