; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dllexport noundef i32 @stage_20_operator_avg_pool2d_backward(ptr noalias nocapture readonly %x, ptr noalias nocapture readnone %y, ptr noalias nocapture writeonly %T_pool_grad) local_unnamed_addr #0 !dbg !5 {
for_end_ww.1.3.2:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata ptr %y, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata ptr %T_pool_grad, metadata !14, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  %0 = load float, ptr %x, align 4, !dbg !15, !tbaa !23
  %1 = fmul float %0, 2.500000e-01, !dbg !15
  %2 = fadd float %1, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %2, ptr %T_pool_grad, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  %3 = getelementptr inbounds float, ptr %T_pool_grad, i64 1
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  %4 = getelementptr inbounds float, ptr %x, i64 1, !dbg !15
  %5 = load float, ptr %4, align 4, !dbg !15, !tbaa !23
  %6 = fmul float %5, 2.500000e-01, !dbg !15
  %7 = fadd float %6, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  %8 = load float, ptr %x, align 4, !dbg !15, !tbaa !23
  %9 = fmul float %8, 2.500000e-01, !dbg !15
  %10 = fadd float %9, %7, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %10, ptr %3, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  %11 = getelementptr inbounds float, ptr %T_pool_grad, i64 2
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  %12 = getelementptr inbounds float, ptr %x, i64 2, !dbg !15
  %13 = load float, ptr %12, align 4, !dbg !15, !tbaa !23
  %14 = fmul float %13, 2.500000e-01, !dbg !15
  %15 = fadd float %14, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  %16 = getelementptr inbounds float, ptr %x, i64 1, !dbg !15
  %17 = load float, ptr %16, align 4, !dbg !15, !tbaa !23
  %18 = fmul float %17, 2.500000e-01, !dbg !15
  %19 = fadd float %18, %15, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %19, ptr %11, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  %20 = getelementptr inbounds float, ptr %T_pool_grad, i64 3
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  %21 = getelementptr inbounds float, ptr %x, i64 2, !dbg !15
  %22 = load float, ptr %21, align 4, !dbg !15, !tbaa !23
  %23 = fmul float %22, 2.500000e-01, !dbg !15
  %24 = fadd float %23, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %24, ptr %20, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !15
  %25 = getelementptr inbounds float, ptr %T_pool_grad, i64 4
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  %26 = getelementptr inbounds float, ptr %x, i64 3, !dbg !15
  %27 = load float, ptr %26, align 4, !dbg !15, !tbaa !23
  %28 = fmul float %27, 2.500000e-01, !dbg !15
  %29 = fadd float %28, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %30 = load float, ptr %x, align 4, !dbg !15, !tbaa !23
  %31 = fmul float %30, 2.500000e-01, !dbg !15
  %32 = fadd float %31, %29, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %32, ptr %25, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  %33 = getelementptr inbounds float, ptr %T_pool_grad, i64 5
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  %34 = getelementptr inbounds float, ptr %x, i64 4, !dbg !15
  %35 = load float, ptr %34, align 4, !dbg !15, !tbaa !23
  %36 = fmul float %35, 2.500000e-01, !dbg !15
  %37 = fadd float %36, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  %38 = getelementptr inbounds float, ptr %x, i64 3, !dbg !15
  %39 = load float, ptr %38, align 4, !dbg !15, !tbaa !23
  %40 = fmul float %39, 2.500000e-01, !dbg !15
  %41 = fadd float %40, %37, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  %42 = getelementptr inbounds float, ptr %x, i64 1, !dbg !15
  %43 = load float, ptr %42, align 4, !dbg !15, !tbaa !23
  %44 = fmul float %43, 2.500000e-01, !dbg !15
  %45 = fadd float %44, %41, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %46 = load float, ptr %x, align 4, !dbg !15, !tbaa !23
  %47 = fmul float %46, 2.500000e-01, !dbg !15
  %48 = fadd float %47, %45, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %48, ptr %33, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  %49 = getelementptr inbounds float, ptr %T_pool_grad, i64 6
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  %50 = getelementptr inbounds float, ptr %x, i64 5, !dbg !15
  %51 = load float, ptr %50, align 4, !dbg !15, !tbaa !23
  %52 = fmul float %51, 2.500000e-01, !dbg !15
  %53 = fadd float %52, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  %54 = getelementptr inbounds float, ptr %x, i64 4, !dbg !15
  %55 = load float, ptr %54, align 4, !dbg !15, !tbaa !23
  %56 = fmul float %55, 2.500000e-01, !dbg !15
  %57 = fadd float %56, %53, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  %58 = getelementptr inbounds float, ptr %x, i64 2, !dbg !15
  %59 = load float, ptr %58, align 4, !dbg !15, !tbaa !23
  %60 = fmul float %59, 2.500000e-01, !dbg !15
  %61 = fadd float %60, %57, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %62 = getelementptr inbounds float, ptr %x, i64 1, !dbg !15
  %63 = load float, ptr %62, align 4, !dbg !15, !tbaa !23
  %64 = fmul float %63, 2.500000e-01, !dbg !15
  %65 = fadd float %64, %61, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %65, ptr %49, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  %66 = getelementptr inbounds float, ptr %T_pool_grad, i64 7
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  %67 = getelementptr inbounds float, ptr %x, i64 5, !dbg !15
  %68 = load float, ptr %67, align 4, !dbg !15, !tbaa !23
  %69 = fmul float %68, 2.500000e-01, !dbg !15
  %70 = fadd float %69, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %71 = getelementptr inbounds float, ptr %x, i64 2, !dbg !15
  %72 = load float, ptr %71, align 4, !dbg !15, !tbaa !23
  %73 = fmul float %72, 2.500000e-01, !dbg !15
  %74 = fadd float %73, %70, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %74, ptr %66, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !15
  %75 = getelementptr inbounds float, ptr %T_pool_grad, i64 8
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  %76 = getelementptr inbounds float, ptr %x, i64 3, !dbg !15
  %77 = load float, ptr %76, align 4, !dbg !15, !tbaa !23
  %78 = fmul float %77, 2.500000e-01, !dbg !15
  %79 = fadd float %78, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 -1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %79, ptr %75, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  %80 = getelementptr inbounds float, ptr %T_pool_grad, i64 9
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  %81 = getelementptr inbounds float, ptr %x, i64 4, !dbg !15
  %82 = load float, ptr %81, align 4, !dbg !15, !tbaa !23
  %83 = fmul float %82, 2.500000e-01, !dbg !15
  %84 = fadd float %83, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %85 = getelementptr inbounds float, ptr %x, i64 3, !dbg !15
  %86 = load float, ptr %85, align 4, !dbg !15, !tbaa !23
  %87 = fmul float %86, 2.500000e-01, !dbg !15
  %88 = fadd float %87, %84, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %88, ptr %80, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  %89 = getelementptr inbounds float, ptr %T_pool_grad, i64 10
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  %90 = getelementptr inbounds float, ptr %x, i64 5, !dbg !15
  %91 = load float, ptr %90, align 4, !dbg !15, !tbaa !23
  %92 = fmul float %91, 2.500000e-01, !dbg !15
  %93 = fadd float %92, 0.000000e+00, !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %94 = getelementptr inbounds float, ptr %x, i64 4, !dbg !15
  %95 = load float, ptr %94, align 4, !dbg !15, !tbaa !23
  %96 = fmul float %95, 2.500000e-01, !dbg !15
  %97 = fadd float %96, %93, !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %97, ptr %89, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 2, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !20, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !22, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !15
  %98 = getelementptr inbounds float, ptr %x, i64 5, !dbg !15
  %99 = load float, ptr %98, align 4, !dbg !15, !tbaa !23
  %100 = fmul float %99, 2.500000e-01, !dbg !15
  %101 = fadd float %100, 0.000000e+00, !dbg !15
  %102 = getelementptr inbounds float, ptr %T_pool_grad, i64 11
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !15
  store float %101, ptr %102, align 4, !tbaa !26
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !16, metadata !DIExpression()), !dbg !15
  ret i32 0, !dbg !15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

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

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
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
!5 = distinct !DISubprogram(name: "stage_20_operator_avg_pool2d_backward", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocalVariable(name: "T_pool_grad", arg: 3, scope: !5, file: !1, type: !9)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocalVariable(name: "ax2", scope: !5, file: !1, type: !8)
!17 = !DILocalVariable(name: "ax3", scope: !5, file: !1, type: !8)
!18 = !DILocalVariable(name: "wh", scope: !5, file: !1, type: !8)
!19 = !DILocalVariable(name: "ww", scope: !5, file: !1, type: !8)
!20 = !DILocalVariable(name: "cse_var_3", scope: !5, file: !1, type: !8)
!21 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!22 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!23 = !{!24, !24, i64 0}
!24 = !{!"0x40c68850", !25, i64 0}
!25 = !{!"tvm-tbaa"}
!26 = !{!27, !27, i64 0}
!27 = !{!"0x40c8e2d0", !25, i64 0}
