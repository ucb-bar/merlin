; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dllexport noundef i32 @stage_21_operator_adaptive_pool_backward(ptr noalias nocapture readonly %x, ptr noalias nocapture writeonly %adaptive_input_gradient) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata ptr %adaptive_input_gradient, metadata !13, metadata !DIExpression()), !dbg !14
  %adaptive_gradient_delta9 = alloca [4 x float], align 16, !dbg !14
  call void @llvm.dbg.declare(metadata ptr %adaptive_gradient_delta9, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %0 = load float, ptr %x, align 4, !dbg !14, !tbaa !19
  %1 = fmul float %0, 5.000000e-01, !dbg !14
  %2 = fmul float %1, 5.000000e-01, !dbg !14
  store float %2, ptr %adaptive_gradient_delta9, align 16, !dbg !14, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  %3 = getelementptr inbounds float, ptr %x, i64 1, !dbg !14
  %4 = load float, ptr %3, align 4, !dbg !14, !tbaa !19
  %5 = fmul float %4, 5.000000e-01, !dbg !14
  %6 = fmul float %5, 5.000000e-01, !dbg !14
  %7 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 1, !dbg !14
  store float %6, ptr %7, align 4, !dbg !14, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  %8 = getelementptr inbounds float, ptr %x, i64 2, !dbg !14
  %9 = load float, ptr %8, align 4, !dbg !14, !tbaa !19
  %10 = fmul float %9, 5.000000e-01, !dbg !14
  %11 = fmul float %10, 5.000000e-01, !dbg !14
  %12 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 2, !dbg !14
  store float %11, ptr %12, align 8, !dbg !14, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  %13 = getelementptr inbounds float, ptr %x, i64 3, !dbg !14
  %14 = load float, ptr %13, align 4, !dbg !14, !tbaa !19
  %15 = fmul float %14, 5.000000e-01, !dbg !14
  %16 = fmul float %15, 5.000000e-01, !dbg !14
  %17 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 3, !dbg !14
  store float %16, ptr %17, align 4, !dbg !14, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %18 = load float, ptr %adaptive_gradient_delta9, align 16, !dbg !14, !tbaa !22
  %19 = fadd float %18, 0.000000e+00, !dbg !14
  store float %19, ptr %adaptive_input_gradient, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %20 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 1
  store float %19, ptr %20, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %21 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 1, !dbg !14
  %22 = load float, ptr %21, align 4, !dbg !14, !tbaa !22
  %23 = fadd float %22, 0.000000e+00, !dbg !14
  %24 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 2
  store float %23, ptr %24, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %25 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 3
  store float %23, ptr %25, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  %26 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 4
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %27 = load float, ptr %adaptive_gradient_delta9, align 16, !dbg !14, !tbaa !22
  %28 = fadd float %27, 0.000000e+00, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  br label %for_body_window_0.144, !dbg !14

for_body_window_0.144:                            ; preds = %for_body_window_0.144, %entry
  %indvars.iv.142 = phi i64 [ %indvars.iv.next.143, %for_body_window_0.144 ], [ 1, %entry ]
  %29 = phi float [ %33, %for_body_window_0.144 ], [ %28, %entry ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.142, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %30 = shl nuw i64 %indvars.iv.142, 1, !dbg !14
  %31 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 %30, !dbg !14
  %32 = load float, ptr %31, align 8, !dbg !14, !tbaa !22
  %33 = fadd float %32, %29, !dbg !14
  %indvars.iv.next.143 = add nuw nsw i64 %indvars.iv.142, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.143, metadata !24, metadata !DIExpression()), !dbg !14
  %exitcond26.not.1 = icmp eq i64 %indvars.iv.next.143, 2, !dbg !14
  br i1 %exitcond26.not.1, label %for_end_window_0.146, label %for_body_window_0.144, !dbg !14, !prof !29, !llvm.loop !30

for_end_window_0.146:                             ; preds = %for_body_window_0.144
  store float %33, ptr %26, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  br label %for_body_window_0.1.1, !dbg !14

for_body_window_0.1.1:                            ; preds = %for_body_window_0.1.1, %for_end_window_0.146
  %indvars.iv.1.1 = phi i64 [ %indvars.iv.next.1.1, %for_body_window_0.1.1 ], [ 1, %for_end_window_0.146 ]
  %34 = phi float [ %38, %for_body_window_0.1.1 ], [ %28, %for_end_window_0.146 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.1.1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %35 = shl nuw i64 %indvars.iv.1.1, 1, !dbg !14
  %36 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 %35, !dbg !14
  %37 = load float, ptr %36, align 8, !dbg !14, !tbaa !22
  %38 = fadd float %37, %34, !dbg !14
  %indvars.iv.next.1.1 = add nuw nsw i64 %indvars.iv.1.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.1.1, metadata !24, metadata !DIExpression()), !dbg !14
  %exitcond25.not.1 = icmp eq i64 %indvars.iv.next.1.1, 2, !dbg !14
  br i1 %exitcond25.not.1, label %for_end_window_0.1.1, label %for_body_window_0.1.1, !dbg !14, !prof !29, !llvm.loop !30

for_end_window_0.1.1:                             ; preds = %for_body_window_0.1.1
  %39 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 5
  store float %38, ptr %39, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  %40 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 6
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %41 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 1, !dbg !14
  %42 = load float, ptr %41, align 4, !dbg !14, !tbaa !22
  %43 = fadd float %42, 0.000000e+00, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  br label %for_body_window_0.2.1, !dbg !14

for_body_window_0.2.1:                            ; preds = %for_body_window_0.2.1, %for_end_window_0.1.1
  %indvars.iv.2.1 = phi i64 [ %indvars.iv.next.2.1, %for_body_window_0.2.1 ], [ 1, %for_end_window_0.1.1 ]
  %44 = phi float [ %49, %for_body_window_0.2.1 ], [ %43, %for_end_window_0.1.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.2.1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %45 = shl nuw i64 %indvars.iv.2.1, 1, !dbg !14
  %46 = or disjoint i64 %45, 1, !dbg !14
  %47 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 %46, !dbg !14
  %48 = load float, ptr %47, align 4, !dbg !14, !tbaa !22
  %49 = fadd float %48, %44, !dbg !14
  %indvars.iv.next.2.1 = add nuw nsw i64 %indvars.iv.2.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.2.1, metadata !24, metadata !DIExpression()), !dbg !14
  %exitcond24.not.1 = icmp eq i64 %indvars.iv.next.2.1, 2, !dbg !14
  br i1 %exitcond24.not.1, label %for_end_window_0.2.1, label %for_body_window_0.2.1, !dbg !14, !prof !29, !llvm.loop !30

for_end_window_0.2.1:                             ; preds = %for_body_window_0.2.1
  store float %49, ptr %40, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  br label %for_body_window_0.3.1, !dbg !14

for_body_window_0.3.1:                            ; preds = %for_body_window_0.3.1, %for_end_window_0.2.1
  %indvars.iv.3.1 = phi i64 [ %indvars.iv.next.3.1, %for_body_window_0.3.1 ], [ 1, %for_end_window_0.2.1 ]
  %50 = phi float [ %55, %for_body_window_0.3.1 ], [ %43, %for_end_window_0.2.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.3.1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !26, metadata !DIExpression()), !dbg !14
  %51 = shl nuw i64 %indvars.iv.3.1, 1, !dbg !14
  %52 = or disjoint i64 %51, 1, !dbg !14
  %53 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 %52, !dbg !14
  %54 = load float, ptr %53, align 4, !dbg !14, !tbaa !22
  %55 = fadd float %54, %50, !dbg !14
  %indvars.iv.next.3.1 = add nuw nsw i64 %indvars.iv.3.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next.3.1, metadata !24, metadata !DIExpression()), !dbg !14
  %exitcond.not.1 = icmp eq i64 %indvars.iv.next.3.1, 2, !dbg !14
  br i1 %exitcond.not.1, label %for_end_window_0.3.1, label %for_body_window_0.3.1, !dbg !14, !prof !29, !llvm.loop !30

for_end_window_0.3.1:                             ; preds = %for_body_window_0.3.1
  %56 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 7
  store float %55, ptr %56, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %57 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 2, !dbg !14
  %58 = load float, ptr %57, align 8, !dbg !14, !tbaa !22
  %59 = fadd float %58, 0.000000e+00, !dbg !14
  %60 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 8
  store float %59, ptr %60, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression(DW_OP_constu, 1, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %61 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 9
  store float %59, ptr %61, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 10, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 10, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %62 = getelementptr inbounds float, ptr %adaptive_gradient_delta9, i64 3, !dbg !14
  %63 = load float, ptr %62, align 4, !dbg !14, !tbaa !22
  %64 = fadd float %63, 0.000000e+00, !dbg !14
  %65 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 10
  store float %64, ptr %65, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression(DW_OP_constu, 3, DW_OP_or)), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  %66 = getelementptr inbounds float, ptr %adaptive_input_gradient, i64 11
  store float %64, ptr %66, align 4, !dbg !14, !tbaa !27
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !16, metadata !DIExpression()), !dbg !14
  ret i32 0, !dbg !14
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

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
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
!5 = distinct !DISubprogram(name: "stage_21_operator_adaptive_pool_backward", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "adaptive_input_gradient", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocation(line: 0, scope: !5)
!15 = !DILocalVariable(name: "adaptive_gradient_delta", scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "i2", scope: !5, file: !1, type: !8)
!17 = !DILocalVariable(name: "i3", scope: !5, file: !1, type: !8)
!18 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!19 = !{!20, !20, i64 0}
!20 = !{!"0x409a9650", !21, i64 0}
!21 = !{!"tvm-tbaa"}
!22 = !{!23, !23, i64 0}
!23 = !{!"0x409b4a00", !21, i64 0}
!24 = !DILocalVariable(name: "window_0", scope: !5, file: !1, type: !8)
!25 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!26 = !DILocalVariable(name: "cse_var_3", scope: !5, file: !1, type: !8)
!27 = !{!28, !28, i64 0}
!28 = !{!"0x409b6370", !21, i64 0}
!29 = !{!"branch_weights", i32 1, i32 1048574}
!30 = distinct !{!30, !31}
!31 = !{!"llvm.loop.peeled.count", i32 1}
