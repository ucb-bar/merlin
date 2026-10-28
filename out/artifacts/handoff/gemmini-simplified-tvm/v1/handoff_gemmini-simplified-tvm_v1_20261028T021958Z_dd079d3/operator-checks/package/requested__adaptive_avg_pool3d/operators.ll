; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dllexport noundef i32 @stage_19_operator_adaptive_avg_pool3d(ptr noalias nocapture readonly %x, ptr noalias nocapture writeonly %adaptive_pool_avg) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata ptr %adaptive_pool_avg, metadata !13, metadata !DIExpression()), !dbg !14
  %adaptive_pool_sum13 = alloca [8 x float], align 16, !dbg !14
  call void @llvm.dbg.declare(metadata ptr %adaptive_pool_sum13, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !14
  %invariant.gep = getelementptr float, ptr %x, i64 1
  %invariant.gep109 = getelementptr float, ptr %x, i64 4
  %invariant.gep111 = getelementptr float, ptr %x, i64 1
  %invariant.gep113 = getelementptr float, ptr %x, i64 2
  %invariant.gep115 = getelementptr float, ptr %x, i64 2
  %invariant.gep117 = getelementptr float, ptr %x, i64 1
  %invariant.gep119 = getelementptr float, ptr %x, i64 2
  %invariant.gep121 = getelementptr float, ptr %x, i64 1
  br label %for_begin_ax3.preheader, !dbg !14

for_begin_ax3.preheader:                          ; preds = %entry, %for_end_ax3
  %indvars.iv41 = phi i32 [ 0, %entry ], [ %indvars.iv.next42, %for_end_ax3 ]
  %0 = phi i1 [ false, %entry ], [ true, %for_end_ax3 ]
  %1 = phi i1 [ true, %entry ], [ false, %for_end_ax3 ]
  %indvar = phi i64 [ 0, %entry ], [ 1, %for_end_ax3 ]
  %ax229 = phi i32 [ 0, %entry ], [ %113, %for_end_ax3 ]
  call void @llvm.dbg.declare(metadata i64 %indvar, metadata !16, metadata !DIExpression()), !dbg !14
  %2 = shl nuw nsw i64 %indvar, 2
  %3 = trunc i64 %indvar to i32
  %4 = mul nuw nsw i32 %3, 3
  %5 = add nuw nsw i32 %4, 3
  %cse_var_1 = lshr i32 %5, 1
  %6 = add nuw nsw i32 %cse_var_1, 1
  %7 = select i1 %0, i32 3, i32 %6
  %8 = icmp ugt i32 %7, %ax229
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  br i1 %8, label %for_body_ax4.split.us.split.us.us.us, label %for_begin_ax4.preheader.preheader, !dbg !14, !prof !18

for_begin_ax4.preheader.preheader:                ; preds = %for_begin_ax3.preheader
  %9 = shl nuw nsw i64 %indvar, 4
  %scevgep36 = getelementptr i8, ptr %adaptive_pool_sum13, i64 %9
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %scevgep36, i8 0, i64 16, i1 false), !dbg !14, !tbaa !19
  call void @llvm.dbg.declare(metadata i32 poison, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %cse_var_1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  br label %for_end_ax3, !dbg !14

for_body_ax4.split.us.split.us.us.us.1:           ; preds = %for_body_rv0.us.us.us.us.us.1
  call void @llvm.dbg.declare(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !14
  store float %112, ptr %87, align 16, !tbaa !19
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  %10 = or disjoint i64 %2, 1, !dbg !14
  %11 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 %10, !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %cse_var_1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  %wide.trip.count43.149 = zext i32 %86 to i64, !dbg !14
  br label %for_body_rv0.us.us.us.us.us.153, !dbg !14

for_body_rv0.us.us.us.us.us.153:                  ; preds = %for_body_rv0.us.us.us.us.us.153, %for_body_ax4.split.us.split.us.us.us.1
  %indvars.iv39.150 = phi i64 [ %indvars.iv.next40.159, %for_body_rv0.us.us.us.us.us.153 ], [ 0, %for_body_ax4.split.us.split.us.us.us.1 ]
  %.lcssa.us16.us.us.us.us.151 = phi float [ %23, %for_body_rv0.us.us.us.us.us.153 ], [ 0.000000e+00, %for_body_ax4.split.us.split.us.us.us.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.150, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %12 = add nuw nsw i64 %indvars.iv39.150, %indvar
  %13 = mul i64 %12, 85899345920, !dbg !14
  %sext96 = ashr exact i64 %13, 32, !dbg !14
  %14 = or disjoint i64 %sext96, 2, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %15 = getelementptr inbounds float, ptr %x, i64 %14, !dbg !14
  %16 = load float, ptr %15, align 4, !dbg !14, !tbaa !29
  %17 = fadd float %16, %.lcssa.us16.us.us.us.us.151, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %18 = or disjoint i64 %sext96, 3, !dbg !14
  %19 = getelementptr inbounds float, ptr %x, i64 %18, !dbg !14
  %20 = load float, ptr %19, align 4, !dbg !14, !tbaa !29
  %21 = fadd float %20, %17, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %gep110 = getelementptr float, ptr %invariant.gep109, i64 %sext96, !dbg !14
  %22 = load float, ptr %gep110, align 4, !dbg !14, !tbaa !29
  %23 = fadd float %22, %21, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.159 = add nuw nsw i64 %indvars.iv39.150, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.159, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.160.not = icmp eq i64 %indvars.iv.next40.159, %wide.trip.count43.149, !dbg !14
  br i1 %exitcond44.160.not, label %for_body_rv0.us.us.us.us.us.1.1, label %for_body_rv0.us.us.us.us.us.153, !dbg !14, !prof !31

for_body_rv0.us.us.us.us.us.1.1:                  ; preds = %for_body_rv0.us.us.us.us.us.153, %for_body_rv0.us.us.us.us.us.1.1
  %indvars.iv39.1.1 = phi i64 [ %indvars.iv.next40.1.1, %for_body_rv0.us.us.us.us.us.1.1 ], [ 0, %for_body_rv0.us.us.us.us.us.153 ]
  %.lcssa.us16.us.us.us.us.1.1 = phi float [ %33, %for_body_rv0.us.us.us.us.us.1.1 ], [ %23, %for_body_rv0.us.us.us.us.us.153 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.1.1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %24 = add nuw nsw i64 %indvars.iv39.1.1, %indvar
  %25 = mul i64 %24, 85899345920, !dbg !14
  %sext97 = add i64 %25, 30064771072, !dbg !14
  %26 = ashr exact i64 %sext97, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %27 = getelementptr inbounds float, ptr %x, i64 %26, !dbg !14
  %28 = load float, ptr %27, align 4, !dbg !14, !tbaa !29
  %29 = fadd float %28, %.lcssa.us16.us.us.us.us.1.1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %gep112 = getelementptr float, ptr %invariant.gep111, i64 %26, !dbg !14
  %30 = load float, ptr %gep112, align 4, !dbg !14, !tbaa !29
  %31 = fadd float %30, %29, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %10, metadata !28, metadata !DIExpression()), !dbg !14
  %gep114 = getelementptr float, ptr %invariant.gep113, i64 %26, !dbg !14
  %32 = load float, ptr %gep114, align 4, !dbg !14, !tbaa !29
  %33 = fadd float %32, %31, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.1.1 = add nuw nsw i64 %indvars.iv39.1.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.1.1, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.1.1.not = icmp eq i64 %indvars.iv.next40.1.1, %wide.trip.count43.149, !dbg !14
  br i1 %exitcond44.1.1.not, label %for_body_ax4.split.us.split.us.us.us.172, label %for_body_rv0.us.us.us.us.us.1.1, !dbg !14, !prof !31

for_body_ax4.split.us.split.us.us.us.172:         ; preds = %for_body_rv0.us.us.us.us.us.1.1
  call void @llvm.dbg.declare(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !14
  store float %33, ptr %11, align 4, !tbaa !19
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  %34 = or disjoint i64 %2, 2
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  %35 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 %34, !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %cse_var_1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  %wide.trip.count43.173 = zext i32 %86 to i64, !dbg !14
  br label %for_body_rv0.us.us.us.us.us.177, !dbg !14

for_body_rv0.us.us.us.us.us.177:                  ; preds = %for_body_rv0.us.us.us.us.us.177, %for_body_ax4.split.us.split.us.us.us.172
  %indvars.iv39.175 = phi i64 [ %indvars.iv.next40.178, %for_body_rv0.us.us.us.us.us.177 ], [ 0, %for_body_ax4.split.us.split.us.us.us.172 ]
  %.lcssa.us16.us.us.us.us.176 = phi float [ %47, %for_body_rv0.us.us.us.us.us.177 ], [ 0.000000e+00, %for_body_ax4.split.us.split.us.us.us.172 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.175, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %36 = add nuw nsw i64 %indvars.iv39.175, %indvar
  %37 = mul i64 %36, 85899345920, !dbg !14
  %sext98 = add i64 %37, 42949672960, !dbg !14
  %38 = ashr exact i64 %sext98, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %39 = getelementptr inbounds float, ptr %x, i64 %38, !dbg !14
  %40 = load float, ptr %39, align 4, !dbg !14, !tbaa !29
  %41 = fadd float %40, %.lcssa.us16.us.us.us.us.176, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %42 = or disjoint i64 %38, 1, !dbg !14
  %43 = getelementptr inbounds float, ptr %x, i64 %42, !dbg !14
  %44 = load float, ptr %43, align 4, !dbg !14, !tbaa !29
  %45 = fadd float %44, %41, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %gep116 = getelementptr float, ptr %invariant.gep115, i64 %38, !dbg !14
  %46 = load float, ptr %gep116, align 4, !dbg !14, !tbaa !29
  %47 = fadd float %46, %45, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.178 = add nuw nsw i64 %indvars.iv39.175, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.178, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.179.not = icmp eq i64 %indvars.iv.next40.178, %wide.trip.count43.173, !dbg !14
  br i1 %exitcond44.179.not, label %for_body_rv0.us.us.us.us.us.1.186, label %for_body_rv0.us.us.us.us.us.177, !dbg !14, !prof !31

for_body_rv0.us.us.us.us.us.1.186:                ; preds = %for_body_rv0.us.us.us.us.us.177, %for_body_rv0.us.us.us.us.us.1.186
  %indvars.iv39.1.184 = phi i64 [ %indvars.iv.next40.1.187, %for_body_rv0.us.us.us.us.us.1.186 ], [ 0, %for_body_rv0.us.us.us.us.us.177 ]
  %.lcssa.us16.us.us.us.us.1.185 = phi float [ %57, %for_body_rv0.us.us.us.us.us.1.186 ], [ %47, %for_body_rv0.us.us.us.us.us.177 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.1.184, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %48 = add nuw nsw i64 %indvars.iv39.1.184, %indvar
  %49 = mul i64 %48, 85899345920, !dbg !14
  %sext99 = add i64 %49, 64424509440, !dbg !14
  %50 = ashr exact i64 %sext99, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %51 = getelementptr inbounds float, ptr %x, i64 %50, !dbg !14
  %52 = load float, ptr %51, align 4, !dbg !14, !tbaa !29
  %53 = fadd float %52, %.lcssa.us16.us.us.us.us.1.185, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %gep118 = getelementptr float, ptr %invariant.gep117, i64 %50, !dbg !14
  %54 = load float, ptr %gep118, align 4, !dbg !14, !tbaa !29
  %55 = fadd float %54, %53, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %34, metadata !28, metadata !DIExpression()), !dbg !14
  %gep120 = getelementptr float, ptr %invariant.gep119, i64 %50, !dbg !14
  %56 = load float, ptr %gep120, align 4, !dbg !14, !tbaa !29
  %57 = fadd float %56, %55, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.1.187 = add nuw nsw i64 %indvars.iv39.1.184, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.1.187, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.1.188.not = icmp eq i64 %indvars.iv.next40.1.187, %wide.trip.count43.173, !dbg !14
  br i1 %exitcond44.1.188.not, label %for_body_ax4.split.us.split.us.us.us.1.1, label %for_body_rv0.us.us.us.us.us.1.186, !dbg !14, !prof !31

for_body_ax4.split.us.split.us.us.us.1.1:         ; preds = %for_body_rv0.us.us.us.us.us.1.186
  call void @llvm.dbg.declare(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !14
  store float %57, ptr %35, align 8, !tbaa !19
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  %58 = or disjoint i64 %2, 3, !dbg !14
  %59 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 %58, !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %cse_var_1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  %wide.trip.count43.149.1 = zext i32 %86 to i64, !dbg !14
  br label %for_body_rv0.us.us.us.us.us.153.1, !dbg !14

for_body_rv0.us.us.us.us.us.153.1:                ; preds = %for_body_rv0.us.us.us.us.us.153.1, %for_body_ax4.split.us.split.us.us.us.1.1
  %indvars.iv39.150.1 = phi i64 [ %indvars.iv.next40.159.1, %for_body_rv0.us.us.us.us.us.153.1 ], [ 0, %for_body_ax4.split.us.split.us.us.us.1.1 ]
  %.lcssa.us16.us.us.us.us.151.1 = phi float [ %73, %for_body_rv0.us.us.us.us.us.153.1 ], [ 0.000000e+00, %for_body_ax4.split.us.split.us.us.us.1.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.150.1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %60 = add nuw nsw i64 %indvars.iv39.150.1, %indvar
  %61 = mul i64 %60, 85899345920, !dbg !14
  %sext100 = add i64 %61, 51539607552, !dbg !14
  %62 = ashr exact i64 %sext100, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %63 = getelementptr inbounds float, ptr %x, i64 %62, !dbg !14
  %64 = load float, ptr %63, align 4, !dbg !14, !tbaa !29
  %65 = fadd float %64, %.lcssa.us16.us.us.us.us.151.1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %66 = or disjoint i64 %62, 1, !dbg !14
  %67 = getelementptr inbounds float, ptr %x, i64 %66, !dbg !14
  %68 = load float, ptr %67, align 4, !dbg !14, !tbaa !29
  %69 = fadd float %68, %65, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %70 = or disjoint i64 %62, 2, !dbg !14
  %71 = getelementptr inbounds float, ptr %x, i64 %70, !dbg !14
  %72 = load float, ptr %71, align 4, !dbg !14, !tbaa !29
  %73 = fadd float %72, %69, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.159.1 = add nuw nsw i64 %indvars.iv39.150.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.159.1, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.160.1.not = icmp eq i64 %indvars.iv.next40.159.1, %wide.trip.count43.149.1, !dbg !14
  br i1 %exitcond44.160.1.not, label %for_body_rv0.us.us.us.us.us.1.1.1, label %for_body_rv0.us.us.us.us.us.153.1, !dbg !14, !prof !31

for_body_rv0.us.us.us.us.us.1.1.1:                ; preds = %for_body_rv0.us.us.us.us.us.153.1, %for_body_rv0.us.us.us.us.us.1.1.1
  %indvars.iv39.1.1.1 = phi i64 [ %indvars.iv.next40.1.1.1, %for_body_rv0.us.us.us.us.us.1.1.1 ], [ 0, %for_body_rv0.us.us.us.us.us.153.1 ]
  %.lcssa.us16.us.us.us.us.1.1.1 = phi float [ %85, %for_body_rv0.us.us.us.us.us.1.1.1 ], [ %73, %for_body_rv0.us.us.us.us.us.153.1 ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.1.1.1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %74 = add nuw nsw i64 %indvars.iv39.1.1.1, %indvar
  %75 = mul i64 %74, 85899345920, !dbg !14
  %sext101 = add i64 %75, 73014444032, !dbg !14
  %76 = ashr exact i64 %sext101, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %77 = getelementptr inbounds float, ptr %x, i64 %76, !dbg !14
  %78 = load float, ptr %77, align 4, !dbg !14, !tbaa !29
  %79 = fadd float %78, %.lcssa.us16.us.us.us.us.1.1.1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %gep122 = getelementptr float, ptr %invariant.gep121, i64 %76, !dbg !14
  %80 = load float, ptr %gep122, align 4, !dbg !14, !tbaa !29
  %81 = fadd float %80, %79, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %58, metadata !28, metadata !DIExpression()), !dbg !14
  %82 = or disjoint i64 %76, 2, !dbg !14
  %83 = getelementptr inbounds float, ptr %x, i64 %82, !dbg !14
  %84 = load float, ptr %83, align 4, !dbg !14, !tbaa !29
  %85 = fadd float %84, %81, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.1.1.1 = add nuw nsw i64 %indvars.iv39.1.1.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.1.1.1, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.1.1.1.not = icmp eq i64 %indvars.iv.next40.1.1.1, %wide.trip.count43.149.1, !dbg !14
  br i1 %exitcond44.1.1.1.not, label %for_end_rv1.split.us.us.us.1.1, label %for_body_rv0.us.us.us.us.us.1.1.1, !dbg !14, !prof !31

for_end_rv1.split.us.us.us.1.1:                   ; preds = %for_body_rv0.us.us.us.us.us.1.1.1
  call void @llvm.dbg.declare(metadata i32 2, metadata !23, metadata !DIExpression()), !dbg !14
  store float %85, ptr %59, align 4, !tbaa !19
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  br label %for_end_ax3, !dbg !14

for_body_ax4.split.us.split.us.us.us:             ; preds = %for_begin_ax3.preheader
  %86 = add nsw i32 %7, %indvars.iv41, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  %87 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 %2, !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !23, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %cse_var_1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  %wide.trip.count43 = zext i32 %86 to i64, !dbg !14
  br label %for_body_rv0.us.us.us.us.us, !dbg !14

for_body_rv0.us.us.us.us.us:                      ; preds = %for_body_rv0.us.us.us.us.us, %for_body_ax4.split.us.split.us.us.us
  %indvars.iv39 = phi i64 [ %indvars.iv.next40, %for_body_rv0.us.us.us.us.us ], [ 0, %for_body_ax4.split.us.split.us.us.us ]
  %.lcssa.us16.us.us.us.us = phi float [ %100, %for_body_rv0.us.us.us.us.us ], [ 0.000000e+00, %for_body_ax4.split.us.split.us.us.us ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %88 = add nuw nsw i64 %indvars.iv39, %indvar
  %sext = mul i64 %88, 85899345920, !dbg !14
  %89 = ashr exact i64 %sext, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %90 = getelementptr inbounds float, ptr %x, i64 %89, !dbg !14
  %91 = load float, ptr %90, align 4, !dbg !14, !tbaa !29
  %92 = fadd float %91, %.lcssa.us16.us.us.us.us, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %93 = or disjoint i64 %89, 1, !dbg !14
  %94 = getelementptr inbounds float, ptr %x, i64 %93, !dbg !14
  %95 = load float, ptr %94, align 4, !dbg !14, !tbaa !29
  %96 = fadd float %95, %92, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %97 = or disjoint i64 %89, 2, !dbg !14
  %98 = getelementptr inbounds float, ptr %x, i64 %97, !dbg !14
  %99 = load float, ptr %98, align 4, !dbg !14, !tbaa !29
  %100 = fadd float %99, %96, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.not = icmp eq i64 %indvars.iv.next40, %wide.trip.count43, !dbg !14
  br i1 %exitcond44.not, label %for_body_rv0.us.us.us.us.us.1, label %for_body_rv0.us.us.us.us.us, !dbg !14, !prof !31

for_body_rv0.us.us.us.us.us.1:                    ; preds = %for_body_rv0.us.us.us.us.us, %for_body_rv0.us.us.us.us.us.1
  %indvars.iv39.1 = phi i64 [ %indvars.iv.next40.1, %for_body_rv0.us.us.us.us.us.1 ], [ 0, %for_body_rv0.us.us.us.us.us ]
  %.lcssa.us16.us.us.us.us.1 = phi float [ %112, %for_body_rv0.us.us.us.us.us.1 ], [ %100, %for_body_rv0.us.us.us.us.us ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv39.1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !26, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !27, metadata !DIExpression()), !dbg !14
  %101 = add nuw nsw i64 %indvars.iv39.1, %indvar
  %102 = mul i64 %101, 85899345920, !dbg !14
  %sext95 = add i64 %102, 21474836480, !dbg !14
  %103 = ashr exact i64 %sext95, 32, !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %104 = getelementptr inbounds float, ptr %x, i64 %103, !dbg !14
  %105 = load float, ptr %104, align 4, !dbg !14, !tbaa !29
  %106 = fadd float %105, %.lcssa.us16.us.us.us.us.1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %gep = getelementptr float, ptr %invariant.gep, i64 %103, !dbg !14
  %107 = load float, ptr %gep, align 4, !dbg !14, !tbaa !29
  %108 = fadd float %107, %106, !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !27, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 %2, metadata !28, metadata !DIExpression()), !dbg !14
  %109 = or disjoint i64 %103, 2, !dbg !14
  %110 = getelementptr inbounds float, ptr %x, i64 %109, !dbg !14
  %111 = load float, ptr %110, align 4, !dbg !14, !tbaa !29
  %112 = fadd float %111, %108, !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !27, metadata !DIExpression()), !dbg !14
  %indvars.iv.next40.1 = add nuw nsw i64 %indvars.iv39.1, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next40.1, metadata !25, metadata !DIExpression()), !dbg !14
  %exitcond44.1.not = icmp eq i64 %indvars.iv.next40.1, %wide.trip.count43, !dbg !14
  br i1 %exitcond44.1.not, label %for_body_ax4.split.us.split.us.us.us.1, label %for_body_rv0.us.us.us.us.us.1, !dbg !14, !prof !31

for_end_ax3:                                      ; preds = %for_begin_ax4.preheader.preheader, %for_end_rv1.split.us.us.us.1.1
  %113 = add nuw nsw i32 %ax229, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvar, metadata !16, metadata !DIExpression(DW_OP_plus_uconst, 1)), !dbg !14
  %indvars.iv.next42 = add nsw i32 %indvars.iv41, -1, !dbg !14
  br i1 %1, label %for_begin_ax3.preheader, label %for_begin_ax35.preheader.preheader, !dbg !14, !prof !32

for_begin_ax35.preheader.preheader:               ; preds = %for_end_ax3
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !35, metadata !DIExpression()), !dbg !14
  %114 = load float, ptr %adaptive_pool_sum13, align 16, !dbg !14, !tbaa !19
  %115 = fdiv float %114, 1.200000e+01, !dbg !14
  store float %115, ptr %adaptive_pool_avg, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !35, metadata !DIExpression()), !dbg !14
  %116 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 1, !dbg !14
  %117 = load float, ptr %116, align 4, !dbg !14, !tbaa !19
  %118 = fdiv float %117, 1.200000e+01, !dbg !14
  %119 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 1, !dbg !14
  store float %118, ptr %119, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !35, metadata !DIExpression()), !dbg !14
  %120 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 2, !dbg !14
  %121 = load float, ptr %120, align 8, !dbg !14, !tbaa !19
  %122 = fdiv float %121, 1.200000e+01, !dbg !14
  %123 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 2, !dbg !14
  store float %122, ptr %123, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !35, metadata !DIExpression()), !dbg !14
  %124 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 3, !dbg !14
  %125 = load float, ptr %124, align 4, !dbg !14, !tbaa !19
  %126 = fdiv float %125, 1.200000e+01, !dbg !14
  %127 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 3, !dbg !14
  store float %126, ptr %127, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !35, metadata !DIExpression()), !dbg !14
  %128 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 4, !dbg !14
  %129 = load float, ptr %128, align 16, !dbg !14, !tbaa !19
  %130 = fdiv float %129, 1.200000e+01, !dbg !14
  %131 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 4, !dbg !14
  store float %130, ptr %131, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !35, metadata !DIExpression()), !dbg !14
  %132 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 5, !dbg !14
  %133 = load float, ptr %132, align 4, !dbg !14, !tbaa !19
  %134 = fdiv float %133, 1.200000e+01, !dbg !14
  %135 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 5, !dbg !14
  store float %134, ptr %135, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 2, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !35, metadata !DIExpression()), !dbg !14
  %136 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 6, !dbg !14
  %137 = load float, ptr %136, align 8, !dbg !14, !tbaa !19
  %138 = fdiv float %137, 1.200000e+01, !dbg !14
  %139 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 6, !dbg !14
  store float %138, ptr %139, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 3, metadata !33, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 5, metadata !34, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !35, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !35, metadata !DIExpression()), !dbg !14
  %140 = getelementptr inbounds float, ptr %adaptive_pool_sum13, i64 7, !dbg !14
  %141 = load float, ptr %140, align 4, !dbg !14, !tbaa !19
  %142 = fdiv float %141, 1.200000e+01, !dbg !14
  %143 = getelementptr inbounds float, ptr %adaptive_pool_avg, i64 7, !dbg !14
  store float %142, ptr %143, align 4, !dbg !14, !tbaa !36
  call void @llvm.dbg.declare(metadata i64 2, metadata !22, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nosync nounwind memory(none) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d -fp-contract=off"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 4}
!5 = distinct !DISubprogram(name: "stage_19_operator_adaptive_avg_pool3d", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "float32", size: 32, encoding: DW_ATE_float)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "adaptive_pool_avg", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocation(line: 0, scope: !5)
!15 = !DILocalVariable(name: "adaptive_pool_sum", scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "ax2", scope: !5, file: !1, type: !8)
!17 = !DILocalVariable(name: "ax3", scope: !5, file: !1, type: !8)
!18 = !{!"branch_weights", i32 127, i32 1}
!19 = !{!20, !20, i64 0}
!20 = !{!"0x40c91b30", !21, i64 0}
!21 = !{!"tvm-tbaa"}
!22 = !DILocalVariable(name: "ax4", scope: !5, file: !1, type: !8)
!23 = !DILocalVariable(name: "rv1", scope: !5, file: !1, type: !8)
!24 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!25 = !DILocalVariable(name: "rv0", scope: !5, file: !1, type: !8)
!26 = !DILocalVariable(name: "cse_var_2", scope: !5, file: !1, type: !8)
!27 = !DILocalVariable(name: "rv2", scope: !5, file: !1, type: !8)
!28 = !DILocalVariable(name: "cse_var_3", scope: !5, file: !1, type: !8)
!29 = !{!30, !30, i64 0}
!30 = !{!"0x40c9acd0", !21, i64 0}
!31 = !{!"branch_weights", i32 127, i32 134217601}
!32 = !{!"branch_weights", i32 1048575, i32 1}
!33 = !DILocalVariable(name: "cse_var_6", scope: !5, file: !1, type: !8)
!34 = !DILocalVariable(name: "cse_var_5", scope: !5, file: !1, type: !8)
!35 = !DILocalVariable(name: "cse_var_4", scope: !5, file: !1, type: !8)
!36 = !{!37, !37, i64 0}
!37 = !{!"0x40c9b7a0", !21, i64 0}
