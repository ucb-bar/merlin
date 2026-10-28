; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_50_operator_fused_cast_sum(ptr noalias %x, ptr noalias %lv_red) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !14, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %lv_red, metadata !15, metadata !DIExpression()), !dbg !16
  %lv_red_global5 = alloca [2 x i32], align 4, !dbg !16
  %one = alloca i8, align 1, !dbg !16
  %x_global6 = alloca [6 x i8], align 1, !dbg !16
  call void @llvm.dbg.declare(metadata ptr %x_global6, metadata !17, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %lv_red_global5, metadata !18, metadata !DIExpression()), !dbg !16
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %x, i64 6, ptr %x, i64 6, ptr %lv_red, i64 16), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !20, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !20, metadata !DIExpression()), !dbg !16
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !16
  br i1 %.not, label %if_else, label %if_then, !dbg !16

common.ret:                                       ; preds = %if_else, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %if_else ]
  ret i32 %common.ret.op, !dbg !16

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !16
  br label %common.ret

if_else:                                          ; preds = %entry
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !16
  tail call void @tvm_gemmini_zero_acc(i32 0, i32 2, i32 1), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(6) %x_global6, ptr noundef nonnull align 1 dereferenceable(6) %x, i64 6, i1 false), !dbg !16
  call void @llvm.dbg.declare(metadata i32 poison, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !22, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 poison, metadata !23, metadata !DIExpression()), !dbg !16
  call void @tvm_gemmini_dma_a(ptr nonnull %x_global6, i32 0, i32 2, i32 3, i64 3, i32 1), !dbg !16
  tail call void @llvm.dbg.value(metadata i8 1, metadata !24, metadata !DIExpression()), !dbg !16
  store i8 1, ptr %one, align 1, !dbg !16, !tbaa !25
  tail call void @llvm.dbg.value(metadata ptr %one, metadata !24, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @tvm_gemmini_dma_b(ptr nonnull %one, i32 16, i32 3, i32 1, i64 0, i32 1), !dbg !16
  call void @tvm_gemmini_compute(i32 0, i32 16, i32 0, i32 2, i32 1, i32 3, i32 1), !dbg !16
  call void @tvm_gemmini_dma_out(ptr nonnull %lv_red_global5, i32 0, i32 2, i32 1, i64 1, i32 1), !dbg !16
  call void @llvm.dbg.declare(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 0, metadata !21, metadata !DIExpression()), !dbg !16
  %0 = load i32, ptr %lv_red_global5, align 4, !dbg !16, !tbaa !39
  %1 = sext i32 %0 to i64, !dbg !16
  store i64 %1, ptr %lv_red, align 8, !dbg !16, !tbaa !41
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata i64 1, metadata !21, metadata !DIExpression()), !dbg !16
  %2 = getelementptr inbounds i32, ptr %lv_red_global5, i64 1, !dbg !16
  %3 = load i32, ptr %2, align 4, !dbg !16, !tbaa !39
  %4 = sext i32 %3 to i64, !dbg !16
  %5 = getelementptr inbounds i64, ptr %lv_red, i64 1, !dbg !16
  store i64 %4, ptr %5, align 8, !dbg !16, !tbaa !41
  call void @llvm.dbg.declare(metadata i64 2, metadata !21, metadata !DIExpression()), !dbg !16
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

attributes #0 = { "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree nosync nounwind memory(none) "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "TVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "IRModule.CodeGenLLVM", directory: ".")
!2 = !{i32 2, !"tvm_target", !"llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d -fp-contract=off"}
!3 = !{i32 4, !"Debug Info Version", i32 3}
!4 = !{i32 4, !"Dwarf Version", i32 4}
!5 = distinct !DISubprogram(name: "stage_50_operator_fused_cast_sum", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !11}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12)
!12 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!15 = !DILocalVariable(name: "lv_red", arg: 2, scope: !5, file: !1, type: !11)
!16 = !DILocation(line: 0, scope: !5)
!17 = !DILocalVariable(name: "x_global", scope: !5, file: !1, type: !9)
!18 = !DILocalVariable(name: "lv_red_global", scope: !5, file: !1, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!20 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!21 = !DILocalVariable(name: "ax0", scope: !5, file: !1, type: !8)
!22 = !DILocalVariable(name: "ax1", scope: !5, file: !1, type: !8)
!23 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!24 = !DILocalVariable(name: "one", scope: !5, file: !1, type: !9)
!25 = !{!26, !26, i64 0}
!26 = !{!"0x40e939f0.w1.b0", !27, i64 0}
!27 = !{!"0x40e939f0.w2.b0", !28, i64 0}
!28 = !{!"0x40e939f0.w4.b0", !29, i64 0}
!29 = !{!"0x40e939f0.w8.b0", !30, i64 0}
!30 = !{!"0x40e939f0.w16.b0", !31, i64 0}
!31 = !{!"0x40e939f0.w32.b0", !32, i64 0}
!32 = !{!"0x40e939f0.w64.b0", !33, i64 0}
!33 = !{!"0x40e939f0.w128.b0", !34, i64 0}
!34 = !{!"0x40e939f0.w256.b0", !35, i64 0}
!35 = !{!"0x40e939f0.w512.b0", !36, i64 0}
!36 = !{!"0x40e939f0.w1024.b0", !37, i64 0}
!37 = !{!"0x40e939f0", !38, i64 0}
!38 = !{!"tvm-tbaa"}
!39 = !{!40, !40, i64 0}
!40 = !{!"0x40e84120", !38, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"0x40dd7ba0", !38, i64 0}
