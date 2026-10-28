; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_33_operator_pad(ptr noalias %x, ptr noalias %MirrorPadInput) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %x, metadata !12, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata ptr %MirrorPadInput, metadata !13, metadata !DIExpression()), !dbg !14
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %x, i64 6, ptr %x, i64 6, ptr %MirrorPadInput, i64 20), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !15, metadata !DIExpression()), !dbg !14
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !14
  br i1 %.not, label %for_begin_i3_0.preheader.peel.next, label %if_then, !dbg !14

common.ret:                                       ; preds = %for_begin_i3_0.preheader.peel.next, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %for_begin_i3_0.preheader.peel.next ]
  ret i32 %common.ret.op, !dbg !14

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !14
  br label %common.ret

for_begin_i3_0.preheader.peel.next:               ; preds = %entry
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !16, metadata !DIExpression()), !dbg !14
  %0 = getelementptr i8, ptr %x, i64 3
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %1 = getelementptr i8, ptr %x, i64 4, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %1, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %MirrorPadInput, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %0, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %2 = getelementptr i8, ptr %MirrorPadInput, i64 1, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %2, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %1, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %3 = getelementptr i8, ptr %MirrorPadInput, i64 2, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %3, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  %4 = getelementptr i8, ptr %x, i64 5, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %4, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %5 = getelementptr i8, ptr %MirrorPadInput, i64 3, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %5, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %1, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %6 = getelementptr i8, ptr %MirrorPadInput, i64 4, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %6, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %7 = getelementptr i8, ptr %MirrorPadInput, i64 5
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %8 = getelementptr i8, ptr %x, i64 1, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %8, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %7, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %x, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %9 = getelementptr i8, ptr %MirrorPadInput, i64 6, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %9, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %8, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %10 = getelementptr i8, ptr %MirrorPadInput, i64 7, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %10, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  %11 = getelementptr i8, ptr %x, i64 2, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %11, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %12 = getelementptr i8, ptr %MirrorPadInput, i64 8, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %12, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %8, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %13 = getelementptr i8, ptr %MirrorPadInput, i64 9, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %13, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !16, metadata !DIExpression()), !dbg !14
  %14 = getelementptr i8, ptr %x, i64 3
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %15 = getelementptr i8, ptr %MirrorPadInput, i64 10
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %16 = getelementptr i8, ptr %x, i64 4, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %16, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %15, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %14, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %17 = getelementptr i8, ptr %MirrorPadInput, i64 11, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %17, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %16, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %18 = getelementptr i8, ptr %MirrorPadInput, i64 12, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %18, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  %19 = getelementptr i8, ptr %x, i64 5, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %19, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %20 = getelementptr i8, ptr %MirrorPadInput, i64 13, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %20, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %16, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %21 = getelementptr i8, ptr %MirrorPadInput, i64 14, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %21, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %22 = getelementptr i8, ptr %MirrorPadInput, i64 15
  call void @llvm.dbg.declare(metadata i64 0, metadata !18, metadata !DIExpression()), !dbg !14
  %23 = getelementptr i8, ptr %x, i64 1, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %23, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %22, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %x, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %24 = getelementptr i8, ptr %MirrorPadInput, i64 16, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %24, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %23, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %25 = getelementptr i8, ptr %MirrorPadInput, i64 17, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %25, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !18, metadata !DIExpression()), !dbg !14
  %26 = getelementptr i8, ptr %x, i64 2, !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %26, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %27 = getelementptr i8, ptr %MirrorPadInput, i64 18, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %27, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !18, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_dma_a(ptr %23, i32 0, i32 1, i32 1, i64 3, i32 0), !dbg !14
  %28 = getelementptr i8, ptr %MirrorPadInput, i64 19, !dbg !14
  tail call void @tvm_gemmini_dma_spad_out(ptr %28, i32 0, i32 1, i32 1, i64 5, i32 0), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !16, metadata !DIExpression()), !dbg !14
  tail call void @tvm_gemmini_end(), !dbg !14
  br label %common.ret
}

declare i32 @tvm_gemmini_validate_tensor_buffers(ptr, i64, ptr, i64, ptr, i64) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @TVMAPISetLastError(ptr) local_unnamed_addr

declare void @tvm_gemmini_begin(i64, i64, i64) local_unnamed_addr

declare void @tvm_gemmini_dma_a(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_spad_out(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

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
!5 = distinct !DISubprogram(name: "stage_33_operator_pad", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "MirrorPadInput", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocation(line: 0, scope: !5)
!15 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!16 = !DILocalVariable(name: "i2_0", scope: !5, file: !1, type: !17)
!17 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!18 = !DILocalVariable(name: "i3_0", scope: !5, file: !1, type: !17)
