; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@.str = private constant [32 x i8] c"Gemmini tensor buffers rejected\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define dllexport noundef i32 @stage_66_operator_kernel(ptr noalias %image, ptr noalias %pool_max) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %image, metadata !12, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata ptr %pool_max, metadata !13, metadata !DIExpression()), !dbg !14
  %pad_temp_global8 = alloca [9 x i8], align 1, !dbg !14
  %pool_max_global9 = alloca [36 x i8], align 16, !dbg !14
  %pad_temp10 = alloca [189 x i8], align 1, !dbg !14
  call void @llvm.dbg.declare(metadata ptr %pad_temp10, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata ptr %pool_max_global9, metadata !16, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata ptr %pad_temp_global8, metadata !17, metadata !DIExpression()), !dbg !14
  %gemmini_admission = tail call i32 @tvm_gemmini_validate_tensor_buffers(ptr %image, i64 105, ptr %image, i64 105, ptr %pool_max, i64 36), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !18, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %gemmini_admission, metadata !18, metadata !DIExpression()), !dbg !14
  %.not = icmp eq i32 %gemmini_admission, 0, !dbg !14
  br i1 %.not, label %if_else, label %if_then, !dbg !14

common.ret:                                       ; preds = %for_end_ax0_0, %if_then
  %common.ret.op = phi i32 [ -1, %if_then ], [ 0, %for_end_ax0_0 ]
  ret i32 %common.ret.op, !dbg !14

if_then:                                          ; preds = %entry
  tail call void @TVMAPISetLastError(ptr nonnull @.str), !dbg !14
  br label %common.ret

if_else:                                          ; preds = %entry
  tail call void @tvm_gemmini_begin(i64 1, i64 1, i64 1), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !14
  %invariant.gep43 = getelementptr inbounds i8, ptr %pad_temp10, i64 10, !dbg !14
  %invariant.gep44 = getelementptr i8, ptr %image, i64 1, !dbg !14
  %invariant.gep46 = getelementptr inbounds i8, ptr %pad_temp10, i64 11, !dbg !14
  %invariant.gep48 = getelementptr i8, ptr %image, i64 2, !dbg !14
  %invariant.gep50 = getelementptr inbounds i8, ptr %pad_temp10, i64 12, !dbg !14
  %invariant.gep52 = getelementptr i8, ptr %image, i64 3, !dbg !14
  %invariant.gep54 = getelementptr inbounds i8, ptr %pad_temp10, i64 13, !dbg !14
  %invariant.gep56 = getelementptr i8, ptr %image, i64 4, !dbg !14
  %invariant.gep58 = getelementptr inbounds i8, ptr %pad_temp10, i64 14, !dbg !14
  %invariant.gep60 = getelementptr i8, ptr %image, i64 5, !dbg !14
  %invariant.gep62 = getelementptr inbounds i8, ptr %pad_temp10, i64 15, !dbg !14
  %invariant.gep64 = getelementptr i8, ptr %image, i64 6, !dbg !14
  %invariant.gep66 = getelementptr inbounds i8, ptr %pad_temp10, i64 16, !dbg !14
  %invariant.gep68 = getelementptr inbounds i8, ptr %pad_temp10, i64 17, !dbg !14
  %invariant.gep70 = getelementptr inbounds i8, ptr %pad_temp10, i64 18, !dbg !14
  %invariant.gep72 = getelementptr i8, ptr %image, i64 7, !dbg !14
  %invariant.gep74 = getelementptr inbounds i8, ptr %pad_temp10, i64 19, !dbg !14
  %invariant.gep76 = getelementptr i8, ptr %image, i64 8, !dbg !14
  %invariant.gep78 = getelementptr inbounds i8, ptr %pad_temp10, i64 20, !dbg !14
  %invariant.gep80 = getelementptr i8, ptr %image, i64 9, !dbg !14
  %invariant.gep82 = getelementptr inbounds i8, ptr %pad_temp10, i64 21, !dbg !14
  %invariant.gep84 = getelementptr i8, ptr %image, i64 10, !dbg !14
  %invariant.gep86 = getelementptr inbounds i8, ptr %pad_temp10, i64 22, !dbg !14
  %invariant.gep88 = getelementptr i8, ptr %image, i64 11, !dbg !14
  %invariant.gep90 = getelementptr inbounds i8, ptr %pad_temp10, i64 23, !dbg !14
  %invariant.gep92 = getelementptr i8, ptr %image, i64 12, !dbg !14
  %invariant.gep94 = getelementptr inbounds i8, ptr %pad_temp10, i64 24, !dbg !14
  %invariant.gep96 = getelementptr i8, ptr %image, i64 13, !dbg !14
  %invariant.gep98 = getelementptr inbounds i8, ptr %pad_temp10, i64 25, !dbg !14
  %invariant.gep100 = getelementptr inbounds i8, ptr %pad_temp10, i64 26, !dbg !14
  %invariant.gep102 = getelementptr inbounds i8, ptr %pad_temp10, i64 27, !dbg !14
  %invariant.gep104 = getelementptr i8, ptr %image, i64 14, !dbg !14
  %invariant.gep106 = getelementptr inbounds i8, ptr %pad_temp10, i64 28, !dbg !14
  %invariant.gep108 = getelementptr i8, ptr %image, i64 15, !dbg !14
  %invariant.gep110 = getelementptr inbounds i8, ptr %pad_temp10, i64 29, !dbg !14
  %invariant.gep112 = getelementptr i8, ptr %image, i64 16, !dbg !14
  %invariant.gep114 = getelementptr inbounds i8, ptr %pad_temp10, i64 30, !dbg !14
  %invariant.gep116 = getelementptr i8, ptr %image, i64 17, !dbg !14
  %invariant.gep118 = getelementptr inbounds i8, ptr %pad_temp10, i64 31, !dbg !14
  %invariant.gep120 = getelementptr i8, ptr %image, i64 18, !dbg !14
  %invariant.gep122 = getelementptr inbounds i8, ptr %pad_temp10, i64 32, !dbg !14
  %invariant.gep124 = getelementptr i8, ptr %image, i64 19, !dbg !14
  %invariant.gep126 = getelementptr inbounds i8, ptr %pad_temp10, i64 33, !dbg !14
  %invariant.gep128 = getelementptr i8, ptr %image, i64 20, !dbg !14
  %invariant.gep130 = getelementptr inbounds i8, ptr %pad_temp10, i64 34, !dbg !14
  %invariant.gep132 = getelementptr inbounds i8, ptr %pad_temp10, i64 35, !dbg !14
  %invariant.gep134 = getelementptr inbounds i8, ptr %pad_temp10, i64 36, !dbg !14
  %invariant.gep136 = getelementptr i8, ptr %image, i64 21, !dbg !14
  %invariant.gep138 = getelementptr inbounds i8, ptr %pad_temp10, i64 37, !dbg !14
  %invariant.gep140 = getelementptr i8, ptr %image, i64 22, !dbg !14
  %invariant.gep142 = getelementptr inbounds i8, ptr %pad_temp10, i64 38, !dbg !14
  %invariant.gep144 = getelementptr i8, ptr %image, i64 23, !dbg !14
  %invariant.gep146 = getelementptr inbounds i8, ptr %pad_temp10, i64 39, !dbg !14
  %invariant.gep148 = getelementptr i8, ptr %image, i64 24, !dbg !14
  %invariant.gep150 = getelementptr inbounds i8, ptr %pad_temp10, i64 40, !dbg !14
  %invariant.gep152 = getelementptr i8, ptr %image, i64 25, !dbg !14
  %invariant.gep154 = getelementptr inbounds i8, ptr %pad_temp10, i64 41, !dbg !14
  %invariant.gep156 = getelementptr i8, ptr %image, i64 26, !dbg !14
  %invariant.gep158 = getelementptr inbounds i8, ptr %pad_temp10, i64 42, !dbg !14
  %invariant.gep160 = getelementptr i8, ptr %image, i64 27, !dbg !14
  %invariant.gep162 = getelementptr inbounds i8, ptr %pad_temp10, i64 43, !dbg !14
  %invariant.gep164 = getelementptr inbounds i8, ptr %pad_temp10, i64 44, !dbg !14
  %invariant.gep166 = getelementptr inbounds i8, ptr %pad_temp10, i64 45, !dbg !14
  %invariant.gep168 = getelementptr i8, ptr %image, i64 28, !dbg !14
  %invariant.gep170 = getelementptr inbounds i8, ptr %pad_temp10, i64 46, !dbg !14
  %invariant.gep172 = getelementptr i8, ptr %image, i64 29, !dbg !14
  %invariant.gep174 = getelementptr inbounds i8, ptr %pad_temp10, i64 47, !dbg !14
  %invariant.gep176 = getelementptr i8, ptr %image, i64 30, !dbg !14
  %invariant.gep178 = getelementptr inbounds i8, ptr %pad_temp10, i64 48, !dbg !14
  %invariant.gep180 = getelementptr i8, ptr %image, i64 31, !dbg !14
  %invariant.gep182 = getelementptr inbounds i8, ptr %pad_temp10, i64 49, !dbg !14
  %invariant.gep184 = getelementptr i8, ptr %image, i64 32, !dbg !14
  %invariant.gep186 = getelementptr inbounds i8, ptr %pad_temp10, i64 50, !dbg !14
  %invariant.gep188 = getelementptr i8, ptr %image, i64 33, !dbg !14
  %invariant.gep190 = getelementptr inbounds i8, ptr %pad_temp10, i64 51, !dbg !14
  %invariant.gep192 = getelementptr i8, ptr %image, i64 34, !dbg !14
  %invariant.gep194 = getelementptr inbounds i8, ptr %pad_temp10, i64 52, !dbg !14
  %invariant.gep196 = getelementptr inbounds i8, ptr %pad_temp10, i64 53, !dbg !14
  br label %for_end_ax3.6, !dbg !14

for_body_ax0_0_init.preheader:                    ; preds = %for_end_ax3.6
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(36) %pool_max_global9, i8 -128, i64 36, i1 false), !dbg !14, !tbaa !20
  call void @llvm.dbg.declare(metadata i64 poison, metadata !23, metadata !DIExpression()), !dbg !14
  %0 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 1
  %1 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 2
  %2 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 3
  %3 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 4
  %4 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 5
  %5 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 6
  %6 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 7
  %7 = getelementptr inbounds i8, ptr %pad_temp_global8, i64 8
  br label %for_begin_ax14.preheader, !dbg !14

for_end_ax3.6:                                    ; preds = %if_else, %for_end_ax3.6
  %indvar = phi i64 [ 0, %if_else ], [ %indvar.next, %for_end_ax3.6 ]
  call void @llvm.dbg.declare(metadata i64 %indvar, metadata !19, metadata !DIExpression()), !dbg !14
  %8 = mul nuw nsw i64 %indvar, 63
  %9 = mul nuw nsw i64 %indvar, 35
  call void @llvm.dbg.declare(metadata i32 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  %scevgep = getelementptr i8, ptr %pad_temp10, i64 %8
  call void @llvm.dbg.declare(metadata i64 poison, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  %gep.1.1 = getelementptr i8, ptr %image, i64 %9, !dbg !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %scevgep, i8 -128, i64 10, i1 false), !dbg !14
  %10 = load i8, ptr %gep.1.1, align 1, !dbg !14, !tbaa !26
  %gep = getelementptr i8, ptr %invariant.gep43, i64 %8, !dbg !14
  store i8 %10, ptr %gep, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  %gep45 = getelementptr i8, ptr %invariant.gep44, i64 %9, !dbg !14
  %11 = load i8, ptr %gep45, align 1, !dbg !14, !tbaa !26
  %gep47 = getelementptr i8, ptr %invariant.gep46, i64 %8, !dbg !14
  store i8 %11, ptr %gep47, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  %gep49 = getelementptr i8, ptr %invariant.gep48, i64 %9, !dbg !14
  %12 = load i8, ptr %gep49, align 1, !dbg !14, !tbaa !26
  %gep51 = getelementptr i8, ptr %invariant.gep50, i64 %8, !dbg !14
  store i8 %12, ptr %gep51, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  %gep53 = getelementptr i8, ptr %invariant.gep52, i64 %9, !dbg !14
  %13 = load i8, ptr %gep53, align 1, !dbg !14, !tbaa !26
  %gep55 = getelementptr i8, ptr %invariant.gep54, i64 %8, !dbg !14
  store i8 %13, ptr %gep55, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  %gep57 = getelementptr i8, ptr %invariant.gep56, i64 %9, !dbg !14
  %14 = load i8, ptr %gep57, align 1, !dbg !14, !tbaa !26
  %gep59 = getelementptr i8, ptr %invariant.gep58, i64 %8, !dbg !14
  store i8 %14, ptr %gep59, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  %gep61 = getelementptr i8, ptr %invariant.gep60, i64 %9, !dbg !14
  %15 = load i8, ptr %gep61, align 1, !dbg !14, !tbaa !26
  %gep63 = getelementptr i8, ptr %invariant.gep62, i64 %8, !dbg !14
  store i8 %15, ptr %gep63, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  %gep65 = getelementptr i8, ptr %invariant.gep64, i64 %9, !dbg !14
  %16 = load i8, ptr %gep65, align 1, !dbg !14, !tbaa !26
  %gep67 = getelementptr i8, ptr %invariant.gep66, i64 %8, !dbg !14
  store i8 %16, ptr %gep67, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  %gep69 = getelementptr i8, ptr %invariant.gep68, i64 %8, !dbg !14
  store i8 -128, ptr %gep69, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !24, metadata !DIExpression()), !dbg !14
  %gep71 = getelementptr i8, ptr %invariant.gep70, i64 %8
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  store i8 -128, ptr %gep71, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  %gep73 = getelementptr i8, ptr %invariant.gep72, i64 %9, !dbg !14
  %17 = load i8, ptr %gep73, align 1, !dbg !14, !tbaa !26
  %gep75 = getelementptr i8, ptr %invariant.gep74, i64 %8, !dbg !14
  store i8 %17, ptr %gep75, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  %gep77 = getelementptr i8, ptr %invariant.gep76, i64 %9, !dbg !14
  %18 = load i8, ptr %gep77, align 1, !dbg !14, !tbaa !26
  %gep79 = getelementptr i8, ptr %invariant.gep78, i64 %8, !dbg !14
  store i8 %18, ptr %gep79, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  %gep81 = getelementptr i8, ptr %invariant.gep80, i64 %9, !dbg !14
  %19 = load i8, ptr %gep81, align 1, !dbg !14, !tbaa !26
  %gep83 = getelementptr i8, ptr %invariant.gep82, i64 %8, !dbg !14
  store i8 %19, ptr %gep83, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  %gep85 = getelementptr i8, ptr %invariant.gep84, i64 %9, !dbg !14
  %20 = load i8, ptr %gep85, align 1, !dbg !14, !tbaa !26
  %gep87 = getelementptr i8, ptr %invariant.gep86, i64 %8, !dbg !14
  store i8 %20, ptr %gep87, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  %gep89 = getelementptr i8, ptr %invariant.gep88, i64 %9, !dbg !14
  %21 = load i8, ptr %gep89, align 1, !dbg !14, !tbaa !26
  %gep91 = getelementptr i8, ptr %invariant.gep90, i64 %8, !dbg !14
  store i8 %21, ptr %gep91, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  %gep93 = getelementptr i8, ptr %invariant.gep92, i64 %9, !dbg !14
  %22 = load i8, ptr %gep93, align 1, !dbg !14, !tbaa !26
  %gep95 = getelementptr i8, ptr %invariant.gep94, i64 %8, !dbg !14
  store i8 %22, ptr %gep95, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  %gep97 = getelementptr i8, ptr %invariant.gep96, i64 %9, !dbg !14
  %23 = load i8, ptr %gep97, align 1, !dbg !14, !tbaa !26
  %gep99 = getelementptr i8, ptr %invariant.gep98, i64 %8, !dbg !14
  store i8 %23, ptr %gep99, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  %gep101 = getelementptr i8, ptr %invariant.gep100, i64 %8, !dbg !14
  store i8 -128, ptr %gep101, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !24, metadata !DIExpression()), !dbg !14
  %gep103 = getelementptr i8, ptr %invariant.gep102, i64 %8
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  store i8 -128, ptr %gep103, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  %gep105 = getelementptr i8, ptr %invariant.gep104, i64 %9, !dbg !14
  %24 = load i8, ptr %gep105, align 1, !dbg !14, !tbaa !26
  %gep107 = getelementptr i8, ptr %invariant.gep106, i64 %8, !dbg !14
  store i8 %24, ptr %gep107, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  %gep109 = getelementptr i8, ptr %invariant.gep108, i64 %9, !dbg !14
  %25 = load i8, ptr %gep109, align 1, !dbg !14, !tbaa !26
  %gep111 = getelementptr i8, ptr %invariant.gep110, i64 %8, !dbg !14
  store i8 %25, ptr %gep111, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  %gep113 = getelementptr i8, ptr %invariant.gep112, i64 %9, !dbg !14
  %26 = load i8, ptr %gep113, align 1, !dbg !14, !tbaa !26
  %gep115 = getelementptr i8, ptr %invariant.gep114, i64 %8, !dbg !14
  store i8 %26, ptr %gep115, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  %gep117 = getelementptr i8, ptr %invariant.gep116, i64 %9, !dbg !14
  %27 = load i8, ptr %gep117, align 1, !dbg !14, !tbaa !26
  %gep119 = getelementptr i8, ptr %invariant.gep118, i64 %8, !dbg !14
  store i8 %27, ptr %gep119, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  %gep121 = getelementptr i8, ptr %invariant.gep120, i64 %9, !dbg !14
  %28 = load i8, ptr %gep121, align 1, !dbg !14, !tbaa !26
  %gep123 = getelementptr i8, ptr %invariant.gep122, i64 %8, !dbg !14
  store i8 %28, ptr %gep123, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  %gep125 = getelementptr i8, ptr %invariant.gep124, i64 %9, !dbg !14
  %29 = load i8, ptr %gep125, align 1, !dbg !14, !tbaa !26
  %gep127 = getelementptr i8, ptr %invariant.gep126, i64 %8, !dbg !14
  store i8 %29, ptr %gep127, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  %gep129 = getelementptr i8, ptr %invariant.gep128, i64 %9, !dbg !14
  %30 = load i8, ptr %gep129, align 1, !dbg !14, !tbaa !26
  %gep131 = getelementptr i8, ptr %invariant.gep130, i64 %8, !dbg !14
  store i8 %30, ptr %gep131, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  %gep133 = getelementptr i8, ptr %invariant.gep132, i64 %8, !dbg !14
  store i8 -128, ptr %gep133, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !24, metadata !DIExpression()), !dbg !14
  %gep135 = getelementptr i8, ptr %invariant.gep134, i64 %8
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  store i8 -128, ptr %gep135, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  %gep137 = getelementptr i8, ptr %invariant.gep136, i64 %9, !dbg !14
  %31 = load i8, ptr %gep137, align 1, !dbg !14, !tbaa !26
  %gep139 = getelementptr i8, ptr %invariant.gep138, i64 %8, !dbg !14
  store i8 %31, ptr %gep139, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  %gep141 = getelementptr i8, ptr %invariant.gep140, i64 %9, !dbg !14
  %32 = load i8, ptr %gep141, align 1, !dbg !14, !tbaa !26
  %gep143 = getelementptr i8, ptr %invariant.gep142, i64 %8, !dbg !14
  store i8 %32, ptr %gep143, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  %gep145 = getelementptr i8, ptr %invariant.gep144, i64 %9, !dbg !14
  %33 = load i8, ptr %gep145, align 1, !dbg !14, !tbaa !26
  %gep147 = getelementptr i8, ptr %invariant.gep146, i64 %8, !dbg !14
  store i8 %33, ptr %gep147, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  %gep149 = getelementptr i8, ptr %invariant.gep148, i64 %9, !dbg !14
  %34 = load i8, ptr %gep149, align 1, !dbg !14, !tbaa !26
  %gep151 = getelementptr i8, ptr %invariant.gep150, i64 %8, !dbg !14
  store i8 %34, ptr %gep151, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  %gep153 = getelementptr i8, ptr %invariant.gep152, i64 %9, !dbg !14
  %35 = load i8, ptr %gep153, align 1, !dbg !14, !tbaa !26
  %gep155 = getelementptr i8, ptr %invariant.gep154, i64 %8, !dbg !14
  store i8 %35, ptr %gep155, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  %gep157 = getelementptr i8, ptr %invariant.gep156, i64 %9, !dbg !14
  %36 = load i8, ptr %gep157, align 1, !dbg !14, !tbaa !26
  %gep159 = getelementptr i8, ptr %invariant.gep158, i64 %8, !dbg !14
  store i8 %36, ptr %gep159, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  %gep161 = getelementptr i8, ptr %invariant.gep160, i64 %9, !dbg !14
  %37 = load i8, ptr %gep161, align 1, !dbg !14, !tbaa !26
  %gep163 = getelementptr i8, ptr %invariant.gep162, i64 %8, !dbg !14
  store i8 %37, ptr %gep163, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  %gep165 = getelementptr i8, ptr %invariant.gep164, i64 %8, !dbg !14
  store i8 -128, ptr %gep165, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !24, metadata !DIExpression()), !dbg !14
  %gep167 = getelementptr i8, ptr %invariant.gep166, i64 %8
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !25, metadata !DIExpression()), !dbg !14
  store i8 -128, ptr %gep167, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !25, metadata !DIExpression()), !dbg !14
  %gep169 = getelementptr i8, ptr %invariant.gep168, i64 %9, !dbg !14
  %38 = load i8, ptr %gep169, align 1, !dbg !14, !tbaa !26
  %gep171 = getelementptr i8, ptr %invariant.gep170, i64 %8, !dbg !14
  store i8 %38, ptr %gep171, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !25, metadata !DIExpression()), !dbg !14
  %gep173 = getelementptr i8, ptr %invariant.gep172, i64 %9, !dbg !14
  %39 = load i8, ptr %gep173, align 1, !dbg !14, !tbaa !26
  %gep175 = getelementptr i8, ptr %invariant.gep174, i64 %8, !dbg !14
  store i8 %39, ptr %gep175, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !25, metadata !DIExpression()), !dbg !14
  %gep177 = getelementptr i8, ptr %invariant.gep176, i64 %9, !dbg !14
  %40 = load i8, ptr %gep177, align 1, !dbg !14, !tbaa !26
  %gep179 = getelementptr i8, ptr %invariant.gep178, i64 %8, !dbg !14
  store i8 %40, ptr %gep179, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !25, metadata !DIExpression()), !dbg !14
  %gep181 = getelementptr i8, ptr %invariant.gep180, i64 %9, !dbg !14
  %41 = load i8, ptr %gep181, align 1, !dbg !14, !tbaa !26
  %gep183 = getelementptr i8, ptr %invariant.gep182, i64 %8, !dbg !14
  store i8 %41, ptr %gep183, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !25, metadata !DIExpression()), !dbg !14
  %gep185 = getelementptr i8, ptr %invariant.gep184, i64 %9, !dbg !14
  %42 = load i8, ptr %gep185, align 1, !dbg !14, !tbaa !26
  %gep187 = getelementptr i8, ptr %invariant.gep186, i64 %8, !dbg !14
  store i8 %42, ptr %gep187, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !25, metadata !DIExpression()), !dbg !14
  %gep189 = getelementptr i8, ptr %invariant.gep188, i64 %9, !dbg !14
  %43 = load i8, ptr %gep189, align 1, !dbg !14, !tbaa !26
  %gep191 = getelementptr i8, ptr %invariant.gep190, i64 %8, !dbg !14
  store i8 %43, ptr %gep191, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !25, metadata !DIExpression()), !dbg !14
  %gep193 = getelementptr i8, ptr %invariant.gep192, i64 %9, !dbg !14
  %44 = load i8, ptr %gep193, align 1, !dbg !14, !tbaa !26
  %gep195 = getelementptr i8, ptr %invariant.gep194, i64 %8, !dbg !14
  store i8 %44, ptr %gep195, align 1, !dbg !14, !tbaa !28
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !25, metadata !DIExpression()), !dbg !14
  %gep197 = getelementptr i8, ptr %invariant.gep196, i64 %8, !dbg !14
  call void @llvm.dbg.declare(metadata i64 9, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !24, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32 0, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 poison, metadata !25, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !24, metadata !DIExpression()), !dbg !14
  %indvar.next = add nuw nsw i64 %indvar, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvar.next, metadata !19, metadata !DIExpression()), !dbg !14
  %exitcond.not = icmp eq i64 %indvar.next, 3, !dbg !14
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %gep197, i8 -128, i64 10, i1 false), !dbg !14
  br i1 %exitcond.not, label %for_body_ax0_0_init.preheader, label %for_end_ax3.6, !dbg !14, !prof !30

for_begin_ax14.preheader:                         ; preds = %for_body_ax0_0_init.preheader, %for_begin_ax14.preheader
  %indvars.iv = phi i64 [ 0, %for_body_ax0_0_init.preheader ], [ %indvars.iv.next, %for_begin_ax14.preheader ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv, metadata !31, metadata !DIExpression()), !dbg !14
  %indvars.iv.tr = trunc i64 %indvars.iv to i32
  %45 = shl i32 %indvars.iv.tr, 1
  %46 = and i32 %45, 6
  %47 = trunc i64 %indvars.iv to i32
  %.frozen = freeze i32 %47
  %48 = udiv i32 %.frozen, 12
  %49 = mul i32 %48, 12
  %.decomposed = sub i32 %.frozen, %49
  %50 = lshr i32 %.decomposed, 2
  %51 = mul nuw nsw i32 %50, 18
  %52 = mul nuw nsw i32 %48, 63
  %53 = add nuw i32 %46, %52
  %54 = add nuw i32 %53, %51
  call void @llvm.dbg.declare(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 0, metadata !19, metadata !DIExpression()), !dbg !14
  %55 = zext nneg i32 %54 to i64, !dbg !14
  %56 = getelementptr inbounds i8, ptr %pad_temp10, i64 %55, !dbg !14
  %57 = load i8, ptr %56, align 1, !dbg !14, !tbaa !28
  store i8 %57, ptr %pad_temp_global8, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 1, metadata !19, metadata !DIExpression()), !dbg !14
  %58 = add nuw i32 %54, 1, !dbg !14
  %59 = zext nneg i32 %58 to i64, !dbg !14
  %60 = getelementptr inbounds i8, ptr %pad_temp10, i64 %59, !dbg !14
  %61 = load i8, ptr %60, align 1, !dbg !14, !tbaa !28
  store i8 %61, ptr %0, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 2, metadata !19, metadata !DIExpression()), !dbg !14
  %62 = add nuw i32 %54, 2, !dbg !14
  %63 = zext nneg i32 %62 to i64, !dbg !14
  %64 = getelementptr inbounds i8, ptr %pad_temp10, i64 %63, !dbg !14
  %65 = load i8, ptr %64, align 1, !dbg !14, !tbaa !28
  store i8 %65, ptr %1, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 3, metadata !19, metadata !DIExpression()), !dbg !14
  %66 = add nuw i32 %54, 9, !dbg !14
  %67 = zext nneg i32 %66 to i64, !dbg !14
  %68 = getelementptr inbounds i8, ptr %pad_temp10, i64 %67, !dbg !14
  %69 = load i8, ptr %68, align 1, !dbg !14, !tbaa !28
  store i8 %69, ptr %2, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 4, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 4, metadata !19, metadata !DIExpression()), !dbg !14
  %70 = add nuw i32 %54, 10, !dbg !14
  %71 = zext nneg i32 %70 to i64, !dbg !14
  %72 = getelementptr inbounds i8, ptr %pad_temp10, i64 %71, !dbg !14
  %73 = load i8, ptr %72, align 1, !dbg !14, !tbaa !28
  store i8 %73, ptr %3, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 5, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 5, metadata !19, metadata !DIExpression()), !dbg !14
  %74 = add nuw i32 %54, 11, !dbg !14
  %75 = zext nneg i32 %74 to i64, !dbg !14
  %76 = getelementptr inbounds i8, ptr %pad_temp10, i64 %75, !dbg !14
  %77 = load i8, ptr %76, align 1, !dbg !14, !tbaa !28
  store i8 %77, ptr %4, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 6, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 6, metadata !19, metadata !DIExpression()), !dbg !14
  %78 = add nuw i32 %54, 18, !dbg !14
  %79 = zext nneg i32 %78 to i64, !dbg !14
  %80 = getelementptr inbounds i8, ptr %pad_temp10, i64 %79, !dbg !14
  %81 = load i8, ptr %80, align 1, !dbg !14, !tbaa !28
  store i8 %81, ptr %5, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 7, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 7, metadata !19, metadata !DIExpression()), !dbg !14
  %82 = add nuw i32 %54, 19, !dbg !14
  %83 = zext nneg i32 %82 to i64, !dbg !14
  %84 = getelementptr inbounds i8, ptr %pad_temp10, i64 %83, !dbg !14
  %85 = load i8, ptr %84, align 1, !dbg !14, !tbaa !28
  store i8 %85, ptr %6, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 8, metadata !19, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i64 8, metadata !19, metadata !DIExpression()), !dbg !14
  %86 = add nuw i32 %54, 20, !dbg !14
  %87 = zext nneg i32 %86 to i64, !dbg !14
  %88 = getelementptr inbounds i8, ptr %pad_temp10, i64 %87, !dbg !14
  %89 = load i8, ptr %88, align 1, !dbg !14, !tbaa !28
  store i8 %89, ptr %7, align 1, !dbg !14, !tbaa !32
  call void @llvm.dbg.declare(metadata i64 9, metadata !19, metadata !DIExpression()), !dbg !14
  call void @tvm_gemmini_dma_widen(ptr nonnull %pad_temp_global8, i32 0, i32 9, i32 1, i64 1, i32 1), !dbg !14
  %90 = getelementptr inbounds i8, ptr %pool_max_global9, i64 %indvars.iv, !dbg !14
  call void @tvm_gemmini_dma_pool_out(ptr nonnull %90, i32 0, i32 3, i32 1), !dbg !14
  %91 = load i8, ptr %90, align 1, !dbg !14, !tbaa !20
  %92 = getelementptr inbounds i8, ptr %pool_max, i64 %indvars.iv, !dbg !14
  store i8 %91, ptr %92, align 1, !dbg !14, !tbaa !34
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !14
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next, metadata !31, metadata !DIExpression()), !dbg !14
  %exitcond42.not = icmp eq i64 %indvars.iv.next, 36, !dbg !14
  br i1 %exitcond42.not, label %for_end_ax0_0, label %for_begin_ax14.preheader, !dbg !14, !prof !30

for_end_ax0_0:                                    ; preds = %for_begin_ax14.preheader
  call void @tvm_gemmini_end(), !dbg !14
  br label %common.ret
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @tvm_gemmini_validate_tensor_buffers(ptr, i64, ptr, i64, ptr, i64) local_unnamed_addr

declare void @TVMAPISetLastError(ptr) local_unnamed_addr

declare void @tvm_gemmini_begin(i64, i64, i64) local_unnamed_addr

declare void @tvm_gemmini_dma_widen(ptr, i32, i32, i32, i64, i32) local_unnamed_addr

declare void @tvm_gemmini_dma_pool_out(ptr, i32, i32, i32) local_unnamed_addr

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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #4

attributes #0 = { "target-cpu"="generic-rv64" "target-features"="+m,+a,+f,+d,+c" }
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
!5 = distinct !DISubprogram(name: "stage_66_operator_kernel", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !9}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10)
!10 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "image", arg: 1, scope: !5, file: !1, type: !9)
!13 = !DILocalVariable(name: "pool_max", arg: 2, scope: !5, file: !1, type: !9)
!14 = !DILocation(line: 0, scope: !5)
!15 = !DILocalVariable(name: "pad_temp", scope: !5, file: !1, type: !9)
!16 = !DILocalVariable(name: "pool_max_global", scope: !5, file: !1, type: !9)
!17 = !DILocalVariable(name: "pad_temp_global", scope: !5, file: !1, type: !9)
!18 = !DILocalVariable(name: "gemmini_admission", scope: !5, file: !1, type: !8)
!19 = !DILocalVariable(name: "ax1", scope: !5, file: !1, type: !8)
!20 = !{!21, !21, i64 0}
!21 = !{!"0x40edf3b0", !22, i64 0}
!22 = !{!"tvm-tbaa"}
!23 = !DILocalVariable(name: "ax0_0_init", scope: !5, file: !1, type: !8)
!24 = !DILocalVariable(name: "ax2", scope: !5, file: !1, type: !8)
!25 = !DILocalVariable(name: "ax3", scope: !5, file: !1, type: !8)
!26 = !{!27, !27, i64 0}
!27 = !{!"0x40ad9660", !22, i64 0}
!28 = !{!29, !29, i64 0}
!29 = !{!"0x40ad9a70", !22, i64 0}
!30 = !{!"branch_weights", i32 1, i32 1048575}
!31 = !DILocalVariable(name: "ax0_0", scope: !5, file: !1, type: !8)
!32 = !{!33, !33, i64 0}
!33 = !{!"0x40da44a0", !22, i64 0}
!34 = !{!35, !35, i64 0}
!35 = !{!"0x409f5a40", !22, i64 0}
