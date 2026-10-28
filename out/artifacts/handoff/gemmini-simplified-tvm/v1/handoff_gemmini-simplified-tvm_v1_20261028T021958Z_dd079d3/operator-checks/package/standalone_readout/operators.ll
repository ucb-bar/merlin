; ModuleID = 'TVMMod'
source_filename = "TVMMod"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-elf"

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dllexport noundef i32 @stage_69_operator_kernel(ptr noalias nocapture readonly %accumulator, ptr noalias nocapture writeonly %readout) local_unnamed_addr #0 !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %accumulator, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @llvm.dbg.value(metadata ptr %readout, metadata !14, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !15
  br label %for_begin_j.preheader, !dbg !15

for_begin_j.preheader:                            ; preds = %entry, %for_begin_j.preheader
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for_begin_j.preheader ]
  call void @llvm.dbg.declare(metadata i64 %indvars.iv, metadata !16, metadata !DIExpression()), !dbg !15
  %0 = mul nuw nsw i64 %indvars.iv, 17
  call void @llvm.dbg.declare(metadata i32 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 0, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %0, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %0, metadata !18, metadata !DIExpression()), !dbg !15
  %1 = getelementptr inbounds i32, ptr %accumulator, i64 %0, !dbg !15
  %2 = load i32, ptr %1, align 4, !dbg !15, !tbaa !19
  %3 = sitofp i32 %2 to float, !dbg !15
  %4 = fmul float %3, 0x3F69803980000000, !dbg !15
  %5 = fcmp ogt float %4, 0.000000e+00, !dbg !15
  %6 = select i1 %5, float %4, float 0.000000e+00, !dbg !15
  %7 = tail call float @llvm.nearbyint.f32(float %6), !dbg !15
  %8 = fcmp ogt float %7, -1.280000e+02, !dbg !15
  %9 = select i1 %8, float %7, float -1.280000e+02, !dbg !15
  %10 = fcmp olt float %9, 1.270000e+02, !dbg !15
  %11 = select i1 %10, float %9, float 1.270000e+02, !dbg !15
  %12 = fptosi float %11 to i8, !dbg !15
  %13 = getelementptr inbounds i8, ptr %readout, i64 %0, !dbg !15
  store i8 %12, ptr %13, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 1, metadata !17, metadata !DIExpression()), !dbg !15
  %14 = add nuw nsw i64 %0, 1, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %14, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %14, metadata !18, metadata !DIExpression()), !dbg !15
  %15 = getelementptr inbounds i32, ptr %accumulator, i64 %14, !dbg !15
  %16 = load i32, ptr %15, align 4, !dbg !15, !tbaa !19
  %17 = sitofp i32 %16 to float, !dbg !15
  %18 = fmul float %17, 0x3F69803980000000, !dbg !15
  %19 = fcmp ogt float %18, 0.000000e+00, !dbg !15
  %20 = select i1 %19, float %18, float 0.000000e+00, !dbg !15
  %21 = tail call float @llvm.nearbyint.f32(float %20), !dbg !15
  %22 = fcmp ogt float %21, -1.280000e+02, !dbg !15
  %23 = select i1 %22, float %21, float -1.280000e+02, !dbg !15
  %24 = fcmp olt float %23, 1.270000e+02, !dbg !15
  %25 = select i1 %24, float %23, float 1.270000e+02, !dbg !15
  %26 = fptosi float %25 to i8, !dbg !15
  %27 = getelementptr inbounds i8, ptr %readout, i64 %14, !dbg !15
  store i8 %26, ptr %27, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 2, metadata !17, metadata !DIExpression()), !dbg !15
  %28 = add nuw nsw i64 %0, 2, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %28, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %28, metadata !18, metadata !DIExpression()), !dbg !15
  %29 = getelementptr inbounds i32, ptr %accumulator, i64 %28, !dbg !15
  %30 = load i32, ptr %29, align 4, !dbg !15, !tbaa !19
  %31 = sitofp i32 %30 to float, !dbg !15
  %32 = fmul float %31, 0x3F69803980000000, !dbg !15
  %33 = fcmp ogt float %32, 0.000000e+00, !dbg !15
  %34 = select i1 %33, float %32, float 0.000000e+00, !dbg !15
  %35 = tail call float @llvm.nearbyint.f32(float %34), !dbg !15
  %36 = fcmp ogt float %35, -1.280000e+02, !dbg !15
  %37 = select i1 %36, float %35, float -1.280000e+02, !dbg !15
  %38 = fcmp olt float %37, 1.270000e+02, !dbg !15
  %39 = select i1 %38, float %37, float 1.270000e+02, !dbg !15
  %40 = fptosi float %39 to i8, !dbg !15
  %41 = getelementptr inbounds i8, ptr %readout, i64 %28, !dbg !15
  store i8 %40, ptr %41, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 3, metadata !17, metadata !DIExpression()), !dbg !15
  %42 = add nuw nsw i64 %0, 3, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %42, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %42, metadata !18, metadata !DIExpression()), !dbg !15
  %43 = getelementptr inbounds i32, ptr %accumulator, i64 %42, !dbg !15
  %44 = load i32, ptr %43, align 4, !dbg !15, !tbaa !19
  %45 = sitofp i32 %44 to float, !dbg !15
  %46 = fmul float %45, 0x3F69803980000000, !dbg !15
  %47 = fcmp ogt float %46, 0.000000e+00, !dbg !15
  %48 = select i1 %47, float %46, float 0.000000e+00, !dbg !15
  %49 = tail call float @llvm.nearbyint.f32(float %48), !dbg !15
  %50 = fcmp ogt float %49, -1.280000e+02, !dbg !15
  %51 = select i1 %50, float %49, float -1.280000e+02, !dbg !15
  %52 = fcmp olt float %51, 1.270000e+02, !dbg !15
  %53 = select i1 %52, float %51, float 1.270000e+02, !dbg !15
  %54 = fptosi float %53 to i8, !dbg !15
  %55 = getelementptr inbounds i8, ptr %readout, i64 %42, !dbg !15
  store i8 %54, ptr %55, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 4, metadata !17, metadata !DIExpression()), !dbg !15
  %56 = add nuw nsw i64 %0, 4, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %56, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %56, metadata !18, metadata !DIExpression()), !dbg !15
  %57 = getelementptr inbounds i32, ptr %accumulator, i64 %56, !dbg !15
  %58 = load i32, ptr %57, align 4, !dbg !15, !tbaa !19
  %59 = sitofp i32 %58 to float, !dbg !15
  %60 = fmul float %59, 0x3F69803980000000, !dbg !15
  %61 = fcmp ogt float %60, 0.000000e+00, !dbg !15
  %62 = select i1 %61, float %60, float 0.000000e+00, !dbg !15
  %63 = tail call float @llvm.nearbyint.f32(float %62), !dbg !15
  %64 = fcmp ogt float %63, -1.280000e+02, !dbg !15
  %65 = select i1 %64, float %63, float -1.280000e+02, !dbg !15
  %66 = fcmp olt float %65, 1.270000e+02, !dbg !15
  %67 = select i1 %66, float %65, float 1.270000e+02, !dbg !15
  %68 = fptosi float %67 to i8, !dbg !15
  %69 = getelementptr inbounds i8, ptr %readout, i64 %56, !dbg !15
  store i8 %68, ptr %69, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 5, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 5, metadata !17, metadata !DIExpression()), !dbg !15
  %70 = add nuw nsw i64 %0, 5, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %70, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %70, metadata !18, metadata !DIExpression()), !dbg !15
  %71 = getelementptr inbounds i32, ptr %accumulator, i64 %70, !dbg !15
  %72 = load i32, ptr %71, align 4, !dbg !15, !tbaa !19
  %73 = sitofp i32 %72 to float, !dbg !15
  %74 = fmul float %73, 0x3F69803980000000, !dbg !15
  %75 = fcmp ogt float %74, 0.000000e+00, !dbg !15
  %76 = select i1 %75, float %74, float 0.000000e+00, !dbg !15
  %77 = tail call float @llvm.nearbyint.f32(float %76), !dbg !15
  %78 = fcmp ogt float %77, -1.280000e+02, !dbg !15
  %79 = select i1 %78, float %77, float -1.280000e+02, !dbg !15
  %80 = fcmp olt float %79, 1.270000e+02, !dbg !15
  %81 = select i1 %80, float %79, float 1.270000e+02, !dbg !15
  %82 = fptosi float %81 to i8, !dbg !15
  %83 = getelementptr inbounds i8, ptr %readout, i64 %70, !dbg !15
  store i8 %82, ptr %83, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 6, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 6, metadata !17, metadata !DIExpression()), !dbg !15
  %84 = add nuw nsw i64 %0, 6, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %84, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %84, metadata !18, metadata !DIExpression()), !dbg !15
  %85 = getelementptr inbounds i32, ptr %accumulator, i64 %84, !dbg !15
  %86 = load i32, ptr %85, align 4, !dbg !15, !tbaa !19
  %87 = sitofp i32 %86 to float, !dbg !15
  %88 = fmul float %87, 0x3F69803980000000, !dbg !15
  %89 = fcmp ogt float %88, 0.000000e+00, !dbg !15
  %90 = select i1 %89, float %88, float 0.000000e+00, !dbg !15
  %91 = tail call float @llvm.nearbyint.f32(float %90), !dbg !15
  %92 = fcmp ogt float %91, -1.280000e+02, !dbg !15
  %93 = select i1 %92, float %91, float -1.280000e+02, !dbg !15
  %94 = fcmp olt float %93, 1.270000e+02, !dbg !15
  %95 = select i1 %94, float %93, float 1.270000e+02, !dbg !15
  %96 = fptosi float %95 to i8, !dbg !15
  %97 = getelementptr inbounds i8, ptr %readout, i64 %84, !dbg !15
  store i8 %96, ptr %97, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 7, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 7, metadata !17, metadata !DIExpression()), !dbg !15
  %98 = add nuw nsw i64 %0, 7, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %98, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %98, metadata !18, metadata !DIExpression()), !dbg !15
  %99 = getelementptr inbounds i32, ptr %accumulator, i64 %98, !dbg !15
  %100 = load i32, ptr %99, align 4, !dbg !15, !tbaa !19
  %101 = sitofp i32 %100 to float, !dbg !15
  %102 = fmul float %101, 0x3F69803980000000, !dbg !15
  %103 = fcmp ogt float %102, 0.000000e+00, !dbg !15
  %104 = select i1 %103, float %102, float 0.000000e+00, !dbg !15
  %105 = tail call float @llvm.nearbyint.f32(float %104), !dbg !15
  %106 = fcmp ogt float %105, -1.280000e+02, !dbg !15
  %107 = select i1 %106, float %105, float -1.280000e+02, !dbg !15
  %108 = fcmp olt float %107, 1.270000e+02, !dbg !15
  %109 = select i1 %108, float %107, float 1.270000e+02, !dbg !15
  %110 = fptosi float %109 to i8, !dbg !15
  %111 = getelementptr inbounds i8, ptr %readout, i64 %98, !dbg !15
  store i8 %110, ptr %111, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 8, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 8, metadata !17, metadata !DIExpression()), !dbg !15
  %112 = add nuw nsw i64 %0, 8, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %112, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %112, metadata !18, metadata !DIExpression()), !dbg !15
  %113 = getelementptr inbounds i32, ptr %accumulator, i64 %112, !dbg !15
  %114 = load i32, ptr %113, align 4, !dbg !15, !tbaa !19
  %115 = sitofp i32 %114 to float, !dbg !15
  %116 = fmul float %115, 0x3F69803980000000, !dbg !15
  %117 = fcmp ogt float %116, 0.000000e+00, !dbg !15
  %118 = select i1 %117, float %116, float 0.000000e+00, !dbg !15
  %119 = tail call float @llvm.nearbyint.f32(float %118), !dbg !15
  %120 = fcmp ogt float %119, -1.280000e+02, !dbg !15
  %121 = select i1 %120, float %119, float -1.280000e+02, !dbg !15
  %122 = fcmp olt float %121, 1.270000e+02, !dbg !15
  %123 = select i1 %122, float %121, float 1.270000e+02, !dbg !15
  %124 = fptosi float %123 to i8, !dbg !15
  %125 = getelementptr inbounds i8, ptr %readout, i64 %112, !dbg !15
  store i8 %124, ptr %125, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 9, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 9, metadata !17, metadata !DIExpression()), !dbg !15
  %126 = add nuw nsw i64 %0, 9, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %126, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %126, metadata !18, metadata !DIExpression()), !dbg !15
  %127 = getelementptr inbounds i32, ptr %accumulator, i64 %126, !dbg !15
  %128 = load i32, ptr %127, align 4, !dbg !15, !tbaa !19
  %129 = sitofp i32 %128 to float, !dbg !15
  %130 = fmul float %129, 0x3F69803980000000, !dbg !15
  %131 = fcmp ogt float %130, 0.000000e+00, !dbg !15
  %132 = select i1 %131, float %130, float 0.000000e+00, !dbg !15
  %133 = tail call float @llvm.nearbyint.f32(float %132), !dbg !15
  %134 = fcmp ogt float %133, -1.280000e+02, !dbg !15
  %135 = select i1 %134, float %133, float -1.280000e+02, !dbg !15
  %136 = fcmp olt float %135, 1.270000e+02, !dbg !15
  %137 = select i1 %136, float %135, float 1.270000e+02, !dbg !15
  %138 = fptosi float %137 to i8, !dbg !15
  %139 = getelementptr inbounds i8, ptr %readout, i64 %126, !dbg !15
  store i8 %138, ptr %139, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 10, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 10, metadata !17, metadata !DIExpression()), !dbg !15
  %140 = add nuw nsw i64 %0, 10, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %140, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %140, metadata !18, metadata !DIExpression()), !dbg !15
  %141 = getelementptr inbounds i32, ptr %accumulator, i64 %140, !dbg !15
  %142 = load i32, ptr %141, align 4, !dbg !15, !tbaa !19
  %143 = sitofp i32 %142 to float, !dbg !15
  %144 = fmul float %143, 0x3F69803980000000, !dbg !15
  %145 = fcmp ogt float %144, 0.000000e+00, !dbg !15
  %146 = select i1 %145, float %144, float 0.000000e+00, !dbg !15
  %147 = tail call float @llvm.nearbyint.f32(float %146), !dbg !15
  %148 = fcmp ogt float %147, -1.280000e+02, !dbg !15
  %149 = select i1 %148, float %147, float -1.280000e+02, !dbg !15
  %150 = fcmp olt float %149, 1.270000e+02, !dbg !15
  %151 = select i1 %150, float %149, float 1.270000e+02, !dbg !15
  %152 = fptosi float %151 to i8, !dbg !15
  %153 = getelementptr inbounds i8, ptr %readout, i64 %140, !dbg !15
  store i8 %152, ptr %153, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 11, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 11, metadata !17, metadata !DIExpression()), !dbg !15
  %154 = add nuw nsw i64 %0, 11, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %154, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %154, metadata !18, metadata !DIExpression()), !dbg !15
  %155 = getelementptr inbounds i32, ptr %accumulator, i64 %154, !dbg !15
  %156 = load i32, ptr %155, align 4, !dbg !15, !tbaa !19
  %157 = sitofp i32 %156 to float, !dbg !15
  %158 = fmul float %157, 0x3F69803980000000, !dbg !15
  %159 = fcmp ogt float %158, 0.000000e+00, !dbg !15
  %160 = select i1 %159, float %158, float 0.000000e+00, !dbg !15
  %161 = tail call float @llvm.nearbyint.f32(float %160), !dbg !15
  %162 = fcmp ogt float %161, -1.280000e+02, !dbg !15
  %163 = select i1 %162, float %161, float -1.280000e+02, !dbg !15
  %164 = fcmp olt float %163, 1.270000e+02, !dbg !15
  %165 = select i1 %164, float %163, float 1.270000e+02, !dbg !15
  %166 = fptosi float %165 to i8, !dbg !15
  %167 = getelementptr inbounds i8, ptr %readout, i64 %154, !dbg !15
  store i8 %166, ptr %167, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 12, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 12, metadata !17, metadata !DIExpression()), !dbg !15
  %168 = add nuw nsw i64 %0, 12, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %168, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %168, metadata !18, metadata !DIExpression()), !dbg !15
  %169 = getelementptr inbounds i32, ptr %accumulator, i64 %168, !dbg !15
  %170 = load i32, ptr %169, align 4, !dbg !15, !tbaa !19
  %171 = sitofp i32 %170 to float, !dbg !15
  %172 = fmul float %171, 0x3F69803980000000, !dbg !15
  %173 = fcmp ogt float %172, 0.000000e+00, !dbg !15
  %174 = select i1 %173, float %172, float 0.000000e+00, !dbg !15
  %175 = tail call float @llvm.nearbyint.f32(float %174), !dbg !15
  %176 = fcmp ogt float %175, -1.280000e+02, !dbg !15
  %177 = select i1 %176, float %175, float -1.280000e+02, !dbg !15
  %178 = fcmp olt float %177, 1.270000e+02, !dbg !15
  %179 = select i1 %178, float %177, float 1.270000e+02, !dbg !15
  %180 = fptosi float %179 to i8, !dbg !15
  %181 = getelementptr inbounds i8, ptr %readout, i64 %168, !dbg !15
  store i8 %180, ptr %181, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 13, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 13, metadata !17, metadata !DIExpression()), !dbg !15
  %182 = add nuw nsw i64 %0, 13, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %182, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %182, metadata !18, metadata !DIExpression()), !dbg !15
  %183 = getelementptr inbounds i32, ptr %accumulator, i64 %182, !dbg !15
  %184 = load i32, ptr %183, align 4, !dbg !15, !tbaa !19
  %185 = sitofp i32 %184 to float, !dbg !15
  %186 = fmul float %185, 0x3F69803980000000, !dbg !15
  %187 = fcmp ogt float %186, 0.000000e+00, !dbg !15
  %188 = select i1 %187, float %186, float 0.000000e+00, !dbg !15
  %189 = tail call float @llvm.nearbyint.f32(float %188), !dbg !15
  %190 = fcmp ogt float %189, -1.280000e+02, !dbg !15
  %191 = select i1 %190, float %189, float -1.280000e+02, !dbg !15
  %192 = fcmp olt float %191, 1.270000e+02, !dbg !15
  %193 = select i1 %192, float %191, float 1.270000e+02, !dbg !15
  %194 = fptosi float %193 to i8, !dbg !15
  %195 = getelementptr inbounds i8, ptr %readout, i64 %182, !dbg !15
  store i8 %194, ptr %195, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 14, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 14, metadata !17, metadata !DIExpression()), !dbg !15
  %196 = add nuw nsw i64 %0, 14, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %196, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %196, metadata !18, metadata !DIExpression()), !dbg !15
  %197 = getelementptr inbounds i32, ptr %accumulator, i64 %196, !dbg !15
  %198 = load i32, ptr %197, align 4, !dbg !15, !tbaa !19
  %199 = sitofp i32 %198 to float, !dbg !15
  %200 = fmul float %199, 0x3F69803980000000, !dbg !15
  %201 = fcmp ogt float %200, 0.000000e+00, !dbg !15
  %202 = select i1 %201, float %200, float 0.000000e+00, !dbg !15
  %203 = tail call float @llvm.nearbyint.f32(float %202), !dbg !15
  %204 = fcmp ogt float %203, -1.280000e+02, !dbg !15
  %205 = select i1 %204, float %203, float -1.280000e+02, !dbg !15
  %206 = fcmp olt float %205, 1.270000e+02, !dbg !15
  %207 = select i1 %206, float %205, float 1.270000e+02, !dbg !15
  %208 = fptosi float %207 to i8, !dbg !15
  %209 = getelementptr inbounds i8, ptr %readout, i64 %196, !dbg !15
  store i8 %208, ptr %209, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 15, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 15, metadata !17, metadata !DIExpression()), !dbg !15
  %210 = add nuw nsw i64 %0, 15, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %210, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %210, metadata !18, metadata !DIExpression()), !dbg !15
  %211 = getelementptr inbounds i32, ptr %accumulator, i64 %210, !dbg !15
  %212 = load i32, ptr %211, align 4, !dbg !15, !tbaa !19
  %213 = sitofp i32 %212 to float, !dbg !15
  %214 = fmul float %213, 0x3F69803980000000, !dbg !15
  %215 = fcmp ogt float %214, 0.000000e+00, !dbg !15
  %216 = select i1 %215, float %214, float 0.000000e+00, !dbg !15
  %217 = tail call float @llvm.nearbyint.f32(float %216), !dbg !15
  %218 = fcmp ogt float %217, -1.280000e+02, !dbg !15
  %219 = select i1 %218, float %217, float -1.280000e+02, !dbg !15
  %220 = fcmp olt float %219, 1.270000e+02, !dbg !15
  %221 = select i1 %220, float %219, float 1.270000e+02, !dbg !15
  %222 = fptosi float %221 to i8, !dbg !15
  %223 = getelementptr inbounds i8, ptr %readout, i64 %210, !dbg !15
  store i8 %222, ptr %223, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 16, metadata !17, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 16, metadata !17, metadata !DIExpression()), !dbg !15
  %224 = add nuw nsw i64 %0, 16, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %224, metadata !18, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i64 %224, metadata !18, metadata !DIExpression()), !dbg !15
  %225 = getelementptr inbounds i32, ptr %accumulator, i64 %224, !dbg !15
  %226 = load i32, ptr %225, align 4, !dbg !15, !tbaa !19
  %227 = sitofp i32 %226 to float, !dbg !15
  %228 = fmul float %227, 0x3F69803980000000, !dbg !15
  %229 = fcmp ogt float %228, 0.000000e+00, !dbg !15
  %230 = select i1 %229, float %228, float 0.000000e+00, !dbg !15
  %231 = tail call float @llvm.nearbyint.f32(float %230), !dbg !15
  %232 = fcmp ogt float %231, -1.280000e+02, !dbg !15
  %233 = select i1 %232, float %231, float -1.280000e+02, !dbg !15
  %234 = fcmp olt float %233, 1.270000e+02, !dbg !15
  %235 = select i1 %234, float %233, float 1.270000e+02, !dbg !15
  %236 = fptosi float %235 to i8, !dbg !15
  %237 = getelementptr inbounds i8, ptr %readout, i64 %224, !dbg !15
  store i8 %236, ptr %237, align 1, !dbg !15, !tbaa !22
  call void @llvm.dbg.declare(metadata i64 17, metadata !17, metadata !DIExpression()), !dbg !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !15
  call void @llvm.dbg.declare(metadata i64 %indvars.iv.next, metadata !16, metadata !DIExpression()), !dbg !15
  %exitcond.not = icmp eq i64 %indvars.iv.next, 3, !dbg !15
  br i1 %exitcond.not, label %for_end_i, label %for_begin_j.preheader, !dbg !15, !prof !24

for_end_i:                                        ; preds = %for_begin_j.preheader
  ret i32 0, !dbg !15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nearbyint.f32(float) #1

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
!5 = distinct !DISubprogram(name: "stage_69_operator_kernel", scope: !1, file: !1, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !10}
!8 = !DIBasicType(name: "int32", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11)
!11 = !DIBasicType(name: "int8", size: 8, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "accumulator", arg: 1, scope: !5, file: !1, type: !9)
!14 = !DILocalVariable(name: "readout", arg: 2, scope: !5, file: !1, type: !10)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocalVariable(name: "i", scope: !5, file: !1, type: !8)
!17 = !DILocalVariable(name: "j", scope: !5, file: !1, type: !8)
!18 = !DILocalVariable(name: "cse_var_1", scope: !5, file: !1, type: !8)
!19 = !{!20, !20, i64 0}
!20 = !{!"0x40a58440", !21, i64 0}
!21 = !{!"tvm-tbaa"}
!22 = !{!23, !23, i64 0}
!23 = !{!"0x409f3f30", !21, i64 0}
!24 = !{!"branch_weights", i32 1, i32 1048575}
