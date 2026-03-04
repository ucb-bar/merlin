; ModuleID = 'matmul_dispatch_0'
source_filename = "matmul_dispatch_0"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-eabi-elf"

%iree_hal_executable_library_header_t = type { i32, ptr, i32, i32 }
%iree_hal_executable_dispatch_attrs_v0_t = type { i64, i16, i8, i8, i32, i32, i16, i16, i64, i64, i64, i64, i64 }
%iree_hal_executable_source_location_v0_t = type { i32, i32, ptr }
%iree_hal_executable_stage_location_table_v0_t = type { i32, ptr, ptr }
%iree_hal_executable_library_v0_t = type { ptr, %iree_hal_executable_import_table_v0_t, %iree_hal_executable_export_table_v0_t, %iree_hal_executable_constant_table_v0_t, %iree_hal_executable_source_file_table_v0_t }
%iree_hal_executable_import_table_v0_t = type { i32, ptr }
%iree_hal_executable_export_table_v0_t = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%iree_hal_executable_constant_table_v0_t = type { i32 }
%iree_hal_executable_source_file_table_v0_t = type { i32, ptr }
%iree_hal_executable_dispatch_state_v0_t = type { i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr }
%iree_hal_executable_environment_v0_t = type { ptr, ptr, ptr, ptr, %iree_hal_processor_v0_t }
%iree_hal_processor_v0_t = type { [8 x i64] }

@__import_ordinal_malloc = internal constant i32 1
@__import_ordinal_free = internal constant i32 0
@0 = private constant [18 x i8] c"matmul_dispatch_0\00", align 1
@iree_hal_executable_library_query_v0_header = private constant %iree_hal_executable_library_header_t { i32 6, ptr @0, i32 0, i32 0 }
@1 = private constant [5 x i8] c"free\00", align 1
@2 = private constant [7 x i8] c"malloc\00", align 1
@iree_hal_executable_library_query_v0_import_names = private constant [2 x ptr] [ptr @1, ptr @2]
@iree_hal_executable_library_query_v0_funcs = private constant [1 x ptr] [ptr @matmul_dispatch_0_matmul_4x4x4_f32]
@iree_hal_executable_library_query_v0_attrs = private constant [1 x %iree_hal_executable_dispatch_attrs_v0_t] [%iree_hal_executable_dispatch_attrs_v0_t { i64 0, i16 0, i8 0, i8 3, i32 1, i32 1, i16 1, i16 0, i64 0, i64 0, i64 0, i64 0, i64 0 }]
@3 = private constant [35 x i8] c"matmul_dispatch_0_matmul_4x4x4_f32\00", align 1
@iree_hal_executable_library_query_v0_names = private constant [1 x ptr] [ptr @3]
@4 = private constant [153 x i8] c"/Users/sparshsingh/work/merlin/compiler/plugins/target/Gemmini/test/Output/gemmini_matmul_lowering.mlir.tmp.dir/configured_module_matmul_dispatch_0.mlir\00", align 1
@iree_hal_executable_library_query_v0_source_locations = private constant [1 x %iree_hal_executable_source_location_v0_t] [%iree_hal_executable_source_location_v0_t { i32 3, i32 152, ptr @4 }]
@iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names = private constant [0 x ptr] zeroinitializer
@iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations = private constant [0 x %iree_hal_executable_source_location_v0_t] zeroinitializer
@iree_hal_executable_library_query_v0_stage_location_tables = private constant [1 x %iree_hal_executable_stage_location_table_v0_t] [%iree_hal_executable_stage_location_table_v0_t { i32 0, ptr @iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_names, ptr @iree_hal_executable_library_query_v0_matmul_dispatch_0_matmul_4x4x4_f32_stage_source_locations }]
@iree_hal_executable_library_query_v0 = private constant %iree_hal_executable_library_v0_t { ptr @iree_hal_executable_library_query_v0_header, %iree_hal_executable_import_table_v0_t { i32 2, ptr @iree_hal_executable_library_query_v0_import_names }, %iree_hal_executable_export_table_v0_t { i32 1, ptr @iree_hal_executable_library_query_v0_funcs, ptr @iree_hal_executable_library_query_v0_attrs, ptr null, ptr null, ptr @iree_hal_executable_library_query_v0_names, ptr null, ptr null, ptr @iree_hal_executable_library_query_v0_source_locations, ptr @iree_hal_executable_library_query_v0_stage_location_tables }, %iree_hal_executable_constant_table_v0_t zeroinitializer, %iree_hal_executable_source_file_table_v0_t zeroinitializer }

declare void @free(ptr) #0

declare ptr @malloc(i64) #0

define internal i32 @matmul_dispatch_0_matmul_4x4x4_f32(ptr noalias noundef nonnull align 16 %0, ptr noalias noundef nonnull align 16 %1, ptr noalias noundef nonnull align 16 %2) #0 !dbg !3 {
  %4 = load %iree_hal_executable_dispatch_state_v0_t, ptr %1, align 8, !dbg !79
  %5 = extractvalue %iree_hal_executable_dispatch_state_v0_t %4, 10, !dbg !79
  %6 = load ptr, ptr %5, align 8, !dbg !79
  call void @llvm.assume(i1 true) [ "align"(ptr %6, i64 64) ], !dbg !79
  %7 = load %iree_hal_executable_dispatch_state_v0_t, ptr %1, align 8, !dbg !80
  %8 = extractvalue %iree_hal_executable_dispatch_state_v0_t %7, 10, !dbg !80
  %9 = getelementptr ptr, ptr %8, i32 1, !dbg !80
  %10 = load ptr, ptr %9, align 8, !dbg !80
  call void @llvm.assume(i1 true) [ "align"(ptr %10, i64 64) ], !dbg !80
  %11 = load %iree_hal_executable_dispatch_state_v0_t, ptr %1, align 8, !dbg !81
  %12 = extractvalue %iree_hal_executable_dispatch_state_v0_t %11, 10, !dbg !81
  %13 = getelementptr ptr, ptr %12, i32 2, !dbg !81
  %14 = load ptr, ptr %13, align 8, !dbg !81
  call void @llvm.assume(i1 true) [ "align"(ptr %14, i64 64) ], !dbg !81
  %15 = alloca { ptr, i64 }, i64 1, align 8, !dbg !82
  store { ptr, i64 } { ptr undef, i64 64 }, ptr %15, align 8, !dbg !82
  %16 = load i32, ptr @__import_ordinal_malloc, align 4, !dbg !82
  %17 = load %iree_hal_executable_environment_v0_t, ptr %0, align 8, !dbg !82
  %18 = extractvalue %iree_hal_executable_environment_v0_t %17, 1, !dbg !82
  %19 = extractvalue %iree_hal_executable_environment_v0_t %17, 2, !dbg !82
  %20 = getelementptr ptr, ptr %19, i32 %16, !dbg !82
  %21 = extractvalue %iree_hal_executable_environment_v0_t %17, 3, !dbg !82
  %22 = getelementptr ptr, ptr %21, i32 %16, !dbg !82
  %23 = load ptr, ptr %20, align 8, !dbg !82
  %24 = load ptr, ptr %22, align 8, !dbg !82
  %25 = call i32 %18(ptr %23, ptr %15, ptr %24, ptr null), !dbg !82
  %26 = icmp eq i32 %25, 0, !dbg !82
  br i1 %26, label %29, label %27, !dbg !82, !prof !83

27:                                               ; preds = %45, %3
  %28 = phi i32 [ %61, %45 ], [ %25, %3 ], !dbg !82
  ret i32 %28, !dbg !82

29:                                               ; preds = %3
  %30 = load { ptr, i64 }, ptr %15, align 8, !dbg !82
  %31 = extractvalue { ptr, i64 } %30, 0, !dbg !82
  br label %32, !dbg !82

32:                                               ; preds = %43, %29
  %33 = phi i64 [ %44, %43 ], [ 0, %29 ], !dbg !82
  %34 = icmp slt i64 %33, 4, !dbg !82
  br i1 %34, label %35, label %45, !dbg !82

35:                                               ; preds = %38, %32
  %36 = phi i64 [ %42, %38 ], [ 0, %32 ], !dbg !82
  %37 = icmp slt i64 %36, 4, !dbg !82
  br i1 %37, label %38, label %43, !dbg !82

38:                                               ; preds = %35
  %39 = mul nuw nsw i64 %33, 4, !dbg !82
  %40 = add nuw nsw i64 %39, %36, !dbg !82
  %41 = getelementptr inbounds nuw i32, ptr %31, i64 %40, !dbg !82
  store i32 0, ptr %41, align 4, !dbg !82
  %42 = add i64 %36, 1, !dbg !82
  br label %35, !dbg !82

43:                                               ; preds = %35
  %44 = add i64 %33, 1, !dbg !82
  br label %32, !dbg !82

45:                                               ; preds = %32
  %46 = ptrtoint ptr %6 to i64, !dbg !82
  %47 = ptrtoint ptr %10 to i64, !dbg !82
  %48 = ptrtoint ptr %14 to i64, !dbg !82
  %49 = ptrtoint ptr %31 to i64, !dbg !82
  call void @llvm.riscv.config.ex(i64 4575657221408489476, i64 281474976710656), !dbg !82
  call void @llvm.riscv.config.st(i64 2, i64 4575657221408423940), !dbg !82
  call void @llvm.riscv.config.ld(i64 4575657221409472769, i64 4), !dbg !82
  call void @llvm.riscv.config.ld(i64 4575657221409472777, i64 4), !dbg !82
  call void @llvm.riscv.config.ld(i64 4575657221409472785, i64 16), !dbg !82
  call void @llvm.riscv.loop.ws.config.bounds(i64 51540393996, i64 4295032833), !dbg !82
  call void @llvm.riscv.loop.ws.config.addrs.ab(i64 %46, i64 %47), !dbg !82
  call void @llvm.riscv.loop.ws.config.addrs.dc(i64 %49, i64 %48), !dbg !82
  call void @llvm.riscv.loop.ws.config.strides.ab(i64 4, i64 4), !dbg !82
  call void @llvm.riscv.loop.ws.config.strides.dc(i64 4, i64 4), !dbg !82
  call void @llvm.riscv.loop.ws(i64 1, i64 0), !dbg !82
  call void @llvm.riscv.flush(i64 0, i64 0), !dbg !82
  %50 = alloca { ptr }, i64 1, align 8, !dbg !82
  %51 = insertvalue { ptr } undef, ptr %31, 0, !dbg !82
  store { ptr } %51, ptr %50, align 8, !dbg !82
  %52 = load i32, ptr @__import_ordinal_free, align 4, !dbg !82
  %53 = load %iree_hal_executable_environment_v0_t, ptr %0, align 8, !dbg !82
  %54 = extractvalue %iree_hal_executable_environment_v0_t %53, 1, !dbg !82
  %55 = extractvalue %iree_hal_executable_environment_v0_t %53, 2, !dbg !82
  %56 = getelementptr ptr, ptr %55, i32 %52, !dbg !82
  %57 = extractvalue %iree_hal_executable_environment_v0_t %53, 3, !dbg !82
  %58 = getelementptr ptr, ptr %57, i32 %52, !dbg !82
  %59 = load ptr, ptr %56, align 8, !dbg !82
  %60 = load ptr, ptr %58, align 8, !dbg !82
  %61 = call i32 %54(ptr %59, ptr %50, ptr %60, ptr null), !dbg !82
  %62 = icmp eq i32 %61, 0, !dbg !82
  br i1 %62, label %63, label %27, !dbg !82, !prof !83

63:                                               ; preds = %45
  ret i32 0, !dbg !84
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nounwind
declare void @llvm.riscv.config.ex(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.config.st(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.config.ld(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws.config.bounds(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws.config.addrs.ab(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws.config.addrs.dc(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws.config.strides.ab(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws.config.strides.dc(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.loop.ws(i64, i64) #2

; Function Attrs: nounwind
declare void @llvm.riscv.flush(i64, i64) #2

; Function Attrs: uwtable
define dso_local dllexport ptr @iree_hal_executable_library_query(i32 %0, ptr %1) #3 {
entry:
  %2 = icmp eq i32 %0, 6
  %3 = select i1 %2, ptr @iree_hal_executable_library_query_v0, ptr null
  ret ptr %3
}

attributes #0 = { "frame-pointer"="all" "hot" "no-builtins" "nonlazybind" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) "frame-pointer"="all" "hot" "no-builtins" "nonlazybind" }
attributes #2 = { nounwind "frame-pointer"="all" "hot" "no-builtins" "nonlazybind" }
attributes #3 = { uwtable "nonlazybind" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C17, file: !1, producer: "IREE", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "configured_module_matmul_dispatch_0.mlir", directory: "/Users/sparshsingh/work/merlin/compiler/plugins/target/Gemmini/test/Output/gemmini_matmul_lowering.mlir.tmp.dir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "matmul_dispatch_0_matmul_4x4x4_f32", linkageName: "matmul_dispatch_0_matmul_4x4x4_f32", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{!6, !7, !38, !67}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "iree_hal_executable_environment_v0_t", baseType: !10)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "iree_hal_executable_environment_v0_t", scope: !11, file: !11, line: 246, size: 768, elements: !12)
!11 = !DIFile(filename: "runtime/src/iree/hal/local/executable_library.h", directory: ".")
!12 = !{!13, !21, !24, !27, !29}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "constants", baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, scope: !11, file: !11, line: 227, baseType: !17, size: 2048, elements: !19)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", baseType: !18)
!18 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!19 = !{!20}
!20 = !DISubrange(count: 64)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "import_thunk", baseType: !22, size: 64, offset: 64)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DIBasicType(name: "void", encoding: DW_ATE_address)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "import_funcs", baseType: !25, size: 64, offset: 128)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64)
!26 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "import_contexts", baseType: !28, size: 64, offset: 192)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 64)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "processor", baseType: !30, offset: 256)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "iree_hal_processor_v0_t", scope: !11, file: !11, line: 227, size: 512, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "data", baseType: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, scope: !11, file: !11, line: 227, baseType: !34, size: 512, elements: !36)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", baseType: !35)
!35 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!36 = !{!37}
!37 = !DISubrange(count: 8)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64)
!39 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !40)
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "iree_hal_executable_dispatch_state_v0_t", baseType: !41)
!41 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "iree_hal_executable_dispatch_state_v0_t", scope: !11, file: !11, line: 275, size: 384, elements: !42)
!42 = !{!43, !44, !45, !48, !49, !50, !51, !52, !55, !56, !57, !62}
!43 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_size_x", baseType: !17, size: 32)
!44 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_size_y", baseType: !17, size: 32, offset: 32)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_size_z", baseType: !46, size: 16, offset: 64)
!46 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint16_t", baseType: !47)
!47 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "constant_count", baseType: !46, size: 16, offset: 80)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_count_x", baseType: !17, size: 32, offset: 96)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_count_y", baseType: !17, size: 32, offset: 128)
!51 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_count_z", baseType: !46, size: 16, offset: 160)
!52 = !DIDerivedType(tag: DW_TAG_member, name: "max_concurrency", baseType: !53, size: 8, offset: 176)
!53 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", baseType: !54)
!54 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!55 = !DIDerivedType(tag: DW_TAG_member, name: "binding_count", baseType: !53, size: 8, offset: 184)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "constants", baseType: !14, size: 64, offset: 192)
!57 = !DIDerivedType(tag: DW_TAG_member, name: "binding_ptrs", baseType: !58, size: 64, offset: 256)
!58 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !59, size: 64)
!59 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !60)
!60 = !DICompositeType(tag: DW_TAG_array_type, scope: !11, file: !11, line: 227, baseType: !61, size: 4096, elements: !19)
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !53, size: 64)
!62 = !DIDerivedType(tag: DW_TAG_member, name: "binding_lengths", baseType: !63, size: 64, offset: 320)
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !64, size: 64)
!64 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !65)
!65 = !DICompositeType(tag: DW_TAG_array_type, scope: !11, file: !11, line: 227, baseType: !66, size: 4096, elements: !19)
!66 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", baseType: !34)
!67 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !68, size: 64)
!68 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !69)
!69 = !DIDerivedType(tag: DW_TAG_typedef, name: "iree_hal_executable_workgroup_state_v0_t", baseType: !70)
!70 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "iree_hal_executable_workgroup_state_v0_t", scope: !11, file: !11, line: 321, size: 256, elements: !71)
!71 = !{!72, !73, !74, !75, !76, !77, !78}
!72 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_id_x", baseType: !17, size: 32)
!73 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_id_y", baseType: !17, size: 32, offset: 32)
!74 = !DIDerivedType(tag: DW_TAG_member, name: "workgroup_id_z", baseType: !46, size: 16, offset: 64)
!75 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", baseType: !46, size: 16, offset: 80)
!76 = !DIDerivedType(tag: DW_TAG_member, name: "processor_id", baseType: !17, size: 32, offset: 96)
!77 = !DIDerivedType(tag: DW_TAG_member, name: "local_memory", baseType: !22, size: 64, offset: 128)
!78 = !DIDerivedType(tag: DW_TAG_member, name: "local_memory_size", baseType: !17, size: 32, offset: 192)
!79 = !DILocation(line: 10, column: 8, scope: !3)
!80 = !DILocation(line: 11, column: 8, scope: !3)
!81 = !DILocation(line: 12, column: 8, scope: !3)
!82 = !DILocation(line: 16, column: 8, scope: !3)
!83 = !{!"branch_weights", i32 1, i32 0}
!84 = !DILocation(line: 18, column: 8, scope: !3)
