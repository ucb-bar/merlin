// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  func.func @matmul(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<64x64xf32>) {
    %alloc = memref.alloc() : memref<64x64xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    gemmini.tile_matmul %arg0 %arg1 %arg2 %alloc : memref<64x64xf16> memref<64x64xf16> memref<64x64xf32> memref<64x64xi32>
    memref.dealloc %alloc : memref<64x64xi32>
    return
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg16, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg17, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg19, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg18, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg20, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg0, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg1, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg2, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg3, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg5, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg4, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg6, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(64 : index) : i64
    %25 = llvm.mlir.constant(64 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(4096 : index) : i64
    %28 = llvm.mlir.zero : !llvm.ptr
    %29 = llvm.getelementptr %28[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %24, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %25, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %26, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = builtin.unrealized_conversion_cast %40 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x64xi32>
    %42 = llvm.mlir.constant(0 : i32) : i32
    linalg.fill ins(%42 : i32) outs(%41 : memref<64x64xi32>)
    %43 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %52 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%51, %52) : (i64, i64) -> ()
    %53 = llvm.mlir.constant(64 : i64) : i64
    %54 = llvm.mlir.constant(2 : i64) : i64
    %55 = llvm.mlir.constant(4575657221408424000 : i64) : i64
    "gemmini.intr.config_st"(%54, %55) : (i64, i64) -> ()
    %56 = llvm.mlir.constant(64 : i64) : i64
    %57 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%57, %56) : (i64, i64) -> ()
    %58 = llvm.mlir.constant(64 : i64) : i64
    %59 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%59, %58) : (i64, i64) -> ()
    %60 = llvm.mlir.constant(256 : i64) : i64
    %61 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%61, %60) : (i64, i64) -> ()
    %62 = llvm.mlir.constant(0 : i64) : i64
    %63 = llvm.mlir.constant(0 : i64) : i64
    %64 = llvm.mlir.constant(0 : i64) : i64
    %65 = llvm.mlir.constant(0 : i64) : i64
    %66 = llvm.mlir.constant(0 : i64) : i64
    %67 = llvm.mlir.constant(17180131332 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%66, %67) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%44, %46) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%50, %48) : (i64, i64) -> ()
    %68 = llvm.mlir.constant(64 : i64) : i64
    %69 = llvm.mlir.constant(64 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%68, %69) : (i64, i64) -> ()
    %70 = llvm.mlir.constant(64 : i64) : i64
    %71 = llvm.mlir.constant(64 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%70, %71) : (i64, i64) -> ()
    %72 = llvm.mlir.constant(1 : i64) : i64
    %73 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%72, %73) : (i64, i64) -> ()
    %74 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%74, %74) : (i64, i64) -> ()
    %75 = llvm.extractvalue %40[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%75) : (!llvm.ptr) -> ()
    llvm.return
  }
}


