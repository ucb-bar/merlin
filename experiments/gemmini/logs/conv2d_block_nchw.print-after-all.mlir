// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  func.func @conv2d_block_nchw(%arg0: memref<1x3x32x32xf32>, %arg1: memref<64x3x3x3xf32>, %arg2: memref<1x64x30x30xf32>) {
    %alloc = memref.alloc() : memref<1x32x32x3xf32>
    %alloc_0 = memref.alloc() : memref<27x64xf32>
    %alloc_1 = memref.alloc() : memref<64xi32>
    %alloc_2 = memref.alloc() : memref<900x64xf32>
    %c30_i64 = arith.constant 30 : i64
    %c3 = arith.constant 3 : index
    %c3_3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c1 step %c1_4 {
      %c0_10 = arith.constant 0 : index
      %c3_11 = arith.constant 3 : index
      %c1_12 = arith.constant 1 : index
      scf.for %arg4 = %c0_10 to %c3_11 step %c1_12 {
        %c0_13 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg5 = %c0_13 to %c32 step %c1_14 {
          %c0_15 = arith.constant 0 : index
          %c32_16 = arith.constant 32 : index
          %c1_17 = arith.constant 1 : index
          scf.for %arg6 = %c0_15 to %c32_16 step %c1_17 {
            %0 = memref.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<1x3x32x32xf32>
            memref.store %0, %alloc[%arg3, %arg5, %arg6, %arg4] : memref<1x32x32x3xf32>
          }
        }
      }
    }
    %c0_5 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_6 = arith.constant 1 : index
    scf.for %arg3 = %c0_5 to %c64 step %c1_6 {
      %c0_10 = arith.constant 0 : index
      %c3_11 = arith.constant 3 : index
      %c1_12 = arith.constant 1 : index
      scf.for %arg4 = %c0_10 to %c3_11 step %c1_12 {
        %c0_13 = arith.constant 0 : index
        %c3_14 = arith.constant 3 : index
        %c1_15 = arith.constant 1 : index
        scf.for %arg5 = %c0_13 to %c3_14 step %c1_15 {
          %c0_16 = arith.constant 0 : index
          %c3_17 = arith.constant 3 : index
          %c1_18 = arith.constant 1 : index
          scf.for %arg6 = %c0_16 to %c3_17 step %c1_18 {
            %0 = arith.muli %arg5, %c3 : index
            %1 = arith.muli %0, %c3_3 : index
            %2 = arith.muli %arg6, %c3_3 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg4 : index
            %5 = memref.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<64x3x3x3xf32>
            memref.store %5, %alloc_0[%4, %arg3] : memref<27x64xf32>
          }
        }
      }
    }
    %c3_i64 = arith.constant 3 : i64
    gemmini.tile_conv %alloc %alloc_0 %alloc_1 %alloc_2 %c30_i64 %c30_i64 %c3_i64 : memref<1x32x32x3xf32> memref<27x64xf32> memref<64xi32> memref<900x64xf32> i64 i64 i64
    %c0_7 = arith.constant 0 : index
    %c1_8 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    scf.for %arg3 = %c0_7 to %c1_8 step %c1_9 {
      %c0_10 = arith.constant 0 : index
      %c64_11 = arith.constant 64 : index
      %c1_12 = arith.constant 1 : index
      scf.for %arg4 = %c0_10 to %c64_11 step %c1_12 {
        %c0_13 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_14 = arith.constant 1 : index
        scf.for %arg5 = %c0_13 to %c30 step %c1_14 {
          %c0_15 = arith.constant 0 : index
          %c30_16 = arith.constant 30 : index
          %c1_17 = arith.constant 1 : index
          scf.for %arg6 = %c0_15 to %c30_16 step %c1_17 {
            %c30_18 = arith.constant 30 : index
            %0 = arith.muli %arg3, %c30_18 : index
            %1 = arith.muli %0, %c30_18 : index
            %2 = arith.muli %arg5, %c30_18 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg6 : index
            %5 = memref.load %alloc_2[%4, %arg4] : memref<900x64xf32>
            memref.store %5, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x64x30x30xf32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<1x32x32x3xf32>
    memref.dealloc %alloc_0 : memref<27x64xf32>
    memref.dealloc %alloc_2 : memref<900x64xf32>
    memref.dealloc %alloc_1 : memref<64xi32>
    return
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @conv2d_block_nchw(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg22, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg23, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg24, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg25, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg29, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg26, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg30, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg27, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg31, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg28, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg32, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg18, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg19, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg16, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg20, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg17, %21[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg21, %22[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %25 = llvm.insertvalue %arg0, %24[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %arg1, %25[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %arg2, %26[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %arg3, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %arg7, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %arg4, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %arg8, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %arg5, %31[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %arg9, %32[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %arg6, %33[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %arg10, %34[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(32 : index) : i64
    %38 = llvm.mlir.constant(32 : index) : i64
    %39 = llvm.mlir.constant(3 : index) : i64
    %40 = llvm.mlir.constant(1 : index) : i64
    %41 = llvm.mlir.constant(96 : index) : i64
    %42 = llvm.mlir.constant(3072 : index) : i64
    %43 = llvm.mlir.constant(3072 : index) : i64
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.insertvalue %36, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.insertvalue %37, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %38, %54[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %39, %55[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %42, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %41, %57[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %39, %58[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.insertvalue %40, %59[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.mlir.constant(27 : index) : i64
    %62 = llvm.mlir.constant(64 : index) : i64
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.mlir.constant(1728 : index) : i64
    %65 = llvm.mlir.zero : !llvm.ptr
    %66 = llvm.getelementptr %65[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr
    %69 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %68, %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %61, %73[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %62, %74[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %62, %75[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %63, %76[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.mlir.constant(64 : index) : i64
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.mlir.zero : !llvm.ptr
    %81 = llvm.getelementptr %80[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %82 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %83 = llvm.call @malloc(%82) : (i64) -> !llvm.ptr
    %84 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.insertvalue %83, %85[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %87 = llvm.mlir.constant(0 : index) : i64
    %88 = llvm.insertvalue %87, %86[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.insertvalue %78, %88[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %90 = llvm.insertvalue %79, %89[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %91 = llvm.mlir.constant(900 : index) : i64
    %92 = llvm.mlir.constant(64 : index) : i64
    %93 = llvm.mlir.constant(1 : index) : i64
    %94 = llvm.mlir.constant(57600 : index) : i64
    %95 = llvm.mlir.zero : !llvm.ptr
    %96 = llvm.getelementptr %95[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.call @malloc(%97) : (i64) -> !llvm.ptr
    %99 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %100 = llvm.insertvalue %98, %99[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.insertvalue %98, %100[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.mlir.constant(0 : index) : i64
    %103 = llvm.insertvalue %102, %101[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %91, %103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %92, %104[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %92, %105[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %93, %106[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(30 : i64) : i64
    %109 = llvm.mlir.constant(3 : index) : i64
    %110 = llvm.mlir.constant(3 : index) : i64
    %111 = llvm.mlir.constant(0 : index) : i64
    %112 = llvm.mlir.constant(1 : index) : i64
    %113 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%111 : i64)
  ^bb1(%114: i64):  // 2 preds: ^bb0, ^bb11
    %115 = llvm.icmp "slt" %114, %112 : i64
    llvm.cond_br %115, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %116 = llvm.mlir.constant(0 : index) : i64
    %117 = llvm.mlir.constant(3 : index) : i64
    %118 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%116 : i64)
  ^bb3(%119: i64):  // 2 preds: ^bb2, ^bb10
    %120 = llvm.icmp "slt" %119, %117 : i64
    llvm.cond_br %120, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %121 = llvm.mlir.constant(0 : index) : i64
    %122 = llvm.mlir.constant(32 : index) : i64
    %123 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%121 : i64)
  ^bb5(%124: i64):  // 2 preds: ^bb4, ^bb9
    %125 = llvm.icmp "slt" %124, %122 : i64
    llvm.cond_br %125, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %126 = llvm.mlir.constant(0 : index) : i64
    %127 = llvm.mlir.constant(32 : index) : i64
    %128 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%126 : i64)
  ^bb7(%129: i64):  // 2 preds: ^bb6, ^bb8
    %130 = llvm.icmp "slt" %129, %127 : i64
    llvm.cond_br %130, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %131 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %132 = llvm.mlir.constant(3072 : index) : i64
    %133 = llvm.mul %114, %132 : i64
    %134 = llvm.mlir.constant(1024 : index) : i64
    %135 = llvm.mul %119, %134 : i64
    %136 = llvm.add %133, %135 : i64
    %137 = llvm.mlir.constant(32 : index) : i64
    %138 = llvm.mul %124, %137 : i64
    %139 = llvm.add %136, %138 : i64
    %140 = llvm.add %139, %129 : i64
    %141 = llvm.getelementptr %131[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %142 = llvm.load %141 : !llvm.ptr -> f32
    %143 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %144 = llvm.mlir.constant(3072 : index) : i64
    %145 = llvm.mul %114, %144 : i64
    %146 = llvm.mlir.constant(96 : index) : i64
    %147 = llvm.mul %124, %146 : i64
    %148 = llvm.add %145, %147 : i64
    %149 = llvm.mlir.constant(3 : index) : i64
    %150 = llvm.mul %129, %149 : i64
    %151 = llvm.add %148, %150 : i64
    %152 = llvm.add %151, %119 : i64
    %153 = llvm.getelementptr %143[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %142, %153 : f32, !llvm.ptr
    %154 = llvm.add %129, %128 : i64
    llvm.br ^bb7(%154 : i64)
  ^bb9:  // pred: ^bb7
    %155 = llvm.add %124, %123 : i64
    llvm.br ^bb5(%155 : i64)
  ^bb10:  // pred: ^bb5
    %156 = llvm.add %119, %118 : i64
    llvm.br ^bb3(%156 : i64)
  ^bb11:  // pred: ^bb3
    %157 = llvm.add %114, %113 : i64
    llvm.br ^bb1(%157 : i64)
  ^bb12:  // pred: ^bb1
    %158 = llvm.mlir.constant(0 : index) : i64
    %159 = llvm.mlir.constant(64 : index) : i64
    %160 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%158 : i64)
  ^bb13(%161: i64):  // 2 preds: ^bb12, ^bb23
    %162 = llvm.icmp "slt" %161, %159 : i64
    llvm.cond_br %162, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    %163 = llvm.mlir.constant(0 : index) : i64
    %164 = llvm.mlir.constant(3 : index) : i64
    %165 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%163 : i64)
  ^bb15(%166: i64):  // 2 preds: ^bb14, ^bb22
    %167 = llvm.icmp "slt" %166, %164 : i64
    llvm.cond_br %167, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    %168 = llvm.mlir.constant(0 : index) : i64
    %169 = llvm.mlir.constant(3 : index) : i64
    %170 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb17(%168 : i64)
  ^bb17(%171: i64):  // 2 preds: ^bb16, ^bb21
    %172 = llvm.icmp "slt" %171, %169 : i64
    llvm.cond_br %172, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %173 = llvm.mlir.constant(0 : index) : i64
    %174 = llvm.mlir.constant(3 : index) : i64
    %175 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%173 : i64)
  ^bb19(%176: i64):  // 2 preds: ^bb18, ^bb20
    %177 = llvm.icmp "slt" %176, %174 : i64
    llvm.cond_br %177, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %178 = llvm.mul %171, %109 : i64
    %179 = llvm.mul %178, %110 : i64
    %180 = llvm.mul %176, %110 : i64
    %181 = llvm.add %179, %180 : i64
    %182 = llvm.add %181, %166 : i64
    %183 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %184 = llvm.mlir.constant(27 : index) : i64
    %185 = llvm.mul %161, %184 : i64
    %186 = llvm.mlir.constant(9 : index) : i64
    %187 = llvm.mul %166, %186 : i64
    %188 = llvm.add %185, %187 : i64
    %189 = llvm.mlir.constant(3 : index) : i64
    %190 = llvm.mul %171, %189 : i64
    %191 = llvm.add %188, %190 : i64
    %192 = llvm.add %191, %176 : i64
    %193 = llvm.getelementptr %183[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %194 = llvm.load %193 : !llvm.ptr -> f32
    %195 = llvm.extractvalue %77[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %196 = llvm.mlir.constant(64 : index) : i64
    %197 = llvm.mul %182, %196 : i64
    %198 = llvm.add %197, %161 : i64
    %199 = llvm.getelementptr %195[%198] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %194, %199 : f32, !llvm.ptr
    %200 = llvm.add %176, %175 : i64
    llvm.br ^bb19(%200 : i64)
  ^bb21:  // pred: ^bb19
    %201 = llvm.add %171, %170 : i64
    llvm.br ^bb17(%201 : i64)
  ^bb22:  // pred: ^bb17
    %202 = llvm.add %166, %165 : i64
    llvm.br ^bb15(%202 : i64)
  ^bb23:  // pred: ^bb15
    %203 = llvm.add %161, %160 : i64
    llvm.br ^bb13(%203 : i64)
  ^bb24:  // pred: ^bb13
    %204 = llvm.mlir.constant(3 : i64) : i64
    %205 = llvm.extractvalue %60[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %206 = llvm.ptrtoint %205 : !llvm.ptr to i64
    %207 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.ptrtoint %207 : !llvm.ptr to i64
    %209 = llvm.extractvalue %90[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %210 = llvm.ptrtoint %209 : !llvm.ptr to i64
    %211 = llvm.extractvalue %77[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %212 = llvm.ptrtoint %211 : !llvm.ptr to i64
    %213 = llvm.mlir.constant(64 : i64) : i64
    %214 = llvm.mlir.constant(2 : i64) : i64
    %215 = llvm.mlir.constant(4575657221408424000 : i64) : i64
    "gemmini.intr.config_st"(%214, %215) : (i64, i64) -> ()
    %216 = llvm.mlir.constant(65540 : i64) : i64
    %217 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%216, %217) : (i64, i64) -> ()
    %218 = llvm.mlir.constant(0 : i64) : i64
    %219 = llvm.mlir.constant(0 : i64) : i64
    %220 = llvm.mlir.constant(0 : i64) : i64
    %221 = llvm.mlir.constant(0 : i64) : i64
    %222 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %223 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%222, %223) : (i64, i64) -> ()
    %224 = llvm.mlir.constant(844429225164800 : i64) : i64
    %225 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%224, %225) : (i64, i64) -> ()
    %226 = llvm.mlir.constant(844437815230464 : i64) : i64
    %227 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%226, %227) : (i64, i64) -> ()
    %228 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %229 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%228, %229) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%212, %208) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%210, %206) : (i64, i64) -> ()
    %230 = llvm.mlir.constant(768 : i64) : i64
    %231 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%230, %231) : (i64, i64) -> ()
    %232 = llvm.mlir.constant(16 : i64) : i64
    %233 = llvm.add %208, %232 : i64
    %234 = llvm.mlir.constant(64 : i64) : i64
    %235 = llvm.add %210, %234 : i64
    %236 = llvm.mlir.constant(16 : i64) : i64
    %237 = llvm.add %212, %236 : i64
    %238 = llvm.mlir.constant(0 : i64) : i64
    %239 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %240 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%239, %240) : (i64, i64) -> ()
    %241 = llvm.mlir.constant(844429225164800 : i64) : i64
    %242 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%241, %242) : (i64, i64) -> ()
    %243 = llvm.mlir.constant(844437815230464 : i64) : i64
    %244 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%243, %244) : (i64, i64) -> ()
    %245 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %246 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%245, %246) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%237, %233) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%235, %206) : (i64, i64) -> ()
    %247 = llvm.mlir.constant(768 : i64) : i64
    %248 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%247, %248) : (i64, i64) -> ()
    %249 = llvm.mlir.constant(32 : i64) : i64
    %250 = llvm.add %208, %249 : i64
    %251 = llvm.mlir.constant(128 : i64) : i64
    %252 = llvm.add %210, %251 : i64
    %253 = llvm.mlir.constant(32 : i64) : i64
    %254 = llvm.add %212, %253 : i64
    %255 = llvm.mlir.constant(0 : i64) : i64
    %256 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %257 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%256, %257) : (i64, i64) -> ()
    %258 = llvm.mlir.constant(844429225164800 : i64) : i64
    %259 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%258, %259) : (i64, i64) -> ()
    %260 = llvm.mlir.constant(844437815230464 : i64) : i64
    %261 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%260, %261) : (i64, i64) -> ()
    %262 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %263 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%262, %263) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%254, %250) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%252, %206) : (i64, i64) -> ()
    %264 = llvm.mlir.constant(768 : i64) : i64
    %265 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%264, %265) : (i64, i64) -> ()
    %266 = llvm.mlir.constant(48 : i64) : i64
    %267 = llvm.add %208, %266 : i64
    %268 = llvm.mlir.constant(192 : i64) : i64
    %269 = llvm.add %210, %268 : i64
    %270 = llvm.mlir.constant(48 : i64) : i64
    %271 = llvm.add %212, %270 : i64
    %272 = llvm.mlir.constant(0 : i64) : i64
    %273 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %274 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%273, %274) : (i64, i64) -> ()
    %275 = llvm.mlir.constant(844429225164800 : i64) : i64
    %276 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%275, %276) : (i64, i64) -> ()
    %277 = llvm.mlir.constant(844437815230464 : i64) : i64
    %278 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%277, %278) : (i64, i64) -> ()
    %279 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %280 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%279, %280) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%271, %267) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%269, %206) : (i64, i64) -> ()
    %281 = llvm.mlir.constant(768 : i64) : i64
    %282 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%281, %282) : (i64, i64) -> ()
    %283 = llvm.mlir.constant(1472 : i64) : i64
    %284 = llvm.add %208, %283 : i64
    %285 = llvm.mlir.constant(0 : i64) : i64
    %286 = llvm.mlir.constant(0 : i64) : i64
    %287 = llvm.mlir.constant(69 : i64) : i64
    %288 = llvm.add %206, %287 : i64
    %289 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %290 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%289, %290) : (i64, i64) -> ()
    %291 = llvm.mlir.constant(844429225164800 : i64) : i64
    %292 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%291, %292) : (i64, i64) -> ()
    %293 = llvm.mlir.constant(844437815230464 : i64) : i64
    %294 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%293, %294) : (i64, i64) -> ()
    %295 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %296 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%295, %296) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%212, %284) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%210, %288) : (i64, i64) -> ()
    %297 = llvm.mlir.constant(768 : i64) : i64
    %298 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%297, %298) : (i64, i64) -> ()
    %299 = llvm.mlir.constant(1488 : i64) : i64
    %300 = llvm.add %208, %299 : i64
    %301 = llvm.mlir.constant(64 : i64) : i64
    %302 = llvm.add %210, %301 : i64
    %303 = llvm.mlir.constant(16 : i64) : i64
    %304 = llvm.add %212, %303 : i64
    %305 = llvm.mlir.constant(69 : i64) : i64
    %306 = llvm.add %206, %305 : i64
    %307 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %308 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%307, %308) : (i64, i64) -> ()
    %309 = llvm.mlir.constant(844429225164800 : i64) : i64
    %310 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%309, %310) : (i64, i64) -> ()
    %311 = llvm.mlir.constant(844437815230464 : i64) : i64
    %312 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%311, %312) : (i64, i64) -> ()
    %313 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %314 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%313, %314) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%304, %300) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%302, %306) : (i64, i64) -> ()
    %315 = llvm.mlir.constant(768 : i64) : i64
    %316 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%315, %316) : (i64, i64) -> ()
    %317 = llvm.mlir.constant(1504 : i64) : i64
    %318 = llvm.add %208, %317 : i64
    %319 = llvm.mlir.constant(128 : i64) : i64
    %320 = llvm.add %210, %319 : i64
    %321 = llvm.mlir.constant(32 : i64) : i64
    %322 = llvm.add %212, %321 : i64
    %323 = llvm.mlir.constant(69 : i64) : i64
    %324 = llvm.add %206, %323 : i64
    %325 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %326 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%325, %326) : (i64, i64) -> ()
    %327 = llvm.mlir.constant(844429225164800 : i64) : i64
    %328 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%327, %328) : (i64, i64) -> ()
    %329 = llvm.mlir.constant(844437815230464 : i64) : i64
    %330 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%329, %330) : (i64, i64) -> ()
    %331 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %332 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%331, %332) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%322, %318) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%320, %324) : (i64, i64) -> ()
    %333 = llvm.mlir.constant(768 : i64) : i64
    %334 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%333, %334) : (i64, i64) -> ()
    %335 = llvm.mlir.constant(1520 : i64) : i64
    %336 = llvm.add %208, %335 : i64
    %337 = llvm.mlir.constant(192 : i64) : i64
    %338 = llvm.add %210, %337 : i64
    %339 = llvm.mlir.constant(48 : i64) : i64
    %340 = llvm.add %212, %339 : i64
    %341 = llvm.mlir.constant(69 : i64) : i64
    %342 = llvm.add %206, %341 : i64
    %343 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %344 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%343, %344) : (i64, i64) -> ()
    %345 = llvm.mlir.constant(844429225164800 : i64) : i64
    %346 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%345, %346) : (i64, i64) -> ()
    %347 = llvm.mlir.constant(844437815230464 : i64) : i64
    %348 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%347, %348) : (i64, i64) -> ()
    %349 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %350 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%349, %350) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%340, %336) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%338, %342) : (i64, i64) -> ()
    %351 = llvm.mlir.constant(768 : i64) : i64
    %352 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%351, %352) : (i64, i64) -> ()
    %353 = llvm.mlir.constant(42240 : i64) : i64
    %354 = llvm.add %208, %353 : i64
    %355 = llvm.mlir.constant(0 : i64) : i64
    %356 = llvm.mlir.constant(0 : i64) : i64
    %357 = llvm.mlir.constant(2112 : i64) : i64
    %358 = llvm.add %206, %357 : i64
    %359 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %360 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%359, %360) : (i64, i64) -> ()
    %361 = llvm.mlir.constant(844429225164800 : i64) : i64
    %362 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%361, %362) : (i64, i64) -> ()
    %363 = llvm.mlir.constant(844437815230464 : i64) : i64
    %364 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%363, %364) : (i64, i64) -> ()
    %365 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %366 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%365, %366) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%212, %354) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%210, %358) : (i64, i64) -> ()
    %367 = llvm.mlir.constant(768 : i64) : i64
    %368 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%367, %368) : (i64, i64) -> ()
    %369 = llvm.mlir.constant(42256 : i64) : i64
    %370 = llvm.add %208, %369 : i64
    %371 = llvm.mlir.constant(64 : i64) : i64
    %372 = llvm.add %210, %371 : i64
    %373 = llvm.mlir.constant(16 : i64) : i64
    %374 = llvm.add %212, %373 : i64
    %375 = llvm.mlir.constant(2112 : i64) : i64
    %376 = llvm.add %206, %375 : i64
    %377 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %378 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%377, %378) : (i64, i64) -> ()
    %379 = llvm.mlir.constant(844429225164800 : i64) : i64
    %380 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%379, %380) : (i64, i64) -> ()
    %381 = llvm.mlir.constant(844437815230464 : i64) : i64
    %382 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%381, %382) : (i64, i64) -> ()
    %383 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %384 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%383, %384) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%374, %370) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%372, %376) : (i64, i64) -> ()
    %385 = llvm.mlir.constant(768 : i64) : i64
    %386 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%385, %386) : (i64, i64) -> ()
    %387 = llvm.mlir.constant(42272 : i64) : i64
    %388 = llvm.add %208, %387 : i64
    %389 = llvm.mlir.constant(128 : i64) : i64
    %390 = llvm.add %210, %389 : i64
    %391 = llvm.mlir.constant(32 : i64) : i64
    %392 = llvm.add %212, %391 : i64
    %393 = llvm.mlir.constant(2112 : i64) : i64
    %394 = llvm.add %206, %393 : i64
    %395 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %396 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%395, %396) : (i64, i64) -> ()
    %397 = llvm.mlir.constant(844429225164800 : i64) : i64
    %398 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%397, %398) : (i64, i64) -> ()
    %399 = llvm.mlir.constant(844437815230464 : i64) : i64
    %400 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%399, %400) : (i64, i64) -> ()
    %401 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %402 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%401, %402) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%392, %388) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%390, %394) : (i64, i64) -> ()
    %403 = llvm.mlir.constant(768 : i64) : i64
    %404 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%403, %404) : (i64, i64) -> ()
    %405 = llvm.mlir.constant(42288 : i64) : i64
    %406 = llvm.add %208, %405 : i64
    %407 = llvm.mlir.constant(192 : i64) : i64
    %408 = llvm.add %210, %407 : i64
    %409 = llvm.mlir.constant(48 : i64) : i64
    %410 = llvm.add %212, %409 : i64
    %411 = llvm.mlir.constant(2112 : i64) : i64
    %412 = llvm.add %206, %411 : i64
    %413 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %414 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%413, %414) : (i64, i64) -> ()
    %415 = llvm.mlir.constant(844429225164800 : i64) : i64
    %416 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%415, %416) : (i64, i64) -> ()
    %417 = llvm.mlir.constant(844437815230464 : i64) : i64
    %418 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%417, %418) : (i64, i64) -> ()
    %419 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %420 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%419, %420) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%410, %406) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%408, %412) : (i64, i64) -> ()
    %421 = llvm.mlir.constant(768 : i64) : i64
    %422 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%421, %422) : (i64, i64) -> ()
    %423 = llvm.mlir.constant(43712 : i64) : i64
    %424 = llvm.add %208, %423 : i64
    %425 = llvm.mlir.constant(0 : i64) : i64
    %426 = llvm.mlir.constant(0 : i64) : i64
    %427 = llvm.mlir.constant(2181 : i64) : i64
    %428 = llvm.add %206, %427 : i64
    %429 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %430 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%429, %430) : (i64, i64) -> ()
    %431 = llvm.mlir.constant(844429225164800 : i64) : i64
    %432 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%431, %432) : (i64, i64) -> ()
    %433 = llvm.mlir.constant(844437815230464 : i64) : i64
    %434 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%433, %434) : (i64, i64) -> ()
    %435 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %436 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%435, %436) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%212, %424) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%210, %428) : (i64, i64) -> ()
    %437 = llvm.mlir.constant(768 : i64) : i64
    %438 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%437, %438) : (i64, i64) -> ()
    %439 = llvm.mlir.constant(43728 : i64) : i64
    %440 = llvm.add %208, %439 : i64
    %441 = llvm.mlir.constant(64 : i64) : i64
    %442 = llvm.add %210, %441 : i64
    %443 = llvm.mlir.constant(16 : i64) : i64
    %444 = llvm.add %212, %443 : i64
    %445 = llvm.mlir.constant(2181 : i64) : i64
    %446 = llvm.add %206, %445 : i64
    %447 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %448 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%447, %448) : (i64, i64) -> ()
    %449 = llvm.mlir.constant(844429225164800 : i64) : i64
    %450 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%449, %450) : (i64, i64) -> ()
    %451 = llvm.mlir.constant(844437815230464 : i64) : i64
    %452 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%451, %452) : (i64, i64) -> ()
    %453 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %454 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%453, %454) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%444, %440) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%442, %446) : (i64, i64) -> ()
    %455 = llvm.mlir.constant(768 : i64) : i64
    %456 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%455, %456) : (i64, i64) -> ()
    %457 = llvm.mlir.constant(43744 : i64) : i64
    %458 = llvm.add %208, %457 : i64
    %459 = llvm.mlir.constant(128 : i64) : i64
    %460 = llvm.add %210, %459 : i64
    %461 = llvm.mlir.constant(32 : i64) : i64
    %462 = llvm.add %212, %461 : i64
    %463 = llvm.mlir.constant(2181 : i64) : i64
    %464 = llvm.add %206, %463 : i64
    %465 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %466 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%465, %466) : (i64, i64) -> ()
    %467 = llvm.mlir.constant(844429225164800 : i64) : i64
    %468 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%467, %468) : (i64, i64) -> ()
    %469 = llvm.mlir.constant(844437815230464 : i64) : i64
    %470 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%469, %470) : (i64, i64) -> ()
    %471 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %472 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%471, %472) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%462, %458) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%460, %464) : (i64, i64) -> ()
    %473 = llvm.mlir.constant(768 : i64) : i64
    %474 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%473, %474) : (i64, i64) -> ()
    %475 = llvm.mlir.constant(43760 : i64) : i64
    %476 = llvm.add %208, %475 : i64
    %477 = llvm.mlir.constant(192 : i64) : i64
    %478 = llvm.add %210, %477 : i64
    %479 = llvm.mlir.constant(48 : i64) : i64
    %480 = llvm.add %212, %479 : i64
    %481 = llvm.mlir.constant(2181 : i64) : i64
    %482 = llvm.add %206, %481 : i64
    %483 = llvm.mlir.constant(18014411396481025 : i64) : i64
    %484 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%483, %484) : (i64, i64) -> ()
    %485 = llvm.mlir.constant(844429225164800 : i64) : i64
    %486 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%485, %486) : (i64, i64) -> ()
    %487 = llvm.mlir.constant(844437815230464 : i64) : i64
    %488 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%487, %488) : (i64, i64) -> ()
    %489 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %490 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%489, %490) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%480, %476) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%478, %482) : (i64, i64) -> ()
    %491 = llvm.mlir.constant(768 : i64) : i64
    %492 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%491, %492) : (i64, i64) -> ()
    %493 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%493, %493) : (i64, i64) -> ()
    %494 = llvm.mlir.constant(0 : index) : i64
    %495 = llvm.mlir.constant(1 : index) : i64
    %496 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb25(%494 : i64)
  ^bb25(%497: i64):  // 2 preds: ^bb24, ^bb35
    %498 = llvm.icmp "slt" %497, %495 : i64
    llvm.cond_br %498, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    %499 = llvm.mlir.constant(0 : index) : i64
    %500 = llvm.mlir.constant(64 : index) : i64
    %501 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb27(%499 : i64)
  ^bb27(%502: i64):  // 2 preds: ^bb26, ^bb34
    %503 = llvm.icmp "slt" %502, %500 : i64
    llvm.cond_br %503, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    %504 = llvm.mlir.constant(0 : index) : i64
    %505 = llvm.mlir.constant(30 : index) : i64
    %506 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb29(%504 : i64)
  ^bb29(%507: i64):  // 2 preds: ^bb28, ^bb33
    %508 = llvm.icmp "slt" %507, %505 : i64
    llvm.cond_br %508, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    %509 = llvm.mlir.constant(0 : index) : i64
    %510 = llvm.mlir.constant(30 : index) : i64
    %511 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb31(%509 : i64)
  ^bb31(%512: i64):  // 2 preds: ^bb30, ^bb32
    %513 = llvm.icmp "slt" %512, %510 : i64
    llvm.cond_br %513, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %514 = llvm.mlir.constant(30 : index) : i64
    %515 = llvm.mul %497, %514 : i64
    %516 = llvm.mul %515, %514 : i64
    %517 = llvm.mul %507, %514 : i64
    %518 = llvm.add %516, %517 : i64
    %519 = llvm.add %518, %512 : i64
    %520 = llvm.extractvalue %107[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %521 = llvm.mlir.constant(64 : index) : i64
    %522 = llvm.mul %519, %521 : i64
    %523 = llvm.add %522, %502 : i64
    %524 = llvm.getelementptr %520[%523] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %525 = llvm.load %524 : !llvm.ptr -> f32
    %526 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %527 = llvm.mlir.constant(57600 : index) : i64
    %528 = llvm.mul %497, %527 : i64
    %529 = llvm.mlir.constant(900 : index) : i64
    %530 = llvm.mul %502, %529 : i64
    %531 = llvm.add %528, %530 : i64
    %532 = llvm.mlir.constant(30 : index) : i64
    %533 = llvm.mul %507, %532 : i64
    %534 = llvm.add %531, %533 : i64
    %535 = llvm.add %534, %512 : i64
    %536 = llvm.getelementptr %526[%535] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %525, %536 : f32, !llvm.ptr
    %537 = llvm.add %512, %511 : i64
    llvm.br ^bb31(%537 : i64)
  ^bb33:  // pred: ^bb31
    %538 = llvm.add %507, %506 : i64
    llvm.br ^bb29(%538 : i64)
  ^bb34:  // pred: ^bb29
    %539 = llvm.add %502, %501 : i64
    llvm.br ^bb27(%539 : i64)
  ^bb35:  // pred: ^bb27
    %540 = llvm.add %497, %496 : i64
    llvm.br ^bb25(%540 : i64)
  ^bb36:  // pred: ^bb25
    %541 = llvm.extractvalue %60[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%541) : (!llvm.ptr) -> ()
    %542 = llvm.extractvalue %77[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%542) : (!llvm.ptr) -> ()
    %543 = llvm.extractvalue %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%543) : (!llvm.ptr) -> ()
    %544 = llvm.extractvalue %90[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%544) : (!llvm.ptr) -> ()
    llvm.return
  }
}


