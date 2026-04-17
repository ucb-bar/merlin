// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  func.func @mini_cnn_block(%arg0: memref<1x3x32x32xf32>, %arg1: memref<16x3x3x3xf32>, %arg2: memref<32x16x3x3xf32>, %arg3: memref<1x32x26x26xf32>) {
    %alloc = memref.alloc() : memref<1x16x30x30xf32>
    %alloc_0 = memref.alloc() : memref<1x32x26x26xf32>
    %alloc_1 = memref.alloc() : memref<1x32x32x3xf32>
    %alloc_2 = memref.alloc() : memref<27x16xf32>
    %alloc_3 = memref.alloc() : memref<16xi32>
    %alloc_4 = memref.alloc() : memref<900x16xf32>
    %c30_i64 = arith.constant 30 : i64
    %c3 = arith.constant 3 : index
    %c3_5 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_6 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %c1 step %c1_6 {
      %c0_27 = arith.constant 0 : index
      %c3_28 = arith.constant 3 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c3_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c32_31 = arith.constant 32 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c32_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c32_34 = arith.constant 32 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c32_34 step %c1_35 {
            %0 = memref.load %arg0[%arg4, %arg5, %arg6, %arg7] : memref<1x3x32x32xf32>
            memref.store %0, %alloc_1[%arg4, %arg6, %arg7, %arg5] : memref<1x32x32x3xf32>
          }
        }
      }
    }
    %c0_7 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1_8 = arith.constant 1 : index
    scf.for %arg4 = %c0_7 to %c16 step %c1_8 {
      %c0_27 = arith.constant 0 : index
      %c3_28 = arith.constant 3 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c3_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c3_31 = arith.constant 3 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c3_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c3_34 = arith.constant 3 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c3_34 step %c1_35 {
            %0 = arith.muli %arg6, %c3 : index
            %1 = arith.muli %0, %c3_5 : index
            %2 = arith.muli %arg7, %c3_5 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %arg1[%arg4, %arg5, %arg6, %arg7] : memref<16x3x3x3xf32>
            memref.store %5, %alloc_2[%4, %arg4] : memref<27x16xf32>
          }
        }
      }
    }
    %c3_i64 = arith.constant 3 : i64
    gemmini.tile_conv %alloc_1 %alloc_2 %alloc_3 %alloc_4 %c30_i64 %c30_i64 %c3_i64 : memref<1x32x32x3xf32> memref<27x16xf32> memref<16xi32> memref<900x16xf32> i64 i64 i64
    %c0_9 = arith.constant 0 : index
    %c1_10 = arith.constant 1 : index
    %c1_11 = arith.constant 1 : index
    scf.for %arg4 = %c0_9 to %c1_10 step %c1_11 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c30 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c30_33 = arith.constant 30 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c30_33 step %c1_34 {
            %c30_35 = arith.constant 30 : index
            %0 = arith.muli %arg4, %c30_35 : index
            %1 = arith.muli %0, %c30_35 : index
            %2 = arith.muli %arg6, %c30_35 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg7 : index
            %5 = memref.load %alloc_4[%4, %arg5] : memref<900x16xf32>
            memref.store %5, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x16x30x30xf32>
          }
        }
      }
    }
    memref.dealloc %alloc_1 : memref<1x32x32x3xf32>
    memref.dealloc %alloc_2 : memref<27x16xf32>
    memref.dealloc %alloc_4 : memref<900x16xf32>
    memref.dealloc %alloc_3 : memref<16xi32>
    %alloc_12 = memref.alloc() : memref<1x30x30x16xf32>
    %alloc_13 = memref.alloc() : memref<144x32xf32>
    %alloc_14 = memref.alloc() : memref<32xi32>
    %alloc_15 = memref.alloc() : memref<676x32xf32>
    %c26_i64 = arith.constant 26 : i64
    %c3_16 = arith.constant 3 : index
    %c16_17 = arith.constant 16 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    %c1_20 = arith.constant 1 : index
    scf.for %arg4 = %c0_18 to %c1_19 step %c1_20 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c30 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c30_33 = arith.constant 30 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c30_33 step %c1_34 {
            %0 = memref.load %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x16x30x30xf32>
            memref.store %0, %alloc_12[%arg4, %arg6, %arg7, %arg5] : memref<1x30x30x16xf32>
          }
        }
      }
    }
    %c0_21 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1_22 = arith.constant 1 : index
    scf.for %arg4 = %c0_21 to %c32 step %c1_22 {
      %c0_27 = arith.constant 0 : index
      %c16_28 = arith.constant 16 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c16_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c3_31 = arith.constant 3 : index
        %c1_32 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c3_31 step %c1_32 {
          %c0_33 = arith.constant 0 : index
          %c3_34 = arith.constant 3 : index
          %c1_35 = arith.constant 1 : index
          scf.for %arg7 = %c0_33 to %c3_34 step %c1_35 {
            %0 = arith.muli %arg6, %c3_16 : index
            %1 = arith.muli %0, %c16_17 : index
            %2 = arith.muli %arg7, %c16_17 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %arg2[%arg4, %arg5, %arg6, %arg7] : memref<32x16x3x3xf32>
            memref.store %5, %alloc_13[%4, %arg4] : memref<144x32xf32>
          }
        }
      }
    }
    %c3_i64_23 = arith.constant 3 : i64
    gemmini.tile_conv %alloc_12 %alloc_13 %alloc_14 %alloc_15 %c26_i64 %c26_i64 %c3_i64_23 : memref<1x30x30x16xf32> memref<144x32xf32> memref<32xi32> memref<676x32xf32> i64 i64 i64
    %c0_24 = arith.constant 0 : index
    %c1_25 = arith.constant 1 : index
    %c1_26 = arith.constant 1 : index
    scf.for %arg4 = %c0_24 to %c1_25 step %c1_26 {
      %c0_27 = arith.constant 0 : index
      %c32_28 = arith.constant 32 : index
      %c1_29 = arith.constant 1 : index
      scf.for %arg5 = %c0_27 to %c32_28 step %c1_29 {
        %c0_30 = arith.constant 0 : index
        %c26 = arith.constant 26 : index
        %c1_31 = arith.constant 1 : index
        scf.for %arg6 = %c0_30 to %c26 step %c1_31 {
          %c0_32 = arith.constant 0 : index
          %c26_33 = arith.constant 26 : index
          %c1_34 = arith.constant 1 : index
          scf.for %arg7 = %c0_32 to %c26_33 step %c1_34 {
            %c26_35 = arith.constant 26 : index
            %0 = arith.muli %arg4, %c26_35 : index
            %1 = arith.muli %0, %c26_35 : index
            %2 = arith.muli %arg6, %c26_35 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg7 : index
            %5 = memref.load %alloc_15[%4, %arg5] : memref<676x32xf32>
            memref.store %5, %alloc_0[%arg4, %arg5, %arg6, %arg7] : memref<1x32x26x26xf32>
          }
        }
      }
    }
    memref.dealloc %alloc_12 : memref<1x30x30x16xf32>
    memref.dealloc %alloc_13 : memref<144x32xf32>
    memref.dealloc %alloc_15 : memref<676x32xf32>
    memref.dealloc %alloc_14 : memref<32xi32>
    linalg.copy ins(%alloc_0 : memref<1x32x26x26xf32>) outs(%arg3 : memref<1x32x26x26xf32>)
    memref.dealloc %alloc : memref<1x16x30x30xf32>
    memref.dealloc %alloc_0 : memref<1x32x26x26xf32>
    return
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @mini_cnn_block(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: !llvm.ptr, %arg34: !llvm.ptr, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg33, %0[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %2 = llvm.insertvalue %arg34, %1[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %3 = llvm.insertvalue %arg35, %2[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %4 = llvm.insertvalue %arg36, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %5 = llvm.insertvalue %arg40, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %6 = llvm.insertvalue %arg37, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %7 = llvm.insertvalue %arg41, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %8 = llvm.insertvalue %arg38, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %9 = llvm.insertvalue %arg42, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %10 = llvm.insertvalue %arg39, %9[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %11 = llvm.insertvalue %arg43, %10[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %12 = builtin.unrealized_conversion_cast %11 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x32x26x26xf32>
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %14 = llvm.insertvalue %arg22, %13[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg23, %14[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg24, %15[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg25, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg29, %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg26, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg30, %19[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg27, %20[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg31, %21[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg28, %22[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %arg32, %23[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %26 = llvm.insertvalue %arg11, %25[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %arg12, %26[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %arg13, %27[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %arg14, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %arg18, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %arg15, %30[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %arg19, %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %arg16, %32[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %arg20, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %arg17, %34[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.insertvalue %arg21, %35[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %38 = llvm.insertvalue %arg0, %37[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %39 = llvm.insertvalue %arg1, %38[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %40 = llvm.insertvalue %arg2, %39[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %41 = llvm.insertvalue %arg3, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %42 = llvm.insertvalue %arg7, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %43 = llvm.insertvalue %arg4, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %44 = llvm.insertvalue %arg8, %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %45 = llvm.insertvalue %arg5, %44[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %arg9, %45[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.insertvalue %arg6, %46[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %48 = llvm.insertvalue %arg10, %47[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.mlir.constant(16 : index) : i64
    %51 = llvm.mlir.constant(30 : index) : i64
    %52 = llvm.mlir.constant(30 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.mlir.constant(900 : index) : i64
    %55 = llvm.mlir.constant(14400 : index) : i64
    %56 = llvm.mlir.constant(14400 : index) : i64
    %57 = llvm.mlir.zero : !llvm.ptr
    %58 = llvm.getelementptr %57[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.call @malloc(%59) : (i64) -> !llvm.ptr
    %61 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.insertvalue %60, %62[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.mlir.constant(0 : index) : i64
    %65 = llvm.insertvalue %64, %63[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %66 = llvm.insertvalue %49, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %67 = llvm.insertvalue %50, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %68 = llvm.insertvalue %51, %67[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %69 = llvm.insertvalue %52, %68[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %70 = llvm.insertvalue %55, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.insertvalue %54, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.insertvalue %52, %71[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %73 = llvm.insertvalue %53, %72[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.mlir.constant(32 : index) : i64
    %76 = llvm.mlir.constant(26 : index) : i64
    %77 = llvm.mlir.constant(26 : index) : i64
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.mlir.constant(676 : index) : i64
    %80 = llvm.mlir.constant(21632 : index) : i64
    %81 = llvm.mlir.constant(21632 : index) : i64
    %82 = llvm.mlir.zero : !llvm.ptr
    %83 = llvm.getelementptr %82[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.ptrtoint %83 : !llvm.ptr to i64
    %85 = llvm.call @malloc(%84) : (i64) -> !llvm.ptr
    %86 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %87 = llvm.insertvalue %85, %86[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %88 = llvm.insertvalue %85, %87[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %89 = llvm.mlir.constant(0 : index) : i64
    %90 = llvm.insertvalue %89, %88[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %91 = llvm.insertvalue %74, %90[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %92 = llvm.insertvalue %75, %91[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %93 = llvm.insertvalue %76, %92[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.insertvalue %77, %93[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %95 = llvm.insertvalue %80, %94[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.insertvalue %79, %95[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.insertvalue %77, %96[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %98 = llvm.insertvalue %78, %97[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %99 = builtin.unrealized_conversion_cast %98 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> to memref<1x32x26x26xf32>
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.mlir.constant(32 : index) : i64
    %102 = llvm.mlir.constant(32 : index) : i64
    %103 = llvm.mlir.constant(3 : index) : i64
    %104 = llvm.mlir.constant(1 : index) : i64
    %105 = llvm.mlir.constant(96 : index) : i64
    %106 = llvm.mlir.constant(3072 : index) : i64
    %107 = llvm.mlir.constant(3072 : index) : i64
    %108 = llvm.mlir.zero : !llvm.ptr
    %109 = llvm.getelementptr %108[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %110 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %111 = llvm.call @malloc(%110) : (i64) -> !llvm.ptr
    %112 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %113 = llvm.insertvalue %111, %112[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %114 = llvm.insertvalue %111, %113[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %115 = llvm.mlir.constant(0 : index) : i64
    %116 = llvm.insertvalue %115, %114[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %117 = llvm.insertvalue %100, %116[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %118 = llvm.insertvalue %101, %117[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %119 = llvm.insertvalue %102, %118[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %120 = llvm.insertvalue %103, %119[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %121 = llvm.insertvalue %106, %120[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %122 = llvm.insertvalue %105, %121[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %123 = llvm.insertvalue %103, %122[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %124 = llvm.insertvalue %104, %123[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %125 = llvm.mlir.constant(27 : index) : i64
    %126 = llvm.mlir.constant(16 : index) : i64
    %127 = llvm.mlir.constant(1 : index) : i64
    %128 = llvm.mlir.constant(432 : index) : i64
    %129 = llvm.mlir.zero : !llvm.ptr
    %130 = llvm.getelementptr %129[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %131 = llvm.ptrtoint %130 : !llvm.ptr to i64
    %132 = llvm.call @malloc(%131) : (i64) -> !llvm.ptr
    %133 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %134 = llvm.insertvalue %132, %133[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %132, %134[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(0 : index) : i64
    %137 = llvm.insertvalue %136, %135[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.insertvalue %125, %137[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.insertvalue %126, %138[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.insertvalue %126, %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.insertvalue %127, %140[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(16 : index) : i64
    %143 = llvm.mlir.constant(1 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%142] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.call @malloc(%146) : (i64) -> !llvm.ptr
    %148 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %149 = llvm.insertvalue %147, %148[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %150 = llvm.insertvalue %147, %149[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %151 = llvm.mlir.constant(0 : index) : i64
    %152 = llvm.insertvalue %151, %150[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %153 = llvm.insertvalue %142, %152[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %154 = llvm.insertvalue %143, %153[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %155 = llvm.mlir.constant(900 : index) : i64
    %156 = llvm.mlir.constant(16 : index) : i64
    %157 = llvm.mlir.constant(1 : index) : i64
    %158 = llvm.mlir.constant(14400 : index) : i64
    %159 = llvm.mlir.zero : !llvm.ptr
    %160 = llvm.getelementptr %159[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %161 = llvm.ptrtoint %160 : !llvm.ptr to i64
    %162 = llvm.call @malloc(%161) : (i64) -> !llvm.ptr
    %163 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %164 = llvm.insertvalue %162, %163[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %162, %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.mlir.constant(0 : index) : i64
    %167 = llvm.insertvalue %166, %165[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.insertvalue %155, %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.insertvalue %156, %168[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.insertvalue %156, %169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.insertvalue %157, %170[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.mlir.constant(30 : i64) : i64
    %173 = llvm.mlir.constant(3 : index) : i64
    %174 = llvm.mlir.constant(3 : index) : i64
    %175 = llvm.mlir.constant(0 : index) : i64
    %176 = llvm.mlir.constant(1 : index) : i64
    %177 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%175 : i64)
  ^bb1(%178: i64):  // 2 preds: ^bb0, ^bb11
    %179 = llvm.icmp "slt" %178, %176 : i64
    llvm.cond_br %179, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %180 = llvm.mlir.constant(0 : index) : i64
    %181 = llvm.mlir.constant(3 : index) : i64
    %182 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%180 : i64)
  ^bb3(%183: i64):  // 2 preds: ^bb2, ^bb10
    %184 = llvm.icmp "slt" %183, %181 : i64
    llvm.cond_br %184, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %185 = llvm.mlir.constant(0 : index) : i64
    %186 = llvm.mlir.constant(32 : index) : i64
    %187 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%185 : i64)
  ^bb5(%188: i64):  // 2 preds: ^bb4, ^bb9
    %189 = llvm.icmp "slt" %188, %186 : i64
    llvm.cond_br %189, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %190 = llvm.mlir.constant(0 : index) : i64
    %191 = llvm.mlir.constant(32 : index) : i64
    %192 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%190 : i64)
  ^bb7(%193: i64):  // 2 preds: ^bb6, ^bb8
    %194 = llvm.icmp "slt" %193, %191 : i64
    llvm.cond_br %194, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %195 = llvm.extractvalue %48[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %196 = llvm.mlir.constant(3072 : index) : i64
    %197 = llvm.mul %178, %196 : i64
    %198 = llvm.mlir.constant(1024 : index) : i64
    %199 = llvm.mul %183, %198 : i64
    %200 = llvm.add %197, %199 : i64
    %201 = llvm.mlir.constant(32 : index) : i64
    %202 = llvm.mul %188, %201 : i64
    %203 = llvm.add %200, %202 : i64
    %204 = llvm.add %203, %193 : i64
    %205 = llvm.getelementptr %195[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %206 = llvm.load %205 : !llvm.ptr -> f32
    %207 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %208 = llvm.mlir.constant(3072 : index) : i64
    %209 = llvm.mul %178, %208 : i64
    %210 = llvm.mlir.constant(96 : index) : i64
    %211 = llvm.mul %188, %210 : i64
    %212 = llvm.add %209, %211 : i64
    %213 = llvm.mlir.constant(3 : index) : i64
    %214 = llvm.mul %193, %213 : i64
    %215 = llvm.add %212, %214 : i64
    %216 = llvm.add %215, %183 : i64
    %217 = llvm.getelementptr %207[%216] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %206, %217 : f32, !llvm.ptr
    %218 = llvm.add %193, %192 : i64
    llvm.br ^bb7(%218 : i64)
  ^bb9:  // pred: ^bb7
    %219 = llvm.add %188, %187 : i64
    llvm.br ^bb5(%219 : i64)
  ^bb10:  // pred: ^bb5
    %220 = llvm.add %183, %182 : i64
    llvm.br ^bb3(%220 : i64)
  ^bb11:  // pred: ^bb3
    %221 = llvm.add %178, %177 : i64
    llvm.br ^bb1(%221 : i64)
  ^bb12:  // pred: ^bb1
    %222 = llvm.mlir.constant(0 : index) : i64
    %223 = llvm.mlir.constant(16 : index) : i64
    %224 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%222 : i64)
  ^bb13(%225: i64):  // 2 preds: ^bb12, ^bb23
    %226 = llvm.icmp "slt" %225, %223 : i64
    llvm.cond_br %226, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    %227 = llvm.mlir.constant(0 : index) : i64
    %228 = llvm.mlir.constant(3 : index) : i64
    %229 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%227 : i64)
  ^bb15(%230: i64):  // 2 preds: ^bb14, ^bb22
    %231 = llvm.icmp "slt" %230, %228 : i64
    llvm.cond_br %231, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    %232 = llvm.mlir.constant(0 : index) : i64
    %233 = llvm.mlir.constant(3 : index) : i64
    %234 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb17(%232 : i64)
  ^bb17(%235: i64):  // 2 preds: ^bb16, ^bb21
    %236 = llvm.icmp "slt" %235, %233 : i64
    llvm.cond_br %236, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %237 = llvm.mlir.constant(0 : index) : i64
    %238 = llvm.mlir.constant(3 : index) : i64
    %239 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%237 : i64)
  ^bb19(%240: i64):  // 2 preds: ^bb18, ^bb20
    %241 = llvm.icmp "slt" %240, %238 : i64
    llvm.cond_br %241, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %242 = llvm.mul %235, %173 : i64
    %243 = llvm.mul %242, %174 : i64
    %244 = llvm.mul %240, %174 : i64
    %245 = llvm.add %243, %244 : i64
    %246 = llvm.add %245, %230 : i64
    %247 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %248 = llvm.mlir.constant(27 : index) : i64
    %249 = llvm.mul %225, %248 : i64
    %250 = llvm.mlir.constant(9 : index) : i64
    %251 = llvm.mul %230, %250 : i64
    %252 = llvm.add %249, %251 : i64
    %253 = llvm.mlir.constant(3 : index) : i64
    %254 = llvm.mul %235, %253 : i64
    %255 = llvm.add %252, %254 : i64
    %256 = llvm.add %255, %240 : i64
    %257 = llvm.getelementptr %247[%256] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %258 = llvm.load %257 : !llvm.ptr -> f32
    %259 = llvm.extractvalue %141[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %260 = llvm.mlir.constant(16 : index) : i64
    %261 = llvm.mul %246, %260 : i64
    %262 = llvm.add %261, %225 : i64
    %263 = llvm.getelementptr %259[%262] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %258, %263 : f32, !llvm.ptr
    %264 = llvm.add %240, %239 : i64
    llvm.br ^bb19(%264 : i64)
  ^bb21:  // pred: ^bb19
    %265 = llvm.add %235, %234 : i64
    llvm.br ^bb17(%265 : i64)
  ^bb22:  // pred: ^bb17
    %266 = llvm.add %230, %229 : i64
    llvm.br ^bb15(%266 : i64)
  ^bb23:  // pred: ^bb15
    %267 = llvm.add %225, %224 : i64
    llvm.br ^bb13(%267 : i64)
  ^bb24:  // pred: ^bb13
    %268 = llvm.mlir.constant(3 : i64) : i64
    %269 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %270 = llvm.ptrtoint %269 : !llvm.ptr to i64
    %271 = llvm.extractvalue %171[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %272 = llvm.ptrtoint %271 : !llvm.ptr to i64
    %273 = llvm.extractvalue %154[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %274 = llvm.ptrtoint %273 : !llvm.ptr to i64
    %275 = llvm.extractvalue %141[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.ptrtoint %275 : !llvm.ptr to i64
    %277 = llvm.mlir.constant(16 : i64) : i64
    %278 = llvm.mlir.constant(2 : i64) : i64
    %279 = llvm.mlir.constant(4575657221408423952 : i64) : i64
    "gemmini.intr.config_st"(%278, %279) : (i64, i64) -> ()
    %280 = llvm.mlir.constant(65540 : i64) : i64
    %281 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%280, %281) : (i64, i64) -> ()
    %282 = llvm.mlir.constant(0 : i64) : i64
    %283 = llvm.mlir.constant(0 : i64) : i64
    %284 = llvm.mlir.constant(0 : i64) : i64
    %285 = llvm.mlir.constant(0 : i64) : i64
    %286 = llvm.mlir.constant(4503612514369537 : i64) : i64
    %287 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%286, %287) : (i64, i64) -> ()
    %288 = llvm.mlir.constant(844429225164800 : i64) : i64
    %289 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%288, %289) : (i64, i64) -> ()
    %290 = llvm.mlir.constant(844437815230464 : i64) : i64
    %291 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%290, %291) : (i64, i64) -> ()
    %292 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %293 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%292, %293) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%276, %272) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%274, %270) : (i64, i64) -> ()
    %294 = llvm.mlir.constant(768 : i64) : i64
    %295 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%294, %295) : (i64, i64) -> ()
    %296 = llvm.mlir.constant(368 : i64) : i64
    %297 = llvm.add %272, %296 : i64
    %298 = llvm.mlir.constant(0 : i64) : i64
    %299 = llvm.mlir.constant(0 : i64) : i64
    %300 = llvm.mlir.constant(69 : i64) : i64
    %301 = llvm.add %270, %300 : i64
    %302 = llvm.mlir.constant(4503612514369537 : i64) : i64
    %303 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%302, %303) : (i64, i64) -> ()
    %304 = llvm.mlir.constant(844429225164800 : i64) : i64
    %305 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%304, %305) : (i64, i64) -> ()
    %306 = llvm.mlir.constant(844437815230464 : i64) : i64
    %307 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%306, %307) : (i64, i64) -> ()
    %308 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %309 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%308, %309) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%276, %297) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%274, %301) : (i64, i64) -> ()
    %310 = llvm.mlir.constant(768 : i64) : i64
    %311 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%310, %311) : (i64, i64) -> ()
    %312 = llvm.mlir.constant(10560 : i64) : i64
    %313 = llvm.add %272, %312 : i64
    %314 = llvm.mlir.constant(0 : i64) : i64
    %315 = llvm.mlir.constant(0 : i64) : i64
    %316 = llvm.mlir.constant(2112 : i64) : i64
    %317 = llvm.add %270, %316 : i64
    %318 = llvm.mlir.constant(4503612514369537 : i64) : i64
    %319 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%318, %319) : (i64, i64) -> ()
    %320 = llvm.mlir.constant(844429225164800 : i64) : i64
    %321 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%320, %321) : (i64, i64) -> ()
    %322 = llvm.mlir.constant(844437815230464 : i64) : i64
    %323 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%322, %323) : (i64, i64) -> ()
    %324 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %325 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%324, %325) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%276, %313) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%274, %317) : (i64, i64) -> ()
    %326 = llvm.mlir.constant(768 : i64) : i64
    %327 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%326, %327) : (i64, i64) -> ()
    %328 = llvm.mlir.constant(10928 : i64) : i64
    %329 = llvm.add %272, %328 : i64
    %330 = llvm.mlir.constant(0 : i64) : i64
    %331 = llvm.mlir.constant(0 : i64) : i64
    %332 = llvm.mlir.constant(2181 : i64) : i64
    %333 = llvm.add %270, %332 : i64
    %334 = llvm.mlir.constant(4503612514369537 : i64) : i64
    %335 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%334, %335) : (i64, i64) -> ()
    %336 = llvm.mlir.constant(844429225164800 : i64) : i64
    %337 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%336, %337) : (i64, i64) -> ()
    %338 = llvm.mlir.constant(844437815230464 : i64) : i64
    %339 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%338, %339) : (i64, i64) -> ()
    %340 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %341 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%340, %341) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%276, %329) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%274, %333) : (i64, i64) -> ()
    %342 = llvm.mlir.constant(768 : i64) : i64
    %343 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%342, %343) : (i64, i64) -> ()
    %344 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%344, %344) : (i64, i64) -> ()
    %345 = llvm.mlir.constant(0 : index) : i64
    %346 = llvm.mlir.constant(1 : index) : i64
    %347 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb25(%345 : i64)
  ^bb25(%348: i64):  // 2 preds: ^bb24, ^bb35
    %349 = llvm.icmp "slt" %348, %346 : i64
    llvm.cond_br %349, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    %350 = llvm.mlir.constant(0 : index) : i64
    %351 = llvm.mlir.constant(16 : index) : i64
    %352 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb27(%350 : i64)
  ^bb27(%353: i64):  // 2 preds: ^bb26, ^bb34
    %354 = llvm.icmp "slt" %353, %351 : i64
    llvm.cond_br %354, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    %355 = llvm.mlir.constant(0 : index) : i64
    %356 = llvm.mlir.constant(30 : index) : i64
    %357 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb29(%355 : i64)
  ^bb29(%358: i64):  // 2 preds: ^bb28, ^bb33
    %359 = llvm.icmp "slt" %358, %356 : i64
    llvm.cond_br %359, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    %360 = llvm.mlir.constant(0 : index) : i64
    %361 = llvm.mlir.constant(30 : index) : i64
    %362 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb31(%360 : i64)
  ^bb31(%363: i64):  // 2 preds: ^bb30, ^bb32
    %364 = llvm.icmp "slt" %363, %361 : i64
    llvm.cond_br %364, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %365 = llvm.mlir.constant(30 : index) : i64
    %366 = llvm.mul %348, %365 : i64
    %367 = llvm.mul %366, %365 : i64
    %368 = llvm.mul %358, %365 : i64
    %369 = llvm.add %367, %368 : i64
    %370 = llvm.add %369, %363 : i64
    %371 = llvm.extractvalue %171[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %372 = llvm.mlir.constant(16 : index) : i64
    %373 = llvm.mul %370, %372 : i64
    %374 = llvm.add %373, %353 : i64
    %375 = llvm.getelementptr %371[%374] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %376 = llvm.load %375 : !llvm.ptr -> f32
    %377 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %378 = llvm.mlir.constant(14400 : index) : i64
    %379 = llvm.mul %348, %378 : i64
    %380 = llvm.mlir.constant(900 : index) : i64
    %381 = llvm.mul %353, %380 : i64
    %382 = llvm.add %379, %381 : i64
    %383 = llvm.mlir.constant(30 : index) : i64
    %384 = llvm.mul %358, %383 : i64
    %385 = llvm.add %382, %384 : i64
    %386 = llvm.add %385, %363 : i64
    %387 = llvm.getelementptr %377[%386] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %376, %387 : f32, !llvm.ptr
    %388 = llvm.add %363, %362 : i64
    llvm.br ^bb31(%388 : i64)
  ^bb33:  // pred: ^bb31
    %389 = llvm.add %358, %357 : i64
    llvm.br ^bb29(%389 : i64)
  ^bb34:  // pred: ^bb29
    %390 = llvm.add %353, %352 : i64
    llvm.br ^bb27(%390 : i64)
  ^bb35:  // pred: ^bb27
    %391 = llvm.add %348, %347 : i64
    llvm.br ^bb25(%391 : i64)
  ^bb36:  // pred: ^bb25
    %392 = llvm.extractvalue %124[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%392) : (!llvm.ptr) -> ()
    %393 = llvm.extractvalue %141[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%393) : (!llvm.ptr) -> ()
    %394 = llvm.extractvalue %171[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%394) : (!llvm.ptr) -> ()
    %395 = llvm.extractvalue %154[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%395) : (!llvm.ptr) -> ()
    %396 = llvm.mlir.constant(1 : index) : i64
    %397 = llvm.mlir.constant(30 : index) : i64
    %398 = llvm.mlir.constant(30 : index) : i64
    %399 = llvm.mlir.constant(16 : index) : i64
    %400 = llvm.mlir.constant(1 : index) : i64
    %401 = llvm.mlir.constant(480 : index) : i64
    %402 = llvm.mlir.constant(14400 : index) : i64
    %403 = llvm.mlir.constant(14400 : index) : i64
    %404 = llvm.mlir.zero : !llvm.ptr
    %405 = llvm.getelementptr %404[%403] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %406 = llvm.ptrtoint %405 : !llvm.ptr to i64
    %407 = llvm.call @malloc(%406) : (i64) -> !llvm.ptr
    %408 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %409 = llvm.insertvalue %407, %408[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %410 = llvm.insertvalue %407, %409[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %411 = llvm.mlir.constant(0 : index) : i64
    %412 = llvm.insertvalue %411, %410[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %413 = llvm.insertvalue %396, %412[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %414 = llvm.insertvalue %397, %413[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %415 = llvm.insertvalue %398, %414[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %416 = llvm.insertvalue %399, %415[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %417 = llvm.insertvalue %402, %416[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %418 = llvm.insertvalue %401, %417[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %419 = llvm.insertvalue %399, %418[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %420 = llvm.insertvalue %400, %419[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %421 = llvm.mlir.constant(144 : index) : i64
    %422 = llvm.mlir.constant(32 : index) : i64
    %423 = llvm.mlir.constant(1 : index) : i64
    %424 = llvm.mlir.constant(4608 : index) : i64
    %425 = llvm.mlir.zero : !llvm.ptr
    %426 = llvm.getelementptr %425[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %427 = llvm.ptrtoint %426 : !llvm.ptr to i64
    %428 = llvm.call @malloc(%427) : (i64) -> !llvm.ptr
    %429 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %430 = llvm.insertvalue %428, %429[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %431 = llvm.insertvalue %428, %430[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %432 = llvm.mlir.constant(0 : index) : i64
    %433 = llvm.insertvalue %432, %431[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %434 = llvm.insertvalue %421, %433[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %435 = llvm.insertvalue %422, %434[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %436 = llvm.insertvalue %422, %435[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %437 = llvm.insertvalue %423, %436[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %438 = llvm.mlir.constant(32 : index) : i64
    %439 = llvm.mlir.constant(1 : index) : i64
    %440 = llvm.mlir.zero : !llvm.ptr
    %441 = llvm.getelementptr %440[%438] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %442 = llvm.ptrtoint %441 : !llvm.ptr to i64
    %443 = llvm.call @malloc(%442) : (i64) -> !llvm.ptr
    %444 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %445 = llvm.insertvalue %443, %444[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %446 = llvm.insertvalue %443, %445[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %447 = llvm.mlir.constant(0 : index) : i64
    %448 = llvm.insertvalue %447, %446[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %449 = llvm.insertvalue %438, %448[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %450 = llvm.insertvalue %439, %449[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %451 = llvm.mlir.constant(676 : index) : i64
    %452 = llvm.mlir.constant(32 : index) : i64
    %453 = llvm.mlir.constant(1 : index) : i64
    %454 = llvm.mlir.constant(21632 : index) : i64
    %455 = llvm.mlir.zero : !llvm.ptr
    %456 = llvm.getelementptr %455[%454] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %457 = llvm.ptrtoint %456 : !llvm.ptr to i64
    %458 = llvm.call @malloc(%457) : (i64) -> !llvm.ptr
    %459 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %460 = llvm.insertvalue %458, %459[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %461 = llvm.insertvalue %458, %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %462 = llvm.mlir.constant(0 : index) : i64
    %463 = llvm.insertvalue %462, %461[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %464 = llvm.insertvalue %451, %463[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %465 = llvm.insertvalue %452, %464[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %466 = llvm.insertvalue %452, %465[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %467 = llvm.insertvalue %453, %466[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %468 = llvm.mlir.constant(26 : i64) : i64
    %469 = llvm.mlir.constant(3 : index) : i64
    %470 = llvm.mlir.constant(16 : index) : i64
    %471 = llvm.mlir.constant(0 : index) : i64
    %472 = llvm.mlir.constant(1 : index) : i64
    %473 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb37(%471 : i64)
  ^bb37(%474: i64):  // 2 preds: ^bb36, ^bb47
    %475 = llvm.icmp "slt" %474, %472 : i64
    llvm.cond_br %475, ^bb38, ^bb48
  ^bb38:  // pred: ^bb37
    %476 = llvm.mlir.constant(0 : index) : i64
    %477 = llvm.mlir.constant(16 : index) : i64
    %478 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb39(%476 : i64)
  ^bb39(%479: i64):  // 2 preds: ^bb38, ^bb46
    %480 = llvm.icmp "slt" %479, %477 : i64
    llvm.cond_br %480, ^bb40, ^bb47
  ^bb40:  // pred: ^bb39
    %481 = llvm.mlir.constant(0 : index) : i64
    %482 = llvm.mlir.constant(30 : index) : i64
    %483 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb41(%481 : i64)
  ^bb41(%484: i64):  // 2 preds: ^bb40, ^bb45
    %485 = llvm.icmp "slt" %484, %482 : i64
    llvm.cond_br %485, ^bb42, ^bb46
  ^bb42:  // pred: ^bb41
    %486 = llvm.mlir.constant(0 : index) : i64
    %487 = llvm.mlir.constant(30 : index) : i64
    %488 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb43(%486 : i64)
  ^bb43(%489: i64):  // 2 preds: ^bb42, ^bb44
    %490 = llvm.icmp "slt" %489, %487 : i64
    llvm.cond_br %490, ^bb44, ^bb45
  ^bb44:  // pred: ^bb43
    %491 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %492 = llvm.mlir.constant(14400 : index) : i64
    %493 = llvm.mul %474, %492 : i64
    %494 = llvm.mlir.constant(900 : index) : i64
    %495 = llvm.mul %479, %494 : i64
    %496 = llvm.add %493, %495 : i64
    %497 = llvm.mlir.constant(30 : index) : i64
    %498 = llvm.mul %484, %497 : i64
    %499 = llvm.add %496, %498 : i64
    %500 = llvm.add %499, %489 : i64
    %501 = llvm.getelementptr %491[%500] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %502 = llvm.load %501 : !llvm.ptr -> f32
    %503 = llvm.extractvalue %420[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %504 = llvm.mlir.constant(14400 : index) : i64
    %505 = llvm.mul %474, %504 : i64
    %506 = llvm.mlir.constant(480 : index) : i64
    %507 = llvm.mul %484, %506 : i64
    %508 = llvm.add %505, %507 : i64
    %509 = llvm.mlir.constant(16 : index) : i64
    %510 = llvm.mul %489, %509 : i64
    %511 = llvm.add %508, %510 : i64
    %512 = llvm.add %511, %479 : i64
    %513 = llvm.getelementptr %503[%512] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %502, %513 : f32, !llvm.ptr
    %514 = llvm.add %489, %488 : i64
    llvm.br ^bb43(%514 : i64)
  ^bb45:  // pred: ^bb43
    %515 = llvm.add %484, %483 : i64
    llvm.br ^bb41(%515 : i64)
  ^bb46:  // pred: ^bb41
    %516 = llvm.add %479, %478 : i64
    llvm.br ^bb39(%516 : i64)
  ^bb47:  // pred: ^bb39
    %517 = llvm.add %474, %473 : i64
    llvm.br ^bb37(%517 : i64)
  ^bb48:  // pred: ^bb37
    %518 = llvm.mlir.constant(0 : index) : i64
    %519 = llvm.mlir.constant(32 : index) : i64
    %520 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb49(%518 : i64)
  ^bb49(%521: i64):  // 2 preds: ^bb48, ^bb59
    %522 = llvm.icmp "slt" %521, %519 : i64
    llvm.cond_br %522, ^bb50, ^bb60
  ^bb50:  // pred: ^bb49
    %523 = llvm.mlir.constant(0 : index) : i64
    %524 = llvm.mlir.constant(16 : index) : i64
    %525 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb51(%523 : i64)
  ^bb51(%526: i64):  // 2 preds: ^bb50, ^bb58
    %527 = llvm.icmp "slt" %526, %524 : i64
    llvm.cond_br %527, ^bb52, ^bb59
  ^bb52:  // pred: ^bb51
    %528 = llvm.mlir.constant(0 : index) : i64
    %529 = llvm.mlir.constant(3 : index) : i64
    %530 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb53(%528 : i64)
  ^bb53(%531: i64):  // 2 preds: ^bb52, ^bb57
    %532 = llvm.icmp "slt" %531, %529 : i64
    llvm.cond_br %532, ^bb54, ^bb58
  ^bb54:  // pred: ^bb53
    %533 = llvm.mlir.constant(0 : index) : i64
    %534 = llvm.mlir.constant(3 : index) : i64
    %535 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb55(%533 : i64)
  ^bb55(%536: i64):  // 2 preds: ^bb54, ^bb56
    %537 = llvm.icmp "slt" %536, %534 : i64
    llvm.cond_br %537, ^bb56, ^bb57
  ^bb56:  // pred: ^bb55
    %538 = llvm.mul %531, %469 : i64
    %539 = llvm.mul %538, %470 : i64
    %540 = llvm.mul %536, %470 : i64
    %541 = llvm.add %539, %540 : i64
    %542 = llvm.add %541, %526 : i64
    %543 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %544 = llvm.mlir.constant(144 : index) : i64
    %545 = llvm.mul %521, %544 : i64
    %546 = llvm.mlir.constant(9 : index) : i64
    %547 = llvm.mul %526, %546 : i64
    %548 = llvm.add %545, %547 : i64
    %549 = llvm.mlir.constant(3 : index) : i64
    %550 = llvm.mul %531, %549 : i64
    %551 = llvm.add %548, %550 : i64
    %552 = llvm.add %551, %536 : i64
    %553 = llvm.getelementptr %543[%552] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %554 = llvm.load %553 : !llvm.ptr -> f32
    %555 = llvm.extractvalue %437[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %556 = llvm.mlir.constant(32 : index) : i64
    %557 = llvm.mul %542, %556 : i64
    %558 = llvm.add %557, %521 : i64
    %559 = llvm.getelementptr %555[%558] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %554, %559 : f32, !llvm.ptr
    %560 = llvm.add %536, %535 : i64
    llvm.br ^bb55(%560 : i64)
  ^bb57:  // pred: ^bb55
    %561 = llvm.add %531, %530 : i64
    llvm.br ^bb53(%561 : i64)
  ^bb58:  // pred: ^bb53
    %562 = llvm.add %526, %525 : i64
    llvm.br ^bb51(%562 : i64)
  ^bb59:  // pred: ^bb51
    %563 = llvm.add %521, %520 : i64
    llvm.br ^bb49(%563 : i64)
  ^bb60:  // pred: ^bb49
    %564 = llvm.mlir.constant(3 : i64) : i64
    %565 = llvm.extractvalue %420[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %566 = llvm.ptrtoint %565 : !llvm.ptr to i64
    %567 = llvm.extractvalue %467[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %568 = llvm.ptrtoint %567 : !llvm.ptr to i64
    %569 = llvm.extractvalue %450[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %570 = llvm.ptrtoint %569 : !llvm.ptr to i64
    %571 = llvm.extractvalue %437[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %572 = llvm.ptrtoint %571 : !llvm.ptr to i64
    %573 = llvm.mlir.constant(32 : i64) : i64
    %574 = llvm.mlir.constant(2 : i64) : i64
    %575 = llvm.mlir.constant(4575657221408423968 : i64) : i64
    "gemmini.intr.config_st"(%574, %575) : (i64, i64) -> ()
    %576 = llvm.mlir.constant(65540 : i64) : i64
    %577 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%576, %577) : (i64, i64) -> ()
    %578 = llvm.mlir.constant(0 : i64) : i64
    %579 = llvm.mlir.constant(0 : i64) : i64
    %580 = llvm.mlir.constant(0 : i64) : i64
    %581 = llvm.mlir.constant(0 : i64) : i64
    %582 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %583 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%582, %583) : (i64, i64) -> ()
    %584 = llvm.mlir.constant(844429225164800 : i64) : i64
    %585 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%584, %585) : (i64, i64) -> ()
    %586 = llvm.mlir.constant(844437816082432 : i64) : i64
    %587 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%586, %587) : (i64, i64) -> ()
    %588 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %589 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%588, %589) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%572, %568) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%570, %566) : (i64, i64) -> ()
    %590 = llvm.mlir.constant(256 : i64) : i64
    %591 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%590, %591) : (i64, i64) -> ()
    %592 = llvm.mlir.constant(16 : i64) : i64
    %593 = llvm.add %568, %592 : i64
    %594 = llvm.mlir.constant(64 : i64) : i64
    %595 = llvm.add %570, %594 : i64
    %596 = llvm.mlir.constant(16 : i64) : i64
    %597 = llvm.add %572, %596 : i64
    %598 = llvm.mlir.constant(0 : i64) : i64
    %599 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %600 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%599, %600) : (i64, i64) -> ()
    %601 = llvm.mlir.constant(844429225164800 : i64) : i64
    %602 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%601, %602) : (i64, i64) -> ()
    %603 = llvm.mlir.constant(844437816082432 : i64) : i64
    %604 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%603, %604) : (i64, i64) -> ()
    %605 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %606 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%605, %606) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%597, %593) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%595, %566) : (i64, i64) -> ()
    %607 = llvm.mlir.constant(256 : i64) : i64
    %608 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%607, %608) : (i64, i64) -> ()
    %609 = llvm.mlir.constant(736 : i64) : i64
    %610 = llvm.add %568, %609 : i64
    %611 = llvm.mlir.constant(0 : i64) : i64
    %612 = llvm.mlir.constant(0 : i64) : i64
    %613 = llvm.mlir.constant(368 : i64) : i64
    %614 = llvm.add %566, %613 : i64
    %615 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %616 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%615, %616) : (i64, i64) -> ()
    %617 = llvm.mlir.constant(844429225164800 : i64) : i64
    %618 = llvm.mlir.constant(281569466187792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%617, %618) : (i64, i64) -> ()
    %619 = llvm.mlir.constant(844437816082432 : i64) : i64
    %620 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%619, %620) : (i64, i64) -> ()
    %621 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %622 = llvm.mlir.constant(65539 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%621, %622) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%572, %610) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%570, %614) : (i64, i64) -> ()
    %623 = llvm.mlir.constant(256 : i64) : i64
    %624 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%623, %624) : (i64, i64) -> ()
    %625 = llvm.mlir.constant(752 : i64) : i64
    %626 = llvm.add %568, %625 : i64
    %627 = llvm.mlir.constant(64 : i64) : i64
    %628 = llvm.add %570, %627 : i64
    %629 = llvm.mlir.constant(16 : i64) : i64
    %630 = llvm.add %572, %629 : i64
    %631 = llvm.mlir.constant(368 : i64) : i64
    %632 = llvm.add %566, %631 : i64
    %633 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %634 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%633, %634) : (i64, i64) -> ()
    %635 = llvm.mlir.constant(844429225164800 : i64) : i64
    %636 = llvm.mlir.constant(281569466187792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%635, %636) : (i64, i64) -> ()
    %637 = llvm.mlir.constant(844437816082432 : i64) : i64
    %638 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%637, %638) : (i64, i64) -> ()
    %639 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %640 = llvm.mlir.constant(65539 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%639, %640) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%630, %626) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%628, %632) : (i64, i64) -> ()
    %641 = llvm.mlir.constant(256 : i64) : i64
    %642 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%641, %642) : (i64, i64) -> ()
    %643 = llvm.mlir.constant(18304 : i64) : i64
    %644 = llvm.add %568, %643 : i64
    %645 = llvm.mlir.constant(0 : i64) : i64
    %646 = llvm.mlir.constant(0 : i64) : i64
    %647 = llvm.mlir.constant(10560 : i64) : i64
    %648 = llvm.add %566, %647 : i64
    %649 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %650 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%649, %650) : (i64, i64) -> ()
    %651 = llvm.mlir.constant(844429225164800 : i64) : i64
    %652 = llvm.mlir.constant(281492158087184 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%651, %652) : (i64, i64) -> ()
    %653 = llvm.mlir.constant(844437816082432 : i64) : i64
    %654 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%653, %654) : (i64, i64) -> ()
    %655 = llvm.mlir.constant(1125899906842624 : i64) : i64
    %656 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%655, %656) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%572, %644) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%570, %648) : (i64, i64) -> ()
    %657 = llvm.mlir.constant(256 : i64) : i64
    %658 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%657, %658) : (i64, i64) -> ()
    %659 = llvm.mlir.constant(18320 : i64) : i64
    %660 = llvm.add %568, %659 : i64
    %661 = llvm.mlir.constant(64 : i64) : i64
    %662 = llvm.add %570, %661 : i64
    %663 = llvm.mlir.constant(16 : i64) : i64
    %664 = llvm.add %572, %663 : i64
    %665 = llvm.mlir.constant(10560 : i64) : i64
    %666 = llvm.add %566, %665 : i64
    %667 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %668 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%667, %668) : (i64, i64) -> ()
    %669 = llvm.mlir.constant(844429225164800 : i64) : i64
    %670 = llvm.mlir.constant(281492158087184 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%669, %670) : (i64, i64) -> ()
    %671 = llvm.mlir.constant(844437816082432 : i64) : i64
    %672 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%671, %672) : (i64, i64) -> ()
    %673 = llvm.mlir.constant(1125899906842624 : i64) : i64
    %674 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%673, %674) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%664, %660) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%662, %666) : (i64, i64) -> ()
    %675 = llvm.mlir.constant(256 : i64) : i64
    %676 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%675, %676) : (i64, i64) -> ()
    %677 = llvm.mlir.constant(19040 : i64) : i64
    %678 = llvm.add %568, %677 : i64
    %679 = llvm.mlir.constant(0 : i64) : i64
    %680 = llvm.mlir.constant(0 : i64) : i64
    %681 = llvm.mlir.constant(10928 : i64) : i64
    %682 = llvm.add %566, %681 : i64
    %683 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %684 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%683, %684) : (i64, i64) -> ()
    %685 = llvm.mlir.constant(844429225164800 : i64) : i64
    %686 = llvm.mlir.constant(281492156776464 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%685, %686) : (i64, i64) -> ()
    %687 = llvm.mlir.constant(844437816082432 : i64) : i64
    %688 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%687, %688) : (i64, i64) -> ()
    %689 = llvm.mlir.constant(1125899906842624 : i64) : i64
    %690 = llvm.mlir.constant(65539 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%689, %690) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%572, %678) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%570, %682) : (i64, i64) -> ()
    %691 = llvm.mlir.constant(256 : i64) : i64
    %692 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%691, %692) : (i64, i64) -> ()
    %693 = llvm.mlir.constant(19056 : i64) : i64
    %694 = llvm.add %568, %693 : i64
    %695 = llvm.mlir.constant(64 : i64) : i64
    %696 = llvm.add %570, %695 : i64
    %697 = llvm.mlir.constant(16 : i64) : i64
    %698 = llvm.add %572, %697 : i64
    %699 = llvm.mlir.constant(10928 : i64) : i64
    %700 = llvm.add %566, %699 : i64
    %701 = llvm.mlir.constant(9007267976183809 : i64) : i64
    %702 = llvm.mlir.constant(4296671258 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%701, %702) : (i64, i64) -> ()
    %703 = llvm.mlir.constant(844429225164800 : i64) : i64
    %704 = llvm.mlir.constant(281492156776464 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%703, %704) : (i64, i64) -> ()
    %705 = llvm.mlir.constant(844437816082432 : i64) : i64
    %706 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%705, %706) : (i64, i64) -> ()
    %707 = llvm.mlir.constant(1125899906842624 : i64) : i64
    %708 = llvm.mlir.constant(65539 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%707, %708) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%698, %694) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%696, %700) : (i64, i64) -> ()
    %709 = llvm.mlir.constant(256 : i64) : i64
    %710 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%709, %710) : (i64, i64) -> ()
    %711 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%711, %711) : (i64, i64) -> ()
    %712 = llvm.mlir.constant(0 : index) : i64
    %713 = llvm.mlir.constant(1 : index) : i64
    %714 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb61(%712 : i64)
  ^bb61(%715: i64):  // 2 preds: ^bb60, ^bb71
    %716 = llvm.icmp "slt" %715, %713 : i64
    llvm.cond_br %716, ^bb62, ^bb72
  ^bb62:  // pred: ^bb61
    %717 = llvm.mlir.constant(0 : index) : i64
    %718 = llvm.mlir.constant(32 : index) : i64
    %719 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb63(%717 : i64)
  ^bb63(%720: i64):  // 2 preds: ^bb62, ^bb70
    %721 = llvm.icmp "slt" %720, %718 : i64
    llvm.cond_br %721, ^bb64, ^bb71
  ^bb64:  // pred: ^bb63
    %722 = llvm.mlir.constant(0 : index) : i64
    %723 = llvm.mlir.constant(26 : index) : i64
    %724 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb65(%722 : i64)
  ^bb65(%725: i64):  // 2 preds: ^bb64, ^bb69
    %726 = llvm.icmp "slt" %725, %723 : i64
    llvm.cond_br %726, ^bb66, ^bb70
  ^bb66:  // pred: ^bb65
    %727 = llvm.mlir.constant(0 : index) : i64
    %728 = llvm.mlir.constant(26 : index) : i64
    %729 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb67(%727 : i64)
  ^bb67(%730: i64):  // 2 preds: ^bb66, ^bb68
    %731 = llvm.icmp "slt" %730, %728 : i64
    llvm.cond_br %731, ^bb68, ^bb69
  ^bb68:  // pred: ^bb67
    %732 = llvm.mlir.constant(26 : index) : i64
    %733 = llvm.mul %715, %732 : i64
    %734 = llvm.mul %733, %732 : i64
    %735 = llvm.mul %725, %732 : i64
    %736 = llvm.add %734, %735 : i64
    %737 = llvm.add %736, %730 : i64
    %738 = llvm.extractvalue %467[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %739 = llvm.mlir.constant(32 : index) : i64
    %740 = llvm.mul %737, %739 : i64
    %741 = llvm.add %740, %720 : i64
    %742 = llvm.getelementptr %738[%741] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %743 = llvm.load %742 : !llvm.ptr -> f32
    %744 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %745 = llvm.mlir.constant(21632 : index) : i64
    %746 = llvm.mul %715, %745 : i64
    %747 = llvm.mlir.constant(676 : index) : i64
    %748 = llvm.mul %720, %747 : i64
    %749 = llvm.add %746, %748 : i64
    %750 = llvm.mlir.constant(26 : index) : i64
    %751 = llvm.mul %725, %750 : i64
    %752 = llvm.add %749, %751 : i64
    %753 = llvm.add %752, %730 : i64
    %754 = llvm.getelementptr %744[%753] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %743, %754 : f32, !llvm.ptr
    %755 = llvm.add %730, %729 : i64
    llvm.br ^bb67(%755 : i64)
  ^bb69:  // pred: ^bb67
    %756 = llvm.add %725, %724 : i64
    llvm.br ^bb65(%756 : i64)
  ^bb70:  // pred: ^bb65
    %757 = llvm.add %720, %719 : i64
    llvm.br ^bb63(%757 : i64)
  ^bb71:  // pred: ^bb63
    %758 = llvm.add %715, %714 : i64
    llvm.br ^bb61(%758 : i64)
  ^bb72:  // pred: ^bb61
    %759 = llvm.extractvalue %420[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%759) : (!llvm.ptr) -> ()
    %760 = llvm.extractvalue %437[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%760) : (!llvm.ptr) -> ()
    %761 = llvm.extractvalue %467[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%761) : (!llvm.ptr) -> ()
    %762 = llvm.extractvalue %450[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%762) : (!llvm.ptr) -> ()
    linalg.copy ins(%99 : memref<1x32x26x26xf32>) outs(%12 : memref<1x32x26x26xf32>)
    %763 = llvm.extractvalue %73[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%763) : (!llvm.ptr) -> ()
    %764 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%764) : (!llvm.ptr) -> ()
    llvm.return
  }
}


