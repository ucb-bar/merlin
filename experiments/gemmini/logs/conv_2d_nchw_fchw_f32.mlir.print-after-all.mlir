// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  memref.global "private" @input : memref<2x2x5x5xf32> = dense<[[[[1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00]], [[-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00]]], [[[1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00]], [[-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00]]]]>
  memref.global "private" @weight : memref<2x2x3x3xf32> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]], [[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]]]>
  func.func @main() -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %0 = memref.get_global @input : memref<2x2x5x5xf32>
    %1 = memref.get_global @weight : memref<2x2x3x3xf32>
    %alloc = memref.alloc() : memref<2x2x3x3xf32>
    %alloc_0 = memref.alloc() : memref<2x5x5x2xf32>
    %alloc_1 = memref.alloc() : memref<18x2xf32>
    %alloc_2 = memref.alloc() : memref<2xi32>
    %alloc_3 = memref.alloc() : memref<18x2xf32>
    %c3_i64 = arith.constant 3 : i64
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c2_4 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c2_4 step %c1 {
      %c0_12 = arith.constant 0 : index
      %c2_13 = arith.constant 2 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg1 = %c0_12 to %c2_13 step %c1_14 {
        %c0_15 = arith.constant 0 : index
        %c5 = arith.constant 5 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg2 = %c0_15 to %c5 step %c1_16 {
          %c0_17 = arith.constant 0 : index
          %c5_18 = arith.constant 5 : index
          %c1_19 = arith.constant 1 : index
          scf.for %arg3 = %c0_17 to %c5_18 step %c1_19 {
            %2 = memref.load %0[%arg0, %arg1, %arg2, %arg3] : memref<2x2x5x5xf32>
            memref.store %2, %alloc_0[%arg0, %arg2, %arg3, %arg1] : memref<2x5x5x2xf32>
          }
        }
      }
    }
    %c0_5 = arith.constant 0 : index
    %c2_6 = arith.constant 2 : index
    %c1_7 = arith.constant 1 : index
    scf.for %arg0 = %c0_5 to %c2_6 step %c1_7 {
      %c0_12 = arith.constant 0 : index
      %c2_13 = arith.constant 2 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg1 = %c0_12 to %c2_13 step %c1_14 {
        %c0_15 = arith.constant 0 : index
        %c3_16 = arith.constant 3 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg2 = %c0_15 to %c3_16 step %c1_17 {
          %c0_18 = arith.constant 0 : index
          %c3_19 = arith.constant 3 : index
          %c1_20 = arith.constant 1 : index
          scf.for %arg3 = %c0_18 to %c3_19 step %c1_20 {
            %2 = arith.muli %arg2, %c3 : index
            %3 = arith.muli %2, %c2 : index
            %4 = arith.muli %arg3, %c2 : index
            %5 = arith.addi %3, %4 : index
            %6 = arith.addi %5, %arg1 : index
            %7 = memref.load %1[%arg0, %arg1, %arg2, %arg3] : memref<2x2x3x3xf32>
            memref.store %7, %alloc_1[%6, %arg0] : memref<18x2xf32>
          }
        }
      }
    }
    %c3_i64_8 = arith.constant 3 : i64
    gemmini.tile_conv %alloc_0 %alloc_1 %alloc_2 %alloc_3 %c3_i64 %c3_i64 %c3_i64_8 : memref<2x5x5x2xf32> memref<18x2xf32> memref<2xi32> memref<18x2xf32> i64 i64 i64
    %c0_9 = arith.constant 0 : index
    %c2_10 = arith.constant 2 : index
    %c1_11 = arith.constant 1 : index
    scf.for %arg0 = %c0_9 to %c2_10 step %c1_11 {
      %c0_12 = arith.constant 0 : index
      %c2_13 = arith.constant 2 : index
      %c1_14 = arith.constant 1 : index
      scf.for %arg1 = %c0_12 to %c2_13 step %c1_14 {
        %c0_15 = arith.constant 0 : index
        %c3_16 = arith.constant 3 : index
        %c1_17 = arith.constant 1 : index
        scf.for %arg2 = %c0_15 to %c3_16 step %c1_17 {
          %c0_18 = arith.constant 0 : index
          %c3_19 = arith.constant 3 : index
          %c1_20 = arith.constant 1 : index
          scf.for %arg3 = %c0_18 to %c3_19 step %c1_20 {
            %c3_21 = arith.constant 3 : index
            %2 = arith.muli %arg0, %c3_21 : index
            %3 = arith.muli %2, %c3_21 : index
            %4 = arith.muli %arg2, %c3_21 : index
            %5 = arith.addi %3, %4 : index
            %6 = arith.addi %5, %arg3 : index
            %7 = memref.load %alloc_3[%6, %arg1] : memref<18x2xf32>
            memref.store %7, %alloc[%arg0, %arg1, %arg2, %arg3] : memref<2x2x3x3xf32>
          }
        }
      }
    }
    memref.dealloc %alloc_0 : memref<2x5x5x2xf32>
    memref.dealloc %alloc_1 : memref<18x2xf32>
    memref.dealloc %alloc_3 : memref<18x2xf32>
    memref.dealloc %alloc_2 : memref<2xi32>
    gemmini.print %alloc : memref<2x2x3x3xf32>
    return %c0_i8 : i8
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private @input(dense<[[[[1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00]], [[-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, -1.000000e+00]]], [[[1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00]], [[-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00], [-1.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, -1.000000e+00]]]]> : tensor<2x2x5x5xf32>) {addr_space = 0 : i32} : !llvm.array<2 x array<2 x array<5 x array<5 x f32>>>>
  llvm.mlir.global private @weight(dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]], [[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]], [[3.000000e+00, 2.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]]]> : tensor<2x2x3x3xf32>) {addr_space = 0 : i32} : !llvm.array<2 x array<2 x array<3 x array<3 x f32>>>>
  llvm.func @main() -> i8 {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(2 : index) : i64
    %3 = llvm.mlir.constant(5 : index) : i64
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(25 : index) : i64
    %7 = llvm.mlir.constant(50 : index) : i64
    %8 = llvm.mlir.constant(100 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.mlir.addressof @input : !llvm.ptr
    %13 = llvm.getelementptr %12[0, 0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x array<5 x array<5 x f32>>>>
    %14 = llvm.mlir.constant(3735928559 : index) : i64
    %15 = llvm.inttoptr %14 : i64 to !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %13, %17[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.insertvalue %19, %18[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %1, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %2, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %3, %22[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.insertvalue %4, %23[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %25 = llvm.insertvalue %7, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %6, %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %4, %26[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %5, %27[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.mlir.constant(2 : index) : i64
    %30 = llvm.mlir.constant(2 : index) : i64
    %31 = llvm.mlir.constant(3 : index) : i64
    %32 = llvm.mlir.constant(3 : index) : i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.mlir.constant(9 : index) : i64
    %35 = llvm.mlir.constant(18 : index) : i64
    %36 = llvm.mlir.constant(36 : index) : i64
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.getelementptr %37[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.mlir.addressof @weight : !llvm.ptr
    %41 = llvm.getelementptr %40[0, 0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x array<3 x array<3 x f32>>>>
    %42 = llvm.mlir.constant(3735928559 : index) : i64
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %46 = llvm.insertvalue %41, %45[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %47 = llvm.mlir.constant(0 : index) : i64
    %48 = llvm.insertvalue %47, %46[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %49 = llvm.insertvalue %29, %48[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %50 = llvm.insertvalue %30, %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %51 = llvm.insertvalue %31, %50[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %52 = llvm.insertvalue %32, %51[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %53 = llvm.insertvalue %35, %52[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %54 = llvm.insertvalue %34, %53[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %55 = llvm.insertvalue %32, %54[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %33, %55[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.mlir.constant(2 : index) : i64
    %58 = llvm.mlir.constant(2 : index) : i64
    %59 = llvm.mlir.constant(3 : index) : i64
    %60 = llvm.mlir.constant(3 : index) : i64
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.mlir.constant(9 : index) : i64
    %63 = llvm.mlir.constant(18 : index) : i64
    %64 = llvm.mlir.constant(36 : index) : i64
    %65 = llvm.mlir.zero : !llvm.ptr
    %66 = llvm.getelementptr %65[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr
    %69 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %71 = llvm.insertvalue %68, %70[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %74 = llvm.insertvalue %57, %73[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %75 = llvm.insertvalue %58, %74[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %76 = llvm.insertvalue %59, %75[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %77 = llvm.insertvalue %60, %76[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %78 = llvm.insertvalue %63, %77[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %79 = llvm.insertvalue %62, %78[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %80 = llvm.insertvalue %60, %79[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %81 = llvm.insertvalue %61, %80[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %82 = llvm.mlir.constant(2 : index) : i64
    %83 = llvm.mlir.constant(5 : index) : i64
    %84 = llvm.mlir.constant(5 : index) : i64
    %85 = llvm.mlir.constant(2 : index) : i64
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.constant(10 : index) : i64
    %88 = llvm.mlir.constant(50 : index) : i64
    %89 = llvm.mlir.constant(100 : index) : i64
    %90 = llvm.mlir.zero : !llvm.ptr
    %91 = llvm.getelementptr %90[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.call @malloc(%92) : (i64) -> !llvm.ptr
    %94 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.mlir.constant(0 : index) : i64
    %98 = llvm.insertvalue %97, %96[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %99 = llvm.insertvalue %82, %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %100 = llvm.insertvalue %83, %99[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %101 = llvm.insertvalue %84, %100[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %102 = llvm.insertvalue %85, %101[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.insertvalue %88, %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %104 = llvm.insertvalue %87, %103[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %105 = llvm.insertvalue %85, %104[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %106 = llvm.insertvalue %86, %105[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.mlir.constant(18 : index) : i64
    %108 = llvm.mlir.constant(2 : index) : i64
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.mlir.constant(36 : index) : i64
    %111 = llvm.mlir.zero : !llvm.ptr
    %112 = llvm.getelementptr %111[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.ptrtoint %112 : !llvm.ptr to i64
    %114 = llvm.call @malloc(%113) : (i64) -> !llvm.ptr
    %115 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.insertvalue %114, %116[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.mlir.constant(0 : index) : i64
    %119 = llvm.insertvalue %118, %117[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.insertvalue %107, %119[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.insertvalue %108, %120[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.insertvalue %108, %121[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.insertvalue %109, %122[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.mlir.constant(2 : index) : i64
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.mlir.zero : !llvm.ptr
    %127 = llvm.getelementptr %126[%124] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %128 = llvm.ptrtoint %127 : !llvm.ptr to i64
    %129 = llvm.call @malloc(%128) : (i64) -> !llvm.ptr
    %130 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %131 = llvm.insertvalue %129, %130[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %132 = llvm.insertvalue %129, %131[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %133 = llvm.mlir.constant(0 : index) : i64
    %134 = llvm.insertvalue %133, %132[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.insertvalue %124, %134[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %136 = llvm.insertvalue %125, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.mlir.constant(18 : index) : i64
    %138 = llvm.mlir.constant(2 : index) : i64
    %139 = llvm.mlir.constant(1 : index) : i64
    %140 = llvm.mlir.constant(36 : index) : i64
    %141 = llvm.mlir.zero : !llvm.ptr
    %142 = llvm.getelementptr %141[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %143 = llvm.ptrtoint %142 : !llvm.ptr to i64
    %144 = llvm.call @malloc(%143) : (i64) -> !llvm.ptr
    %145 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %146 = llvm.insertvalue %144, %145[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.insertvalue %144, %146[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.mlir.constant(0 : index) : i64
    %149 = llvm.insertvalue %148, %147[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %137, %149[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %138, %150[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %138, %151[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %139, %152[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.mlir.constant(3 : i64) : i64
    %155 = llvm.mlir.constant(3 : index) : i64
    %156 = llvm.mlir.constant(2 : index) : i64
    %157 = llvm.mlir.constant(0 : index) : i64
    %158 = llvm.mlir.constant(2 : index) : i64
    %159 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%157 : i64)
  ^bb1(%160: i64):  // 2 preds: ^bb0, ^bb11
    %161 = llvm.icmp "slt" %160, %158 : i64
    llvm.cond_br %161, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %162 = llvm.mlir.constant(0 : index) : i64
    %163 = llvm.mlir.constant(2 : index) : i64
    %164 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%162 : i64)
  ^bb3(%165: i64):  // 2 preds: ^bb2, ^bb10
    %166 = llvm.icmp "slt" %165, %163 : i64
    llvm.cond_br %166, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %167 = llvm.mlir.constant(0 : index) : i64
    %168 = llvm.mlir.constant(5 : index) : i64
    %169 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%167 : i64)
  ^bb5(%170: i64):  // 2 preds: ^bb4, ^bb9
    %171 = llvm.icmp "slt" %170, %168 : i64
    llvm.cond_br %171, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %172 = llvm.mlir.constant(0 : index) : i64
    %173 = llvm.mlir.constant(5 : index) : i64
    %174 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%172 : i64)
  ^bb7(%175: i64):  // 2 preds: ^bb6, ^bb8
    %176 = llvm.icmp "slt" %175, %173 : i64
    llvm.cond_br %176, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %177 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %178 = llvm.mlir.constant(50 : index) : i64
    %179 = llvm.mul %160, %178 : i64
    %180 = llvm.mlir.constant(25 : index) : i64
    %181 = llvm.mul %165, %180 : i64
    %182 = llvm.add %179, %181 : i64
    %183 = llvm.mlir.constant(5 : index) : i64
    %184 = llvm.mul %170, %183 : i64
    %185 = llvm.add %182, %184 : i64
    %186 = llvm.add %185, %175 : i64
    %187 = llvm.getelementptr %177[%186] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %188 = llvm.load %187 : !llvm.ptr -> f32
    %189 = llvm.extractvalue %106[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %190 = llvm.mlir.constant(50 : index) : i64
    %191 = llvm.mul %160, %190 : i64
    %192 = llvm.mlir.constant(10 : index) : i64
    %193 = llvm.mul %170, %192 : i64
    %194 = llvm.add %191, %193 : i64
    %195 = llvm.mlir.constant(2 : index) : i64
    %196 = llvm.mul %175, %195 : i64
    %197 = llvm.add %194, %196 : i64
    %198 = llvm.add %197, %165 : i64
    %199 = llvm.getelementptr %189[%198] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %199 : f32, !llvm.ptr
    %200 = llvm.add %175, %174 : i64
    llvm.br ^bb7(%200 : i64)
  ^bb9:  // pred: ^bb7
    %201 = llvm.add %170, %169 : i64
    llvm.br ^bb5(%201 : i64)
  ^bb10:  // pred: ^bb5
    %202 = llvm.add %165, %164 : i64
    llvm.br ^bb3(%202 : i64)
  ^bb11:  // pred: ^bb3
    %203 = llvm.add %160, %159 : i64
    llvm.br ^bb1(%203 : i64)
  ^bb12:  // pred: ^bb1
    %204 = llvm.mlir.constant(0 : index) : i64
    %205 = llvm.mlir.constant(2 : index) : i64
    %206 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%204 : i64)
  ^bb13(%207: i64):  // 2 preds: ^bb12, ^bb23
    %208 = llvm.icmp "slt" %207, %205 : i64
    llvm.cond_br %208, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    %209 = llvm.mlir.constant(0 : index) : i64
    %210 = llvm.mlir.constant(2 : index) : i64
    %211 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%209 : i64)
  ^bb15(%212: i64):  // 2 preds: ^bb14, ^bb22
    %213 = llvm.icmp "slt" %212, %210 : i64
    llvm.cond_br %213, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    %214 = llvm.mlir.constant(0 : index) : i64
    %215 = llvm.mlir.constant(3 : index) : i64
    %216 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb17(%214 : i64)
  ^bb17(%217: i64):  // 2 preds: ^bb16, ^bb21
    %218 = llvm.icmp "slt" %217, %215 : i64
    llvm.cond_br %218, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %219 = llvm.mlir.constant(0 : index) : i64
    %220 = llvm.mlir.constant(3 : index) : i64
    %221 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb19(%219 : i64)
  ^bb19(%222: i64):  // 2 preds: ^bb18, ^bb20
    %223 = llvm.icmp "slt" %222, %220 : i64
    llvm.cond_br %223, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %224 = llvm.mul %217, %155 : i64
    %225 = llvm.mul %224, %156 : i64
    %226 = llvm.mul %222, %156 : i64
    %227 = llvm.add %225, %226 : i64
    %228 = llvm.add %227, %212 : i64
    %229 = llvm.extractvalue %56[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %230 = llvm.mlir.constant(18 : index) : i64
    %231 = llvm.mul %207, %230 : i64
    %232 = llvm.mlir.constant(9 : index) : i64
    %233 = llvm.mul %212, %232 : i64
    %234 = llvm.add %231, %233 : i64
    %235 = llvm.mlir.constant(3 : index) : i64
    %236 = llvm.mul %217, %235 : i64
    %237 = llvm.add %234, %236 : i64
    %238 = llvm.add %237, %222 : i64
    %239 = llvm.getelementptr %229[%238] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %240 = llvm.load %239 : !llvm.ptr -> f32
    %241 = llvm.extractvalue %123[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %242 = llvm.mlir.constant(2 : index) : i64
    %243 = llvm.mul %228, %242 : i64
    %244 = llvm.add %243, %207 : i64
    %245 = llvm.getelementptr %241[%244] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %240, %245 : f32, !llvm.ptr
    %246 = llvm.add %222, %221 : i64
    llvm.br ^bb19(%246 : i64)
  ^bb21:  // pred: ^bb19
    %247 = llvm.add %217, %216 : i64
    llvm.br ^bb17(%247 : i64)
  ^bb22:  // pred: ^bb17
    %248 = llvm.add %212, %211 : i64
    llvm.br ^bb15(%248 : i64)
  ^bb23:  // pred: ^bb15
    %249 = llvm.add %207, %206 : i64
    llvm.br ^bb13(%249 : i64)
  ^bb24:  // pred: ^bb13
    %250 = llvm.mlir.constant(3 : i64) : i64
    %251 = llvm.extractvalue %106[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %252 = llvm.ptrtoint %251 : !llvm.ptr to i64
    %253 = llvm.extractvalue %153[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.ptrtoint %253 : !llvm.ptr to i64
    %255 = llvm.extractvalue %136[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %256 = llvm.ptrtoint %255 : !llvm.ptr to i64
    %257 = llvm.extractvalue %123[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.ptrtoint %257 : !llvm.ptr to i64
    %259 = llvm.mlir.constant(2 : i64) : i64
    %260 = llvm.mlir.constant(2 : i64) : i64
    %261 = llvm.mlir.constant(4575657221408423938 : i64) : i64
    "gemmini.intr.config_st"(%260, %261) : (i64, i64) -> ()
    %262 = llvm.mlir.constant(65540 : i64) : i64
    %263 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%262, %263) : (i64, i64) -> ()
    %264 = llvm.mlir.constant(0 : i64) : i64
    %265 = llvm.mlir.constant(0 : i64) : i64
    %266 = llvm.mlir.constant(0 : i64) : i64
    %267 = llvm.mlir.constant(0 : i64) : i64
    %268 = llvm.mlir.constant(562958543683586 : i64) : i64
    %269 = llvm.mlir.constant(4295163907 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%268, %269) : (i64, i64) -> ()
    %270 = llvm.mlir.constant(844429225164800 : i64) : i64
    %271 = llvm.mlir.constant(562962838519810 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%270, %271) : (i64, i64) -> ()
    %272 = llvm.mlir.constant(844437815164928 : i64) : i64
    %273 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%272, %273) : (i64, i64) -> ()
    %274 = llvm.mlir.constant(844424930131968 : i64) : i64
    %275 = llvm.mlir.constant(65539 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%274, %275) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%258, %254) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%256, %252) : (i64, i64) -> ()
    %276 = llvm.mlir.constant(768 : i64) : i64
    %277 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%276, %277) : (i64, i64) -> ()
    %278 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%278, %278) : (i64, i64) -> ()
    %279 = llvm.mlir.constant(0 : index) : i64
    %280 = llvm.mlir.constant(2 : index) : i64
    %281 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb25(%279 : i64)
  ^bb25(%282: i64):  // 2 preds: ^bb24, ^bb35
    %283 = llvm.icmp "slt" %282, %280 : i64
    llvm.cond_br %283, ^bb26, ^bb36
  ^bb26:  // pred: ^bb25
    %284 = llvm.mlir.constant(0 : index) : i64
    %285 = llvm.mlir.constant(2 : index) : i64
    %286 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb27(%284 : i64)
  ^bb27(%287: i64):  // 2 preds: ^bb26, ^bb34
    %288 = llvm.icmp "slt" %287, %285 : i64
    llvm.cond_br %288, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    %289 = llvm.mlir.constant(0 : index) : i64
    %290 = llvm.mlir.constant(3 : index) : i64
    %291 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb29(%289 : i64)
  ^bb29(%292: i64):  // 2 preds: ^bb28, ^bb33
    %293 = llvm.icmp "slt" %292, %290 : i64
    llvm.cond_br %293, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    %294 = llvm.mlir.constant(0 : index) : i64
    %295 = llvm.mlir.constant(3 : index) : i64
    %296 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb31(%294 : i64)
  ^bb31(%297: i64):  // 2 preds: ^bb30, ^bb32
    %298 = llvm.icmp "slt" %297, %295 : i64
    llvm.cond_br %298, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %299 = llvm.mlir.constant(3 : index) : i64
    %300 = llvm.mul %282, %299 : i64
    %301 = llvm.mul %300, %299 : i64
    %302 = llvm.mul %292, %299 : i64
    %303 = llvm.add %301, %302 : i64
    %304 = llvm.add %303, %297 : i64
    %305 = llvm.extractvalue %153[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.mlir.constant(2 : index) : i64
    %307 = llvm.mul %304, %306 : i64
    %308 = llvm.add %307, %287 : i64
    %309 = llvm.getelementptr %305[%308] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %310 = llvm.load %309 : !llvm.ptr -> f32
    %311 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %312 = llvm.mlir.constant(18 : index) : i64
    %313 = llvm.mul %282, %312 : i64
    %314 = llvm.mlir.constant(9 : index) : i64
    %315 = llvm.mul %287, %314 : i64
    %316 = llvm.add %313, %315 : i64
    %317 = llvm.mlir.constant(3 : index) : i64
    %318 = llvm.mul %292, %317 : i64
    %319 = llvm.add %316, %318 : i64
    %320 = llvm.add %319, %297 : i64
    %321 = llvm.getelementptr %311[%320] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %310, %321 : f32, !llvm.ptr
    %322 = llvm.add %297, %296 : i64
    llvm.br ^bb31(%322 : i64)
  ^bb33:  // pred: ^bb31
    %323 = llvm.add %292, %291 : i64
    llvm.br ^bb29(%323 : i64)
  ^bb34:  // pred: ^bb29
    %324 = llvm.add %287, %286 : i64
    llvm.br ^bb27(%324 : i64)
  ^bb35:  // pred: ^bb27
    %325 = llvm.add %282, %281 : i64
    llvm.br ^bb25(%325 : i64)
  ^bb36:  // pred: ^bb25
    %326 = llvm.extractvalue %106[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.call @free(%326) : (!llvm.ptr) -> ()
    %327 = llvm.extractvalue %123[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%327) : (!llvm.ptr) -> ()
    %328 = llvm.extractvalue %153[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%328) : (!llvm.ptr) -> ()
    %329 = llvm.extractvalue %136[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%329) : (!llvm.ptr) -> ()
    %330 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %331 = llvm.mlir.constant(0 : index) : i64
    %332 = llvm.getelementptr %330[%331, %331] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %333 = llvm.mlir.addressof @nl : !llvm.ptr
    %334 = llvm.mlir.constant(0 : index) : i64
    %335 = llvm.getelementptr %333[%334, %334] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %336 = llvm.mlir.constant(0 : index) : i64
    %337 = llvm.mlir.constant(2 : index) : i64
    %338 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb37(%336 : i64)
  ^bb37(%339: i64):  // 2 preds: ^bb36, ^bb47
    %340 = llvm.icmp "slt" %339, %337 : i64
    llvm.cond_br %340, ^bb38, ^bb48
  ^bb38:  // pred: ^bb37
    %341 = llvm.mlir.constant(0 : index) : i64
    %342 = llvm.mlir.constant(2 : index) : i64
    %343 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb39(%341 : i64)
  ^bb39(%344: i64):  // 2 preds: ^bb38, ^bb46
    %345 = llvm.icmp "slt" %344, %342 : i64
    llvm.cond_br %345, ^bb40, ^bb47
  ^bb40:  // pred: ^bb39
    %346 = llvm.mlir.constant(0 : index) : i64
    %347 = llvm.mlir.constant(3 : index) : i64
    %348 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb41(%346 : i64)
  ^bb41(%349: i64):  // 2 preds: ^bb40, ^bb45
    %350 = llvm.icmp "slt" %349, %347 : i64
    llvm.cond_br %350, ^bb42, ^bb46
  ^bb42:  // pred: ^bb41
    %351 = llvm.mlir.constant(0 : index) : i64
    %352 = llvm.mlir.constant(3 : index) : i64
    %353 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb43(%351 : i64)
  ^bb43(%354: i64):  // 2 preds: ^bb42, ^bb44
    %355 = llvm.icmp "slt" %354, %352 : i64
    llvm.cond_br %355, ^bb44, ^bb45
  ^bb44:  // pred: ^bb43
    %356 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %357 = llvm.mlir.constant(18 : index) : i64
    %358 = llvm.mul %339, %357 : i64
    %359 = llvm.mlir.constant(9 : index) : i64
    %360 = llvm.mul %344, %359 : i64
    %361 = llvm.add %358, %360 : i64
    %362 = llvm.mlir.constant(3 : index) : i64
    %363 = llvm.mul %349, %362 : i64
    %364 = llvm.add %361, %363 : i64
    %365 = llvm.add %364, %354 : i64
    %366 = llvm.getelementptr %356[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %367 = llvm.load %366 : !llvm.ptr -> f32
    %368 = llvm.fpext %367 : f32 to f64
    %369 = llvm.call @printf(%332, %368) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %370 = llvm.add %354, %353 : i64
    llvm.br ^bb43(%370 : i64)
  ^bb45:  // pred: ^bb43
    %371 = llvm.call @printf(%335) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %372 = llvm.add %349, %348 : i64
    llvm.br ^bb41(%372 : i64)
  ^bb46:  // pred: ^bb41
    %373 = llvm.call @printf(%335) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %374 = llvm.add %344, %343 : i64
    llvm.br ^bb39(%374 : i64)
  ^bb47:  // pred: ^bb39
    %375 = llvm.call @printf(%335) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %376 = llvm.add %339, %338 : i64
    llvm.br ^bb37(%376 : i64)
  ^bb48:  // pred: ^bb37
    llvm.return %0 : i8
  }
}


