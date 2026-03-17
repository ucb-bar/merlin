// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  func.func @conv2d_block1(%arg0: memref<1x32x32x32xf16>, %arg1: memref<3x3x32x64xf16>, %arg2: memref<1x30x30x64xf32>) {
    %alloc = memref.alloc() : memref<288x64xf16>
    %alloc_0 = memref.alloc() : memref<900x64xf32>
    %alloc_1 = memref.alloc() : memref<64xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<64xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %arg3 = %c0 to %c3 step %c1 {
      %c3_3 = arith.constant 3 : index
      scf.for %arg4 = %c0 to %c3_3 step %c1 {
        %c32 = arith.constant 32 : index
        scf.for %arg5 = %c0 to %c32 step %c1 {
          %c64 = arith.constant 64 : index
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %c3_4 = arith.constant 3 : index
            %c32_5 = arith.constant 32 : index
            %0 = arith.muli %arg3, %c3_4 : index
            %1 = arith.muli %0, %c32_5 : index
            %2 = arith.muli %arg4, %c32_5 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<3x3x32x64xf16>
            memref.store %5, %alloc[%4, %arg6] : memref<288x64xf16>
          }
        }
      }
    }
    %c30_i64 = arith.constant 30 : i64
    %c3_i64 = arith.constant 3 : i64
    gemmini.tile_conv %arg0 %alloc %alloc_1 %alloc_0 %c30_i64 %c30_i64 %c3_i64 : memref<1x32x32x32xf16> memref<288x64xf16> memref<64xi32> memref<900x64xf32> i64 i64 i64
    %c1_2 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c1_2 step %c1 {
      %c30 = arith.constant 30 : index
      scf.for %arg4 = %c0 to %c30 step %c1 {
        %c30_3 = arith.constant 30 : index
        scf.for %arg5 = %c0 to %c30_3 step %c1 {
          %c64 = arith.constant 64 : index
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %c30_4 = arith.constant 30 : index
            %0 = arith.muli %arg3, %c30_4 : index
            %1 = arith.muli %0, %c30_4 : index
            %2 = arith.muli %c30_4, %arg4 : index
            %3 = arith.addi %1, %2 : index
            %4 = arith.addi %3, %arg5 : index
            %5 = memref.load %alloc_0[%4, %arg6] : memref<900x64xf32>
            memref.store %5, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<1x30x30x64xf32>
          }
        }
      }
    }
    memref.dealloc %alloc : memref<288x64xf16>
    memref.dealloc %alloc_0 : memref<900x64xf32>
    memref.dealloc %alloc_1 : memref<64xi32>
    return
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @conv2d_block1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr, %arg23: !llvm.ptr, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64) {
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
    %13 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %17 = llvm.insertvalue %arg7, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %18 = llvm.insertvalue %arg4, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %19 = llvm.insertvalue %arg8, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %20 = llvm.insertvalue %arg5, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %21 = llvm.insertvalue %arg9, %20[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %22 = llvm.insertvalue %arg6, %21[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %23 = llvm.insertvalue %arg10, %22[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %25 = llvm.insertvalue %arg11, %24[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %26 = llvm.insertvalue %arg12, %25[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %27 = llvm.insertvalue %arg13, %26[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %28 = llvm.insertvalue %arg14, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %29 = llvm.insertvalue %arg18, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %30 = llvm.insertvalue %arg15, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %31 = llvm.insertvalue %arg19, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %32 = llvm.insertvalue %arg16, %31[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %33 = llvm.insertvalue %arg20, %32[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %34 = llvm.insertvalue %arg17, %33[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %35 = llvm.insertvalue %arg21, %34[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %36 = llvm.mlir.constant(288 : index) : i64
    %37 = llvm.mlir.constant(64 : index) : i64
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mlir.constant(18432 : index) : i64
    %40 = llvm.mlir.zero : !llvm.ptr
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(0 : index) : i64
    %48 = llvm.insertvalue %47, %46[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %36, %48[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %37, %49[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %37, %50[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %38, %51[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.mlir.constant(900 : index) : i64
    %54 = llvm.mlir.constant(64 : index) : i64
    %55 = llvm.mlir.constant(1 : index) : i64
    %56 = llvm.mlir.constant(57600 : index) : i64
    %57 = llvm.mlir.zero : !llvm.ptr
    %58 = llvm.getelementptr %57[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.call @malloc(%59) : (i64) -> !llvm.ptr
    %61 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %60, %62[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mlir.constant(0 : index) : i64
    %65 = llvm.insertvalue %64, %63[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %53, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %54, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %54, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %55, %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.mlir.constant(64 : index) : i64
    %71 = llvm.mlir.constant(1 : index) : i64
    %72 = llvm.mlir.zero : !llvm.ptr
    %73 = llvm.getelementptr %72[%70] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.call @malloc(%74) : (i64) -> !llvm.ptr
    %76 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %77 = llvm.insertvalue %75, %76[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.insertvalue %75, %77[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.mlir.constant(0 : index) : i64
    %80 = llvm.insertvalue %79, %78[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.insertvalue %70, %80[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %82 = llvm.insertvalue %71, %81[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = builtin.unrealized_conversion_cast %82 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<64xi32>
    %84 = llvm.mlir.constant(0 : i32) : i32
    linalg.fill ins(%84 : i32) outs(%83 : memref<64xi32>)
    %85 = llvm.mlir.constant(0 : index) : i64
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb1(%85 : i64)
  ^bb1(%88: i64):  // 2 preds: ^bb0, ^bb11
    %89 = llvm.icmp "slt" %88, %87 : i64
    llvm.cond_br %89, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    %90 = llvm.mlir.constant(3 : index) : i64
    llvm.br ^bb3(%85 : i64)
  ^bb3(%91: i64):  // 2 preds: ^bb2, ^bb10
    %92 = llvm.icmp "slt" %91, %90 : i64
    llvm.cond_br %92, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    %93 = llvm.mlir.constant(32 : index) : i64
    llvm.br ^bb5(%85 : i64)
  ^bb5(%94: i64):  // 2 preds: ^bb4, ^bb9
    %95 = llvm.icmp "slt" %94, %93 : i64
    llvm.cond_br %95, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    %96 = llvm.mlir.constant(64 : index) : i64
    llvm.br ^bb7(%85 : i64)
  ^bb7(%97: i64):  // 2 preds: ^bb6, ^bb8
    %98 = llvm.icmp "slt" %97, %96 : i64
    llvm.cond_br %98, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %99 = llvm.mlir.constant(3 : index) : i64
    %100 = llvm.mlir.constant(32 : index) : i64
    %101 = llvm.mul %88, %99 : i64
    %102 = llvm.mul %101, %100 : i64
    %103 = llvm.mul %91, %100 : i64
    %104 = llvm.add %102, %103 : i64
    %105 = llvm.add %104, %94 : i64
    %106 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %107 = llvm.mlir.constant(6144 : index) : i64
    %108 = llvm.mul %88, %107 : i64
    %109 = llvm.mlir.constant(2048 : index) : i64
    %110 = llvm.mul %91, %109 : i64
    %111 = llvm.add %108, %110 : i64
    %112 = llvm.mlir.constant(64 : index) : i64
    %113 = llvm.mul %94, %112 : i64
    %114 = llvm.add %111, %113 : i64
    %115 = llvm.add %114, %97 : i64
    %116 = llvm.getelementptr %106[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %117 = llvm.load %116 : !llvm.ptr -> f16
    %118 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.mlir.constant(64 : index) : i64
    %120 = llvm.mul %105, %119 : i64
    %121 = llvm.add %120, %97 : i64
    %122 = llvm.getelementptr %118[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %117, %122 : f16, !llvm.ptr
    %123 = llvm.add %97, %86 : i64
    llvm.br ^bb7(%123 : i64)
  ^bb9:  // pred: ^bb7
    %124 = llvm.add %94, %86 : i64
    llvm.br ^bb5(%124 : i64)
  ^bb10:  // pred: ^bb5
    %125 = llvm.add %91, %86 : i64
    llvm.br ^bb3(%125 : i64)
  ^bb11:  // pred: ^bb3
    %126 = llvm.add %88, %86 : i64
    llvm.br ^bb1(%126 : i64)
  ^bb12:  // pred: ^bb1
    %127 = llvm.mlir.constant(30 : i64) : i64
    %128 = llvm.mlir.constant(3 : i64) : i64
    %129 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %130 = llvm.ptrtoint %129 : !llvm.ptr to i64
    %131 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.ptrtoint %131 : !llvm.ptr to i64
    %133 = llvm.extractvalue %82[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %134 = llvm.ptrtoint %133 : !llvm.ptr to i64
    %135 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.ptrtoint %135 : !llvm.ptr to i64
    %137 = llvm.mlir.constant(64 : i64) : i64
    %138 = llvm.mlir.constant(2 : i64) : i64
    %139 = llvm.mlir.constant(4575657221408424000 : i64) : i64
    "gemmini.intr.config_st"(%138, %139) : (i64, i64) -> ()
    %140 = llvm.mlir.constant(65540 : i64) : i64
    %141 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%140, %141) : (i64, i64) -> ()
    %142 = llvm.mlir.constant(0 : i64) : i64
    %143 = llvm.mlir.constant(0 : i64) : i64
    %144 = llvm.mlir.constant(0 : i64) : i64
    %145 = llvm.mlir.constant(0 : i64) : i64
    %146 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %147 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%146, %147) : (i64, i64) -> ()
    %148 = llvm.mlir.constant(844429225164800 : i64) : i64
    %149 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%148, %149) : (i64, i64) -> ()
    %150 = llvm.mlir.constant(844437817131008 : i64) : i64
    %151 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%150, %151) : (i64, i64) -> ()
    %152 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %153 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%152, %153) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%136, %132) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%134, %130) : (i64, i64) -> ()
    %154 = llvm.mlir.constant(256 : i64) : i64
    %155 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%154, %155) : (i64, i64) -> ()
    %156 = llvm.mlir.constant(16 : i64) : i64
    %157 = llvm.add %132, %156 : i64
    %158 = llvm.mlir.constant(64 : i64) : i64
    %159 = llvm.add %134, %158 : i64
    %160 = llvm.mlir.constant(16 : i64) : i64
    %161 = llvm.add %136, %160 : i64
    %162 = llvm.mlir.constant(0 : i64) : i64
    %163 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %164 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%163, %164) : (i64, i64) -> ()
    %165 = llvm.mlir.constant(844429225164800 : i64) : i64
    %166 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%165, %166) : (i64, i64) -> ()
    %167 = llvm.mlir.constant(844437817131008 : i64) : i64
    %168 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%167, %168) : (i64, i64) -> ()
    %169 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %170 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%169, %170) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%161, %157) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%159, %130) : (i64, i64) -> ()
    %171 = llvm.mlir.constant(256 : i64) : i64
    %172 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%171, %172) : (i64, i64) -> ()
    %173 = llvm.mlir.constant(32 : i64) : i64
    %174 = llvm.add %132, %173 : i64
    %175 = llvm.mlir.constant(128 : i64) : i64
    %176 = llvm.add %134, %175 : i64
    %177 = llvm.mlir.constant(32 : i64) : i64
    %178 = llvm.add %136, %177 : i64
    %179 = llvm.mlir.constant(0 : i64) : i64
    %180 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %181 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%180, %181) : (i64, i64) -> ()
    %182 = llvm.mlir.constant(844429225164800 : i64) : i64
    %183 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%182, %183) : (i64, i64) -> ()
    %184 = llvm.mlir.constant(844437817131008 : i64) : i64
    %185 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%184, %185) : (i64, i64) -> ()
    %186 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %187 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%186, %187) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%178, %174) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%176, %130) : (i64, i64) -> ()
    %188 = llvm.mlir.constant(256 : i64) : i64
    %189 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%188, %189) : (i64, i64) -> ()
    %190 = llvm.mlir.constant(48 : i64) : i64
    %191 = llvm.add %132, %190 : i64
    %192 = llvm.mlir.constant(192 : i64) : i64
    %193 = llvm.add %134, %192 : i64
    %194 = llvm.mlir.constant(48 : i64) : i64
    %195 = llvm.add %136, %194 : i64
    %196 = llvm.mlir.constant(0 : i64) : i64
    %197 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %198 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%197, %198) : (i64, i64) -> ()
    %199 = llvm.mlir.constant(844429225164800 : i64) : i64
    %200 = llvm.mlir.constant(281569467498512 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%199, %200) : (i64, i64) -> ()
    %201 = llvm.mlir.constant(844437817131008 : i64) : i64
    %202 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%201, %202) : (i64, i64) -> ()
    %203 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %204 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%203, %204) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%195, %191) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%193, %130) : (i64, i64) -> ()
    %205 = llvm.mlir.constant(256 : i64) : i64
    %206 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%205, %206) : (i64, i64) -> ()
    %207 = llvm.mlir.constant(1472 : i64) : i64
    %208 = llvm.add %132, %207 : i64
    %209 = llvm.mlir.constant(0 : i64) : i64
    %210 = llvm.mlir.constant(0 : i64) : i64
    %211 = llvm.mlir.constant(736 : i64) : i64
    %212 = llvm.add %130, %211 : i64
    %213 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %214 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%213, %214) : (i64, i64) -> ()
    %215 = llvm.mlir.constant(844429225164800 : i64) : i64
    %216 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%215, %216) : (i64, i64) -> ()
    %217 = llvm.mlir.constant(844437817131008 : i64) : i64
    %218 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%217, %218) : (i64, i64) -> ()
    %219 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %220 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%219, %220) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%136, %208) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%134, %212) : (i64, i64) -> ()
    %221 = llvm.mlir.constant(256 : i64) : i64
    %222 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%221, %222) : (i64, i64) -> ()
    %223 = llvm.mlir.constant(1488 : i64) : i64
    %224 = llvm.add %132, %223 : i64
    %225 = llvm.mlir.constant(64 : i64) : i64
    %226 = llvm.add %134, %225 : i64
    %227 = llvm.mlir.constant(16 : i64) : i64
    %228 = llvm.add %136, %227 : i64
    %229 = llvm.mlir.constant(736 : i64) : i64
    %230 = llvm.add %130, %229 : i64
    %231 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %232 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%231, %232) : (i64, i64) -> ()
    %233 = llvm.mlir.constant(844429225164800 : i64) : i64
    %234 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%233, %234) : (i64, i64) -> ()
    %235 = llvm.mlir.constant(844437817131008 : i64) : i64
    %236 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%235, %236) : (i64, i64) -> ()
    %237 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %238 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%237, %238) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%228, %224) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%226, %230) : (i64, i64) -> ()
    %239 = llvm.mlir.constant(256 : i64) : i64
    %240 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%239, %240) : (i64, i64) -> ()
    %241 = llvm.mlir.constant(1504 : i64) : i64
    %242 = llvm.add %132, %241 : i64
    %243 = llvm.mlir.constant(128 : i64) : i64
    %244 = llvm.add %134, %243 : i64
    %245 = llvm.mlir.constant(32 : i64) : i64
    %246 = llvm.add %136, %245 : i64
    %247 = llvm.mlir.constant(736 : i64) : i64
    %248 = llvm.add %130, %247 : i64
    %249 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %250 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%249, %250) : (i64, i64) -> ()
    %251 = llvm.mlir.constant(844429225164800 : i64) : i64
    %252 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%251, %252) : (i64, i64) -> ()
    %253 = llvm.mlir.constant(844437817131008 : i64) : i64
    %254 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%253, %254) : (i64, i64) -> ()
    %255 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %256 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%255, %256) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%246, %242) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%244, %248) : (i64, i64) -> ()
    %257 = llvm.mlir.constant(256 : i64) : i64
    %258 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%257, %258) : (i64, i64) -> ()
    %259 = llvm.mlir.constant(1520 : i64) : i64
    %260 = llvm.add %132, %259 : i64
    %261 = llvm.mlir.constant(192 : i64) : i64
    %262 = llvm.add %134, %261 : i64
    %263 = llvm.mlir.constant(48 : i64) : i64
    %264 = llvm.add %136, %263 : i64
    %265 = llvm.mlir.constant(736 : i64) : i64
    %266 = llvm.add %130, %265 : i64
    %267 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %268 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%267, %268) : (i64, i64) -> ()
    %269 = llvm.mlir.constant(844429225164800 : i64) : i64
    %270 = llvm.mlir.constant(281569466449936 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%269, %270) : (i64, i64) -> ()
    %271 = llvm.mlir.constant(844437817131008 : i64) : i64
    %272 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%271, %272) : (i64, i64) -> ()
    %273 = llvm.mlir.constant(6192449487634432 : i64) : i64
    %274 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%273, %274) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%264, %260) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%262, %266) : (i64, i64) -> ()
    %275 = llvm.mlir.constant(256 : i64) : i64
    %276 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%275, %276) : (i64, i64) -> ()
    %277 = llvm.mlir.constant(42240 : i64) : i64
    %278 = llvm.add %132, %277 : i64
    %279 = llvm.mlir.constant(0 : i64) : i64
    %280 = llvm.mlir.constant(0 : i64) : i64
    %281 = llvm.mlir.constant(22528 : i64) : i64
    %282 = llvm.add %130, %281 : i64
    %283 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %284 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%283, %284) : (i64, i64) -> ()
    %285 = llvm.mlir.constant(844429225164800 : i64) : i64
    %286 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%285, %286) : (i64, i64) -> ()
    %287 = llvm.mlir.constant(844437817131008 : i64) : i64
    %288 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%287, %288) : (i64, i64) -> ()
    %289 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %290 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%289, %290) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%136, %278) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%134, %282) : (i64, i64) -> ()
    %291 = llvm.mlir.constant(256 : i64) : i64
    %292 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%291, %292) : (i64, i64) -> ()
    %293 = llvm.mlir.constant(42256 : i64) : i64
    %294 = llvm.add %132, %293 : i64
    %295 = llvm.mlir.constant(64 : i64) : i64
    %296 = llvm.add %134, %295 : i64
    %297 = llvm.mlir.constant(16 : i64) : i64
    %298 = llvm.add %136, %297 : i64
    %299 = llvm.mlir.constant(22528 : i64) : i64
    %300 = llvm.add %130, %299 : i64
    %301 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %302 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%301, %302) : (i64, i64) -> ()
    %303 = llvm.mlir.constant(844429225164800 : i64) : i64
    %304 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%303, %304) : (i64, i64) -> ()
    %305 = llvm.mlir.constant(844437817131008 : i64) : i64
    %306 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%305, %306) : (i64, i64) -> ()
    %307 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %308 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%307, %308) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%298, %294) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%296, %300) : (i64, i64) -> ()
    %309 = llvm.mlir.constant(256 : i64) : i64
    %310 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%309, %310) : (i64, i64) -> ()
    %311 = llvm.mlir.constant(42272 : i64) : i64
    %312 = llvm.add %132, %311 : i64
    %313 = llvm.mlir.constant(128 : i64) : i64
    %314 = llvm.add %134, %313 : i64
    %315 = llvm.mlir.constant(32 : i64) : i64
    %316 = llvm.add %136, %315 : i64
    %317 = llvm.mlir.constant(22528 : i64) : i64
    %318 = llvm.add %130, %317 : i64
    %319 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %320 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%319, %320) : (i64, i64) -> ()
    %321 = llvm.mlir.constant(844429225164800 : i64) : i64
    %322 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%321, %322) : (i64, i64) -> ()
    %323 = llvm.mlir.constant(844437817131008 : i64) : i64
    %324 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%323, %324) : (i64, i64) -> ()
    %325 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %326 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%325, %326) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%316, %312) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%314, %318) : (i64, i64) -> ()
    %327 = llvm.mlir.constant(256 : i64) : i64
    %328 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%327, %328) : (i64, i64) -> ()
    %329 = llvm.mlir.constant(42288 : i64) : i64
    %330 = llvm.add %132, %329 : i64
    %331 = llvm.mlir.constant(192 : i64) : i64
    %332 = llvm.add %134, %331 : i64
    %333 = llvm.mlir.constant(48 : i64) : i64
    %334 = llvm.add %136, %333 : i64
    %335 = llvm.mlir.constant(22528 : i64) : i64
    %336 = llvm.add %130, %335 : i64
    %337 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %338 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%337, %338) : (i64, i64) -> ()
    %339 = llvm.mlir.constant(844429225164800 : i64) : i64
    %340 = llvm.mlir.constant(281509337956368 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%339, %340) : (i64, i64) -> ()
    %341 = llvm.mlir.constant(844437817131008 : i64) : i64
    %342 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%341, %342) : (i64, i64) -> ()
    %343 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %344 = llvm.mlir.constant(65559 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%343, %344) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%334, %330) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%332, %336) : (i64, i64) -> ()
    %345 = llvm.mlir.constant(256 : i64) : i64
    %346 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%345, %346) : (i64, i64) -> ()
    %347 = llvm.mlir.constant(43712 : i64) : i64
    %348 = llvm.add %132, %347 : i64
    %349 = llvm.mlir.constant(0 : i64) : i64
    %350 = llvm.mlir.constant(0 : i64) : i64
    %351 = llvm.mlir.constant(23264 : i64) : i64
    %352 = llvm.add %130, %351 : i64
    %353 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %354 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%353, %354) : (i64, i64) -> ()
    %355 = llvm.mlir.constant(844429225164800 : i64) : i64
    %356 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%355, %356) : (i64, i64) -> ()
    %357 = llvm.mlir.constant(844437817131008 : i64) : i64
    %358 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%357, %358) : (i64, i64) -> ()
    %359 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %360 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%359, %360) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%136, %348) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%134, %352) : (i64, i64) -> ()
    %361 = llvm.mlir.constant(256 : i64) : i64
    %362 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%361, %362) : (i64, i64) -> ()
    %363 = llvm.mlir.constant(43728 : i64) : i64
    %364 = llvm.add %132, %363 : i64
    %365 = llvm.mlir.constant(64 : i64) : i64
    %366 = llvm.add %134, %365 : i64
    %367 = llvm.mlir.constant(16 : i64) : i64
    %368 = llvm.add %136, %367 : i64
    %369 = llvm.mlir.constant(23264 : i64) : i64
    %370 = llvm.add %130, %369 : i64
    %371 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %372 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%371, %372) : (i64, i64) -> ()
    %373 = llvm.mlir.constant(844429225164800 : i64) : i64
    %374 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%373, %374) : (i64, i64) -> ()
    %375 = llvm.mlir.constant(844437817131008 : i64) : i64
    %376 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%375, %376) : (i64, i64) -> ()
    %377 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %378 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%377, %378) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%368, %364) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%366, %370) : (i64, i64) -> ()
    %379 = llvm.mlir.constant(256 : i64) : i64
    %380 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%379, %380) : (i64, i64) -> ()
    %381 = llvm.mlir.constant(43744 : i64) : i64
    %382 = llvm.add %132, %381 : i64
    %383 = llvm.mlir.constant(128 : i64) : i64
    %384 = llvm.add %134, %383 : i64
    %385 = llvm.mlir.constant(32 : i64) : i64
    %386 = llvm.add %136, %385 : i64
    %387 = llvm.mlir.constant(23264 : i64) : i64
    %388 = llvm.add %130, %387 : i64
    %389 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %390 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%389, %390) : (i64, i64) -> ()
    %391 = llvm.mlir.constant(844429225164800 : i64) : i64
    %392 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%391, %392) : (i64, i64) -> ()
    %393 = llvm.mlir.constant(844437817131008 : i64) : i64
    %394 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%393, %394) : (i64, i64) -> ()
    %395 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %396 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%395, %396) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%386, %382) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%384, %388) : (i64, i64) -> ()
    %397 = llvm.mlir.constant(256 : i64) : i64
    %398 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%397, %398) : (i64, i64) -> ()
    %399 = llvm.mlir.constant(43760 : i64) : i64
    %400 = llvm.add %132, %399 : i64
    %401 = llvm.mlir.constant(192 : i64) : i64
    %402 = llvm.add %134, %401 : i64
    %403 = llvm.mlir.constant(48 : i64) : i64
    %404 = llvm.add %136, %403 : i64
    %405 = llvm.mlir.constant(23264 : i64) : i64
    %406 = llvm.add %130, %405 : i64
    %407 = llvm.mlir.constant(18014535950532609 : i64) : i64
    %408 = llvm.mlir.constant(4296933406 : i64) : i64
    "gemmini.intr.loop_conv_ws_config1"(%407, %408) : (i64, i64) -> ()
    %409 = llvm.mlir.constant(844429225164800 : i64) : i64
    %410 = llvm.mlir.constant(281509336907792 : i64) : i64
    "gemmini.intr.loop_conv_ws_config2"(%409, %410) : (i64, i64) -> ()
    %411 = llvm.mlir.constant(844437817131008 : i64) : i64
    %412 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_conv_ws_config3"(%411, %412) : (i64, i64) -> ()
    %413 = llvm.mlir.constant(2251799813685248 : i64) : i64
    %414 = llvm.mlir.constant(65543 : i64) : i64
    "gemmini.intr.loop_conv_ws_config4"(%413, %414) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config5"(%404, %400) : (i64, i64) -> ()
    "gemmini.intr.loop_conv_ws_config6"(%402, %406) : (i64, i64) -> ()
    %415 = llvm.mlir.constant(256 : i64) : i64
    %416 = llvm.mlir.constant(1 : i64) : i64
    "gemmini.intr.loop_conv_ws"(%415, %416) : (i64, i64) -> ()
    %417 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%417, %417) : (i64, i64) -> ()
    %418 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%85 : i64)
  ^bb13(%419: i64):  // 2 preds: ^bb12, ^bb23
    %420 = llvm.icmp "slt" %419, %418 : i64
    llvm.cond_br %420, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    %421 = llvm.mlir.constant(30 : index) : i64
    llvm.br ^bb15(%85 : i64)
  ^bb15(%422: i64):  // 2 preds: ^bb14, ^bb22
    %423 = llvm.icmp "slt" %422, %421 : i64
    llvm.cond_br %423, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    %424 = llvm.mlir.constant(30 : index) : i64
    llvm.br ^bb17(%85 : i64)
  ^bb17(%425: i64):  // 2 preds: ^bb16, ^bb21
    %426 = llvm.icmp "slt" %425, %424 : i64
    llvm.cond_br %426, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    %427 = llvm.mlir.constant(64 : index) : i64
    llvm.br ^bb19(%85 : i64)
  ^bb19(%428: i64):  // 2 preds: ^bb18, ^bb20
    %429 = llvm.icmp "slt" %428, %427 : i64
    llvm.cond_br %429, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %430 = llvm.mlir.constant(30 : index) : i64
    %431 = llvm.mul %419, %430 : i64
    %432 = llvm.mul %431, %430 : i64
    %433 = llvm.mul %422, %430 : i64
    %434 = llvm.add %432, %433 : i64
    %435 = llvm.add %434, %425 : i64
    %436 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %437 = llvm.mlir.constant(64 : index) : i64
    %438 = llvm.mul %435, %437 : i64
    %439 = llvm.add %438, %428 : i64
    %440 = llvm.getelementptr %436[%439] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %441 = llvm.load %440 : !llvm.ptr -> f32
    %442 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %443 = llvm.mlir.constant(57600 : index) : i64
    %444 = llvm.mul %419, %443 : i64
    %445 = llvm.mlir.constant(1920 : index) : i64
    %446 = llvm.mul %422, %445 : i64
    %447 = llvm.add %444, %446 : i64
    %448 = llvm.mlir.constant(64 : index) : i64
    %449 = llvm.mul %425, %448 : i64
    %450 = llvm.add %447, %449 : i64
    %451 = llvm.add %450, %428 : i64
    %452 = llvm.getelementptr %442[%451] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %441, %452 : f32, !llvm.ptr
    %453 = llvm.add %428, %86 : i64
    llvm.br ^bb19(%453 : i64)
  ^bb21:  // pred: ^bb19
    %454 = llvm.add %425, %86 : i64
    llvm.br ^bb17(%454 : i64)
  ^bb22:  // pred: ^bb17
    %455 = llvm.add %422, %86 : i64
    llvm.br ^bb15(%455 : i64)
  ^bb23:  // pred: ^bb15
    %456 = llvm.add %419, %86 : i64
    llvm.br ^bb13(%456 : i64)
  ^bb24:  // pred: ^bb13
    %457 = llvm.extractvalue %52[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%457) : (!llvm.ptr) -> ()
    %458 = llvm.extractvalue %69[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%458) : (!llvm.ptr) -> ()
    %459 = llvm.extractvalue %82[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%459) : (!llvm.ptr) -> ()
    llvm.return
  }
}


