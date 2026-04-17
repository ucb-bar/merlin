// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  func.func @main() -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c2_i8 = arith.constant 2 : i8
    %alloc = memref.alloc() : memref<3x3x3xi8>
    %alloc_0 = memref.alloc() : memref<3x3x3xi8>
    %alloc_1 = memref.alloc() : memref<3x3x3xi8>
    linalg.fill ins(%c1_i8 : i8) outs(%alloc : memref<3x3x3xi8>)
    linalg.fill ins(%c2_i8 : i8) outs(%alloc_0 : memref<3x3x3xi8>)
    %subview = memref.subview %alloc[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %subview_2 = memref.subview %alloc_0[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %subview_3 = memref.subview %alloc_1[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %alloc_4 = memref.alloc() : memref<3x3xi32>
    %c0_i32 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_4 : memref<3x3xi32>)
    gemmini.tile_matmul %subview %subview_2 %subview_3 %alloc_4 : memref<3x3xi8, strided<[3, 1]>> memref<3x3xi8, strided<[3, 1]>> memref<3x3xi8, strided<[3, 1]>> memref<3x3xi32>
    memref.dealloc %alloc_4 : memref<3x3xi32>
    %subview_5 = memref.subview %alloc[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %subview_6 = memref.subview %alloc_0[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %subview_7 = memref.subview %alloc_1[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %alloc_8 = memref.alloc() : memref<3x3xi32>
    %c0_i32_9 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32_9 : i32) outs(%alloc_8 : memref<3x3xi32>)
    gemmini.tile_matmul %subview_5 %subview_6 %subview_7 %alloc_8 : memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi8, strided<[3, 1], offset: 9>> memref<3x3xi32>
    memref.dealloc %alloc_8 : memref<3x3xi32>
    %subview_10 = memref.subview %alloc[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %subview_11 = memref.subview %alloc_0[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %subview_12 = memref.subview %alloc_1[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %alloc_13 = memref.alloc() : memref<3x3xi32>
    %c0_i32_14 = arith.constant 0 : i32
    linalg.fill ins(%c0_i32_14 : i32) outs(%alloc_13 : memref<3x3xi32>)
    gemmini.tile_matmul %subview_10 %subview_11 %subview_12 %alloc_13 : memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi8, strided<[3, 1], offset: 18>> memref<3x3xi32>
    memref.dealloc %alloc_13 : memref<3x3xi32>
    gemmini.print %alloc_1 : memref<3x3x3xi8>
    memref.dealloc %alloc_1 : memref<3x3x3xi8>
    return %c0_i8 : i8
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%d \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() -> i8 {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(1 : i8) : i8
    %2 = llvm.mlir.constant(2 : i8) : i8
    %3 = llvm.mlir.constant(3 : index) : i64
    %4 = llvm.mlir.constant(3 : index) : i64
    %5 = llvm.mlir.constant(3 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(9 : index) : i64
    %8 = llvm.mlir.constant(27 : index) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.getelementptr %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %3, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %4, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %5, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %7, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %5, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %6, %22[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = builtin.unrealized_conversion_cast %23 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x3x3xi8>
    %25 = llvm.mlir.constant(3 : index) : i64
    %26 = llvm.mlir.constant(3 : index) : i64
    %27 = llvm.mlir.constant(3 : index) : i64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.mlir.constant(9 : index) : i64
    %30 = llvm.mlir.constant(27 : index) : i64
    %31 = llvm.mlir.zero : !llvm.ptr
    %32 = llvm.getelementptr %31[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.call @malloc(%33) : (i64) -> !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.insertvalue %25, %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %26, %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %27, %41[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %29, %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %27, %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %28, %44[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = builtin.unrealized_conversion_cast %45 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x3x3xi8>
    %47 = llvm.mlir.constant(3 : index) : i64
    %48 = llvm.mlir.constant(3 : index) : i64
    %49 = llvm.mlir.constant(3 : index) : i64
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.mlir.constant(9 : index) : i64
    %52 = llvm.mlir.constant(27 : index) : i64
    %53 = llvm.mlir.zero : !llvm.ptr
    %54 = llvm.getelementptr %53[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.call @malloc(%55) : (i64) -> !llvm.ptr
    %57 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %58 = llvm.insertvalue %56, %57[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %59 = llvm.insertvalue %56, %58[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.insertvalue %60, %59[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.insertvalue %47, %61[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.insertvalue %48, %62[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %49, %63[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %51, %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %49, %65[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %50, %66[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = builtin.unrealized_conversion_cast %67 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<3x3x3xi8>
    linalg.fill ins(%1 : i8) outs(%24 : memref<3x3x3xi8>)
    linalg.fill ins(%2 : i8) outs(%46 : memref<3x3x3xi8>)
    %subview = memref.subview %24[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %69 = builtin.unrealized_conversion_cast %subview : memref<3x3xi8, strided<[3, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_0 = memref.subview %46[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %70 = builtin.unrealized_conversion_cast %subview_0 : memref<3x3xi8, strided<[3, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_1 = memref.subview %68[0, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1]>>
    %71 = builtin.unrealized_conversion_cast %subview_1 : memref<3x3xi8, strided<[3, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.mlir.constant(3 : index) : i64
    %73 = llvm.mlir.constant(3 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.mlir.constant(9 : index) : i64
    %76 = llvm.mlir.zero : !llvm.ptr
    %77 = llvm.getelementptr %76[%75] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.call @malloc(%78) : (i64) -> !llvm.ptr
    %80 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %79, %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.mlir.constant(0 : index) : i64
    %84 = llvm.insertvalue %83, %82[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %72, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.insertvalue %73, %85[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %73, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %74, %87[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = builtin.unrealized_conversion_cast %88 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x3xi32>
    %90 = llvm.mlir.constant(0 : i32) : i32
    linalg.fill ins(%90 : i32) outs(%89 : memref<3x3xi32>)
    %91 = llvm.extractvalue %69[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.mlir.constant(0 : index) : i64
    %94 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.mlir.constant(0 : index) : i64
    %97 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
    %99 = llvm.mlir.constant(0 : index) : i64
    %100 = llvm.extractvalue %88[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %103 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%102, %103) : (i64, i64) -> ()
    %104 = llvm.mlir.constant(3 : i64) : i64
    %105 = llvm.mlir.constant(2 : i64) : i64
    %106 = llvm.mlir.constant(4575657221408423939 : i64) : i64
    "gemmini.intr.config_st"(%105, %106) : (i64, i64) -> ()
    %107 = llvm.mlir.constant(3 : i64) : i64
    %108 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%108, %107) : (i64, i64) -> ()
    %109 = llvm.mlir.constant(3 : i64) : i64
    %110 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%110, %109) : (i64, i64) -> ()
    %111 = llvm.mlir.constant(12 : i64) : i64
    %112 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%112, %111) : (i64, i64) -> ()
    %113 = llvm.mlir.constant(0 : i64) : i64
    %114 = llvm.mlir.constant(0 : i64) : i64
    %115 = llvm.mlir.constant(0 : i64) : i64
    %116 = llvm.mlir.constant(0 : i64) : i64
    %117 = llvm.mlir.constant(55835426829 : i64) : i64
    %118 = llvm.mlir.constant(4295032833 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%117, %118) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%92, %95) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%101, %98) : (i64, i64) -> ()
    %119 = llvm.mlir.constant(3 : i64) : i64
    %120 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%119, %120) : (i64, i64) -> ()
    %121 = llvm.mlir.constant(3 : i64) : i64
    %122 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%121, %122) : (i64, i64) -> ()
    %123 = llvm.mlir.constant(1 : i64) : i64
    %124 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%123, %124) : (i64, i64) -> ()
    %125 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%125, %125) : (i64, i64) -> ()
    %126 = llvm.extractvalue %88[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%126) : (!llvm.ptr) -> ()
    %subview_2 = memref.subview %24[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %127 = builtin.unrealized_conversion_cast %subview_2 : memref<3x3xi8, strided<[3, 1], offset: 9>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_3 = memref.subview %46[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %128 = builtin.unrealized_conversion_cast %subview_3 : memref<3x3xi8, strided<[3, 1], offset: 9>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_4 = memref.subview %68[1, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 9>>
    %129 = builtin.unrealized_conversion_cast %subview_4 : memref<3x3xi8, strided<[3, 1], offset: 9>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.mlir.constant(3 : index) : i64
    %131 = llvm.mlir.constant(3 : index) : i64
    %132 = llvm.mlir.constant(1 : index) : i64
    %133 = llvm.mlir.constant(9 : index) : i64
    %134 = llvm.mlir.zero : !llvm.ptr
    %135 = llvm.getelementptr %134[%133] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %136 = llvm.ptrtoint %135 : !llvm.ptr to i64
    %137 = llvm.call @malloc(%136) : (i64) -> !llvm.ptr
    %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %139 = llvm.insertvalue %137, %138[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.insertvalue %130, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.insertvalue %131, %143[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %131, %144[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %132, %145[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = builtin.unrealized_conversion_cast %146 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x3xi32>
    %148 = llvm.mlir.constant(0 : i32) : i32
    linalg.fill ins(%148 : i32) outs(%147 : memref<3x3xi32>)
    %149 = llvm.extractvalue %127[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(9 : index) : i64
    %152 = llvm.add %150, %151 : i64
    %153 = llvm.extractvalue %128[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.ptrtoint %153 : !llvm.ptr to i64
    %155 = llvm.mlir.constant(9 : index) : i64
    %156 = llvm.add %154, %155 : i64
    %157 = llvm.extractvalue %129[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.ptrtoint %157 : !llvm.ptr to i64
    %159 = llvm.mlir.constant(9 : index) : i64
    %160 = llvm.add %158, %159 : i64
    %161 = llvm.extractvalue %146[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.ptrtoint %161 : !llvm.ptr to i64
    %163 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %164 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%163, %164) : (i64, i64) -> ()
    %165 = llvm.mlir.constant(3 : i64) : i64
    %166 = llvm.mlir.constant(2 : i64) : i64
    %167 = llvm.mlir.constant(4575657221408423939 : i64) : i64
    "gemmini.intr.config_st"(%166, %167) : (i64, i64) -> ()
    %168 = llvm.mlir.constant(3 : i64) : i64
    %169 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%169, %168) : (i64, i64) -> ()
    %170 = llvm.mlir.constant(3 : i64) : i64
    %171 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%171, %170) : (i64, i64) -> ()
    %172 = llvm.mlir.constant(12 : i64) : i64
    %173 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%173, %172) : (i64, i64) -> ()
    %174 = llvm.mlir.constant(0 : i64) : i64
    %175 = llvm.mlir.constant(0 : i64) : i64
    %176 = llvm.mlir.constant(0 : i64) : i64
    %177 = llvm.mlir.constant(0 : i64) : i64
    %178 = llvm.mlir.constant(55835426829 : i64) : i64
    %179 = llvm.mlir.constant(4295032833 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%178, %179) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%152, %156) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%162, %160) : (i64, i64) -> ()
    %180 = llvm.mlir.constant(3 : i64) : i64
    %181 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%180, %181) : (i64, i64) -> ()
    %182 = llvm.mlir.constant(3 : i64) : i64
    %183 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%182, %183) : (i64, i64) -> ()
    %184 = llvm.mlir.constant(1 : i64) : i64
    %185 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%184, %185) : (i64, i64) -> ()
    %186 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%186, %186) : (i64, i64) -> ()
    %187 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%187) : (!llvm.ptr) -> ()
    %subview_5 = memref.subview %24[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %188 = builtin.unrealized_conversion_cast %subview_5 : memref<3x3xi8, strided<[3, 1], offset: 18>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_6 = memref.subview %46[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %189 = builtin.unrealized_conversion_cast %subview_6 : memref<3x3xi8, strided<[3, 1], offset: 18>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_7 = memref.subview %68[2, 0, 0] [1, 3, 3] [1, 1, 1] : memref<3x3x3xi8> to memref<3x3xi8, strided<[3, 1], offset: 18>>
    %190 = builtin.unrealized_conversion_cast %subview_7 : memref<3x3xi8, strided<[3, 1], offset: 18>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %191 = llvm.mlir.constant(3 : index) : i64
    %192 = llvm.mlir.constant(3 : index) : i64
    %193 = llvm.mlir.constant(1 : index) : i64
    %194 = llvm.mlir.constant(9 : index) : i64
    %195 = llvm.mlir.zero : !llvm.ptr
    %196 = llvm.getelementptr %195[%194] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %197 = llvm.ptrtoint %196 : !llvm.ptr to i64
    %198 = llvm.call @malloc(%197) : (i64) -> !llvm.ptr
    %199 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %200 = llvm.insertvalue %198, %199[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %201 = llvm.insertvalue %198, %200[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %202 = llvm.mlir.constant(0 : index) : i64
    %203 = llvm.insertvalue %202, %201[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %204 = llvm.insertvalue %191, %203[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %205 = llvm.insertvalue %192, %204[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %206 = llvm.insertvalue %192, %205[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.insertvalue %193, %206[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = builtin.unrealized_conversion_cast %207 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<3x3xi32>
    %209 = llvm.mlir.constant(0 : i32) : i32
    linalg.fill ins(%209 : i32) outs(%208 : memref<3x3xi32>)
    %210 = llvm.extractvalue %188[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %211 = llvm.ptrtoint %210 : !llvm.ptr to i64
    %212 = llvm.mlir.constant(18 : index) : i64
    %213 = llvm.add %211, %212 : i64
    %214 = llvm.extractvalue %189[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %215 = llvm.ptrtoint %214 : !llvm.ptr to i64
    %216 = llvm.mlir.constant(18 : index) : i64
    %217 = llvm.add %215, %216 : i64
    %218 = llvm.extractvalue %190[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %219 = llvm.ptrtoint %218 : !llvm.ptr to i64
    %220 = llvm.mlir.constant(18 : index) : i64
    %221 = llvm.add %219, %220 : i64
    %222 = llvm.extractvalue %207[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.ptrtoint %222 : !llvm.ptr to i64
    %224 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %225 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%224, %225) : (i64, i64) -> ()
    %226 = llvm.mlir.constant(3 : i64) : i64
    %227 = llvm.mlir.constant(2 : i64) : i64
    %228 = llvm.mlir.constant(4575657221408423939 : i64) : i64
    "gemmini.intr.config_st"(%227, %228) : (i64, i64) -> ()
    %229 = llvm.mlir.constant(3 : i64) : i64
    %230 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%230, %229) : (i64, i64) -> ()
    %231 = llvm.mlir.constant(3 : i64) : i64
    %232 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%232, %231) : (i64, i64) -> ()
    %233 = llvm.mlir.constant(12 : i64) : i64
    %234 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%234, %233) : (i64, i64) -> ()
    %235 = llvm.mlir.constant(0 : i64) : i64
    %236 = llvm.mlir.constant(0 : i64) : i64
    %237 = llvm.mlir.constant(0 : i64) : i64
    %238 = llvm.mlir.constant(0 : i64) : i64
    %239 = llvm.mlir.constant(55835426829 : i64) : i64
    %240 = llvm.mlir.constant(4295032833 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%239, %240) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%213, %217) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%223, %221) : (i64, i64) -> ()
    %241 = llvm.mlir.constant(3 : i64) : i64
    %242 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%241, %242) : (i64, i64) -> ()
    %243 = llvm.mlir.constant(3 : i64) : i64
    %244 = llvm.mlir.constant(3 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%243, %244) : (i64, i64) -> ()
    %245 = llvm.mlir.constant(1 : i64) : i64
    %246 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%245, %246) : (i64, i64) -> ()
    %247 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%247, %247) : (i64, i64) -> ()
    %248 = llvm.extractvalue %207[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%248) : (!llvm.ptr) -> ()
    %249 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %250 = llvm.mlir.constant(0 : index) : i64
    %251 = llvm.getelementptr %249[%250, %250] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %252 = llvm.mlir.addressof @nl : !llvm.ptr
    %253 = llvm.mlir.constant(0 : index) : i64
    %254 = llvm.getelementptr %252[%253, %253] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %255 = llvm.mlir.constant(0 : index) : i64
    %256 = llvm.mlir.constant(3 : index) : i64
    %257 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%255 : i64)
  ^bb1(%258: i64):  // 2 preds: ^bb0, ^bb8
    %259 = llvm.icmp "slt" %258, %256 : i64
    llvm.cond_br %259, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %260 = llvm.mlir.constant(0 : index) : i64
    %261 = llvm.mlir.constant(3 : index) : i64
    %262 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%260 : i64)
  ^bb3(%263: i64):  // 2 preds: ^bb2, ^bb7
    %264 = llvm.icmp "slt" %263, %261 : i64
    llvm.cond_br %264, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %265 = llvm.mlir.constant(0 : index) : i64
    %266 = llvm.mlir.constant(3 : index) : i64
    %267 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%265 : i64)
  ^bb5(%268: i64):  // 2 preds: ^bb4, ^bb6
    %269 = llvm.icmp "slt" %268, %266 : i64
    llvm.cond_br %269, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %270 = llvm.extractvalue %67[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %271 = llvm.mlir.constant(9 : index) : i64
    %272 = llvm.mul %258, %271 : i64
    %273 = llvm.mlir.constant(3 : index) : i64
    %274 = llvm.mul %263, %273 : i64
    %275 = llvm.add %272, %274 : i64
    %276 = llvm.add %275, %268 : i64
    %277 = llvm.getelementptr %270[%276] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %278 = llvm.load %277 : !llvm.ptr -> i8
    %279 = llvm.sext %278 : i8 to i32
    %280 = llvm.call @printf(%251, %279) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %281 = llvm.add %268, %267 : i64
    llvm.br ^bb5(%281 : i64)
  ^bb7:  // pred: ^bb5
    %282 = llvm.call @printf(%254) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %283 = llvm.add %263, %262 : i64
    llvm.br ^bb3(%283 : i64)
  ^bb8:  // pred: ^bb3
    %284 = llvm.call @printf(%254) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %285 = llvm.add %258, %257 : i64
    llvm.br ^bb1(%285 : i64)
  ^bb9:  // pred: ^bb1
    %286 = llvm.extractvalue %67[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @free(%286) : (!llvm.ptr) -> ()
    llvm.return %0 : i8
  }
}


