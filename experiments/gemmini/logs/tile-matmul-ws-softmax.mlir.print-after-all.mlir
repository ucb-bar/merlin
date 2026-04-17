// -----// IR Dump After (anonymous namespace)::LowerLinalgToGemminiPass (convert-linalg-to-gemmini) //----- //
module {
  memref.global "private" @g1 : memref<5x5xi8> = dense<[[1, 0, 0, 1, 0], [1, -1, 1, 0, 0], [-1, 0, 1, -1, 1], [1, 0, 0, 1, 0], [-1, 0, 0, -1, 0]]>
  memref.global "private" @g2 : memref<5x5xi8> = dense<[[1, -1, 0, 0, 1], [1, 0, -1, 0, -1], [-1, -1, 0, -1, 1], [-1, 0, 0, 1, 0], [1, 0, 0, -1, 0]]>
  func.func @main() -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c-2_i8 = arith.constant -2 : i8
    %c2_i8 = arith.constant 2 : i8
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @g1 : memref<5x5xi8>
    %1 = memref.get_global @g2 : memref<5x5xi8>
    %alloc = memref.alloc() : memref<5x5xi8>
    %alloc_0 = memref.alloc() : memref<5x5xi32>
    %dim = memref.dim %0, %c0 : memref<5x5xi8>
    %dim_1 = memref.dim %1, %c1 : memref<5x5xi8>
    %dim_2 = memref.dim %0, %c1 : memref<5x5xi8>
    scf.for %arg0 = %c0 to %dim step %c1 {
      scf.for %arg1 = %c0 to %dim_1 step %c1 {
        memref.store %c0_i32, %alloc_0[%arg0, %arg1] : memref<5x5xi32>
      }
    }
    gemmini.tile_matmul %0 %1 %alloc %alloc_0 : memref<5x5xi8> memref<5x5xi8> memref<5x5xi8> memref<5x5xi32>
    gemmini.print %alloc : memref<5x5xi8>
    gemmini.tile_matmul %0 %1 %alloc %alloc_0 {act = 4 : i64, bertScale = 5.000000e-02 : f32} : memref<5x5xi8> memref<5x5xi8> memref<5x5xi8> memref<5x5xi32>
    gemmini.print %alloc : memref<5x5xi8>
    return %c0_i8 : i8
  }
}


// -----// IR Dump After (anonymous namespace)::LowerGemminiToLLVMPass (lower-gemmini) //----- //
module {
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%d \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private @g1(dense<[[1, 0, 0, 1, 0], [1, -1, 1, 0, 0], [-1, 0, 1, -1, 1], [1, 0, 0, 1, 0], [-1, 0, 0, -1, 0]]> : tensor<5x5xi8>) {addr_space = 0 : i32} : !llvm.array<5 x array<5 x i8>>
  llvm.mlir.global private @g2(dense<[[1, -1, 0, 0, 1], [1, 0, -1, 0, -1], [-1, -1, 0, -1, 1], [-1, 0, 0, 1, 0], [1, 0, 0, -1, 0]]> : tensor<5x5xi8>) {addr_space = 0 : i32} : !llvm.array<5 x array<5 x i8>>
  llvm.func @main() -> i8 {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(1 : i8) : i8
    %2 = llvm.mlir.constant(-2 : i8) : i8
    %3 = llvm.mlir.constant(2 : i8) : i8
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(5 : index) : i64
    %9 = llvm.mlir.constant(5 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(25 : index) : i64
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %12[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.mlir.addressof @g1 : !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x array<5 x i8>>
    %17 = llvm.mlir.constant(3735928559 : index) : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %16, %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %8, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %9, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %9, %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %10, %26[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mlir.constant(5 : index) : i64
    %29 = llvm.mlir.constant(5 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(25 : index) : i64
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mlir.addressof @g2 : !llvm.ptr
    %36 = llvm.getelementptr %35[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x array<5 x i8>>
    %37 = llvm.mlir.constant(3735928559 : index) : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %36, %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %28, %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %29, %44[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %29, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %30, %46[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(5 : index) : i64
    %49 = llvm.mlir.constant(5 : index) : i64
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.mlir.constant(25 : index) : i64
    %52 = llvm.mlir.zero : !llvm.ptr
    %53 = llvm.getelementptr %52[%51] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %57 = llvm.insertvalue %55, %56[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %55, %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(0 : index) : i64
    %60 = llvm.insertvalue %59, %58[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %48, %60[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %49, %61[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %49, %62[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %50, %63[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mlir.constant(5 : index) : i64
    %66 = llvm.mlir.constant(5 : index) : i64
    %67 = llvm.mlir.constant(1 : index) : i64
    %68 = llvm.mlir.constant(25 : index) : i64
    %69 = llvm.mlir.zero : !llvm.ptr
    %70 = llvm.getelementptr %69[%68] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %71 = llvm.ptrtoint %70 : !llvm.ptr to i64
    %72 = llvm.call @malloc(%71) : (i64) -> !llvm.ptr
    %73 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %72, %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mlir.constant(0 : index) : i64
    %77 = llvm.insertvalue %76, %75[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %65, %77[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %66, %78[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.insertvalue %66, %79[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %67, %80[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(5 : index) : i64
    %83 = llvm.mlir.constant(5 : index) : i64
    %84 = llvm.mlir.constant(5 : index) : i64
    llvm.br ^bb1(%6 : i64)
  ^bb1(%85: i64):  // 2 preds: ^bb0, ^bb5
    %86 = llvm.icmp "slt" %85, %82 : i64
    llvm.cond_br %86, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%87: i64):  // 2 preds: ^bb2, ^bb4
    %88 = llvm.icmp "slt" %87, %83 : i64
    llvm.cond_br %88, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %89 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.mlir.constant(5 : index) : i64
    %91 = llvm.mul %85, %90 : i64
    %92 = llvm.add %91, %87 : i64
    %93 = llvm.getelementptr %89[%92] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %5, %93 : i32, !llvm.ptr
    %94 = llvm.add %87, %7 : i64
    llvm.br ^bb3(%94 : i64)
  ^bb5:  // pred: ^bb3
    %95 = llvm.add %85, %7 : i64
    llvm.br ^bb1(%95 : i64)
  ^bb6:  // pred: ^bb1
    %96 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    %100 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %104 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %105 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%104, %105) : (i64, i64) -> ()
    %106 = llvm.mlir.constant(5 : i64) : i64
    %107 = llvm.mlir.constant(2 : i64) : i64
    %108 = llvm.mlir.constant(4575657221408423941 : i64) : i64
    "gemmini.intr.config_st"(%107, %108) : (i64, i64) -> ()
    %109 = llvm.mlir.constant(5 : i64) : i64
    %110 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%110, %109) : (i64, i64) -> ()
    %111 = llvm.mlir.constant(5 : i64) : i64
    %112 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%112, %111) : (i64, i64) -> ()
    %113 = llvm.mlir.constant(20 : i64) : i64
    %114 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%114, %113) : (i64, i64) -> ()
    %115 = llvm.mlir.constant(0 : i64) : i64
    %116 = llvm.mlir.constant(0 : i64) : i64
    %117 = llvm.mlir.constant(0 : i64) : i64
    %118 = llvm.mlir.constant(0 : i64) : i64
    %119 = llvm.mlir.constant(47245361163 : i64) : i64
    %120 = llvm.mlir.constant(4295032833 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%119, %120) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%97, %99) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%103, %101) : (i64, i64) -> ()
    %121 = llvm.mlir.constant(5 : i64) : i64
    %122 = llvm.mlir.constant(5 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%121, %122) : (i64, i64) -> ()
    %123 = llvm.mlir.constant(5 : i64) : i64
    %124 = llvm.mlir.constant(5 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%123, %124) : (i64, i64) -> ()
    %125 = llvm.mlir.constant(1 : i64) : i64
    %126 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%125, %126) : (i64, i64) -> ()
    %127 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%127, %127) : (i64, i64) -> ()
    %128 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %129 = llvm.mlir.constant(0 : index) : i64
    %130 = llvm.getelementptr %128[%129, %129] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %131 = llvm.mlir.addressof @nl : !llvm.ptr
    %132 = llvm.mlir.constant(0 : index) : i64
    %133 = llvm.getelementptr %131[%132, %132] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %134 = llvm.mlir.constant(0 : index) : i64
    %135 = llvm.mlir.constant(5 : index) : i64
    %136 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%134 : i64)
  ^bb7(%137: i64):  // 2 preds: ^bb6, ^bb11
    %138 = llvm.icmp "slt" %137, %135 : i64
    llvm.cond_br %138, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %139 = llvm.mlir.constant(0 : index) : i64
    %140 = llvm.mlir.constant(5 : index) : i64
    %141 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%139 : i64)
  ^bb9(%142: i64):  // 2 preds: ^bb8, ^bb10
    %143 = llvm.icmp "slt" %142, %140 : i64
    llvm.cond_br %143, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %144 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.mlir.constant(5 : index) : i64
    %146 = llvm.mul %137, %145 : i64
    %147 = llvm.add %146, %142 : i64
    %148 = llvm.getelementptr %144[%147] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %149 = llvm.load %148 : !llvm.ptr -> i8
    %150 = llvm.sext %149 : i8 to i32
    %151 = llvm.call @printf(%130, %150) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %152 = llvm.add %142, %141 : i64
    llvm.br ^bb9(%152 : i64)
  ^bb11:  // pred: ^bb9
    %153 = llvm.call @printf(%133) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %154 = llvm.add %137, %136 : i64
    llvm.br ^bb7(%154 : i64)
  ^bb12:  // pred: ^bb7
    %155 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.ptrtoint %155 : !llvm.ptr to i64
    %157 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.ptrtoint %157 : !llvm.ptr to i64
    %159 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.ptrtoint %159 : !llvm.ptr to i64
    %161 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.ptrtoint %161 : !llvm.ptr to i64
    %163 = llvm.mlir.constant(4575657221408489476 : i64) : i64
    %164 = llvm.mlir.constant(281474976710656 : i64) : i64
    "gemmini.intr.config_ex"(%163, %164) : (i64, i64) -> ()
    %165 = llvm.mlir.constant(5 : i64) : i64
    %166 = llvm.mlir.constant(2 : i64) : i64
    %167 = llvm.mlir.constant(4575657221408423941 : i64) : i64
    "gemmini.intr.config_st"(%166, %167) : (i64, i64) -> ()
    %168 = llvm.mlir.constant(5 : i64) : i64
    %169 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    "gemmini.intr.config_ld"(%169, %168) : (i64, i64) -> ()
    %170 = llvm.mlir.constant(5 : i64) : i64
    %171 = llvm.mlir.constant(4575657221409472777 : i64) : i64
    "gemmini.intr.config_ld"(%171, %170) : (i64, i64) -> ()
    %172 = llvm.mlir.constant(20 : i64) : i64
    %173 = llvm.mlir.constant(4575657221409472785 : i64) : i64
    "gemmini.intr.config_ld"(%173, %172) : (i64, i64) -> ()
    %174 = llvm.mlir.constant(55834640387 : i64) : i64
    %175 = llvm.mlir.constant(1644972474395 : i64) : i64
    "gemmini.intr.config_norm"(%174, %175) : (i64, i64) -> ()
    %176 = llvm.mlir.constant(21650930466819 : i64) : i64
    %177 = llvm.mlir.constant(1644972474395 : i64) : i64
    "gemmini.intr.config_norm"(%176, %177) : (i64, i64) -> ()
    %178 = llvm.mlir.constant(0 : i64) : i64
    %179 = llvm.mlir.constant(0 : i64) : i64
    %180 = llvm.mlir.constant(0 : i64) : i64
    %181 = llvm.mlir.constant(0 : i64) : i64
    %182 = llvm.mlir.constant(47245361163 : i64) : i64
    %183 = llvm.mlir.constant(4295032833 : i64) : i64
    "gemmini.intr.loop_ws_config_bounds"(%182, %183) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_ab"(%156, %158) : (i64, i64) -> ()
    "gemmini.intr.loop_ws_config_addrs_dc"(%162, %160) : (i64, i64) -> ()
    %184 = llvm.mlir.constant(5 : i64) : i64
    %185 = llvm.mlir.constant(5 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_ab"(%184, %185) : (i64, i64) -> ()
    %186 = llvm.mlir.constant(5 : i64) : i64
    %187 = llvm.mlir.constant(5 : i64) : i64
    "gemmini.intr.loop_ws_config_strides_dc"(%186, %187) : (i64, i64) -> ()
    %188 = llvm.mlir.constant(1025 : i64) : i64
    %189 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.loop_ws"(%188, %189) : (i64, i64) -> ()
    %190 = llvm.mlir.constant(0 : i64) : i64
    "gemmini.intr.flush"(%190, %190) : (i64, i64) -> ()
    %191 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %192 = llvm.mlir.constant(0 : index) : i64
    %193 = llvm.getelementptr %191[%192, %192] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %194 = llvm.mlir.addressof @nl : !llvm.ptr
    %195 = llvm.mlir.constant(0 : index) : i64
    %196 = llvm.getelementptr %194[%195, %195] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %197 = llvm.mlir.constant(0 : index) : i64
    %198 = llvm.mlir.constant(5 : index) : i64
    %199 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%197 : i64)
  ^bb13(%200: i64):  // 2 preds: ^bb12, ^bb17
    %201 = llvm.icmp "slt" %200, %198 : i64
    llvm.cond_br %201, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    %202 = llvm.mlir.constant(0 : index) : i64
    %203 = llvm.mlir.constant(5 : index) : i64
    %204 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb15(%202 : i64)
  ^bb15(%205: i64):  // 2 preds: ^bb14, ^bb16
    %206 = llvm.icmp "slt" %205, %203 : i64
    llvm.cond_br %206, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %207 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %208 = llvm.mlir.constant(5 : index) : i64
    %209 = llvm.mul %200, %208 : i64
    %210 = llvm.add %209, %205 : i64
    %211 = llvm.getelementptr %207[%210] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %212 = llvm.load %211 : !llvm.ptr -> i8
    %213 = llvm.sext %212 : i8 to i32
    %214 = llvm.call @printf(%193, %213) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %215 = llvm.add %205, %204 : i64
    llvm.br ^bb15(%215 : i64)
  ^bb17:  // pred: ^bb15
    %216 = llvm.call @printf(%196) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %217 = llvm.add %200, %199 : i64
    llvm.br ^bb13(%217 : i64)
  ^bb18:  // pred: ^bb13
    llvm.return %0 : i8
  }
}


