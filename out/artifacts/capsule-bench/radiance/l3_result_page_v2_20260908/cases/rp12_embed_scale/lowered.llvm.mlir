builtin.module {
  llvm.func @radiance_kernel(%0: !llvm.ptr, %1: !llvm.ptr) {
    %2 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb0(%2 : i32)
  ^bb0(%3: i32):
    %4 = llvm.mlir.constant(256 : i32) : i32
    %5 = llvm.icmp "ult" %3, %4 : i32
    llvm.cond_br %5, ^bb1(%3 : i32), ^bb2
  ^bb1(%6: i32):
    %7 = llvm.getelementptr %0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %8 = llvm.load %7 : !llvm.ptr -> f32
    %9 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %10 = llvm.fmul %8, %9 : f32
    %11 = llvm.getelementptr %1[%6] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %10, %11 : f32, !llvm.ptr
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.add %6, %12 : i32
    llvm.br ^bb0(%13 : i32)
  ^bb2:
    llvm.return
  }
}
