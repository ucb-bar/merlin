builtin.module {
  llvm.func @radiance_kernel(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr) {
    %3 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb0(%3 : i32)
  ^bb0(%4: i32):
    %5 = llvm.mlir.constant(32 : i32) : i32
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1(%4 : i32), ^bb2
  ^bb1(%7: i32):
    %8 = llvm.mlir.constant(16 : i32) : i32
    %9 = llvm.udiv %7, %8 : i32
    %10 = llvm.urem %7, %8 : i32
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.udiv %10, %11 : i32
    %13 = llvm.urem %10, %11 : i32
    %14 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %15 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb3(%15, %14 : i32, f32)
  ^bb2:
    llvm.return
  ^bb3(%16: i32, %17: f32):
    %18 = llvm.mlir.constant(16 : i32) : i32
    %19 = llvm.icmp "ult" %16, %18 : i32
    llvm.cond_br %19, ^bb4(%16, %17 : i32, f32), ^bb5(%17 : f32)
  ^bb4(%20: i32, %21: f32):
    %22 = llvm.mlir.constant(256 : i32) : i32
    %23 = llvm.mlir.constant(16 : i32) : i32
    %24 = llvm.mul %9, %22 : i32
    %25 = llvm.mul %12, %23 : i32
    %26 = llvm.add %25, %20 : i32
    %27 = llvm.add %24, %26 : i32
    %28 = llvm.mlir.constant(16 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.mul %9, %28 : i32
    %31 = llvm.mul %20, %29 : i32
    %32 = llvm.add %31, %13 : i32
    %33 = llvm.add %30, %32 : i32
    %34 = llvm.getelementptr %0[%27] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %35 = llvm.load %34 : !llvm.ptr -> f32
    %36 = llvm.getelementptr %1[%33] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %37 = llvm.load %36 : !llvm.ptr -> f32
    %38 = llvm.fmul %35, %37 : f32
    %39 = llvm.fadd %21, %38 : f32
    %40 = llvm.mlir.constant(1 : i32) : i32
    %41 = llvm.add %20, %40 : i32
    llvm.br ^bb3(%41, %39 : i32, f32)
  ^bb5(%42: f32):
    %43 = llvm.getelementptr %2[%7] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %42, %43 : f32, !llvm.ptr
    %44 = llvm.mlir.constant(1 : i32) : i32
    %45 = llvm.add %7, %44 : i32
    llvm.br ^bb0(%45 : i32)
  }
}
