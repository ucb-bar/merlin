builtin.module {
  llvm.func @radiance_kernel(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr) {
    %3 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb0(%3 : i32)
  ^bb0(%4: i32):
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1(%4 : i32), ^bb2
  ^bb1(%7: i32):
    %8 = llvm.mlir.constant(16 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %11 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb3(%11, %10 : i32, f32)
  ^bb2:
    llvm.return
  ^bb3(%12: i32, %13: f32):
    %14 = llvm.mlir.constant(16 : i32) : i32
    %15 = llvm.icmp "ult" %12, %14 : i32
    llvm.cond_br %15, ^bb4(%12, %13 : i32, f32), ^bb5(%13 : f32)
  ^bb4(%16: i32, %17: f32):
    %18 = llvm.add %9, %16 : i32
    %19 = llvm.getelementptr %1[%18] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %20 = llvm.load %19 : !llvm.ptr -> f32
    %21 = llvm.fmul %20, %20 : f32
    %22 = llvm.fadd %17, %21 : f32
    %23 = llvm.mlir.constant(1 : i32) : i32
    %24 = llvm.add %16, %23 : i32
    llvm.br ^bb3(%24, %22 : i32, f32)
  ^bb5(%25: f32):
    %26 = llvm.mlir.constant(1.600000e+01 : f32) : f32
    %27 = llvm.fdiv %25, %26 : f32
    %28 = llvm.mlir.constant(1.000000e-05 : f32) : f32
    %29 = llvm.fadd %27, %28 : f32
    %30 = llvm.intr.sqrt(%29) : (f32) -> f32
    %31 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb6(%31 : i32)
  ^bb6(%32: i32):
    %33 = llvm.mlir.constant(16 : i32) : i32
    %34 = llvm.icmp "ult" %32, %33 : i32
    llvm.cond_br %34, ^bb7(%32 : i32), ^bb8
  ^bb7(%35: i32):
    %36 = llvm.add %9, %35 : i32
    %37 = llvm.getelementptr %1[%36] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.fdiv %38, %30 : f32
    %40 = llvm.getelementptr %0[%35] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %41 = llvm.load %40 : !llvm.ptr -> f32
    %42 = llvm.fmul %39, %41 : f32
    %43 = llvm.getelementptr %2[%36] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %42, %43 : f32, !llvm.ptr
    %44 = llvm.mlir.constant(1 : i32) : i32
    %45 = llvm.add %35, %44 : i32
    llvm.br ^bb6(%45 : i32)
  ^bb8:
    %46 = llvm.mlir.constant(1 : i32) : i32
    %47 = llvm.add %7, %46 : i32
    llvm.br ^bb0(%47 : i32)
  }
}
