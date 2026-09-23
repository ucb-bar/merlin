module {
  llvm.func @radiance_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %out: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cN = llvm.mlir.constant(32 : i64) : i64
    llvm.br ^l(%c0 : i64)
  ^l(%i: i64):
    %ic = llvm.icmp "slt" %i, %cN : i64
    llvm.cond_br %ic, ^body, ^end
  ^body:
    %ap = llvm.getelementptr %arg0[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %av = llvm.load %ap : !llvm.ptr -> f32
    %bp = llvm.getelementptr %arg1[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bv = llvm.load %bp : !llvm.ptr -> f32
    %rv = llvm.fmul %av, %bv : f32
    %op = llvm.getelementptr %out[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %rv, %op : f32, !llvm.ptr
    %i2 = llvm.add %i, %c1 : i64
    llvm.br ^l(%i2 : i64)
  ^end:
    llvm.return
  }
}
