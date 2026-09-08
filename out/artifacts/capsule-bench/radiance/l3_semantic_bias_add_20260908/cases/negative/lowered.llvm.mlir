module {
  llvm.func @radiance_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %out: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cM = llvm.mlir.constant(16 : i64) : i64
    %cN = llvm.mlir.constant(16 : i64) : i64
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cM : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mN = llvm.mul %mi, %cN : i64
    llvm.br ^n(%c0 : i64)
  ^n(%ni: i64):
    %nc = llvm.icmp "slt" %ni, %cN : i64
    llvm.cond_br %nc, ^nbody, ^mnext
  ^nbody:
    %idx = llvm.add %mN, %ni : i64
    %ap = llvm.getelementptr %arg0[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %av = llvm.load %ap : !llvm.ptr -> f32
    %bp = llvm.getelementptr %arg1[%ni] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bv = llvm.load %bp : !llvm.ptr -> f32
    %rv = llvm.fadd %av, %bv : f32
    %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %rv, %op : f32, !llvm.ptr
    %ni2 = llvm.add %ni, %c1 : i64
    llvm.br ^n(%ni2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return
  }
}
