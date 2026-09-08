module {
  llvm.func @radiance_kernel(%arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg0: !llvm.ptr, %out: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cR = llvm.mlir.constant(16 : i64) : i64
    %cC = llvm.mlir.constant(16 : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %one = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %cCf = llvm.mlir.constant(1.600000e+01 : f32) : f32
    %eps = llvm.mlir.constant(1.000000e-05 : f32) : f32
    llvm.br ^m(%c0 : i64)
  ^m(%mi: i64):
    %mc = llvm.icmp "slt" %mi, %cR : i64
    llvm.cond_br %mc, ^mbody, ^end
  ^mbody:
    %mC = llvm.mul %mi, %cC : i64
    llvm.br ^r(%c0, %zero, %zero : i64, f32, f32)
  ^r(%ri: i64, %s: f32, %sq: f32):
    %rc = llvm.icmp "slt" %ri, %cC : i64
    llvm.cond_br %rc, ^rbody, ^rdone
  ^rbody:
    %xidx = llvm.add %mC, %ri : i64
    %xp = llvm.getelementptr %arg0[%xidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %xv = llvm.load %xp : !llvm.ptr -> f32
    %s2 = llvm.fadd %s, %xv : f32
    %xsq = llvm.fmul %xv, %xv : f32
    %sq2 = llvm.fadd %sq, %xsq : f32
    %ri2 = llvm.add %ri, %c1 : i64
    llvm.br ^r(%ri2, %s2, %sq2 : i64, f32, f32)
  ^rdone:
    %mean = llvm.fdiv %s, %cCf : f32
    %ex2 = llvm.fdiv %sq, %cCf : f32
    %m2 = llvm.fmul %mean, %mean : f32
    %var = llvm.fsub %ex2, %m2 : f32
    %vare = llvm.fadd %var, %eps : f32
    %rt = llvm.intr.sqrt(%vare) : (f32) -> f32
    %inv = llvm.fdiv %one, %rt : f32
    llvm.br ^w(%c0 : i64)
  ^w(%wi: i64):
    %wc = llvm.icmp "slt" %wi, %cC : i64
    llvm.cond_br %wc, ^wbody, ^mnext
  ^wbody:
    %widx = llvm.add %mC, %wi : i64
    %wxp = llvm.getelementptr %arg0[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %wxv = llvm.load %wxp : !llvm.ptr -> f32
    %cen = llvm.fsub %wxv, %mean : f32
    %gp = llvm.getelementptr %arg1[%wi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %gv = llvm.load %gp : !llvm.ptr -> f32
    %bp = llvm.getelementptr %arg2[%wi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bv = llvm.load %bp : !llvm.ptr -> f32
    %ni = llvm.fmul %cen, %inv : f32
    %ng = llvm.fmul %ni, %gv : f32
    %yv = llvm.fadd %ng, %bv : f32
    %wop = llvm.getelementptr %out[%widx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %yv, %wop : f32, !llvm.ptr
    %wi2 = llvm.add %wi, %c1 : i64
    llvm.br ^w(%wi2 : i64)
  ^mnext:
    %mi2 = llvm.add %mi, %c1 : i64
    llvm.br ^m(%mi2 : i64)
  ^end:
    llvm.return
  }
}
