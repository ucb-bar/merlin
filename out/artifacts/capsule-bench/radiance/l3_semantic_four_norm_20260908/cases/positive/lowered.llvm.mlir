module {
  llvm.func @radiance_kernel(%arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg0: !llvm.ptr, %out: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %cR = llvm.mlir.constant(16 : i64) : i64
    %cC = llvm.mlir.constant(16 : i64) : i64
    %zero = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %one = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %cCf = llvm.mlir.constant(1.600000e+01 : f32) : f32
    %eps = llvm.mlir.constant(1.001358e-05 : f32) : f32
    %hsz = llvm.mlir.constant(256 : i64) : i64
    %H = llvm.alloca %hsz x f32 : (i64) -> !llvm.ptr
    llvm.br ^am(%c0 : i64)
  ^am(%ami: i64):
    %amc = llvm.icmp "slt" %ami, %cR : i64
    llvm.cond_br %amc, ^amb, ^adone
  ^amb:
    %amC = llvm.mul %ami, %cC : i64
    llvm.br ^ar(%c0, %zero : i64, f32)
  ^ar(%ari: i64, %ass: f32):
    %arc = llvm.icmp "slt" %ari, %cC : i64
    llvm.cond_br %arc, ^arb, ^ard
  ^arb:
    %axidx = llvm.add %amC, %ari : i64
    %axp = llvm.getelementptr %arg0[%axidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %axv = llvm.load %axp : !llvm.ptr -> f32
    %asq = llvm.fmul %axv, %axv : f32
    %ass2 = llvm.fadd %ass, %asq : f32
    %ari2 = llvm.add %ari, %c1 : i64
    llvm.br ^ar(%ari2, %ass2 : i64, f32)
  ^ard:
    %ams = llvm.fdiv %ass, %cCf : f32
    %amse = llvm.fadd %ams, %eps : f32
    %art = llvm.intr.sqrt(%amse) : (f32) -> f32
    %ainv = llvm.fdiv %one, %art : f32
    llvm.br ^aw(%c0 : i64)
  ^aw(%awi: i64):
    %awc = llvm.icmp "slt" %awi, %cC : i64
    llvm.cond_br %awc, ^awb, ^amnext
  ^awb:
    %awidx = llvm.add %amC, %awi : i64
    %awxp = llvm.getelementptr %arg0[%awidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %awxv = llvm.load %awxp : !llvm.ptr -> f32
    %agp = llvm.getelementptr %arg1[%awi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %agv = llvm.load %agp : !llvm.ptr -> f32
    %axn = llvm.fmul %awxv, %ainv : f32
    %ayv = llvm.fmul %axn, %agv : f32
    %awop = llvm.getelementptr %H[%awidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %ayv, %awop : f32, !llvm.ptr
    %awi2 = llvm.add %awi, %c1 : i64
    llvm.br ^aw(%awi2 : i64)
  ^amnext:
    %ami2 = llvm.add %ami, %c1 : i64
    llvm.br ^am(%ami2 : i64)
  ^adone:
    llvm.br ^bm(%c0 : i64)
  ^bm(%bmi: i64):
    %bmc = llvm.icmp "slt" %bmi, %cR : i64
    llvm.cond_br %bmc, ^bmb, ^bdone
  ^bmb:
    %bmC = llvm.mul %bmi, %cC : i64
    llvm.br ^br(%c0, %zero : i64, f32)
  ^br(%bri: i64, %bss: f32):
    %brc = llvm.icmp "slt" %bri, %cC : i64
    llvm.cond_br %brc, ^brb, ^brd
  ^brb:
    %bxidx = llvm.add %bmC, %bri : i64
    %bxp = llvm.getelementptr %H[%bxidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bxv = llvm.load %bxp : !llvm.ptr -> f32
    %bsq = llvm.fmul %bxv, %bxv : f32
    %bss2 = llvm.fadd %bss, %bsq : f32
    %bri2 = llvm.add %bri, %c1 : i64
    llvm.br ^br(%bri2, %bss2 : i64, f32)
  ^brd:
    %bms = llvm.fdiv %bss, %cCf : f32
    %bmse = llvm.fadd %bms, %eps : f32
    %brt = llvm.intr.sqrt(%bmse) : (f32) -> f32
    %binv = llvm.fdiv %one, %brt : f32
    llvm.br ^bw(%c0 : i64)
  ^bw(%bwi: i64):
    %bwc = llvm.icmp "slt" %bwi, %cC : i64
    llvm.cond_br %bwc, ^bwb, ^bmnext
  ^bwb:
    %bwidx = llvm.add %bmC, %bwi : i64
    %bwxp = llvm.getelementptr %H[%bwidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bwxv = llvm.load %bwxp : !llvm.ptr -> f32
    %bgp = llvm.getelementptr %arg2[%bwi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %bgv = llvm.load %bgp : !llvm.ptr -> f32
    %bxn = llvm.fmul %bwxv, %binv : f32
    %byv = llvm.fmul %bxn, %bgv : f32
    %bwop = llvm.getelementptr %out[%bwidx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %byv, %bwop : f32, !llvm.ptr
    %bwi2 = llvm.add %bwi, %c1 : i64
    llvm.br ^bw(%bwi2 : i64)
  ^bmnext:
    %bmi2 = llvm.add %bmi, %c1 : i64
    llvm.br ^bm(%bmi2 : i64)
  ^bdone:
    llvm.br ^end
  ^end:
    llvm.return
  }
}
