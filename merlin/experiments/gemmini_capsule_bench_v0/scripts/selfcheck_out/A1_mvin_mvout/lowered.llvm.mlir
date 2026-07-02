module {
  llvm.func @gemmini_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.ptrtoint %arg0 : !llvm.ptr to i64
    %1 = llvm.ptrtoint %arg1 : !llvm.ptr to i64
    llvm.inline_asm has_side_effects "fence", ""  : () -> ()
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 7, x0, $0, $1", "r,r" %3, %2 : (i64, i64) -> ()
    %4 = llvm.mlir.constant(16 : i64) : i64
    %5 = llvm.mlir.constant(4575657221409472769 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 0, x0, $0, $1", "r,r" %5, %4 : (i64, i64) -> ()
    %6 = llvm.mlir.constant(4575657221408423952 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 0, x0, $0, $1", "r,r" %7, %6 : (i64, i64) -> ()
    %8 = llvm.mlir.constant(4503668346847232 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 2, x0, $0, $1", "r,r" %0, %8 : (i64, i64) -> ()
    %9 = llvm.mlir.constant(4503668346847232 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 3, x0, $0, $1", "r,r" %1, %9 : (i64, i64) -> ()
    llvm.inline_asm has_side_effects "fence", ""  : () -> ()
    llvm.return
  }
}
