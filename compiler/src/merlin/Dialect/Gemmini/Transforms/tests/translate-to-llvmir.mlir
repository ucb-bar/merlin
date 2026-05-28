// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(convert-arith-to-llvm,convert-func-to-llvm))' \
// RUN: | mlir-translate --mlir-to-llvmir | FileCheck %s

// Drive the gemmini.intr.* -> llvm.intr.riscv.* translation via the
// auto-generated GemminiConversions.inc. The Gemmini dialect translation
// interface is registered in the Gemmini compiler plugin so iree-opt
// sees the dialect, and the registration extension also wires it into
// mlir-translate's dialect registry.

llvm.func @flush_intrinsic(%rs1: i64, %rs2: i64) {
  // CHECK: call void @llvm.riscv.flush(i64
  gemmini.intr.flush %rs1, %rs2 : (i64, i64) -> ()
  llvm.return
}

llvm.func @mvin_intrinsic(%rs1: i64, %rs2: i64) {
  // CHECK: call void @llvm.riscv.mvin(i64
  gemmini.intr.mvin %rs1, %rs2 : (i64, i64) -> ()
  llvm.return
}

llvm.func @config_ex_intrinsic(%rs1: i64, %rs2: i64) {
  // CHECK: call void @llvm.riscv.config(i64
  gemmini.intr.config %rs1, %rs2 : (i64, i64) -> ()
  llvm.return
}
