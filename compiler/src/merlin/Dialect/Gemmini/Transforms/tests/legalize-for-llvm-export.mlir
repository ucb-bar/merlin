// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(merlin-gemmini-legalize-for-llvm-export))' | FileCheck %s

// Smoke test: the pass should be registered (selectable via
// --pass-pipeline) and rewrite gemmini.flush to a gemmini.intr.flush
// (which auto-translates to llvm.riscv.flush at the LLVM IR boundary).

// CHECK-LABEL: func.func @flush_only
func.func @flush_only() {
  %skip = arith.constant 0 : i64
  // CHECK: gemmini.intr.flush
  gemmini.flush %skip
  return
}
