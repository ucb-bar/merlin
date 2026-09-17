// RUN: %merlin-opt %s -p merlin-add-c-interface | %filecheck %s
//
// OBLIGATION boundary materialization, discharged by `merlin-add-c-interface` -- a PRODUCTION
// catalog pass. Every other check in this suite exercises a prototype-catalog pass, which the
// obligation gate correctly refuses to credit to production; this one gives a production pass a
// static verdict.
//
// The host seam is what makes a compiled kernel callable from C: MLIR emits a `_mlir_ciface_`
// wrapper for a func carrying `llvm.emit_c_interface`. Without the attribute the symbol the runtime
// dlsym's does not exist, and the failure surfaces as a missing symbol at load time rather than as
// anything the compiler said.
//
// KNOWN DISCREPANCY (recorded, not asserted away). The pass's own docstring and its catalog summary
// both say "public funcs", but the implementation walks every `func.func` and marks it regardless of
// `sym_visibility` -- a private helper gets a C wrapper it has no caller for. The CHECK lines below
// pin what the pass ACTUALLY does, so this file is a true regression test; the intent/behaviour
// mismatch is filed as VER-29 for the owner of `merlin/python/merlin/llvmlower/`. Asserting the
// documented behaviour here would have made the suite red against a pass nobody had agreed to
// change, and asserting nothing would have let the mismatch stay invisible.

// CHECK:     func.func @forward
// CHECK-SAME: llvm.emit_c_interface
// CHECK:     func.func private @helper
// CHECK-SAME: llvm.emit_c_interface
builtin.module {
  func.func @forward(%a: tensor<4x4xi8>) -> tensor<4x4xi8> {
    func.return %a : tensor<4x4xi8>
  }
  func.func private @helper(%a: tensor<4x4xi8>) -> tensor<4x4xi8> {
    func.return %a : tensor<4x4xi8>
  }
}
