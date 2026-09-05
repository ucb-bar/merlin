// RUN: %merlin-opt %s --split-input-file -p merlin-outline-dispatches | %filecheck %s
//
// OBLIGATION partition/eligibility, discharged by `merlin-outline-dispatches` -- a PRODUCTION
// catalog pass, and the second of the four to get a static verdict.
//
// Dispatch formation is what turns one monolithic `func @forward` into units of placeable work: a
// kernel func per compute root, plus a driver that calls them in order. Four properties make that
// rewrite trustworthy, and this file pins all four on one input.
//
//   1. PARTITION -- every linalg root LEAVES the driver. A root left behind is compiled twice (once
//      in the driver, once in its kernel) and the driver stops being a schedule.
//   2. DATAFLOW -- kernel_1 consumes kernel_0's result and the driver returns kernel_1's. Outlining
//      is value-preserving; a rewired operand is a wrong answer no shape check catches.
//   3. SELF-CONTAINMENT -- each kernel carries its OWN cloned accumulator init (`linalg.fill`), so
//      per-kernel bufferization can fold the zero-init into the contraction. If the fill stays only
//      in the driver, the kernel takes an uninitialized accumulator as an argument.
//   4. PROVENANCE -- `prov.region_id` survives INTO the symbol (`$kernel_0__rmatmul_0`). That suffix
//      is the one thread reaching the emitted ELF; `outline.region_id_of_symbol` and the section
//      slicer both read it, and a dropped suffix silently builds a slice of the wrong model.
//
// KNOWN DISCREPANCY (recorded, not asserted away), second module below. The catalog summary says
// "split func @forward", and `run_dialect_plane` calls the pass with `forward=None`. With no name
// the pass takes `fns[0]` -- the FIRST func with a body -- and rebuilds the module as
// `ModuleOp([driver, *kernels])`. So a module whose first body-carrying func is a helper gets that
// helper "outlined" (zero kernels), `@forward` and every other func SILENTLY DELETED, and no
// diagnostic: measured end to end, `run_dialect_plane` returns kernels=0, speedup=0.0 and raises
// nothing. The selected func also loses its `sym_visibility` (private -> public). The CHECK lines
// below pin what the pass ACTUALLY does, so this file is a true regression test; the intent/behaviour
// mismatch is filed for the owner of `merlin/python/merlin/xdsl_dialects/lowering/`. If the pass is
// fixed to select `@forward` (or to fail closed on ambiguity), the second module's checks go red --
// which is the point: the pin is what makes the fix visible instead of silent.

// CHECK:      func.func @forward
// CHECK-NOT:  linalg.matmul
// CHECK:      %[[K0:.*]] = func.call @forward$kernel_0__rmatmul_0(%{{.*}}, %{{.*}})
// CHECK-NOT:  linalg.matmul
// CHECK:      %[[K1:.*]] = func.call @forward$kernel_1__rmatmul_1(%[[K0]])
// CHECK:      func.return %[[K1]]
// CHECK:      func.func private @forward$kernel_0__rmatmul_0
// CHECK:        linalg.fill
// CHECK:        linalg.matmul
// CHECK:      func.func private @forward$kernel_1__rmatmul_1
// CHECK:        linalg.fill
// CHECK:        linalg.matmul
builtin.module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<4x4xf32>
    %z = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %m = linalg.matmul {"prov.region_id" = "matmul_0"} ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%z : tensor<4x4xf32>) -> tensor<4x4xf32>
    %e2 = tensor.empty() : tensor<4x4xf32>
    %z2 = linalg.fill ins(%c0 : f32) outs(%e2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %n = linalg.matmul {"prov.region_id" = "matmul_1"} ins(%m, %m : tensor<4x4xf32>, tensor<4x4xf32>) outs(%z2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %n : tensor<4x4xf32>
  }
}

// -----

// The discrepancy, pinned: @helper wins, loses `private`, and @forward is gone.
// CHECK-LABEL: func.func @helper
// CHECK-NOT:   func.func @forward
builtin.module {
  func.func private @helper(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    func.return %a : tensor<4x4xf32>
  }
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<4x4xf32>
    %z = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %m = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%z : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %m : tensor<4x4xf32>
  }
}
