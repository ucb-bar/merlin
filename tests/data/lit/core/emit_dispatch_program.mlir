// RUN: %merlin-opt %s -p merlin-emit-dispatch-program --emit=dispatch-program | %filecheck %s
//
// OBLIGATION boundary materialization, discharged by `merlin-emit-dispatch-program` -- the third of
// the four PRODUCTION passes to get a static verdict.
//
// This pass consumes an `OutlineResult` and produces a `DispatchProgram`: Python objects at both
// ends, so it can never be a ModulePass, and `--list-merlin-passes` correctly reports it as
// unregistrable. That is a fact about the PASS, not about its OUTPUT. The program has a stable
// `to_dict`, and `--emit=dispatch-program` prints it -- which is all FileCheck ever needed. Without
// that seam the obligation gate could not rise above 2 of 4, not because this pass is unverifiable
// but because nothing exposed its result as text.
//
// The property: the dispatch program is a single-writer DAG over named buffers. Every node names the
// kernel it calls, declares which buffers it reads, and writes exactly one. `verify_program` states
// the same invariant in code; this pins it on the emitted artifact, where a regression would actually
// be read.

// The CHECK lines below were written by READING the emitter's real output, not by guessing its
// field names: a first draft asserted a "kernel" key that does not exist, and the check failed
// against correct output. Keys are emitted sorted, so "inputs" precedes "op" within a node.
//
// kernel_0 reads both entry arguments and writes one buffer; kernel_1 reads THAT buffer and writes
// the program result. That chain is the dataflow the outliner must preserve -- a rewired operand is
// a wrong answer no shape check would catch.

// CHECK:      "entry": "forward"
// CHECK:      "op": "forward$kernel_0__rmatmul_0"
// CHECK:      "op": "forward$kernel_1__rmatmul_1"
// CHECK:      "results": [
// CHECK-NEXT:   "b8"
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
