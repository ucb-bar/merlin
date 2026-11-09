// RUN: %merlin-opt %s -p merlin-partition-dispatches --emit=schedule | %filecheck %s
//
// OBLIGATION partition/eligibility, discharged by `merlin-partition-dispatches` -- the LAST of the
// four production passes to get a static verdict, taking the obligation gate to 4 of 4.
//
// Like `merlin-emit-dispatch-program`, this pass takes a `DispatchProgram` and returns a
// `PartitionResult` -- Python objects at both ends, never a ModulePass. But it already had a text
// renderer, `emit_schedule_c`, that nothing was connected to a driver; `--emit=schedule` prints it.
//
// The property this pins is the strongest of the three: a LEVEL BARRIER. Every dependency edge must
// cross from a lower level to a higher one, because harts within a level run concurrently. A node
// scheduled at or below the level of something it reads is a race, and the emitted table is exactly
// where that would be visible -- each row carries its kernel, its level, its hart, its input buffers
// and its single output buffer.
//
// `validate()` asserts the same invariant in code. Pinning it on the ARTIFACT matters because the
// table is what a multicore runtime actually consumes: a schedule that is correct in memory and
// wrong on emission is still a wrong schedule.

// CHECK:      #define MERLIN_SCHEDULE_N
// CHECK:      #define MERLIN_SCHEDULE_LEVELS
// CHECK:      #define MERLIN_SCHEDULE_HARTS
// CHECK:      static const merlin_dispatch_t MERLIN_SCHEDULE
// the producer is scheduled at a strictly LOWER level than the consumer that reads its buffer
// CHECK:      { "forward$kernel_0__rmatmul_0", 0,
// CHECK:      { "forward$kernel_1__rmatmul_1", 1,
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
