// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-lower-tile-to-isa))' | FileCheck %s

// gemmini.matmul_tile is a tensor-domain op today, so the
// gemmini-lower-tile-to-isa pass has nothing to bridge until a separate
// bufferization step lowers it to memrefs. This lit fixture is the
// minimal smoke that the pass is registered and runs without crashing on
// a function that contains a recovered (tensor-domain) matmul_tile op.
// See docs/dev_blog/2026-03-11-gemmini-workstream-log.md for the
// bufferization roadmap.

// CHECK-LABEL: func.func @matmul_tile_pass_through
func.func @matmul_tile_pass_through(%lhs: tensor<16x32xi8>, %rhs: tensor<64x32xi8>) -> tensor<16x64xi32> {
  // CHECK: gemmini.matmul_tile
  %0 = gemmini.matmul_tile %lhs, %rhs {dataflow = #gemmini.dataflow<os>, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64, tileM = 16 : i64, tileN = 16 : i64, tileK = 16 : i64} : tensor<16x32xi8>, tensor<64x32xi8> -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>
}
