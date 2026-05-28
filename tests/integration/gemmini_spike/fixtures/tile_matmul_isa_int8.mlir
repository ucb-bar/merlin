// Hand-authored ISA-tier fixture exercising the legalize-for-llvm-export
// path directly. This skips the (currently broken) tensor->memref
// bufferization step and demonstrates that gemmini.tile_matmul (memref
// inputs) lowers cleanly into the gemmini.intr.* RoCC intrinsic ops.
//
// See docs/dev_blog/2026-03-11-gemmini-workstream-log.md section 14.6
// for why this fixture is needed.

func.func @tile_matmul_8x8x8(%A: memref<16x16xi8, strided<[16, 1], offset: 0>>,
                              %B: memref<16x16xi8, strided<[16, 1], offset: 0>>,
                              %C: memref<16x16xi32, strided<[16, 1], offset: 0>>,
                              %D: memref<16x16xi32, strided<[16, 1], offset: 0>>)
    attributes {iree.preserve_func_visibility = true} {
  gemmini.tile_matmul %A, %B, %C, %D
    {aScaleFactor = 1.0 : f32, bScaleFactor = 1.0 : f32, dScaleFactor = 1.0 : f32,
     act = 0 : i64, accScale = 1.0 : f32, bertScale = 0.0 : f32,
     dataflow = 0 : i64}
    : memref<16x16xi8, strided<[16, 1], offset: 0>>,
      memref<16x16xi8, strided<[16, 1], offset: 0>>,
      memref<16x16xi32, strided<[16, 1], offset: 0>>,
      memref<16x16xi32, strided<[16, 1], offset: 0>>
  %skip = arith.constant 0 : i64
  gemmini.flush %skip
  return
}
