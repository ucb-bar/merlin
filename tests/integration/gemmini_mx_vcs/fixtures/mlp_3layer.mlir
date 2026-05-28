// 3-layer MLP exercised by `./merlin compile --target gemmini_mx_vcs[_fp4]`.
//
// Shapes: 16 → 64 → 64 → 16
//   layer 1: linear         (no activation) — straight matmul
//   layer 2: linear + ReLU                  — matmul then clamp(0, +∞)
//   layer 3: linear + LayerNorm-style       — matmul then output requantize
//                                             (mxGemmini's CONFIG_EX
//                                             rs1[4:3] = 2 = LAYERNORM)
//
// The hidden dim 64 = 4 * DIM, exercising tile loops along K and M
// for both the FP8 and FP4 lowerings.
//
// Inputs are i8-typed at the buffer level — libgemmini handles FP8/FP4
// unpacking inside the systolic array via the format selectors set by
// CONFIG_EX bits [15:10]. Accumulator is i32 (mxGemmini's bf16 acc gets
// readback-quantized to i8 by the requantizer LUT, which matches the
// dialect's elem-bits=8 / acc-bits=32 plumbing).
//
// numpy reference: tests/integration/gemmini_mx_vcs/golden_model.py

func.func @mlp_3layer(%input: tensor<1x16xi8>,
    %w1: tensor<16x64xi8>,
    %w2: tensor<64x64xi8>,
    %w3: tensor<64x16xi8>) -> tensor<1x16xi32>
    attributes {iree.preserve_func_visibility = true} {
  %c0_i32 = arith.constant 0 : i32

  // Layer 1: 1x16 @ 16x64 → 1x64 (no activation).
  %init1 = tensor.empty() : tensor<1x64xi32>
  %fill1 = linalg.fill ins(%c0_i32 : i32) outs(%init1 : tensor<1x64xi32>)
      -> tensor<1x64xi32>
  %h1 = linalg.matmul ins(%input, %w1 : tensor<1x16xi8>, tensor<16x64xi8>)
      outs(%fill1 : tensor<1x64xi32>) -> tensor<1x64xi32>

  // ReLU: clamp at zero. Truncate i32 → i8 to feed back into the next
  // matmul's i8 input. This is what mxGemmini's output requantize
  // produces; we model it directly so the dialect doesn't have to know
  // about the LUT-load step.
  %h1_i8 = tensor.empty() : tensor<1x64xi8>
  %h1_relu = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%h1 : tensor<1x64xi32>) outs(%h1_i8 : tensor<1x64xi8>) {
    ^bb0(%in: i32, %out: i8):
      %z = arith.constant 0 : i32
      %is_neg = arith.cmpi slt, %in, %z : i32
      %clamped = arith.select %is_neg, %z, %in : i32
      %trunc = arith.trunci %clamped : i32 to i8
      linalg.yield %trunc : i8
    } -> tensor<1x64xi8>

  // Layer 2: 1x64 @ 64x64 → 1x64 (with ReLU pre-activation already).
  %init2 = tensor.empty() : tensor<1x64xi32>
  %fill2 = linalg.fill ins(%c0_i32 : i32) outs(%init2 : tensor<1x64xi32>)
      -> tensor<1x64xi32>
  %h2 = linalg.matmul ins(%h1_relu, %w2 : tensor<1x64xi8>, tensor<64x64xi8>)
      outs(%fill2 : tensor<1x64xi32>) -> tensor<1x64xi32>

  // Layer 3 input: requantize i32 → i8 with arithmetic shift right (a
  // very simple stand-in for mxGemmini's LAYERNORM/output requantize —
  // we avoid LayerNorm proper here because its lowering depends on the
  // CONFIG_NORM path which Phase 5 doesn't yet wire up).
  %h2_i8 = tensor.empty() : tensor<1x64xi8>
  %h2_q = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%h2 : tensor<1x64xi32>) outs(%h2_i8 : tensor<1x64xi8>) {
    ^bb0(%in: i32, %out: i8):
      %sh = arith.constant 8 : i32
      %shifted = arith.shrsi %in, %sh : i32
      %t = arith.trunci %shifted : i32 to i8
      linalg.yield %t : i8
    } -> tensor<1x64xi8>

  // Layer 3: 1x64 @ 64x16 → 1x16 (no further activation).
  %init3 = tensor.empty() : tensor<1x16xi32>
  %fill3 = linalg.fill ins(%c0_i32 : i32) outs(%init3 : tensor<1x16xi32>)
      -> tensor<1x16xi32>
  %h3 = linalg.matmul ins(%h2_q, %w3 : tensor<1x64xi8>, tensor<64x16xi8>)
      outs(%fill3 : tensor<1x16xi32>) -> tensor<1x16xi32>

  return %h3 : tensor<1x16xi32>
}
