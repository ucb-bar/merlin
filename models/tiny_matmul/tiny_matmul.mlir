// Tiny single-matmul test for LOOP_WS debugging.
// Shape: M=16, N=16, K=16 — all aligned to DIM=16.
// Uses iree.abi.model = sync so the runner can invoke with just the input
// (matches mlp_wide's main_graph sync wrapper convention).

module {
  func.func @main_graph(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32>
      attributes {iree.abi.declaration = "sync func @main_graph(%input0: tensor<16x16xf32>) -> (%output0: tensor<16x16xf32>)"} {
    // Quantize f32 → i8 (zero point + scale absorbed as identity for simplicity)
    %c0_i8 = arith.constant 0 : i8
    %a_init = tensor.empty() : tensor<16x16xi8>
    %a_i8 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<16x16xf32>) outs(%a_init : tensor<16x16xi8>) {
    ^bb0(%in: f32, %out: i8):
      %0 = arith.fptosi %in : f32 to i8
      linalg.yield %0 : i8
    } -> tensor<16x16xi8>

    %b_cst = arith.constant dense<1> : tensor<16x16xi8>
    %c0_i32 = arith.constant 0 : i32
    %c_init = tensor.empty() : tensor<16x16xi32>
    %c_zero = linalg.fill ins(%c0_i32 : i32) outs(%c_init : tensor<16x16xi32>) -> tensor<16x16xi32>
    %c_i32 = linalg.matmul ins(%a_i8, %b_cst : tensor<16x16xi8>, tensor<16x16xi8>) outs(%c_zero : tensor<16x16xi32>) -> tensor<16x16xi32>

    // Dequantize i32 → f32
    %out_init = tensor.empty() : tensor<16x16xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%c_i32 : tensor<16x16xi32>) outs(%out_init : tensor<16x16xf32>) {
    ^bb0(%in: i32, %out: f32):
      %0 = arith.sitofp %in : i32 to f32
      linalg.yield %0 : f32
    } -> tensor<16x16xf32>

    return %out : tensor<16x16xf32>
  }
}
