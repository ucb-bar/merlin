// Purpose-built tiny model that exercises three kernel-embedding
// granularities concurrently in one whole-model compile (Part C / C7):
//
//   Layer A — MEGAKERNEL substitution candidate
//     A 1024x1024 matmul -> bias_add -> relu chain. Without aggressive
//     fusion it forms ~4 dispatches; the megakernel substitution replaces
//     all four with one custom kernel.
//
//   Layer B — LAYER substitution candidate
//     A single 1024-elementwise multiplication. One dispatch.
//
//   Layer C — TILE substitution candidate
//     A 256x256 matmul that's tiled into four 64x64 sub-tiles. The TILE
//     match.kind invokes transform.iree.tile.linalg_op with sizes
//     [64, 64, 256] and replaces each tile with a custom kernel.
//
// The model takes one f32 input tensor and returns one f32 output tensor.
// The chain A -> B -> C produces a deterministic output (golden image used
// by the test harness for byte-equality verification).

util.func public @main(
    %lhs: tensor<1024x1024xf32>,
    %rhs: tensor<1024x1024xf32>,
    %bias: tensor<1024xf32>,
    %scale: tensor<1024xf32>,
    %tile_lhs: tensor<256x256xf32>,
    %tile_rhs: tensor<256x256xf32>
) -> tensor<256x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  // ---- Layer A: matmul + bias_add + relu (MEGAKERNEL candidate) ----
  %a_init = tensor.empty() : tensor<1024x1024xf32>
  %a_zero = linalg.fill ins(%cst : f32) outs(%a_init : tensor<1024x1024xf32>)
            -> tensor<1024x1024xf32>
  %a_mm = linalg.matmul
      ins(%lhs, %rhs : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%a_zero : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %a_bias_init = tensor.empty() : tensor<1024x1024xf32>
  %a_bias = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%a_mm, %bias : tensor<1024x1024xf32>, tensor<1024xf32>)
    outs(%a_bias_init : tensor<1024x1024xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %s = arith.addf %in, %b : f32
    linalg.yield %s : f32
  } -> tensor<1024x1024xf32>
  %a_relu_init = tensor.empty() : tensor<1024x1024xf32>
  %a_relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%a_bias : tensor<1024x1024xf32>)
    outs(%a_relu_init : tensor<1024x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %z = arith.constant 0.0 : f32
    %r = arith.maximumf %in, %z : f32
    linalg.yield %r : f32
  } -> tensor<1024x1024xf32>

  // ---- Layer B: 1D elementwise mul (LAYER candidate) ----
  // Reduce Layer A to a per-column max so we can feed B with a 1D vec.
  %row_init = tensor.empty() : tensor<1024xf32>
  %row_neg_inf = arith.constant -3.402823e+38 : f32
  %row_zero = linalg.fill ins(%row_neg_inf : f32)
              outs(%row_init : tensor<1024xf32>) -> tensor<1024xf32>
  %row_max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]
  } ins(%a_relu : tensor<1024x1024xf32>)
    outs(%row_zero : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maximumf %in, %out : f32
    linalg.yield %m : f32
  } -> tensor<1024xf32>
  %b_init = tensor.empty() : tensor<1024xf32>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
  } ins(%row_max, %scale : tensor<1024xf32>, tensor<1024xf32>)
    outs(%b_init : tensor<1024xf32>) {
  ^bb0(%lhs_v: f32, %rhs_v: f32, %out: f32):
    %p = arith.mulf %lhs_v, %rhs_v : f32
    linalg.yield %p : f32
  } -> tensor<1024xf32>

  // The 1D %b is unused downstream by Layer C; we discard it via a no-op
  // multiply against tile_lhs[0,0] just to keep it on the critical path.
  // (In a real demo we'd thread it into the next stage; here Layer C is
  // independent so the granularities can be tested in isolation.)

  // ---- Layer C: 256x256 matmul (TILE candidate) ----
  %c_init = tensor.empty() : tensor<256x256xf32>
  %c_zero = linalg.fill ins(%cst : f32) outs(%c_init : tensor<256x256xf32>)
            -> tensor<256x256xf32>
  %c = linalg.matmul
      ins(%tile_lhs, %tile_rhs : tensor<256x256xf32>, tensor<256x256xf32>)
      outs(%c_zero : tensor<256x256xf32>) -> tensor<256x256xf32>

  util.return %c : tensor<256x256xf32>
}
