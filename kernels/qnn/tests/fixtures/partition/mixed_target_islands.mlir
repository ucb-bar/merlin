// Mixed-target adjacency: a conv (default → qnn-hta) followed by a
// concat (default → qnn-gpu). Each island gets its own target; the
// partitioner must not merge them. The post-EraseQNNVariantBodyPass
// IR-validity regression test (#108) builds on this fixture.

module {
  func.func @mixed_target_islands(
      %input_a: tensor<1x16x4x4xi8>,
      %input_b: tensor<1x16x4x4xi8>)
      -> tensor<1x32x4x4xf32> {
    %cst_in_scale = arith.constant 1.000000e-01 : f32

    %deq_init = tensor.empty() : tensor<1x16x4x4xf32>
    %deq_a = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input_a : tensor<1x16x4x4xi8>)
        outs(%deq_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_in_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    %deq_b = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input_b : tensor<1x16x4x4xi8>)
        outs(%deq_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_in_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    %concat = tensor.concat dim(1) %deq_a, %deq_b
        : (tensor<1x16x4x4xf32>, tensor<1x16x4x4xf32>)
        -> tensor<1x32x4x4xf32>

    return %concat : tensor<1x32x4x4xf32>
  }
}
