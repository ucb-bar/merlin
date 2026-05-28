// Elementwise fp32 Add over [1,16] — smallest non-trivial fixture for the
// elementwise binary recogniser. Mirrors the hand-authored
// benchmarks/QRB5165/kernels/abi/add_f32.qnn.cpp shape so we can compare
// emitter-produced .qnn-ctx vs hand-authored .qnn-ctx bytes-equal on GPU.

module {
  func.func @add_f32(%a: tensor<1x16xf32>, %b: tensor<1x16xf32>)
      -> tensor<1x16xf32> {
    %init = tensor.empty() : tensor<1x16xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%a, %b : tensor<1x16xf32>, tensor<1x16xf32>)
        outs(%init : tensor<1x16xf32>) {
      ^bb0(%la: f32, %lb: f32, %lo: f32):
        %s = arith.addf %la, %lb : f32
        linalg.yield %s : f32
    } -> tensor<1x16xf32>
    return %out : tensor<1x16xf32>
  }
}
