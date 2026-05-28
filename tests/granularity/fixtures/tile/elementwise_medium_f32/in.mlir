util.func public @main(
    %lhs: tensor<256xf32>, %rhs: tensor<256xf32>
) -> tensor<256xf32> {
  %out = tensor.empty() : tensor<256xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
  } ins(%lhs, %rhs : tensor<256xf32>, tensor<256xf32>)
    outs(%out : tensor<256xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %p = arith.addf %a, %b : f32
    linalg.yield %p : f32
  } -> tensor<256xf32>
  util.return %0 : tensor<256xf32>
}
