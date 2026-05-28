util.func public @main(
    %lhs: tensor<32xi8>, %rhs: tensor<32xi8>
) -> tensor<32xi8> {
  %out = tensor.empty() : tensor<32xi8>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
  } ins(%lhs, %rhs : tensor<32xi8>, tensor<32xi8>)
    outs(%out : tensor<32xi8>) {
  ^bb0(%a: i8, %b: i8, %c: i8):
    %p = arith.muli %a, %b : i8
    linalg.yield %p : i8
  } -> tensor<32xi8>
  util.return %0 : tensor<32xi8>
}
