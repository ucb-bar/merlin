module {
  func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2x4xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<2x4xi32>) -> tensor<2x4xi32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%f : tensor<2x4xi32>) {
    ^bb0(%x: i8, %y: i8, %s: i32):
      %u = arith.extsi %x : i8 to i32
      %v = arith.extsi %y : i8 to i32
      %product = arith.muli %u, %v : i32
      %sum = arith.addi %s, %product : i32
      linalg.yield %sum : i32
    } -> tensor<2x4xi32>
    return %r : tensor<2x4xi32>
  }
}
