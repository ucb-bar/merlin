func.func @forward(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) -> tensor<4x4xi32> {
  %c0 = arith.constant 0 : i32
  %dead0 = tensor.empty() : tensor<8x4xi8>
  %e = tensor.empty() : tensor<4x4xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
  %dead1 = linalg.transpose ins(%a : tensor<4x8xi8>) outs(%dead0 : tensor<8x4xi8>) permutation = [1, 0]
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<4x8xi8>, tensor<8x4xi8>) outs(%z : tensor<4x4xi32>) {
  ^bb0(%x: i8, %y: i8, %acc: i32):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %m = arith.muli %xe, %ye : i32
    %s = arith.addi %acc, %m : i32
    linalg.yield %s : i32
  } -> tensor<4x4xi32>
  %dead2 = arith.addi %c0, %c0 : i32
  return %r : tensor<4x4xi32>
}
