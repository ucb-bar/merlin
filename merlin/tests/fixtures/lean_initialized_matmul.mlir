func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x2xi8>) -> tensor<2x2xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<2x2xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<2x2xi32>) -> tensor<2x2xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>, affine_map<(d0,d1,d2)->(d0,d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%z : tensor<2x2xi32>) {
  ^bb0(%x: i8, %y: i8, %acc: i32):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %m = arith.muli %xe, %ye : i32
    %s = arith.addi %acc, %m : i32
    linalg.yield %s : i32
  } -> tensor<2x2xi32>
  %unused = arith.addi %c0, %c0 : i32
  return %r : tensor<2x2xi32>
}
