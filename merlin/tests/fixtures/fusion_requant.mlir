module {
  func.func @forward(%x: tensor<2x2xi8>, %w: tensor<2x2xi8>) -> tensor<2x2xi8> {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2x2xi32>
    %init = linalg.fill ins(%zero : i32) outs(%e : tensor<2x2xi32>) -> tensor<2x2xi32>
    %mm = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<2x2xi8>, tensor<2x2xi8>) outs(%init : tensor<2x2xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ax = arith.extsi %a : i8 to i32
      %bx = arith.extsi %b : i8 to i32
      %product = arith.muli %ax, %bx : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<2x2xi32>
    %scale = arith.constant 1.250000e-01 : f32
    %lower = arith.constant 0.000000e+00 : f32
    %upper = arith.constant 1.270000e+02 : f32
    %eq = tensor.empty() : tensor<2x2xi8>
    %q = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2x2xi32>) outs(%eq : tensor<2x2xi8>) {
    ^bb0(%a: i32, %o: i8):
      %f = arith.sitofp %a : i32 to f32
      %m = arith.mulf %f, %scale : f32
      %r = math.roundeven %m : f32
      %lo = arith.maximumf %r, %lower : f32
      %hi = arith.minimumf %lo, %upper : f32
      %v = arith.fptosi %hi : f32 to i8
      linalg.yield %v : i8
    } -> tensor<2x2xi8>
    func.return %q : tensor<2x2xi8>
  }
}
