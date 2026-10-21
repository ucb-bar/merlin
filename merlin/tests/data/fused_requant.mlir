builtin.module {
  func.func @fused(%a: tensor<17x20xi8>, %b: tensor<20x19xi8>) -> tensor<17x19xi8> {
    %zero = arith.constant 0 : i32
    %init = tensor.splat %zero : tensor<17x19xi32>
    %acc = linalg.matmul ins(%a, %b : tensor<17x20xi8>, tensor<20x19xi8>) outs(%init : tensor<17x19xi32>) -> tensor<17x19xi32>
    %scale0 = arith.constant 0.5 : f32
    %scale1 = arith.constant 0.0625 : f32
    %lower = arith.constant -128.0 : f32
    %upper = arith.constant 127.0 : f32
    %bias = arith.constant dense<[-0.0625, -0.03125, 0.0, 0.03125, 0.0625, -0.0625, -0.03125, 0.0, 0.03125, 0.0625, -0.0625, -0.03125, 0.0, 0.03125, 0.0625, -0.0625, -0.03125, 0.0, 0.03125]> : tensor<19xf32>
    %empty = tensor.empty() : tensor<17x19xi8>
    %result = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc,%bias : tensor<17x19xi32>, tensor<19xf32>) outs(%empty : tensor<17x19xi8>) {
    ^bb0(%v: i32, %biasv: f32, %out: i8):
      %f = arith.sitofp %v : i32 to f32
      %m0 = arith.mulf %f, %scale0 : f32
      %m1 = arith.mulf %m0, %scale1 : f32
      %add = arith.addf %m1, %biasv : f32
      %r = math.roundeven %add : f32
      %lo = arith.maximumf %r, %lower : f32
      %hi = arith.minimumf %lo, %upper : f32
      %i = arith.fptosi %hi : f32 to i8
      linalg.yield %i : i8
    } -> tensor<17x19xi8>
    func.return %result : tensor<17x19xi8>
  }
}
