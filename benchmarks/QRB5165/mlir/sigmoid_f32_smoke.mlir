// fp32 Sigmoid over [1,16]. The body shape (negate → exp → 1/(1+x))
// is what the elementwise unary recogniser identifies as Sigmoid.
module {
  func.func @sigmoid_f32(%a: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %one = arith.constant 1.0 : f32
    %init = tensor.empty() : tensor<1x16xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%a : tensor<1x16xf32>)
        outs(%init : tensor<1x16xf32>) {
      ^bb0(%la: f32, %lo: f32):
        %neg = arith.negf %la : f32
        %e = math.exp %neg : f32
        %d = arith.addf %one, %e : f32
        %s = arith.divf %one, %d : f32
        linalg.yield %s : f32
    } -> tensor<1x16xf32>
    return %out : tensor<1x16xf32>
  }
}
