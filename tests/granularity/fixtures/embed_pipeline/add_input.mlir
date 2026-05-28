// Tiny synthetic input that contains a single 1-D elementwise add. The
// kernel manifest at the sibling manifest.json should match this op via
// linalg_dag and rewrite it into a flow.dispatch into the precompiled
// add_8xf32 kernel.

!ty = tensor<8xf32>

module {
  func.func @main(%lhs: !ty, %rhs: !ty) -> !ty {
    %empty = tensor.empty() : !ty
    %sum = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%lhs, %rhs : !ty, !ty)
        outs(%empty : !ty) {
      ^bb_inner(%a: f32, %b: f32, %_out: f32):
        %s = arith.addf %a, %b : f32
        linalg.yield %s : f32
    } -> !ty
    return %sum : !ty
  }
}
