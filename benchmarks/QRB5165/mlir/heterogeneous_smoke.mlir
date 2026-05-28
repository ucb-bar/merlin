// Heterogeneous QNN compile-flow smoke: a tiny model whose body intentionally
// matches multiple manifest entries so the kernel-embedding pipeline rewrites
// each into its own .qnn-ctx-backed flow.dispatch. Useful for inspecting:
//
//   1. Which dispatches got matched by which kernel (transform_spec.mlir).
//   2. Which .qnn-ctx files those dispatches point at (transform_spec.qnn_manifest.json).
//   3. The wrapped util.func @call_qnn_<name> stubs that bridge cast_and_call's
//      matched values to the flow.dispatch.
//
// Body: y = sigmoid(relu(a + b)) over a 1x16 fp32 tensor. Three matches expected:
// add_f32 -> relu_f32 -> sigmoid_f32. All three are in benchmarks/QRB5165/kernels/manifest.json.
module {
  func.func @heterogeneous_smoke(%a: tensor<1x16xf32>, %b: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %init0 = tensor.empty() : tensor<1x16xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%a, %b : tensor<1x16xf32>, tensor<1x16xf32>)
        outs(%init0 : tensor<1x16xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %s = arith.addf %x, %y : f32
        linalg.yield %s : f32
    } -> tensor<1x16xf32>

    %init1 = tensor.empty() : tensor<1x16xf32>
    %relu = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%sum : tensor<1x16xf32>)
        outs(%init1 : tensor<1x16xf32>) {
      ^bb0(%x: f32, %o: f32):
        %zero = arith.constant 0.0 : f32
        %s = arith.maximumf %x, %zero : f32
        linalg.yield %s : f32
    } -> tensor<1x16xf32>

    %init2 = tensor.empty() : tensor<1x16xf32>
    %one = arith.constant 1.0 : f32
    %sig = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%relu : tensor<1x16xf32>)
        outs(%init2 : tensor<1x16xf32>) {
      ^bb0(%x: f32, %o: f32):
        %neg = arith.negf %x : f32
        %exp = math.exp %neg : f32
        %denom = arith.addf %one, %exp : f32
        %sval = arith.divf %one, %denom : f32
        linalg.yield %sval : f32
    } -> tensor<1x16xf32>
    return %sig : tensor<1x16xf32>
  }
}
