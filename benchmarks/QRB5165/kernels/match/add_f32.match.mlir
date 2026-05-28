// Linalg-DAG match pattern for elementwise fp32 add over a 1×16 tensor.
//
// Recognises the canonical IREE form for `arith.addf` over two same-shape
// tensors, emitted by torch-mlir's onnx.Add lowering before any data-tiling
// or fusion. The match is deliberately literal — `cast_compatible_dag_from_root`
// matches by op chain + iterator types + indexing maps + arith body.
//
// We pin shape to 1×16 for this first-cut kernel; later kernels parameterise
// shape via `?` dynamic dims and a sibling kernel manifest entry for each
// shape class.

^bb0(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>):
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x16xf32>
%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x16xf32>, tensor<1x16xf32>)
    outs(%init : tensor<1x16xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
} -> tensor<1x16xf32>
