// Linalg-DAG match for uint8 elementwise add (1×16). Matches the canonical
// IREE form for `arith.addi` over two uint8 tensors. The QNN passthrough
// plugin substitutes the matched op with the prebuilt UFIXED_POINT_8 ctxbin
// authored in `kernels/abi/add_uint8.qnn.cpp` (HTA-compatible q-params).

^bb0(%arg0: tensor<1x16xi8>, %arg1: tensor<1x16xi8>):
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x16xi8>
%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x16xi8>, tensor<1x16xi8>)
    outs(%init : tensor<1x16xi8>) {
  ^bb0(%a: i8, %b: i8, %o: i8):
    %s = arith.addi %a, %b : i8
    linalg.yield %s : i8
} -> tensor<1x16xi8>
