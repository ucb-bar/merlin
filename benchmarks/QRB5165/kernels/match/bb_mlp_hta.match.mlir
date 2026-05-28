^bb0(%arg0: tensor<1x128xi8>, %weight: tensor<128x10xi8>):
%cst = arith.constant 0 : i32
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x10xi32>
%fill = linalg.fill ins(%cst : i32) outs(%init : tensor<1x10xi32>) -> tensor<1x10xi32>
%mm = linalg.matmul ins(%arg0, %weight : tensor<1x128xi8>, tensor<128x10xi8>) outs(%fill : tensor<1x10xi32>) -> tensor<1x10xi32>
%out_init = tensor.empty() {"match.operation_name_only"} : tensor<1x10xi8>
%out = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm : tensor<1x10xi32>) outs(%out_init : tensor<1x10xi8>) {
^bb0(%a: i32, %o: i8):
  %t = arith.trunci %a : i32 to i8
  linalg.yield %t : i8
} -> tensor<1x10xi8>
