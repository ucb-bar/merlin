^bb0(%arg0: tensor<64x128xf32>, %weight: tensor<128x128xf32>):
%cst = arith.constant 0.0 : f32
%init = tensor.empty() {"match.operation_name_only"} : tensor<64x128xf32>
%fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
%out = linalg.matmul ins(%arg0, %weight : tensor<64x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
