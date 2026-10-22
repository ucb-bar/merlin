"builtin.module"() ({
  "func.func"() <{sym_name = "torchao_choose_qparams_affine_default", function_type = (tensor<2x4xf32>) -> tensor<2xf32>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "torchao_quantize_affine_default", function_type = (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "subject", function_type = (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>)}> ({
  ^bb0(%x: tensor<2x4xf32>):
    %s = "func.call"(%x) <{callee = @torchao_choose_qparams_affine_default}> : (tensor<2x4xf32>) -> tensor<2xf32>
    %q = "func.call"(%x, %s) <{callee = @torchao_quantize_affine_default}> : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>
    "func.return"(%s, %q) : (tensor<2xf32>, tensor<2x4xi8>) -> ()
  }) : () -> ()
}) {prov.quantization = "int8_dyn_act_int8_weight"} : () -> ()