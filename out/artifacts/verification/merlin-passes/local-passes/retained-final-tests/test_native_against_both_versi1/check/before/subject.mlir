"builtin.module"() ({
  "func.func"() <{sym_name = "torchao_choose_qparams_affine_default", function_type = (tensor<1x128xf32>) -> tensor<1xf32>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "torchao_quantize_affine_default", function_type = (tensor<1x128xf32>, tensor<1xf32>) -> tensor<1x128xi8>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "forward", function_type = (tensor<1x128xf32>) -> (tensor<1xf32>, tensor<1x128xi8>)}> ({
  ^bb0(%x: tensor<1x128xf32>):
    %s = "func.call"(%x) <{callee = @torchao_choose_qparams_affine_default}> {prov.region_id = "cq_0", prov.dispatch_id = "cq_0"} : (tensor<1x128xf32>) -> tensor<1xf32>
    %q = "func.call"(%x, %s) <{callee = @torchao_quantize_affine_default}> {prov.region_id = "qa_0", prov.dispatch_id = "qa_0"} : (tensor<1x128xf32>, tensor<1xf32>) -> tensor<1x128xi8>
    "func.return"(%s, %q) : (tensor<1xf32>, tensor<1x128xi8>) -> ()
  }) : () -> ()
}) {prov.quantization = "int8_dyn_act_int8_weight"} : () -> ()