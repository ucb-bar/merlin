"builtin.module"() ({
  "func.func"() <{sym_name = "torchao_choose_qparams_affine_default", function_type = (tensor<2x4xf32>) -> tensor<2xf32>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "torchao_quantize_affine_default", function_type = (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "body", function_type = (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>), sym_visibility = "private"}> ({
  ^bb0(%x: tensor<2x4xf32>):
    %s = "func.call"(%x) <{callee = @torchao_choose_qparams_affine_default}> {prov.region_id = "cq_0", prov.dispatch_id = "cq_0"} : (tensor<2x4xf32>) -> tensor<2xf32>
    %q = "func.call"(%x, %s) <{callee = @torchao_quantize_affine_default}> {prov.region_id = "qa_0", prov.dispatch_id = "qa_0"} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>
    "func.return"(%s, %q) : (tensor<2xf32>, tensor<2x4xi8>) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "forward", function_type = (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>)}> ({
  ^bb0(%x: tensor<2x4xf32>):
    %s, %q = "func.call"(%x) <{callee = @body}> : (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>)
    "func.return"(%s, %q) : (tensor<2xf32>, tensor<2x4xi8>) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "unrelated", function_type = () -> i32}> ({
    %c = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %d = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %sum = "arith.addi"(%c, %d) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%sum) : (i32) -> ()
  }) : () -> ()
}) {prov.quantization = "int8_dyn_act_int8_weight"} : () -> ()