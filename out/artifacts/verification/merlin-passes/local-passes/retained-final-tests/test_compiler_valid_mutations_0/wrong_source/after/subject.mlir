"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>)}> ({
  ^bb0(%x: tensor<2x4xf32>):
    %0 = "tensor.empty"() : () -> tensor<2xf32>
    %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %2 = "linalg.fill"(%1, %0) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%3: f32, %4: f32):
      "linalg.yield"(%3) : (f32) -> ()
    }) : (f32, tensor<2xf32>) -> tensor<2xf32>
    %5 = "linalg.generic"(%x, %2) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%6: f32, %7: f32):
      %8 = "math.absf"(%6) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %9 = "arith.maximumf"(%7, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%9) : (f32) -> ()
    }) {prov.region_id = "cq_0", prov.dispatch_id = "cq_0", prov.transforms = "torchao_choose_qparams_affine"} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2xf32>
    %10 = "tensor.empty"() : () -> tensor<2xf32>
    %s = "linalg.generic"(%5, %10) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%11: f32, %12: f32):
      %13 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
      %14 = "arith.constant"() <{value = 1.000000e-05 : f32}> : () -> f32
      %15 = "arith.divf"(%11, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %16 = "arith.maximumf"(%15, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%16) : (f32) -> ()
    }) {prov.region_id = "cq_0", prov.dispatch_id = "cq_0", prov.transforms = "torchao_choose_qparams_affine"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %17 = "tensor.empty"() : () -> tensor<2x4xi8>
    %q = "linalg.generic"(%x, %s, %17) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%18: f32, %19: f32, %20: i8):
      %21 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %22 = "arith.constant"() <{value = -1.270000e+02 : f32}> : () -> f32
      %23 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
      %24 = "arith.divf"(%21, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %25 = "arith.mulf"(%19, %24) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %26 = "math.roundeven"(%25) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %27 = "arith.maximumf"(%26, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "arith.minimumf"(%27, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %29 = "arith.fptosi"(%28) : (f32) -> i8
      "linalg.yield"(%29) : (i8) -> ()
    }) {prov.region_id = "qa_0", prov.dispatch_id = "qa_0", prov.transforms = "torchao_quantize_affine"} : (tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xi8>) -> tensor<2x4xi8>
    "func.return"(%s, %q) : (tensor<2xf32>, tensor<2x4xi8>) -> ()
  }) : () -> ()
}) {prov.quantization = "int8_dyn_act_int8_weight"} : () -> ()