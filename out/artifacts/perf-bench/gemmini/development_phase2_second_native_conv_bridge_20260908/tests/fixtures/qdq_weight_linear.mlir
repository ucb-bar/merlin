builtin.module attributes {prov.level = "linalg-on-tensors", prov.quantization = "explicit_qdq"} {
  func.func @generic_qdq_linear(%activation: tensor<3x5xf32>, %weight: tensor<5x7xi8>, %weight_scale: tensor<7xf32>) -> tensor<3x7xf32> {
    %c0_i32 = arith.constant 0 : i32
    %weight_zero_points = tensor.splat %c0_i32 : tensor<7xi32>
    %weight_f32 = "quant_ext.dequantize_per_channel"(%weight, %weight_scale, %weight_zero_points) <{axis = 1 : i64, input_dtype = "i8"}> : (tensor<5x7xi8>, tensor<7xf32>, tensor<7xi32>) -> tensor<5x7xf32>
    %cscale = arith.constant 1.250000e-01 : f32
    %activation_scale = tensor.splat %cscale : tensor<f32>
    %c0_i64 = arith.constant 0 : i64
    %activation_zero_point = tensor.splat %c0_i64 : tensor<i64>
    %activation_i8 = "quant_ext.quantize_per_tensor"(%activation, %activation_scale, %activation_zero_point) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<3x5xf32>, tensor<f32>, tensor<i64>) -> tensor<3x5xi8>
    %activation_f32 = "quant_ext.dequantize_per_tensor"(%activation_i8, %activation_scale, %activation_zero_point) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<3x5xi8>, tensor<f32>, tensor<i64>) -> tensor<3x5xf32>
    %empty = tensor.empty() : tensor<3x7xf32>
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<3x7xf32>) -> tensor<3x7xf32>
    %result = linalg.matmul {prov.region_id = "qdq_structural_probe"} ins(%activation_f32, %weight_f32 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%zero : tensor<3x7xf32>) -> tensor<3x7xf32>
    return %result : tensor<3x7xf32>
  }
}
