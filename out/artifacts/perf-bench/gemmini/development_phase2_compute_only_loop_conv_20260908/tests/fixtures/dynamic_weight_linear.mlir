builtin.module attributes {prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @generic_linear(%activation: tensor<3x5xf32>, %weight: tensor<5x7xi8>, %scale: tensor<7xf32>) -> tensor<3x7xf32> {
    %c0_i32 = arith.constant 0 : i32
    %zero_points = tensor.splat %c0_i32 : tensor<7xi32>
    %weight_f32 = "quant_ext.dequantize_per_channel"(%weight, %scale, %zero_points) <{axis = 1 : i64, input_dtype = "i8"}> : (tensor<5x7xi8>, tensor<7xf32>, tensor<7xi32>) -> tensor<5x7xf32>
    %empty = tensor.empty() : tensor<3x7xf32>
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<3x7xf32>) -> tensor<3x7xf32>
    %result = linalg.matmul {prov.region_id = "structural_probe"} ins(%activation, %weight_f32 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%zero : tensor<3x7xf32>) -> tensor<3x7xf32>
    return %result : tensor<3x7xf32>
  }
}
