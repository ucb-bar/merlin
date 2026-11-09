builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} {
  func.func @forward(%0: tensor<32x32xi8>, %1: tensor<32x32xi8>, %2: tensor<32x32xi8>, %3: tensor<32x32xi8>, %4: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %5 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00161476119 : f32
    %6 = tensor.splat %5 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %7 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %8 = tensor.splat %7 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %9 = "quant_ext.dequantize_per_tensor"(%0, %6, %8) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %10 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0293055251 : f32
    %11 = tensor.splat %10 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %12 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %13 = tensor.splat %12 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %14 = "quant_ext.quantize_per_tensor"(%4, %11, %13) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<32x32xf32>, tensor<f32>, tensor<i64>) -> tensor<32x32xi8>
    %15 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0293055251 : f32
    %16 = tensor.splat %15 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %17 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %18 = tensor.splat %17 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %19 = "quant_ext.dequantize_per_tensor"(%14, %16, %18) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %20 = tensor.empty() : tensor<32x32xf32>
    %21 = arith.constant 0.000000e+00 : f32
    %22 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%21 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %23 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%19, %9 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %24 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 8.723800e-03 : f32
    %25 = tensor.splat %24 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %26 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %27 = tensor.splat %26 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %28 = "quant_ext.quantize_per_tensor"(%23, %25, %27) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<32x32xf32>, tensor<f32>, tensor<i64>) -> tensor<32x32xi8>
    %29 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 8.723800e-03 : f32
    %30 = tensor.splat %29 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %31 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %32 = tensor.splat %31 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %33 = "quant_ext.dequantize_per_tensor"(%28, %30, %32) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %34 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 8.723800e-03 : f32
    %35 = tensor.splat %34 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %36 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %37 = tensor.splat %36 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %38 = "quant_ext.dequantize_per_tensor"(%28, %35, %37) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_3", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %39 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 8.723800e-03 : f32
    %40 = tensor.splat %39 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %41 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %42 = tensor.splat %41 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %43 = "quant_ext.dequantize_per_tensor"(%28, %40, %42) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_4", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %44 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00146657496 : f32
    %45 = tensor.splat %44 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %46 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %47 = tensor.splat %46 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %48 = "quant_ext.dequantize_per_tensor"(%1, %45, %47) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_5", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %49 = tensor.empty() : tensor<32x32xf32>
    %50 = arith.constant 0.000000e+00 : f32
    %51 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%50 : f32) outs(%49 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %52 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%43, %48 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%51 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %53 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00161170762 : f32
    %54 = tensor.splat %53 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %55 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %56 = tensor.splat %55 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %57 = "quant_ext.dequantize_per_tensor"(%2, %54, %56) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_6", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %58 = tensor.empty() : tensor<32x32xf32>
    %59 = arith.constant 0.000000e+00 : f32
    %60 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%59 : f32) outs(%58 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %61 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%38, %57 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%60 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %62 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00133199769 : f32
    %63 = tensor.splat %62 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %64 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %65 = tensor.splat %64 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %66 = "quant_ext.dequantize_per_tensor"(%3, %63, %65) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_7", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %67 = tensor.empty() : tensor<32x32xf32>
    %68 = arith.constant 0.000000e+00 : f32
    %69 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%68 : f32) outs(%67 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %70 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%33, %66 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%69 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %71 = tensor.empty() : tensor<32x32xf32>
    %72 = linalg.transpose ins(%61:tensor<32x32xf32>) outs(%71:tensor<32x32xf32>) permutation = [1, 0]
    %73 = tensor.empty() : tensor<32x32xf32>
    %74 = arith.constant 0.000000e+00 : f32
    %75 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%74 : f32) outs(%73 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %76 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%52, %72 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%75 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %77 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %78 = tensor.splat %77 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %79 = tensor.empty() : tensor<32x32xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %78 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%79 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%81: f32, %82: f32, %83: f32):
      %84 = arith.mulf %81, %82 : f32
      linalg.yield %84 : f32
    } -> tensor<32x32xf32>
    %85 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %86 = tensor.splat %85 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %87 = linalg.reduce ins(%80:tensor<32x32xf32>) outs(%86:tensor<32xf32>) dimensions = [1]
    (%88: f32, %89: f32) {
      %90 = arith.maximumf %88, %89 : f32
      linalg.yield %90 : f32
    }
    %91 = tensor.expand_shape %87 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %92 = tensor.empty() : tensor<32x32xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %91 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%92 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb1(%94: f32, %95: f32, %96: f32):
      %97 = arith.subf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<32x32xf32>
    %98 = tensor.empty() : tensor<32x32xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93 : tensor<32x32xf32>) outs(%98 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%100: f32, %101: f32):
      %102 = math.exp %100 : f32
      linalg.yield %102 : f32
    } -> tensor<32x32xf32>
    %103 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %104 = tensor.splat %103 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %105 = linalg.reduce ins(%99:tensor<32x32xf32>) outs(%104:tensor<32xf32>) dimensions = [1]
    (%106: f32, %107: f32) {
      %108 = arith.addf %106, %107 : f32
      linalg.yield %108 : f32
    }
    %109 = tensor.expand_shape %105 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %110 = tensor.empty() : tensor<32x32xf32>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %109 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%110 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%112: f32, %113: f32, %114: f32):
      %115 = arith.divf %112, %113 : f32
      linalg.yield %115 : f32
    } -> tensor<32x32xf32>
    %116 = tensor.empty() : tensor<32x32xf32>
    %117 = arith.constant 0.000000e+00 : f32
    %118 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%117 : f32) outs(%116 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %119 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%111, %70 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%118 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %120 = tensor.empty() : tensor<32x32xf32>
    %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %119 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%120 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%122: f32, %123: f32, %124: f32):
      %125 = arith.addf %122, %123 : f32
      linalg.yield %125 : f32
    } -> tensor<32x32xf32>
    %126 = tensor.empty() : tensor<32x32xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%121 : tensor<32x32xf32>) outs(%126 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb5(%128: f32, %129: f32):
      %130 = arith.constant 5.000000e-01 : f32
      %131 = arith.constant 1.000000e+00 : f32
      %132 = arith.constant 0.707106769 : f32
      %133 = arith.mulf %128, %132 : f32
      %134 = math.erf %133 : f32
      %135 = arith.addf %131, %134 : f32
      %136 = arith.mulf %130, %128 : f32
      %137 = arith.mulf %136, %135 : f32
      linalg.yield %137 : f32
    } -> tensor<32x32xf32>
    %138 = tensor.empty() : tensor<32x32xf32>
    %139 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%127 : tensor<32x32xf32>) outs(%138 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb6(%140: f32, %141: f32):
      %142 = arith.constant 2.000000e+00 : f32
      %143 = math.powf %140, %142 : f32
      linalg.yield %143 : f32
    } -> tensor<32x32xf32>
    %144 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %145 = tensor.splat %144 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %146 = linalg.reduce ins(%139:tensor<32x32xf32>) outs(%145:tensor<32xf32>) dimensions = [1]
    (%147: f32, %148: f32) {
      %149 = arith.addf %147, %148 : f32
      linalg.yield %149 : f32
    }
    %150 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %151 = tensor.splat %150 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %152 = tensor.empty() : tensor<32xf32>
    %153 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%146, %151 : tensor<32xf32>, tensor<32xf32>) outs(%152 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb7(%154: f32, %155: f32, %156: f32):
      %157 = arith.divf %154, %155 : f32
      linalg.yield %157 : f32
    } -> tensor<32xf32>
    %158 = tensor.expand_shape %153 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %159 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %160 = tensor.splat %159 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %161 = tensor.empty() : tensor<32x1xf32>
    %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%158, %160 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%161 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb8(%163: f32, %164: f32, %165: f32):
      %166 = arith.addf %163, %164 : f32
      linalg.yield %166 : f32
    } -> tensor<32x1xf32>
    %167 = tensor.empty() : tensor<32x1xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162 : tensor<32x1xf32>) outs(%167 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb9(%169: f32, %170: f32):
      %171 = math.rsqrt %169 : f32
      linalg.yield %171 : f32
    } -> tensor<32x1xf32>
    %172 = tensor.empty() : tensor<32x32xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%127, %168 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%172 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%174: f32, %175: f32, %176: f32):
      %177 = arith.mulf %174, %175 : f32
      linalg.yield %177 : f32
    } -> tensor<32x32xf32>
    %178 = tensor.empty() : tensor<32x32xf32>
    %179 = linalg.transpose ins(%173:tensor<32x32xf32>) outs(%178:tensor<32x32xf32>) permutation = [1, 0]
    %180 = tensor.empty() : tensor<32x32xf32>
    %181 = linalg.transpose ins(%179:tensor<32x32xf32>) outs(%180:tensor<32x32xf32>) permutation = [1, 0]
    %182 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %183 = tensor.splat %182 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %184 = linalg.reduce ins(%181:tensor<32x32xf32>) outs(%183:tensor<32xf32>) dimensions = [1]
    (%185: f32, %186: f32) {
      %187 = arith.addf %185, %186 : f32
      linalg.yield %187 : f32
    }
    %188 = tensor.expand_shape %184 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %189 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %190 = tensor.splat %189 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %191 = tensor.empty() : tensor<32x1xf32>
    %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%188, %190 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%191 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb11(%193: f32, %194: f32, %195: f32):
      %196 = arith.divf %193, %194 : f32
      linalg.yield %196 : f32
    } -> tensor<32x1xf32>
    %197 = tensor.empty() : tensor<32x32xf32>
    %198 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%181, %192 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%197 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb12(%199: f32, %200: f32, %201: f32):
      %202 = arith.subf %199, %200 : f32
      linalg.yield %202 : f32
    } -> tensor<32x32xf32>
    func.return %198 : tensor<32x32xf32>
  }
}
