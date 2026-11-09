builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} {
  func.func @forward(%0: tensor<256x128xf32>, %1: tensor<128xf32>, %2: tensor<128xf32>, %3: tensor<128xf32>, %4: tensor<128xf32>, %5: tensor<128xf32>, %6: tensor<128x128xi8>, %7: tensor<128x128xi8>, %8: tensor<128x128xi8>, %9: tensor<128x128xi8>, %10: tensor<344x128xi8>, %11: tensor<344x128xi8>, %12: tensor<128x344xi8>, %13: tensor<128x128xi8>, %14: tensor<128x128xi8>, %15: tensor<128x128xi8>, %16: tensor<128x128xi8>, %17: tensor<344x128xi8>, %18: tensor<344x128xi8>, %19: tensor<128x344xi8>, %20: tensor<256x128xi8>, %21: tensor<1x8xi64>) -> tensor<1x8x256xf32> {
    %22 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00069596176 : f32
    %23 = tensor.splat %22 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %24 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %25 = tensor.splat %24 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %26 = "quant_ext.dequantize_per_tensor"(%6, %23, %25) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %27 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695930212 : f32
    %28 = tensor.splat %27 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %29 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %30 = tensor.splat %29 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %31 = "quant_ext.dequantize_per_tensor"(%7, %28, %30) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %32 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 6.959600e-04 : f32
    %33 = tensor.splat %32 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %34 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %35 = tensor.splat %34 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %36 = "quant_ext.dequantize_per_tensor"(%8, %33, %35) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %37 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695862633 : f32
    %38 = tensor.splat %37 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %39 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %40 = tensor.splat %39 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %41 = "quant_ext.dequantize_per_tensor"(%9, %38, %40) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_3", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %42 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695961528 : f32
    %43 = tensor.splat %42 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %44 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %45 = tensor.splat %44 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %46 = "quant_ext.dequantize_per_tensor"(%10, %43, %45) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_4", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<344x128xi8>, tensor<f32>, tensor<i64>) -> tensor<344x128xf32>
    %47 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695960072 : f32
    %48 = tensor.splat %47 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %49 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %50 = tensor.splat %49 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %51 = "quant_ext.dequantize_per_tensor"(%11, %48, %50) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_5", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<344x128xi8>, tensor<f32>, tensor<i64>) -> tensor<344x128xf32>
    %52 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000424537167 : f32
    %53 = tensor.splat %52 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %54 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %55 = tensor.splat %54 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %56 = "quant_ext.dequantize_per_tensor"(%12, %53, %55) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_6", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x344xi8>, tensor<f32>, tensor<i64>) -> tensor<128x344xf32>
    %57 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695867289 : f32
    %58 = tensor.splat %57 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %59 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %60 = tensor.splat %59 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %61 = "quant_ext.dequantize_per_tensor"(%13, %58, %60) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_7", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %62 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695934927 : f32
    %63 = tensor.splat %62 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %64 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %65 = tensor.splat %64 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %66 = "quant_ext.dequantize_per_tensor"(%14, %63, %65) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_8", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %67 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695938768 : f32
    %68 = tensor.splat %67 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %69 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %70 = tensor.splat %69 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %71 = "quant_ext.dequantize_per_tensor"(%15, %68, %70) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_9", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %72 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0006959202 : f32
    %73 = tensor.splat %72 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %74 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %75 = tensor.splat %74 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %76 = "quant_ext.dequantize_per_tensor"(%16, %73, %75) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_10", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x128xi8>, tensor<f32>, tensor<i64>) -> tensor<128x128xf32>
    %77 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695969793 : f32
    %78 = tensor.splat %77 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %79 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %80 = tensor.splat %79 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %81 = "quant_ext.dequantize_per_tensor"(%17, %78, %80) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_11", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<344x128xi8>, tensor<f32>, tensor<i64>) -> tensor<344x128xf32>
    %82 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695939816 : f32
    %83 = tensor.splat %82 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %84 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %85 = tensor.splat %84 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %86 = "quant_ext.dequantize_per_tensor"(%18, %83, %85) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_12", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<344x128xi8>, tensor<f32>, tensor<i64>) -> tensor<344x128xf32>
    %87 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000424538332 : f32
    %88 = tensor.splat %87 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %89 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %90 = tensor.splat %89 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %91 = "quant_ext.dequantize_per_tensor"(%19, %88, %90) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_13", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x344xi8>, tensor<f32>, tensor<i64>) -> tensor<128x344xf32>
    %92 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000695921306 : f32
    %93 = tensor.splat %92 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %94 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %95 = tensor.splat %94 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %96 = "quant_ext.dequantize_per_tensor"(%20, %93, %95) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_14", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<256x128xi8>, tensor<f32>, tensor<i64>) -> tensor<256x128xf32>
    %97 = tensor.empty() : tensor<8xi64>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%97 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%99: i64):
      %100 = linalg.index 0 : index
      %101 = arith.index_cast %100 : index to i64
      %102 = arith.constant 1 : i64
      %103 = arith.muli %101, %102 : i64
      %104 = arith.constant 0 : i64
      %105 = arith.addi %104, %103 : i64
      linalg.yield %105 : i64
    } -> tensor<8xi64>
    %106 = tensor.empty() : tensor<1x8x128xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x8xi64>) outs(%106 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "emb", prov.fqn = "emb"} {
    ^bb1(%108: i64, %109: f32):
      %110 = arith.index_cast %108 : i64 to index
      %111 = linalg.index 2 : index
      %112 = tensor.extract %0[%110, %111] : tensor<256x128xf32>
      linalg.yield %112 : f32
    } -> tensor<1x8x128xf32>
    %113 = tensor.empty() : tensor<1x8x128xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107 : tensor<1x8x128xf32>) outs(%113 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb2(%115: f32, %116: f32):
      %117 = arith.constant 2.000000e+00 : f32
      %118 = math.powf %115, %117 : f32
      linalg.yield %118 : f32
    } -> tensor<1x8x128xf32>
    %119 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 0.000000e+00 : f32
    %120 = tensor.splat %119 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %121 = linalg.reduce ins(%114:tensor<1x8x128xf32>) outs(%120:tensor<1x8xf32>) dimensions = [2]
    (%122: f32, %123: f32) {
      %124 = arith.addf %122, %123 : f32
      linalg.yield %124 : f32
    }
    %125 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.280000e+02 : f32
    %126 = tensor.splat %125 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %127 = tensor.empty() : tensor<1x8xf32>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%121, %126 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%127 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb3(%129: f32, %130: f32, %131: f32):
      %132 = arith.divf %129, %130 : f32
      linalg.yield %132 : f32
    } -> tensor<1x8xf32>
    %133 = tensor.collapse_shape %128 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %134 = tensor.expand_shape %133 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %135 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.000000e-05 : f32
    %136 = tensor.splat %135 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8x1xf32>
    %137 = tensor.empty() : tensor<1x8x1xf32>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%134, %136 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%137 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb4(%139: f32, %140: f32, %141: f32):
      %142 = arith.addf %139, %140 : f32
      linalg.yield %142 : f32
    } -> tensor<1x8x1xf32>
    %143 = tensor.empty() : tensor<1x8x1xf32>
    %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%138 : tensor<1x8x1xf32>) outs(%143 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb5(%145: f32, %146: f32):
      %147 = math.rsqrt %145 : f32
      linalg.yield %147 : f32
    } -> tensor<1x8x1xf32>
    %148 = tensor.empty() : tensor<1x8x128xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107, %144 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%148 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb6(%150: f32, %151: f32, %152: f32):
      %153 = arith.mulf %150, %151 : f32
      linalg.yield %153 : f32
    } -> tensor<1x8x128xf32>
    %154 = tensor.empty() : tensor<1x8x128xf32>
    %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%149, %1 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%154 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb7(%156: f32, %157: f32, %158: f32):
      %159 = arith.mulf %156, %157 : f32
      linalg.yield %159 : f32
    } -> tensor<1x8x128xf32>
    %160 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0274261627 : f32
    %161 = tensor.splat %160 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %162 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %163 = tensor.splat %162 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %164 = "quant_ext.quantize_per_tensor"(%155, %161, %163) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %165 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0274261627 : f32
    %166 = tensor.splat %165 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %167 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %168 = tensor.splat %167 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %169 = "quant_ext.dequantize_per_tensor"(%164, %166, %168) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_15", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %170 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0274261627 : f32
    %171 = tensor.splat %170 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %172 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %173 = tensor.splat %172 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %174 = "quant_ext.dequantize_per_tensor"(%164, %171, %173) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_16", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %175 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0274261627 : f32
    %176 = tensor.splat %175 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %177 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %178 = tensor.splat %177 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %179 = "quant_ext.dequantize_per_tensor"(%164, %176, %178) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_17", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %180 = tensor.empty() : tensor<128x128xf32>
    %181 = linalg.transpose ins(%26:tensor<128x128xf32>) outs(%180:tensor<128x128xf32>) permutation = [1, 0]
    %182 = tensor.collapse_shape %179 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %183 = tensor.expand_shape %182 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %184 = tensor.empty() : tensor<8x128xf32>
    %185 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %186 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%185 : f32) outs(%184 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %187 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q", prov.transposed_b = "true"} ins(%183, %181 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%186 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %188 = tensor.collapse_shape %187 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %189 = tensor.expand_shape %188 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %190 = tensor.collapse_shape %189 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %191 = tensor.expand_shape %190 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %192 = tensor.empty() : tensor<1x4x8x32xf32>
    %193 = linalg.transpose ins(%191:tensor<1x8x4x32xf32>) outs(%192:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %194 = tensor.empty() : tensor<128x128xf32>
    %195 = linalg.transpose ins(%31:tensor<128x128xf32>) outs(%194:tensor<128x128xf32>) permutation = [1, 0]
    %196 = tensor.collapse_shape %174 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %197 = tensor.expand_shape %196 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %198 = tensor.empty() : tensor<8x128xf32>
    %199 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %200 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%199 : f32) outs(%198 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %201 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k", prov.transposed_b = "true"} ins(%197, %195 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%200 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %202 = tensor.collapse_shape %201 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %203 = tensor.expand_shape %202 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %204 = tensor.collapse_shape %203 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %205 = tensor.expand_shape %204 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %206 = tensor.empty() : tensor<1x4x8x32xf32>
    %207 = linalg.transpose ins(%205:tensor<1x8x4x32xf32>) outs(%206:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %208 = tensor.empty() : tensor<128x128xf32>
    %209 = linalg.transpose ins(%36:tensor<128x128xf32>) outs(%208:tensor<128x128xf32>) permutation = [1, 0]
    %210 = tensor.collapse_shape %169 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %211 = tensor.expand_shape %210 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %212 = tensor.empty() : tensor<8x128xf32>
    %213 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %214 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%213 : f32) outs(%212 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %215 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v", prov.transposed_b = "true"} ins(%211, %209 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%214 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %216 = tensor.collapse_shape %215 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %217 = tensor.expand_shape %216 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %218 = tensor.collapse_shape %217 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %219 = tensor.expand_shape %218 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %220 = tensor.empty() : tensor<1x4x8x32xf32>
    %221 = linalg.transpose ins(%219:tensor<1x8x4x32xf32>) outs(%220:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %222 = tensor.empty() : tensor<16xf32>
    %223 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%222 : tensor<16xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb8(%224: f32):
      %225 = linalg.index 0 : index
      %226 = arith.index_cast %225 : index to i64
      %227 = arith.sitofp %226 : i64 to f32
      %228 = arith.constant 1.000000e+00 : f32
      %229 = arith.mulf %227, %228 : f32
      %230 = arith.constant 0.000000e+00 : f32
      %231 = arith.addf %230, %229 : f32
      linalg.yield %231 : f32
    } -> tensor<16xf32>
    %232 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %233 = tensor.splat %232 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %234 = tensor.empty() : tensor<16xf32>
    %235 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%223, %233 : tensor<16xf32>, tensor<16xf32>) outs(%234 : tensor<16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb9(%236: f32, %237: f32, %238: f32):
      %239 = arith.divf %236, %237 : f32
      linalg.yield %239 : f32
    } -> tensor<16xf32>
    %240 = tensor.empty() : tensor<16xf32>
    %241 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%235 : tensor<16xf32>) outs(%240 : tensor<16xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb10(%242: f32, %243: f32):
      %244 = arith.constant 1.000000e+04 : f32
      %245 = math.powf %244, %242 : f32
      linalg.yield %245 : f32
    } -> tensor<16xf32>
    %246 = tensor.empty() : tensor<16xf32>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%241 : tensor<16xf32>) outs(%246 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb11(%248: f32, %249: f32):
      %250 = arith.constant 1.000000e+00 : f32
      %251 = arith.divf %250, %248 : f32
      linalg.yield %251 : f32
    } -> tensor<16xf32>
    %252 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %253 = tensor.splat %252 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %254 = tensor.empty() : tensor<16xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%247, %253 : tensor<16xf32>, tensor<16xf32>) outs(%254 : tensor<16xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb12(%256: f32, %257: f32, %258: f32):
      %259 = arith.mulf %256, %257 : f32
      linalg.yield %259 : f32
    } -> tensor<16xf32>
    %260 = tensor.expand_shape %98 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %261 = tensor.empty() : tensor<8x1xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%260 : tensor<8x1xi64>) outs(%261 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb13(%263: i64, %264: f32):
      %265 = arith.sitofp %263 : i64 to f32
      linalg.yield %265 : f32
    } -> tensor<8x1xf32>
    %266 = tensor.expand_shape %255 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %267 = tensor.empty() : tensor<8x16xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%262, %266 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%267 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb14(%269: f32, %270: f32, %271: f32):
      %272 = arith.mulf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<8x16xf32>
    %273 = tensor.empty() : tensor<8x16xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268 : tensor<8x16xf32>) outs(%273 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb15(%275: f32, %276: f32):
      %277 = math.cos %275 : f32
      linalg.yield %277 : f32
    } -> tensor<8x16xf32>
    %278 = tensor.empty() : tensor<8x16xf32>
    %279 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268 : tensor<8x16xf32>) outs(%278 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb16(%280: f32, %281: f32):
      %282 = math.cos %280 : f32
      linalg.yield %282 : f32
    } -> tensor<8x16xf32>
    %283 = tensor.concat dim(1) %274, %279 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %284 = tensor.collapse_shape %283 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %285 = tensor.expand_shape %284 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %286 = tensor.collapse_shape %285 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %287 = tensor.expand_shape %286 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %288 = tensor.empty() : tensor<8x16xf32>
    %289 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268 : tensor<8x16xf32>) outs(%288 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb17(%290: f32, %291: f32):
      %292 = math.sin %290 : f32
      linalg.yield %292 : f32
    } -> tensor<8x16xf32>
    %293 = tensor.empty() : tensor<8x16xf32>
    %294 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268 : tensor<8x16xf32>) outs(%293 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb18(%295: f32, %296: f32):
      %297 = math.sin %295 : f32
      linalg.yield %297 : f32
    } -> tensor<8x16xf32>
    %298 = tensor.concat dim(1) %289, %294 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %299 = tensor.collapse_shape %298 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %300 = tensor.expand_shape %299 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %301 = tensor.collapse_shape %300 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %302 = tensor.expand_shape %301 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %303 = "tensor.extract_slice"(%193) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %304 = "tensor.extract_slice"(%193) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %305 = tensor.empty() : tensor<1x4x8x16xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304 : tensor<1x4x8x16xf32>) outs(%305 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb19(%307: f32, %308: f32):
      %309 = arith.negf %307 : f32
      linalg.yield %309 : f32
    } -> tensor<1x4x8x16xf32>
    %310 = tensor.concat dim(3) %306, %303 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %311 = tensor.empty() : tensor<1x4x8x32xf32>
    %312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%193, %287 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%311 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb20(%313: f32, %314: f32, %315: f32):
      %316 = arith.mulf %313, %314 : f32
      linalg.yield %316 : f32
    } -> tensor<1x4x8x32xf32>
    %317 = tensor.empty() : tensor<1x4x8x32xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %302 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%317 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb21(%319: f32, %320: f32, %321: f32):
      %322 = arith.mulf %319, %320 : f32
      linalg.yield %322 : f32
    } -> tensor<1x4x8x32xf32>
    %323 = tensor.empty() : tensor<1x4x8x32xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312, %318 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%323 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb22(%325: f32, %326: f32, %327: f32):
      %328 = arith.addf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x4x8x32xf32>
    %329 = tensor.empty() : tensor<16xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%329 : tensor<16xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb23(%331: f32):
      %332 = linalg.index 0 : index
      %333 = arith.index_cast %332 : index to i64
      %334 = arith.sitofp %333 : i64 to f32
      %335 = arith.constant 1.000000e+00 : f32
      %336 = arith.mulf %334, %335 : f32
      %337 = arith.constant 0.000000e+00 : f32
      %338 = arith.addf %337, %336 : f32
      linalg.yield %338 : f32
    } -> tensor<16xf32>
    %339 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %340 = tensor.splat %339 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %341 = tensor.empty() : tensor<16xf32>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%330, %340 : tensor<16xf32>, tensor<16xf32>) outs(%341 : tensor<16xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb24(%343: f32, %344: f32, %345: f32):
      %346 = arith.divf %343, %344 : f32
      linalg.yield %346 : f32
    } -> tensor<16xf32>
    %347 = tensor.empty() : tensor<16xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%342 : tensor<16xf32>) outs(%347 : tensor<16xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb25(%349: f32, %350: f32):
      %351 = arith.constant 1.000000e+04 : f32
      %352 = math.powf %351, %349 : f32
      linalg.yield %352 : f32
    } -> tensor<16xf32>
    %353 = tensor.empty() : tensor<16xf32>
    %354 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%348 : tensor<16xf32>) outs(%353 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb26(%355: f32, %356: f32):
      %357 = arith.constant 1.000000e+00 : f32
      %358 = arith.divf %357, %355 : f32
      linalg.yield %358 : f32
    } -> tensor<16xf32>
    %359 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %360 = tensor.splat %359 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %361 = tensor.empty() : tensor<16xf32>
    %362 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%354, %360 : tensor<16xf32>, tensor<16xf32>) outs(%361 : tensor<16xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb27(%363: f32, %364: f32, %365: f32):
      %366 = arith.mulf %363, %364 : f32
      linalg.yield %366 : f32
    } -> tensor<16xf32>
    %367 = tensor.expand_shape %98 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %368 = tensor.empty() : tensor<8x1xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%367 : tensor<8x1xi64>) outs(%368 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb28(%370: i64, %371: f32):
      %372 = arith.sitofp %370 : i64 to f32
      linalg.yield %372 : f32
    } -> tensor<8x1xf32>
    %373 = tensor.expand_shape %362 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %374 = tensor.empty() : tensor<8x16xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%369, %373 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%374 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb29(%376: f32, %377: f32, %378: f32):
      %379 = arith.mulf %376, %377 : f32
      linalg.yield %379 : f32
    } -> tensor<8x16xf32>
    %380 = tensor.empty() : tensor<8x16xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%375 : tensor<8x16xf32>) outs(%380 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb30(%382: f32, %383: f32):
      %384 = math.cos %382 : f32
      linalg.yield %384 : f32
    } -> tensor<8x16xf32>
    %385 = tensor.empty() : tensor<8x16xf32>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%375 : tensor<8x16xf32>) outs(%385 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb31(%387: f32, %388: f32):
      %389 = math.cos %387 : f32
      linalg.yield %389 : f32
    } -> tensor<8x16xf32>
    %390 = tensor.concat dim(1) %381, %386 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %391 = tensor.collapse_shape %390 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %392 = tensor.expand_shape %391 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %393 = tensor.collapse_shape %392 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %394 = tensor.expand_shape %393 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %395 = tensor.empty() : tensor<8x16xf32>
    %396 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%375 : tensor<8x16xf32>) outs(%395 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb32(%397: f32, %398: f32):
      %399 = math.sin %397 : f32
      linalg.yield %399 : f32
    } -> tensor<8x16xf32>
    %400 = tensor.empty() : tensor<8x16xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%375 : tensor<8x16xf32>) outs(%400 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb33(%402: f32, %403: f32):
      %404 = math.sin %402 : f32
      linalg.yield %404 : f32
    } -> tensor<8x16xf32>
    %405 = tensor.concat dim(1) %396, %401 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %406 = tensor.collapse_shape %405 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %407 = tensor.expand_shape %406 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %408 = tensor.collapse_shape %407 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %409 = tensor.expand_shape %408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %410 = "tensor.extract_slice"(%207) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %411 = "tensor.extract_slice"(%207) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %412 = tensor.empty() : tensor<1x4x8x16xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x4x8x16xf32>) outs(%412 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb34(%414: f32, %415: f32):
      %416 = arith.negf %414 : f32
      linalg.yield %416 : f32
    } -> tensor<1x4x8x16xf32>
    %417 = tensor.concat dim(3) %413, %410 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %418 = tensor.empty() : tensor<1x4x8x32xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%207, %394 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%418 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb35(%420: f32, %421: f32, %422: f32):
      %423 = arith.mulf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x4x8x32xf32>
    %424 = tensor.empty() : tensor<1x4x8x32xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%417, %409 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%424 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb36(%426: f32, %427: f32, %428: f32):
      %429 = arith.mulf %426, %427 : f32
      linalg.yield %429 : f32
    } -> tensor<1x4x8x32xf32>
    %430 = tensor.empty() : tensor<1x4x8x32xf32>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %425 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%430 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb37(%432: f32, %433: f32, %434: f32):
      %435 = arith.addf %432, %433 : f32
      linalg.yield %435 : f32
    } -> tensor<1x4x8x32xf32>
    %436 = tensor.empty() : tensor<1x4x32x8xf32>
    %437 = linalg.transpose ins(%431:tensor<1x4x8x32xf32>) outs(%436:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %438 = tensor.empty() : tensor<1x4x8x32xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324 : tensor<1x4x8x32xf32>) outs(%438 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb38(%440: f32, %441: f32):
      linalg.yield %440 : f32
    } -> tensor<1x4x8x32xf32>
    %442 = tensor.collapse_shape %439 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %443 = tensor.expand_shape %442 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %444 = tensor.empty() : tensor<1x4x32x8xf32>
    %445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%437 : tensor<1x4x32x8xf32>) outs(%444 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb39(%446: f32, %447: f32):
      linalg.yield %446 : f32
    } -> tensor<1x4x32x8xf32>
    %448 = tensor.collapse_shape %445 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %449 = tensor.expand_shape %448 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %450 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %451 = tensor.splat %450 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%443, %449 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%451 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb40(%453: f32, %454: f32, %455: f32):
      %456 = arith.mulf %453, %454 : f32
      %457 = arith.addf %455, %456 : f32
      linalg.yield %457 : f32
    } -> tensor<4x8x8xf32>
    %458 = tensor.collapse_shape %452 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %459 = tensor.expand_shape %458 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %460 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 5.65685415 : f32
    %461 = tensor.splat %460 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32>
    %462 = tensor.empty() : tensor<1x4x8x8xf32>
    %463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459, %461 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%462 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb41(%464: f32, %465: f32, %466: f32):
      %467 = arith.divf %464, %465 : f32
      linalg.yield %467 : f32
    } -> tensor<1x4x8x8xf32>
    %468 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %469 = tensor.splat %468 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xf32>
    %470 = tensor.empty() : tensor<8xi64>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%470 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb42(%472: i64):
      %473 = linalg.index 0 : index
      %474 = arith.index_cast %473 : index to i64
      %475 = arith.constant 1 : i64
      %476 = arith.muli %474, %475 : i64
      %477 = arith.constant 0 : i64
      %478 = arith.addi %477, %476 : i64
      linalg.yield %478 : i64
    } -> tensor<8xi64>
    %479 = tensor.expand_shape %471 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %480 = tensor.empty() : tensor<8xi64>
    %481 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%480 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb43(%482: i64):
      %483 = linalg.index 0 : index
      %484 = arith.index_cast %483 : index to i64
      %485 = arith.constant 1 : i64
      %486 = arith.muli %484, %485 : i64
      %487 = arith.constant 0 : i64
      %488 = arith.addi %487, %486 : i64
      linalg.yield %488 : i64
    } -> tensor<8xi64>
    %489 = tensor.expand_shape %481 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %490 = tensor.empty() : tensor<8x8xi64>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%479, %489 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%490 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb44(%492: i64, %493: i64, %494: i64):
      %495 = arith.subi %492, %493 : i64
      linalg.yield %495 : i64
    } -> tensor<8x8xi64>
    %496 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1 : i64
    %497 = tensor.splat %496 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xi64>
    %498 = tensor.empty() : tensor<8x8xi1>
    %499 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%491, %497 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%498 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb45(%500: i64, %501: i64, %502: i1):
      %503 = arith.cmpi sge, %500, %501 : i64
      linalg.yield %503 : i1
    } -> tensor<8x8xi1>
    %504 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %505 = tensor.splat %504 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<f32>
    %506 = tensor.empty() : tensor<8x8xf32>
    %507 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%499, %469, %505 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%506 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb46(%508: i1, %509: f32, %510: f32, %511: f32):
      %512 = arith.select %508, %509, %510 : f32
      linalg.yield %512 : f32
    } -> tensor<8x8xf32>
    %513 = tensor.empty() : tensor<1x4x8x8xf32>
    %514 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %507 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%513 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb47(%515: f32, %516: f32, %517: f32):
      %518 = arith.addf %515, %516 : f32
      linalg.yield %518 : f32
    } -> tensor<1x4x8x8xf32>
    %519 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %520 = tensor.splat %519 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %521 = linalg.reduce ins(%514:tensor<1x4x8x8xf32>) outs(%520:tensor<1x4x8xf32>) dimensions = [3]
    (%522: f32, %523: f32) {
      %524 = arith.maximumf %522, %523 : f32
      linalg.yield %524 : f32
    }
    %525 = tensor.collapse_shape %521 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %526 = tensor.expand_shape %525 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %527 = tensor.empty() : tensor<1x4x8x8xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%514, %526 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%527 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb48(%529: f32, %530: f32, %531: f32):
      %532 = arith.subf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<1x4x8x8xf32>
    %533 = tensor.empty() : tensor<1x4x8x8xf32>
    %534 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528 : tensor<1x4x8x8xf32>) outs(%533 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb49(%535: f32, %536: f32):
      %537 = math.exp %535 : f32
      linalg.yield %537 : f32
    } -> tensor<1x4x8x8xf32>
    %538 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %539 = tensor.splat %538 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %540 = linalg.reduce ins(%534:tensor<1x4x8x8xf32>) outs(%539:tensor<1x4x8xf32>) dimensions = [3]
    (%541: f32, %542: f32) {
      %543 = arith.addf %541, %542 : f32
      linalg.yield %543 : f32
    }
    %544 = tensor.collapse_shape %540 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %545 = tensor.expand_shape %544 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %546 = tensor.empty() : tensor<1x4x8x8xf32>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534, %545 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%546 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb50(%548: f32, %549: f32, %550: f32):
      %551 = arith.divf %548, %549 : f32
      linalg.yield %551 : f32
    } -> tensor<1x4x8x8xf32>
    %552 = tensor.empty() : tensor<1x4x8x8xf32>
    %553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%547 : tensor<1x4x8x8xf32>) outs(%552 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb51(%554: f32, %555: f32):
      linalg.yield %554 : f32
    } -> tensor<1x4x8x8xf32>
    %556 = tensor.collapse_shape %553 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %557 = tensor.expand_shape %556 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %558 = tensor.empty() : tensor<1x4x8x32xf32>
    %559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%221 : tensor<1x4x8x32xf32>) outs(%558 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb52(%560: f32, %561: f32):
      linalg.yield %560 : f32
    } -> tensor<1x4x8x32xf32>
    %562 = tensor.collapse_shape %559 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %563 = tensor.expand_shape %562 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %564 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %565 = tensor.splat %564 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%557, %563 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%565 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb53(%567: f32, %568: f32, %569: f32):
      %570 = arith.mulf %567, %568 : f32
      %571 = arith.addf %569, %570 : f32
      linalg.yield %571 : f32
    } -> tensor<4x8x32xf32>
    %572 = tensor.collapse_shape %566 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %573 = tensor.expand_shape %572 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %574 = tensor.empty() : tensor<1x8x4x32xf32>
    %575 = linalg.transpose ins(%573:tensor<1x4x8x32xf32>) outs(%574:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %576 = tensor.collapse_shape %575 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %577 = tensor.expand_shape %576 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %578 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0113536799 : f32
    %579 = tensor.splat %578 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %580 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %581 = tensor.splat %580 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %582 = "quant_ext.quantize_per_tensor"(%577, %579, %581) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %583 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0113536799 : f32
    %584 = tensor.splat %583 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %585 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %586 = tensor.splat %585 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %587 = "quant_ext.dequantize_per_tensor"(%582, %584, %586) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_18", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %588 = tensor.empty() : tensor<128x128xf32>
    %589 = linalg.transpose ins(%41:tensor<128x128xf32>) outs(%588:tensor<128x128xf32>) permutation = [1, 0]
    %590 = tensor.collapse_shape %587 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %591 = tensor.expand_shape %590 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %592 = tensor.empty() : tensor<8x128xf32>
    %593 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %594 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%593 : f32) outs(%592 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %595 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o", prov.transposed_b = "true"} ins(%591, %589 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%594 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %596 = tensor.collapse_shape %595 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %597 = tensor.expand_shape %596 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %598 = tensor.empty() : tensor<1x8x128xf32>
    %599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107, %597 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%598 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb54(%600: f32, %601: f32, %602: f32):
      %603 = arith.addf %600, %601 : f32
      linalg.yield %603 : f32
    } -> tensor<1x8x128xf32>
    %604 = tensor.empty() : tensor<1x8x128xf32>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599 : tensor<1x8x128xf32>) outs(%604 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb55(%606: f32, %607: f32):
      %608 = arith.constant 2.000000e+00 : f32
      %609 = math.powf %606, %608 : f32
      linalg.yield %609 : f32
    } -> tensor<1x8x128xf32>
    %610 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 0.000000e+00 : f32
    %611 = tensor.splat %610 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %612 = linalg.reduce ins(%605:tensor<1x8x128xf32>) outs(%611:tensor<1x8xf32>) dimensions = [2]
    (%613: f32, %614: f32) {
      %615 = arith.addf %613, %614 : f32
      linalg.yield %615 : f32
    }
    %616 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.280000e+02 : f32
    %617 = tensor.splat %616 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %618 = tensor.empty() : tensor<1x8xf32>
    %619 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%612, %617 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%618 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb56(%620: f32, %621: f32, %622: f32):
      %623 = arith.divf %620, %621 : f32
      linalg.yield %623 : f32
    } -> tensor<1x8xf32>
    %624 = tensor.collapse_shape %619 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %625 = tensor.expand_shape %624 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %626 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.000000e-05 : f32
    %627 = tensor.splat %626 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8x1xf32>
    %628 = tensor.empty() : tensor<1x8x1xf32>
    %629 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%625, %627 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%628 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb57(%630: f32, %631: f32, %632: f32):
      %633 = arith.addf %630, %631 : f32
      linalg.yield %633 : f32
    } -> tensor<1x8x1xf32>
    %634 = tensor.empty() : tensor<1x8x1xf32>
    %635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629 : tensor<1x8x1xf32>) outs(%634 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb58(%636: f32, %637: f32):
      %638 = math.rsqrt %636 : f32
      linalg.yield %638 : f32
    } -> tensor<1x8x1xf32>
    %639 = tensor.empty() : tensor<1x8x128xf32>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599, %635 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%639 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb59(%641: f32, %642: f32, %643: f32):
      %644 = arith.mulf %641, %642 : f32
      linalg.yield %644 : f32
    } -> tensor<1x8x128xf32>
    %645 = tensor.empty() : tensor<1x8x128xf32>
    %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%640, %2 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%645 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb60(%647: f32, %648: f32, %649: f32):
      %650 = arith.mulf %647, %648 : f32
      linalg.yield %650 : f32
    } -> tensor<1x8x128xf32>
    %651 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0250222664 : f32
    %652 = tensor.splat %651 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %653 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %654 = tensor.splat %653 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %655 = "quant_ext.quantize_per_tensor"(%646, %652, %654) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_2", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %656 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0250222664 : f32
    %657 = tensor.splat %656 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %658 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %659 = tensor.splat %658 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %660 = "quant_ext.dequantize_per_tensor"(%655, %657, %659) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_19", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %661 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0250222664 : f32
    %662 = tensor.splat %661 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %663 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %664 = tensor.splat %663 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %665 = "quant_ext.dequantize_per_tensor"(%655, %662, %664) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_20", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %666 = tensor.empty() : tensor<128x344xf32>
    %667 = linalg.transpose ins(%46:tensor<344x128xf32>) outs(%666:tensor<128x344xf32>) permutation = [1, 0]
    %668 = tensor.collapse_shape %665 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %669 = tensor.expand_shape %668 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %670 = tensor.empty() : tensor<8x344xf32>
    %671 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %672 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%671 : f32) outs(%670 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %673 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g", prov.transposed_b = "true"} ins(%669, %667 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%672 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %674 = tensor.collapse_shape %673 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %675 = tensor.expand_shape %674 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %676 = tensor.empty() : tensor<1x8x344xf32>
    %677 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675 : tensor<1x8x344xf32>) outs(%676 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb61(%678: f32, %679: f32):
      %680 = arith.constant 1.000000e+00 : f32
      %681 = arith.negf %678 : f32
      %682 = math.exp %681 : f32
      %683 = arith.addf %680, %682 : f32
      %684 = arith.divf %680, %683 : f32
      linalg.yield %684 : f32
    } -> tensor<1x8x344xf32>
    %685 = tensor.empty() : tensor<1x8x344xf32>
    %686 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675, %677 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%685 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb62(%687: f32, %688: f32, %689: f32):
      %690 = arith.mulf %687, %688 : f32
      linalg.yield %690 : f32
    } -> tensor<1x8x344xf32>
    %691 = tensor.empty() : tensor<128x344xf32>
    %692 = linalg.transpose ins(%51:tensor<344x128xf32>) outs(%691:tensor<128x344xf32>) permutation = [1, 0]
    %693 = tensor.collapse_shape %660 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %694 = tensor.expand_shape %693 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %695 = tensor.empty() : tensor<8x344xf32>
    %696 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %697 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%696 : f32) outs(%695 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %698 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u", prov.transposed_b = "true"} ins(%694, %692 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%697 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %699 = tensor.collapse_shape %698 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %700 = tensor.expand_shape %699 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %701 = tensor.empty() : tensor<1x8x344xf32>
    %702 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686, %700 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%701 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb63(%703: f32, %704: f32, %705: f32):
      %706 = arith.mulf %703, %704 : f32
      linalg.yield %706 : f32
    } -> tensor<1x8x344xf32>
    %707 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0182978343 : f32
    %708 = tensor.splat %707 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %709 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %710 = tensor.splat %709 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %711 = "quant_ext.quantize_per_tensor"(%702, %708, %710) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_3", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x344xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x344xi8>
    %712 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0182978343 : f32
    %713 = tensor.splat %712 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %714 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %715 = tensor.splat %714 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %716 = "quant_ext.dequantize_per_tensor"(%711, %713, %715) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_21", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x344xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x344xf32>
    %717 = tensor.empty() : tensor<344x128xf32>
    %718 = linalg.transpose ins(%56:tensor<128x344xf32>) outs(%717:tensor<344x128xf32>) permutation = [1, 0]
    %719 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %720 = tensor.expand_shape %719 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %721 = tensor.empty() : tensor<8x128xf32>
    %722 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %723 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%722 : f32) outs(%721 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %724 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn", prov.transposed_b = "true"} ins(%720, %718 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%723 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %725 = tensor.collapse_shape %724 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %726 = tensor.expand_shape %725 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %727 = tensor.empty() : tensor<1x8x128xf32>
    %728 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599, %726 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%727 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb64(%729: f32, %730: f32, %731: f32):
      %732 = arith.addf %729, %730 : f32
      linalg.yield %732 : f32
    } -> tensor<1x8x128xf32>
    %733 = tensor.empty() : tensor<1x8x128xf32>
    %734 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728 : tensor<1x8x128xf32>) outs(%733 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb65(%735: f32, %736: f32):
      %737 = arith.constant 2.000000e+00 : f32
      %738 = math.powf %735, %737 : f32
      linalg.yield %738 : f32
    } -> tensor<1x8x128xf32>
    %739 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 0.000000e+00 : f32
    %740 = tensor.splat %739 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %741 = linalg.reduce ins(%734:tensor<1x8x128xf32>) outs(%740:tensor<1x8xf32>) dimensions = [2]
    (%742: f32, %743: f32) {
      %744 = arith.addf %742, %743 : f32
      linalg.yield %744 : f32
    }
    %745 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.280000e+02 : f32
    %746 = tensor.splat %745 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %747 = tensor.empty() : tensor<1x8xf32>
    %748 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%741, %746 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%747 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb66(%749: f32, %750: f32, %751: f32):
      %752 = arith.divf %749, %750 : f32
      linalg.yield %752 : f32
    } -> tensor<1x8xf32>
    %753 = tensor.collapse_shape %748 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %754 = tensor.expand_shape %753 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %755 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.000000e-05 : f32
    %756 = tensor.splat %755 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8x1xf32>
    %757 = tensor.empty() : tensor<1x8x1xf32>
    %758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%754, %756 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%757 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb67(%759: f32, %760: f32, %761: f32):
      %762 = arith.addf %759, %760 : f32
      linalg.yield %762 : f32
    } -> tensor<1x8x1xf32>
    %763 = tensor.empty() : tensor<1x8x1xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%758 : tensor<1x8x1xf32>) outs(%763 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb68(%765: f32, %766: f32):
      %767 = math.rsqrt %765 : f32
      linalg.yield %767 : f32
    } -> tensor<1x8x1xf32>
    %768 = tensor.empty() : tensor<1x8x128xf32>
    %769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728, %764 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%768 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb69(%770: f32, %771: f32, %772: f32):
      %773 = arith.mulf %770, %771 : f32
      linalg.yield %773 : f32
    } -> tensor<1x8x128xf32>
    %774 = tensor.empty() : tensor<1x8x128xf32>
    %775 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%769, %3 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%774 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb70(%776: f32, %777: f32, %778: f32):
      %779 = arith.mulf %776, %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x8x128xf32>
    %780 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0266733076 : f32
    %781 = tensor.splat %780 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %782 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %783 = tensor.splat %782 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %784 = "quant_ext.quantize_per_tensor"(%775, %781, %783) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_4", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %785 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0266733076 : f32
    %786 = tensor.splat %785 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %787 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %788 = tensor.splat %787 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %789 = "quant_ext.dequantize_per_tensor"(%784, %786, %788) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_22", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %790 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0266733076 : f32
    %791 = tensor.splat %790 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %792 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %793 = tensor.splat %792 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %794 = "quant_ext.dequantize_per_tensor"(%784, %791, %793) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_23", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %795 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0266733076 : f32
    %796 = tensor.splat %795 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %797 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %798 = tensor.splat %797 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %799 = "quant_ext.dequantize_per_tensor"(%784, %796, %798) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_24", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %800 = tensor.empty() : tensor<128x128xf32>
    %801 = linalg.transpose ins(%61:tensor<128x128xf32>) outs(%800:tensor<128x128xf32>) permutation = [1, 0]
    %802 = tensor.collapse_shape %799 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %803 = tensor.expand_shape %802 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %804 = tensor.empty() : tensor<8x128xf32>
    %805 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %806 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%805 : f32) outs(%804 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %807 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q", prov.transposed_b = "true"} ins(%803, %801 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%806 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %808 = tensor.collapse_shape %807 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %809 = tensor.expand_shape %808 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %810 = tensor.collapse_shape %809 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %811 = tensor.expand_shape %810 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %812 = tensor.empty() : tensor<1x4x8x32xf32>
    %813 = linalg.transpose ins(%811:tensor<1x8x4x32xf32>) outs(%812:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %814 = tensor.empty() : tensor<128x128xf32>
    %815 = linalg.transpose ins(%66:tensor<128x128xf32>) outs(%814:tensor<128x128xf32>) permutation = [1, 0]
    %816 = tensor.collapse_shape %794 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %817 = tensor.expand_shape %816 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %818 = tensor.empty() : tensor<8x128xf32>
    %819 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %820 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%819 : f32) outs(%818 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %821 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k", prov.transposed_b = "true"} ins(%817, %815 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%820 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %822 = tensor.collapse_shape %821 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %823 = tensor.expand_shape %822 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %824 = tensor.collapse_shape %823 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %825 = tensor.expand_shape %824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %826 = tensor.empty() : tensor<1x4x8x32xf32>
    %827 = linalg.transpose ins(%825:tensor<1x8x4x32xf32>) outs(%826:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %828 = tensor.empty() : tensor<128x128xf32>
    %829 = linalg.transpose ins(%71:tensor<128x128xf32>) outs(%828:tensor<128x128xf32>) permutation = [1, 0]
    %830 = tensor.collapse_shape %789 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %831 = tensor.expand_shape %830 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %832 = tensor.empty() : tensor<8x128xf32>
    %833 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %834 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%833 : f32) outs(%832 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %835 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v", prov.transposed_b = "true"} ins(%831, %829 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%834 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %836 = tensor.collapse_shape %835 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %837 = tensor.expand_shape %836 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %838 = tensor.collapse_shape %837 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %839 = tensor.expand_shape %838 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %840 = tensor.empty() : tensor<1x4x8x32xf32>
    %841 = linalg.transpose ins(%839:tensor<1x8x4x32xf32>) outs(%840:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %842 = tensor.empty() : tensor<16xf32>
    %843 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%842 : tensor<16xf32>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb71(%844: f32):
      %845 = linalg.index 0 : index
      %846 = arith.index_cast %845 : index to i64
      %847 = arith.sitofp %846 : i64 to f32
      %848 = arith.constant 1.000000e+00 : f32
      %849 = arith.mulf %847, %848 : f32
      %850 = arith.constant 0.000000e+00 : f32
      %851 = arith.addf %850, %849 : f32
      linalg.yield %851 : f32
    } -> tensor<16xf32>
    %852 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %853 = tensor.splat %852 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %854 = tensor.empty() : tensor<16xf32>
    %855 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%843, %853 : tensor<16xf32>, tensor<16xf32>) outs(%854 : tensor<16xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb72(%856: f32, %857: f32, %858: f32):
      %859 = arith.divf %856, %857 : f32
      linalg.yield %859 : f32
    } -> tensor<16xf32>
    %860 = tensor.empty() : tensor<16xf32>
    %861 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%855 : tensor<16xf32>) outs(%860 : tensor<16xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb73(%862: f32, %863: f32):
      %864 = arith.constant 1.000000e+04 : f32
      %865 = math.powf %864, %862 : f32
      linalg.yield %865 : f32
    } -> tensor<16xf32>
    %866 = tensor.empty() : tensor<16xf32>
    %867 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%861 : tensor<16xf32>) outs(%866 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb74(%868: f32, %869: f32):
      %870 = arith.constant 1.000000e+00 : f32
      %871 = arith.divf %870, %868 : f32
      linalg.yield %871 : f32
    } -> tensor<16xf32>
    %872 = arith.constant {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %873 = tensor.splat %872 {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %874 = tensor.empty() : tensor<16xf32>
    %875 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%867, %873 : tensor<16xf32>, tensor<16xf32>) outs(%874 : tensor<16xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb75(%876: f32, %877: f32, %878: f32):
      %879 = arith.mulf %876, %877 : f32
      linalg.yield %879 : f32
    } -> tensor<16xf32>
    %880 = tensor.expand_shape %98 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %881 = tensor.empty() : tensor<8x1xf32>
    %882 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%880 : tensor<8x1xi64>) outs(%881 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb76(%883: i64, %884: f32):
      %885 = arith.sitofp %883 : i64 to f32
      linalg.yield %885 : f32
    } -> tensor<8x1xf32>
    %886 = tensor.expand_shape %875 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %887 = tensor.empty() : tensor<8x16xf32>
    %888 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%882, %886 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%887 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb77(%889: f32, %890: f32, %891: f32):
      %892 = arith.mulf %889, %890 : f32
      linalg.yield %892 : f32
    } -> tensor<8x16xf32>
    %893 = tensor.empty() : tensor<8x16xf32>
    %894 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%888 : tensor<8x16xf32>) outs(%893 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb78(%895: f32, %896: f32):
      %897 = math.cos %895 : f32
      linalg.yield %897 : f32
    } -> tensor<8x16xf32>
    %898 = tensor.empty() : tensor<8x16xf32>
    %899 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%888 : tensor<8x16xf32>) outs(%898 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb79(%900: f32, %901: f32):
      %902 = math.cos %900 : f32
      linalg.yield %902 : f32
    } -> tensor<8x16xf32>
    %903 = tensor.concat dim(1) %894, %899 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %904 = tensor.collapse_shape %903 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %905 = tensor.expand_shape %904 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %906 = tensor.collapse_shape %905 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %907 = tensor.expand_shape %906 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %908 = tensor.empty() : tensor<8x16xf32>
    %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%888 : tensor<8x16xf32>) outs(%908 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb80(%910: f32, %911: f32):
      %912 = math.sin %910 : f32
      linalg.yield %912 : f32
    } -> tensor<8x16xf32>
    %913 = tensor.empty() : tensor<8x16xf32>
    %914 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%888 : tensor<8x16xf32>) outs(%913 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb81(%915: f32, %916: f32):
      %917 = math.sin %915 : f32
      linalg.yield %917 : f32
    } -> tensor<8x16xf32>
    %918 = tensor.concat dim(1) %909, %914 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %919 = tensor.collapse_shape %918 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %920 = tensor.expand_shape %919 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %921 = tensor.collapse_shape %920 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %922 = tensor.expand_shape %921 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %923 = "tensor.extract_slice"(%813) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %924 = "tensor.extract_slice"(%813) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %925 = tensor.empty() : tensor<1x4x8x16xf32>
    %926 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%924 : tensor<1x4x8x16xf32>) outs(%925 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb82(%927: f32, %928: f32):
      %929 = arith.negf %927 : f32
      linalg.yield %929 : f32
    } -> tensor<1x4x8x16xf32>
    %930 = tensor.concat dim(3) %926, %923 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %931 = tensor.empty() : tensor<1x4x8x32xf32>
    %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%813, %907 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%931 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb83(%933: f32, %934: f32, %935: f32):
      %936 = arith.mulf %933, %934 : f32
      linalg.yield %936 : f32
    } -> tensor<1x4x8x32xf32>
    %937 = tensor.empty() : tensor<1x4x8x32xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%930, %922 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%937 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb84(%939: f32, %940: f32, %941: f32):
      %942 = arith.mulf %939, %940 : f32
      linalg.yield %942 : f32
    } -> tensor<1x4x8x32xf32>
    %943 = tensor.empty() : tensor<1x4x8x32xf32>
    %944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%932, %938 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%943 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb85(%945: f32, %946: f32, %947: f32):
      %948 = arith.addf %945, %946 : f32
      linalg.yield %948 : f32
    } -> tensor<1x4x8x32xf32>
    %949 = tensor.empty() : tensor<16xf32>
    %950 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%949 : tensor<16xf32>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb86(%951: f32):
      %952 = linalg.index 0 : index
      %953 = arith.index_cast %952 : index to i64
      %954 = arith.sitofp %953 : i64 to f32
      %955 = arith.constant 1.000000e+00 : f32
      %956 = arith.mulf %954, %955 : f32
      %957 = arith.constant 0.000000e+00 : f32
      %958 = arith.addf %957, %956 : f32
      linalg.yield %958 : f32
    } -> tensor<16xf32>
    %959 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %960 = tensor.splat %959 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %961 = tensor.empty() : tensor<16xf32>
    %962 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%950, %960 : tensor<16xf32>, tensor<16xf32>) outs(%961 : tensor<16xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb87(%963: f32, %964: f32, %965: f32):
      %966 = arith.divf %963, %964 : f32
      linalg.yield %966 : f32
    } -> tensor<16xf32>
    %967 = tensor.empty() : tensor<16xf32>
    %968 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%962 : tensor<16xf32>) outs(%967 : tensor<16xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb88(%969: f32, %970: f32):
      %971 = arith.constant 1.000000e+04 : f32
      %972 = math.powf %971, %969 : f32
      linalg.yield %972 : f32
    } -> tensor<16xf32>
    %973 = tensor.empty() : tensor<16xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%968 : tensor<16xf32>) outs(%973 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb89(%975: f32, %976: f32):
      %977 = arith.constant 1.000000e+00 : f32
      %978 = arith.divf %977, %975 : f32
      linalg.yield %978 : f32
    } -> tensor<16xf32>
    %979 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %980 = tensor.splat %979 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %981 = tensor.empty() : tensor<16xf32>
    %982 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%974, %980 : tensor<16xf32>, tensor<16xf32>) outs(%981 : tensor<16xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb90(%983: f32, %984: f32, %985: f32):
      %986 = arith.mulf %983, %984 : f32
      linalg.yield %986 : f32
    } -> tensor<16xf32>
    %987 = tensor.expand_shape %98 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %988 = tensor.empty() : tensor<8x1xf32>
    %989 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%987 : tensor<8x1xi64>) outs(%988 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb91(%990: i64, %991: f32):
      %992 = arith.sitofp %990 : i64 to f32
      linalg.yield %992 : f32
    } -> tensor<8x1xf32>
    %993 = tensor.expand_shape %982 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %994 = tensor.empty() : tensor<8x16xf32>
    %995 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%989, %993 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%994 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb92(%996: f32, %997: f32, %998: f32):
      %999 = arith.mulf %996, %997 : f32
      linalg.yield %999 : f32
    } -> tensor<8x16xf32>
    %1000 = tensor.empty() : tensor<8x16xf32>
    %1001 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%995 : tensor<8x16xf32>) outs(%1000 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb93(%1002: f32, %1003: f32):
      %1004 = math.cos %1002 : f32
      linalg.yield %1004 : f32
    } -> tensor<8x16xf32>
    %1005 = tensor.empty() : tensor<8x16xf32>
    %1006 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%995 : tensor<8x16xf32>) outs(%1005 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb94(%1007: f32, %1008: f32):
      %1009 = math.cos %1007 : f32
      linalg.yield %1009 : f32
    } -> tensor<8x16xf32>
    %1010 = tensor.concat dim(1) %1001, %1006 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %1011 = tensor.collapse_shape %1010 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %1012 = tensor.expand_shape %1011 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %1013 = tensor.collapse_shape %1012 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1014 = tensor.expand_shape %1013 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %1015 = tensor.empty() : tensor<8x16xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%995 : tensor<8x16xf32>) outs(%1015 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb95(%1017: f32, %1018: f32):
      %1019 = math.sin %1017 : f32
      linalg.yield %1019 : f32
    } -> tensor<8x16xf32>
    %1020 = tensor.empty() : tensor<8x16xf32>
    %1021 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%995 : tensor<8x16xf32>) outs(%1020 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb96(%1022: f32, %1023: f32):
      %1024 = math.sin %1022 : f32
      linalg.yield %1024 : f32
    } -> tensor<8x16xf32>
    %1025 = tensor.concat dim(1) %1016, %1021 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %1026 = tensor.collapse_shape %1025 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %1027 = tensor.expand_shape %1026 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %1028 = tensor.collapse_shape %1027 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1029 = tensor.expand_shape %1028 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %1030 = "tensor.extract_slice"(%827) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %1031 = "tensor.extract_slice"(%827) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %1032 = tensor.empty() : tensor<1x4x8x16xf32>
    %1033 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1031 : tensor<1x4x8x16xf32>) outs(%1032 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb97(%1034: f32, %1035: f32):
      %1036 = arith.negf %1034 : f32
      linalg.yield %1036 : f32
    } -> tensor<1x4x8x16xf32>
    %1037 = tensor.concat dim(3) %1033, %1030 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %1038 = tensor.empty() : tensor<1x4x8x32xf32>
    %1039 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%827, %1014 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%1038 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb98(%1040: f32, %1041: f32, %1042: f32):
      %1043 = arith.mulf %1040, %1041 : f32
      linalg.yield %1043 : f32
    } -> tensor<1x4x8x32xf32>
    %1044 = tensor.empty() : tensor<1x4x8x32xf32>
    %1045 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1037, %1029 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%1044 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb99(%1046: f32, %1047: f32, %1048: f32):
      %1049 = arith.mulf %1046, %1047 : f32
      linalg.yield %1049 : f32
    } -> tensor<1x4x8x32xf32>
    %1050 = tensor.empty() : tensor<1x4x8x32xf32>
    %1051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1039, %1045 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%1050 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb100(%1052: f32, %1053: f32, %1054: f32):
      %1055 = arith.addf %1052, %1053 : f32
      linalg.yield %1055 : f32
    } -> tensor<1x4x8x32xf32>
    %1056 = tensor.empty() : tensor<1x4x32x8xf32>
    %1057 = linalg.transpose ins(%1051:tensor<1x4x8x32xf32>) outs(%1056:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %1058 = tensor.empty() : tensor<1x4x8x32xf32>
    %1059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%944 : tensor<1x4x8x32xf32>) outs(%1058 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb101(%1060: f32, %1061: f32):
      linalg.yield %1060 : f32
    } -> tensor<1x4x8x32xf32>
    %1062 = tensor.collapse_shape %1059 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1063 = tensor.expand_shape %1062 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1064 = tensor.empty() : tensor<1x4x32x8xf32>
    %1065 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1057 : tensor<1x4x32x8xf32>) outs(%1064 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb102(%1066: f32, %1067: f32):
      linalg.yield %1066 : f32
    } -> tensor<1x4x32x8xf32>
    %1068 = tensor.collapse_shape %1065 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %1069 = tensor.expand_shape %1068 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %1070 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1071 = tensor.splat %1070 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1063, %1069 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%1071 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb103(%1073: f32, %1074: f32, %1075: f32):
      %1076 = arith.mulf %1073, %1074 : f32
      %1077 = arith.addf %1075, %1076 : f32
      linalg.yield %1077 : f32
    } -> tensor<4x8x8xf32>
    %1078 = tensor.collapse_shape %1072 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %1079 = tensor.expand_shape %1078 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %1080 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 5.65685415 : f32
    %1081 = tensor.splat %1080 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32>
    %1082 = tensor.empty() : tensor<1x4x8x8xf32>
    %1083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1079, %1081 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%1082 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb104(%1084: f32, %1085: f32, %1086: f32):
      %1087 = arith.divf %1084, %1085 : f32
      linalg.yield %1087 : f32
    } -> tensor<1x4x8x8xf32>
    %1088 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1089 = tensor.splat %1088 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xf32>
    %1090 = tensor.empty() : tensor<8xi64>
    %1091 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1090 : tensor<8xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb105(%1092: i64):
      %1093 = linalg.index 0 : index
      %1094 = arith.index_cast %1093 : index to i64
      %1095 = arith.constant 1 : i64
      %1096 = arith.muli %1094, %1095 : i64
      %1097 = arith.constant 0 : i64
      %1098 = arith.addi %1097, %1096 : i64
      linalg.yield %1098 : i64
    } -> tensor<8xi64>
    %1099 = tensor.expand_shape %1091 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %1100 = tensor.empty() : tensor<8xi64>
    %1101 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1100 : tensor<8xi64>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb106(%1102: i64):
      %1103 = linalg.index 0 : index
      %1104 = arith.index_cast %1103 : index to i64
      %1105 = arith.constant 1 : i64
      %1106 = arith.muli %1104, %1105 : i64
      %1107 = arith.constant 0 : i64
      %1108 = arith.addi %1107, %1106 : i64
      linalg.yield %1108 : i64
    } -> tensor<8xi64>
    %1109 = tensor.expand_shape %1101 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %1110 = tensor.empty() : tensor<8x8xi64>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1099, %1109 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%1110 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb107(%1112: i64, %1113: i64, %1114: i64):
      %1115 = arith.subi %1112, %1113 : i64
      linalg.yield %1115 : i64
    } -> tensor<8x8xi64>
    %1116 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1 : i64
    %1117 = tensor.splat %1116 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xi64>
    %1118 = tensor.empty() : tensor<8x8xi1>
    %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1111, %1117 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%1118 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb108(%1120: i64, %1121: i64, %1122: i1):
      %1123 = arith.cmpi sge, %1120, %1121 : i64
      linalg.yield %1123 : i1
    } -> tensor<8x8xi1>
    %1124 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1125 = tensor.splat %1124 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<f32>
    %1126 = tensor.empty() : tensor<8x8xf32>
    %1127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1119, %1089, %1125 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%1126 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb109(%1128: i1, %1129: f32, %1130: f32, %1131: f32):
      %1132 = arith.select %1128, %1129, %1130 : f32
      linalg.yield %1132 : f32
    } -> tensor<8x8xf32>
    %1133 = tensor.empty() : tensor<1x4x8x8xf32>
    %1134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1083, %1127 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%1133 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb110(%1135: f32, %1136: f32, %1137: f32):
      %1138 = arith.addf %1135, %1136 : f32
      linalg.yield %1138 : f32
    } -> tensor<1x4x8x8xf32>
    %1139 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1140 = tensor.splat %1139 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1141 = linalg.reduce ins(%1134:tensor<1x4x8x8xf32>) outs(%1140:tensor<1x4x8xf32>) dimensions = [3]
    (%1142: f32, %1143: f32) {
      %1144 = arith.maximumf %1142, %1143 : f32
      linalg.yield %1144 : f32
    }
    %1145 = tensor.collapse_shape %1141 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1146 = tensor.expand_shape %1145 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1147 = tensor.empty() : tensor<1x4x8x8xf32>
    %1148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1134, %1146 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1147 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb111(%1149: f32, %1150: f32, %1151: f32):
      %1152 = arith.subf %1149, %1150 : f32
      linalg.yield %1152 : f32
    } -> tensor<1x4x8x8xf32>
    %1153 = tensor.empty() : tensor<1x4x8x8xf32>
    %1154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1148 : tensor<1x4x8x8xf32>) outs(%1153 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb112(%1155: f32, %1156: f32):
      %1157 = math.exp %1155 : f32
      linalg.yield %1157 : f32
    } -> tensor<1x4x8x8xf32>
    %1158 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1159 = tensor.splat %1158 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1160 = linalg.reduce ins(%1154:tensor<1x4x8x8xf32>) outs(%1159:tensor<1x4x8xf32>) dimensions = [3]
    (%1161: f32, %1162: f32) {
      %1163 = arith.addf %1161, %1162 : f32
      linalg.yield %1163 : f32
    }
    %1164 = tensor.collapse_shape %1160 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1165 = tensor.expand_shape %1164 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1166 = tensor.empty() : tensor<1x4x8x8xf32>
    %1167 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1154, %1165 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1166 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb113(%1168: f32, %1169: f32, %1170: f32):
      %1171 = arith.divf %1168, %1169 : f32
      linalg.yield %1171 : f32
    } -> tensor<1x4x8x8xf32>
    %1172 = tensor.empty() : tensor<1x4x8x8xf32>
    %1173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1167 : tensor<1x4x8x8xf32>) outs(%1172 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb114(%1174: f32, %1175: f32):
      linalg.yield %1174 : f32
    } -> tensor<1x4x8x8xf32>
    %1176 = tensor.collapse_shape %1173 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %1177 = tensor.expand_shape %1176 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %1178 = tensor.empty() : tensor<1x4x8x32xf32>
    %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%841 : tensor<1x4x8x32xf32>) outs(%1178 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb115(%1180: f32, %1181: f32):
      linalg.yield %1180 : f32
    } -> tensor<1x4x8x32xf32>
    %1182 = tensor.collapse_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1183 = tensor.expand_shape %1182 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1184 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1185 = tensor.splat %1184 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32>
    %1186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1177, %1183 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%1185 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb116(%1187: f32, %1188: f32, %1189: f32):
      %1190 = arith.mulf %1187, %1188 : f32
      %1191 = arith.addf %1189, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<4x8x32xf32>
    %1192 = tensor.collapse_shape %1186 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %1193 = tensor.expand_shape %1192 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %1194 = tensor.empty() : tensor<1x8x4x32xf32>
    %1195 = linalg.transpose ins(%1193:tensor<1x4x8x32xf32>) outs(%1194:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %1196 = tensor.collapse_shape %1195 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %1197 = tensor.expand_shape %1196 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1198 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0109007452 : f32
    %1199 = tensor.splat %1198 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1200 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1201 = tensor.splat %1200 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1202 = "quant_ext.quantize_per_tensor"(%1197, %1199, %1201) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_5", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %1203 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0109007452 : f32
    %1204 = tensor.splat %1203 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1205 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1206 = tensor.splat %1205 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1207 = "quant_ext.dequantize_per_tensor"(%1202, %1204, %1206) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_25", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %1208 = tensor.empty() : tensor<128x128xf32>
    %1209 = linalg.transpose ins(%76:tensor<128x128xf32>) outs(%1208:tensor<128x128xf32>) permutation = [1, 0]
    %1210 = tensor.collapse_shape %1207 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1211 = tensor.expand_shape %1210 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %1212 = tensor.empty() : tensor<8x128xf32>
    %1213 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1214 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1213 : f32) outs(%1212 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1215 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o", prov.transposed_b = "true"} ins(%1211, %1209 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%1214 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1216 = tensor.collapse_shape %1215 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %1217 = tensor.expand_shape %1216 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1218 = tensor.empty() : tensor<1x8x128xf32>
    %1219 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728, %1217 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1218 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb117(%1220: f32, %1221: f32, %1222: f32):
      %1223 = arith.addf %1220, %1221 : f32
      linalg.yield %1223 : f32
    } -> tensor<1x8x128xf32>
    %1224 = tensor.empty() : tensor<1x8x128xf32>
    %1225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1219 : tensor<1x8x128xf32>) outs(%1224 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb118(%1226: f32, %1227: f32):
      %1228 = arith.constant 2.000000e+00 : f32
      %1229 = math.powf %1226, %1228 : f32
      linalg.yield %1229 : f32
    } -> tensor<1x8x128xf32>
    %1230 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 0.000000e+00 : f32
    %1231 = tensor.splat %1230 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1232 = linalg.reduce ins(%1225:tensor<1x8x128xf32>) outs(%1231:tensor<1x8xf32>) dimensions = [2]
    (%1233: f32, %1234: f32) {
      %1235 = arith.addf %1233, %1234 : f32
      linalg.yield %1235 : f32
    }
    %1236 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.280000e+02 : f32
    %1237 = tensor.splat %1236 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1238 = tensor.empty() : tensor<1x8xf32>
    %1239 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1232, %1237 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1238 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb119(%1240: f32, %1241: f32, %1242: f32):
      %1243 = arith.divf %1240, %1241 : f32
      linalg.yield %1243 : f32
    } -> tensor<1x8xf32>
    %1244 = tensor.collapse_shape %1239 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %1245 = tensor.expand_shape %1244 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1246 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.000000e-05 : f32
    %1247 = tensor.splat %1246 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8x1xf32>
    %1248 = tensor.empty() : tensor<1x8x1xf32>
    %1249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1247 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1248 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb120(%1250: f32, %1251: f32, %1252: f32):
      %1253 = arith.addf %1250, %1251 : f32
      linalg.yield %1253 : f32
    } -> tensor<1x8x1xf32>
    %1254 = tensor.empty() : tensor<1x8x1xf32>
    %1255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1249 : tensor<1x8x1xf32>) outs(%1254 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb121(%1256: f32, %1257: f32):
      %1258 = math.rsqrt %1256 : f32
      linalg.yield %1258 : f32
    } -> tensor<1x8x1xf32>
    %1259 = tensor.empty() : tensor<1x8x128xf32>
    %1260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1219, %1255 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1259 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb122(%1261: f32, %1262: f32, %1263: f32):
      %1264 = arith.mulf %1261, %1262 : f32
      linalg.yield %1264 : f32
    } -> tensor<1x8x128xf32>
    %1265 = tensor.empty() : tensor<1x8x128xf32>
    %1266 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1260, %4 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1265 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb123(%1267: f32, %1268: f32, %1269: f32):
      %1270 = arith.mulf %1267, %1268 : f32
      linalg.yield %1270 : f32
    } -> tensor<1x8x128xf32>
    %1271 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0259607546 : f32
    %1272 = tensor.splat %1271 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1273 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1274 = tensor.splat %1273 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1275 = "quant_ext.quantize_per_tensor"(%1266, %1272, %1274) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_6", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %1276 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0259607546 : f32
    %1277 = tensor.splat %1276 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1278 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1279 = tensor.splat %1278 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1280 = "quant_ext.dequantize_per_tensor"(%1275, %1277, %1279) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_26", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %1281 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0259607546 : f32
    %1282 = tensor.splat %1281 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1283 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1284 = tensor.splat %1283 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1285 = "quant_ext.dequantize_per_tensor"(%1275, %1282, %1284) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_27", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %1286 = tensor.empty() : tensor<128x344xf32>
    %1287 = linalg.transpose ins(%81:tensor<344x128xf32>) outs(%1286:tensor<128x344xf32>) permutation = [1, 0]
    %1288 = tensor.collapse_shape %1285 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1289 = tensor.expand_shape %1288 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %1290 = tensor.empty() : tensor<8x344xf32>
    %1291 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1292 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1291 : f32) outs(%1290 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1293 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g", prov.transposed_b = "true"} ins(%1289, %1287 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1292 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1294 = tensor.collapse_shape %1293 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %1295 = tensor.expand_shape %1294 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1296 = tensor.empty() : tensor<1x8x344xf32>
    %1297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295 : tensor<1x8x344xf32>) outs(%1296 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb124(%1298: f32, %1299: f32):
      %1300 = arith.constant 1.000000e+00 : f32
      %1301 = arith.negf %1298 : f32
      %1302 = math.exp %1301 : f32
      %1303 = arith.addf %1300, %1302 : f32
      %1304 = arith.divf %1300, %1303 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x8x344xf32>
    %1305 = tensor.empty() : tensor<1x8x344xf32>
    %1306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295, %1297 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1305 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb125(%1307: f32, %1308: f32, %1309: f32):
      %1310 = arith.mulf %1307, %1308 : f32
      linalg.yield %1310 : f32
    } -> tensor<1x8x344xf32>
    %1311 = tensor.empty() : tensor<128x344xf32>
    %1312 = linalg.transpose ins(%86:tensor<344x128xf32>) outs(%1311:tensor<128x344xf32>) permutation = [1, 0]
    %1313 = tensor.collapse_shape %1280 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1314 = tensor.expand_shape %1313 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %1315 = tensor.empty() : tensor<8x344xf32>
    %1316 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1317 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1316 : f32) outs(%1315 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1318 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u", prov.transposed_b = "true"} ins(%1314, %1312 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1317 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1319 = tensor.collapse_shape %1318 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %1320 = tensor.expand_shape %1319 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1321 = tensor.empty() : tensor<1x8x344xf32>
    %1322 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1320 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1321 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb126(%1323: f32, %1324: f32, %1325: f32):
      %1326 = arith.mulf %1323, %1324 : f32
      linalg.yield %1326 : f32
    } -> tensor<1x8x344xf32>
    %1327 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00946410559 : f32
    %1328 = tensor.splat %1327 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1329 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1330 = tensor.splat %1329 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1331 = "quant_ext.quantize_per_tensor"(%1322, %1328, %1330) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_7", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x344xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x344xi8>
    %1332 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00946410559 : f32
    %1333 = tensor.splat %1332 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1334 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1335 = tensor.splat %1334 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1336 = "quant_ext.dequantize_per_tensor"(%1331, %1333, %1335) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_28", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x344xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x344xf32>
    %1337 = tensor.empty() : tensor<344x128xf32>
    %1338 = linalg.transpose ins(%91:tensor<128x344xf32>) outs(%1337:tensor<344x128xf32>) permutation = [1, 0]
    %1339 = tensor.collapse_shape %1336 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %1340 = tensor.expand_shape %1339 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %1341 = tensor.empty() : tensor<8x128xf32>
    %1342 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1343 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1342 : f32) outs(%1341 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1344 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn", prov.transposed_b = "true"} ins(%1340, %1338 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%1343 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1345 = tensor.collapse_shape %1344 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %1346 = tensor.expand_shape %1345 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1347 = tensor.empty() : tensor<1x8x128xf32>
    %1348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1219, %1346 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1347 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb127(%1349: f32, %1350: f32, %1351: f32):
      %1352 = arith.addf %1349, %1350 : f32
      linalg.yield %1352 : f32
    } -> tensor<1x8x128xf32>
    %1353 = tensor.empty() : tensor<1x8x128xf32>
    %1354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1348 : tensor<1x8x128xf32>) outs(%1353 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb128(%1355: f32, %1356: f32):
      %1357 = arith.constant 2.000000e+00 : f32
      %1358 = math.powf %1355, %1357 : f32
      linalg.yield %1358 : f32
    } -> tensor<1x8x128xf32>
    %1359 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %1360 = tensor.splat %1359 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1361 = linalg.reduce ins(%1354:tensor<1x8x128xf32>) outs(%1360:tensor<1x8xf32>) dimensions = [2]
    (%1362: f32, %1363: f32) {
      %1364 = arith.addf %1362, %1363 : f32
      linalg.yield %1364 : f32
    }
    %1365 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.280000e+02 : f32
    %1366 = tensor.splat %1365 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1367 = tensor.empty() : tensor<1x8xf32>
    %1368 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1361, %1366 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1367 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb129(%1369: f32, %1370: f32, %1371: f32):
      %1372 = arith.divf %1369, %1370 : f32
      linalg.yield %1372 : f32
    } -> tensor<1x8xf32>
    %1373 = tensor.collapse_shape %1368 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32> into tensor<8xf32>
    %1374 = tensor.expand_shape %1373 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1375 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.000000e-05 : f32
    %1376 = tensor.splat %1375 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8x1xf32>
    %1377 = tensor.empty() : tensor<1x8x1xf32>
    %1378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1374, %1376 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1377 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb130(%1379: f32, %1380: f32, %1381: f32):
      %1382 = arith.addf %1379, %1380 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x8x1xf32>
    %1383 = tensor.empty() : tensor<1x8x1xf32>
    %1384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1378 : tensor<1x8x1xf32>) outs(%1383 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb131(%1385: f32, %1386: f32):
      %1387 = math.rsqrt %1385 : f32
      linalg.yield %1387 : f32
    } -> tensor<1x8x1xf32>
    %1388 = tensor.empty() : tensor<1x8x128xf32>
    %1389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1348, %1384 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1388 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb132(%1390: f32, %1391: f32, %1392: f32):
      %1393 = arith.mulf %1390, %1391 : f32
      linalg.yield %1393 : f32
    } -> tensor<1x8x128xf32>
    %1394 = tensor.empty() : tensor<1x8x128xf32>
    %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1389, %5 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1394 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb133(%1396: f32, %1397: f32, %1398: f32):
      %1399 = arith.mulf %1396, %1397 : f32
      linalg.yield %1399 : f32
    } -> tensor<1x8x128xf32>
    %1400 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.026208479 : f32
    %1401 = tensor.splat %1400 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1402 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1403 = tensor.splat %1402 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1404 = "quant_ext.quantize_per_tensor"(%1395, %1401, %1403) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_8", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x8x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xi8>
    %1405 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.026208479 : f32
    %1406 = tensor.splat %1405 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1407 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1408 = tensor.splat %1407 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1409 = "quant_ext.dequantize_per_tensor"(%1404, %1406, %1408) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_29", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x8x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8x128xf32>
    %1410 = tensor.empty() : tensor<128x256xf32>
    %1411 = linalg.transpose ins(%96:tensor<256x128xf32>) outs(%1410:tensor<128x256xf32>) permutation = [1, 0]
    %1412 = tensor.collapse_shape %1409 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1413 = tensor.expand_shape %1412 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1024xf32> into tensor<8x128xf32>
    %1414 = tensor.empty() : tensor<8x256xf32>
    %1415 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1416 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1415 : f32) outs(%1414 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1417 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm", prov.transposed_b = "true"} ins(%1413, %1411 : tensor<8x128xf32>, tensor<128x256xf32>) outs(%1416 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1418 = tensor.collapse_shape %1417 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<8x256xf32> into tensor<2048xf32>
    %1419 = tensor.expand_shape %1418 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<2048xf32> into tensor<1x8x256xf32>
    func.return %1419 : tensor<1x8x256xf32>
  }
}
