builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} {
  func.func @forward(%0: tensor<256x32xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32xf32>, %5: tensor<32xf32>, %6: tensor<32x3x16x16xi8>, %7: tensor<32x32xi8>, %8: tensor<32x32xi8>, %9: tensor<32x32xi8>, %10: tensor<32x32xi8>, %11: tensor<32x32xi8>, %12: tensor<64x32xi8>, %13: tensor<64x32xi8>, %14: tensor<32x64xi8>, %15: tensor<32x32xi8>, %16: tensor<32x32xi8>, %17: tensor<32x32xi8>, %18: tensor<32x32xi8>, %19: tensor<32x32xi8>, %20: tensor<32x32xi8>, %21: tensor<32x32xi8>, %22: tensor<32x32xi8>, %23: tensor<32x32xi8>, %24: tensor<32x32xi8>, %25: tensor<64x32xi8>, %26: tensor<32x64xi8>, %27: tensor<32x32xi8>, %28: tensor<1x3x32x32xf32>, %29: tensor<1xi1>, %30: tensor<1x11xi64>, %31: tensor<1x11xi1>, %32: tensor<1x32xf32>, %33: tensor<1x16x32xf32>) -> tensor<1x16x32xf32> {
    %34 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000284120732 : f32
    %35 = tensor.splat %34 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %36 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %37 = tensor.splat %36 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %38 = "quant_ext.dequantize_per_tensor"(%6, %35, %37) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x3x16x16xi8>, tensor<f32>, tensor<i64>) -> tensor<32x3x16x16xf32>
    %39 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139179907 : f32
    %40 = tensor.splat %39 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %41 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %42 = tensor.splat %41 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %43 = "quant_ext.dequantize_per_tensor"(%7, %40, %42) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %44 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138446689 : f32
    %45 = tensor.splat %44 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %46 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %47 = tensor.splat %46 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %48 = "quant_ext.dequantize_per_tensor"(%8, %45, %47) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %49 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138657703 : f32
    %50 = tensor.splat %49 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %51 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %52 = tensor.splat %51 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %53 = "quant_ext.dequantize_per_tensor"(%9, %50, %52) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_3", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %54 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139146182 : f32
    %55 = tensor.splat %54 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %56 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %57 = tensor.splat %56 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %58 = "quant_ext.dequantize_per_tensor"(%10, %55, %57) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_4", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %59 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139190163 : f32
    %60 = tensor.splat %59 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %61 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %62 = tensor.splat %61 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %63 = "quant_ext.dequantize_per_tensor"(%11, %60, %62) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_5", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %64 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139121036 : f32
    %65 = tensor.splat %64 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %66 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %67 = tensor.splat %66 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %68 = "quant_ext.dequantize_per_tensor"(%12, %65, %67) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_6", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %69 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139179872 : f32
    %70 = tensor.splat %69 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %71 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %72 = tensor.splat %71 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %73 = "quant_ext.dequantize_per_tensor"(%13, %70, %72) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_7", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %74 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000984238577 : f32
    %75 = tensor.splat %74 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %76 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %77 = tensor.splat %76 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %78 = "quant_ext.dequantize_per_tensor"(%14, %75, %77) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_8", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x64xi8>, tensor<f32>, tensor<i64>) -> tensor<32x64xf32>
    %79 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138849823 : f32
    %80 = tensor.splat %79 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %81 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %82 = tensor.splat %81 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %83 = "quant_ext.dequantize_per_tensor"(%15, %80, %82) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_9", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %84 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139160384 : f32
    %85 = tensor.splat %84 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %86 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %87 = tensor.splat %86 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %88 = "quant_ext.dequantize_per_tensor"(%16, %85, %87) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_10", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %89 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139066449 : f32
    %90 = tensor.splat %89 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %91 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %92 = tensor.splat %91 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %93 = "quant_ext.dequantize_per_tensor"(%17, %90, %92) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_11", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %94 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138944248 : f32
    %95 = tensor.splat %94 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %96 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %97 = tensor.splat %96 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %98 = "quant_ext.dequantize_per_tensor"(%18, %95, %97) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_12", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %99 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0013912085 : f32
    %100 = tensor.splat %99 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %101 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %102 = tensor.splat %101 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %103 = "quant_ext.dequantize_per_tensor"(%19, %100, %102) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_13", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %104 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138616748 : f32
    %105 = tensor.splat %104 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %106 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %107 = tensor.splat %106 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %108 = "quant_ext.dequantize_per_tensor"(%20, %105, %107) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_14", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %109 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139072898 : f32
    %110 = tensor.splat %109 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %111 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %112 = tensor.splat %111 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %113 = "quant_ext.dequantize_per_tensor"(%21, %110, %112) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_15", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %114 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0013903965 : f32
    %115 = tensor.splat %114 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %116 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %117 = tensor.splat %116 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %118 = "quant_ext.dequantize_per_tensor"(%22, %115, %117) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_16", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %119 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138967822 : f32
    %120 = tensor.splat %119 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %121 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %122 = tensor.splat %121 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %123 = "quant_ext.dequantize_per_tensor"(%23, %120, %122) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_17", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %124 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0013885845 : f32
    %125 = tensor.splat %124 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %126 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %127 = tensor.splat %126 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %128 = "quant_ext.dequantize_per_tensor"(%24, %125, %127) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_18", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %129 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139177206 : f32
    %130 = tensor.splat %129 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %131 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %132 = tensor.splat %131 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %133 = "quant_ext.dequantize_per_tensor"(%25, %130, %132) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_19", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %134 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 9.841940e-04 : f32
    %135 = tensor.splat %134 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %136 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %137 = tensor.splat %136 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %138 = "quant_ext.dequantize_per_tensor"(%26, %135, %137) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_20", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x64xi8>, tensor<f32>, tensor<i64>) -> tensor<32x64xf32>
    %139 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138796284 : f32
    %140 = tensor.splat %139 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %141 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %142 = tensor.splat %141 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %143 = "quant_ext.dequantize_per_tensor"(%27, %140, %142) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_21", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %144 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0143915368 : f32
    %145 = tensor.splat %144 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %146 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %147 = tensor.splat %146 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %148 = "quant_ext.quantize_per_tensor"(%28, %145, %147) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x3x32x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x3x32x32xi8>
    %149 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0143915368 : f32
    %150 = tensor.splat %149 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %151 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %152 = tensor.splat %151 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %153 = "quant_ext.dequantize_per_tensor"(%148, %150, %152) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_22", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x3x32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x3x32x32xf32>
    %154 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0103853745 : f32
    %155 = tensor.splat %154 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %156 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %157 = tensor.splat %156 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %158 = "quant_ext.quantize_per_tensor"(%32, %155, %157) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32xi8>
    %159 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0103853745 : f32
    %160 = tensor.splat %159 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %161 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %162 = tensor.splat %161 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %163 = "quant_ext.dequantize_per_tensor"(%158, %160, %162) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_23", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32xf32>
    %164 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0307341404 : f32
    %165 = tensor.splat %164 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %166 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %167 = tensor.splat %166 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %168 = "quant_ext.quantize_per_tensor"(%33, %165, %167) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_2", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %169 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0307341404 : f32
    %170 = tensor.splat %169 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %171 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %172 = tensor.splat %171 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %173 = "quant_ext.dequantize_per_tensor"(%168, %170, %172) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_24", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %174 = tensor.empty() : tensor<3x16x16x1x2x2xf32>
    %175 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 16) + d1), ((d5 * 16) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%153 : tensor<1x3x32x32xf32>) outs(%174 : tensor<3x16x16x1x2x2xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} {
    ^bb0(%176: f32, %177: f32):
      linalg.yield %176 : f32
    } -> tensor<3x16x16x1x2x2xf32>
    %178 = tensor.collapse_shape %175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<3x16x16x1x2x2xf32> into tensor<3072xf32>
    %179 = tensor.expand_shape %178 [[0 : i64, 1 : i64]] output_shape [768, 4] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<3072xf32> into tensor<768x4xf32>
    %180 = tensor.collapse_shape %38 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<32x3x16x16xf32> into tensor<24576xf32>
    %181 = tensor.expand_shape %180 [[0 : i64, 1 : i64]] output_shape [32, 768] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<24576xf32> into tensor<32x768xf32>
    %182 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} 0.000000e+00 : f32
    %183 = tensor.splat %182 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<32x4xf32>
    %184 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} ins(%181, %179 : tensor<32x768xf32>, tensor<768x4xf32>) outs(%183 : tensor<32x4xf32>) -> tensor<32x4xf32>
    %185 = tensor.collapse_shape %184 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<32x4xf32> into tensor<128xf32>
    %186 = tensor.expand_shape %185 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 2, 2] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<128xf32> into tensor<32x1x2x2xf32>
    %187 = tensor.collapse_shape %186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<32x1x2x2xf32> into tensor<128xf32>
    %188 = tensor.expand_shape %187 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 2, 2] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : tensor<128xf32> into tensor<1x32x2x2xf32>
    %189 = tensor.collapse_shape %188 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x2x2xf32> into tensor<128xf32>
    %190 = tensor.expand_shape %189 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 4] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x32x4xf32>
    %191 = tensor.empty() : tensor<1x4x32xf32>
    %192 = linalg.transpose ins(%190:tensor<1x32x4xf32>) outs(%191:tensor<1x4x32xf32>) permutation = [0, 2, 1]
    %193 = tensor.empty() : tensor<1xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%29 : tensor<1xi1>) outs(%193 : tensor<1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb1(%195: i1, %196: f32):
      %197 = arith.sitofp %195 : i1 to f32
      linalg.yield %197 : f32
    } -> tensor<1xf32>
    %198 = tensor.expand_shape %194 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1xf32>
    %199 = tensor.collapse_shape %198 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
    %200 = tensor.expand_shape %199 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
    %201 = tensor.empty() : tensor<1x4x32xf32>
    %202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %200 : tensor<1x4x32xf32>, tensor<1x1x1xf32>) outs(%201 : tensor<1x4x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%203: f32, %204: f32, %205: f32):
      %206 = arith.mulf %203, %204 : f32
      linalg.yield %206 : f32
    } -> tensor<1x4x32xf32>
    %207 = tensor.empty() : tensor<1x11x32xf32>
    %208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<1x11xi64>) outs(%207 : tensor<1x11x32xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "tok", prov.fqn = "tok"} {
    ^bb3(%209: i64, %210: f32):
      %211 = arith.index_cast %209 : i64 to index
      %212 = linalg.index 2 : index
      %213 = tensor.extract %0[%211, %212] : tensor<256x32xf32>
      linalg.yield %213 : f32
    } -> tensor<1x11x32xf32>
    %214 = tensor.empty() : tensor<32x32xf32>
    %215 = linalg.transpose ins(%43:tensor<32x32xf32>) outs(%214:tensor<32x32xf32>) permutation = [1, 0]
    %216 = tensor.empty() : tensor<1x32xf32>
    %217 = arith.constant {prov.module = "state_in"} 0.000000e+00 : f32
    %218 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "state_in"} ins(%217 : f32) outs(%216 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %219 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "state_in", prov.fqn = "state_in", prov.transposed_b = "true"} ins(%163, %215 : tensor<1x32xf32>, tensor<32x32xf32>) outs(%218 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %220 = tensor.collapse_shape %219 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %221 = tensor.expand_shape %220 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
    %222 = tensor.concat dim(1) %202, %208, %221 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x32xf32>, tensor<1x11x32xf32>, tensor<1x1x32xf32>) -> tensor<1x16x32xf32>
    %223 = tensor.expand_shape %29 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1xi1> into tensor<1x1xi1>
    %224 = tensor.empty() : tensor<1x4xi1>
    %225 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%223 : tensor<1x1xi1>) outs(%224 : tensor<1x4xi1>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool"} {
    ^bb4(%226: i1, %227: i1):
      linalg.yield %226 : i1
    } -> tensor<1x4xi1>
    %228 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bool"} true
    %229 = tensor.splat %228 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bool"} : tensor<1xi1>
    %230 = tensor.expand_shape %229 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1xi1> into tensor<1x1xi1>
    %231 = tensor.concat dim(1) %225, %31, %230 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "bool"} : (tensor<1x4xi1>, tensor<1x11xi1>, tensor<1x1xi1>) -> tensor<1x16xi1>
    %232 = tensor.empty() : tensor<1x16xf32>
    %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%231 : tensor<1x16xi1>) outs(%232 : tensor<1x16xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb5(%234: i1, %235: f32):
      %236 = arith.sitofp %234 : i1 to f32
      linalg.yield %236 : f32
    } -> tensor<1x16xf32>
    %237 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %238 = tensor.splat %237 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %239 = tensor.empty() : tensor<1x16xf32>
    %240 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%238, %233 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%239 : tensor<1x16xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%241: f32, %242: f32, %243: f32):
      %244 = arith.subf %241, %242 : f32
      linalg.yield %244 : f32
    } -> tensor<1x16xf32>
    %245 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -1.000000e+04 : f32
    %246 = tensor.splat %245 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %247 = tensor.empty() : tensor<1x16xf32>
    %248 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%240, %246 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%247 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%249: f32, %250: f32, %251: f32):
      %252 = arith.mulf %249, %250 : f32
      linalg.yield %252 : f32
    } -> tensor<1x16xf32>
    %253 = tensor.collapse_shape %248 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x16xf32> into tensor<16xf32>
    %254 = tensor.expand_shape %253 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 16] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<1x1x16xf32>
    %255 = tensor.collapse_shape %254 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x16xf32> into tensor<16xf32>
    %256 = tensor.expand_shape %255 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 16] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<1x1x1x16xf32>
    %257 = tensor.empty() : tensor<16xi64>
    %258 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%257 : tensor<16xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb8(%259: i64):
      %260 = linalg.index 0 : index
      %261 = arith.index_cast %260 : index to i64
      %262 = arith.constant 1 : i64
      %263 = arith.muli %261, %262 : i64
      %264 = arith.constant 0 : i64
      %265 = arith.addi %264, %263 : i64
      linalg.yield %265 : i64
    } -> tensor<16xi64>
    %266 = tensor.empty() : tensor<1x16x32xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222 : tensor<1x16x32xf32>) outs(%266 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb9(%268: f32, %269: f32):
      %270 = arith.constant 2.000000e+00 : f32
      %271 = math.powf %268, %270 : f32
      linalg.yield %271 : f32
    } -> tensor<1x16x32xf32>
    %272 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} 0.000000e+00 : f32
    %273 = tensor.splat %272 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : tensor<1x16xf32>
    %274 = linalg.reduce ins(%267:tensor<1x16x32xf32>) outs(%273:tensor<1x16xf32>) dimensions = [2]
    (%275: f32, %276: f32) {
      %277 = arith.addf %275, %276 : f32
      linalg.yield %277 : f32
    }
    %278 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} 3.200000e+01 : f32
    %279 = tensor.splat %278 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : tensor<1x16xf32>
    %280 = tensor.empty() : tensor<1x16xf32>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%274, %279 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%280 : tensor<1x16xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb10(%282: f32, %283: f32, %284: f32):
      %285 = arith.divf %282, %283 : f32
      linalg.yield %285 : f32
    } -> tensor<1x16xf32>
    %286 = tensor.collapse_shape %281 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : tensor<1x16xf32> into tensor<16xf32>
    %287 = tensor.expand_shape %286 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : tensor<16xf32> into tensor<1x16x1xf32>
    %288 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} 1.000000e-05 : f32
    %289 = tensor.splat %288 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : tensor<1x16x1xf32>
    %290 = tensor.empty() : tensor<1x16x1xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287, %289 : tensor<1x16x1xf32>, tensor<1x16x1xf32>) outs(%290 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb11(%292: f32, %293: f32, %294: f32):
      %295 = arith.addf %292, %293 : f32
      linalg.yield %295 : f32
    } -> tensor<1x16x1xf32>
    %296 = tensor.empty() : tensor<1x16x1xf32>
    %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%291 : tensor<1x16x1xf32>) outs(%296 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb12(%298: f32, %299: f32):
      %300 = math.rsqrt %298 : f32
      linalg.yield %300 : f32
    } -> tensor<1x16x1xf32>
    %301 = tensor.empty() : tensor<1x16x32xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %297 : tensor<1x16x32xf32>, tensor<1x16x1xf32>) outs(%301 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb13(%303: f32, %304: f32, %305: f32):
      %306 = arith.mulf %303, %304 : f32
      linalg.yield %306 : f32
    } -> tensor<1x16x32xf32>
    %307 = tensor.empty() : tensor<1x16x32xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%302, %1 : tensor<1x16x32xf32>, tensor<32xf32>) outs(%307 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} {
    ^bb14(%309: f32, %310: f32, %311: f32):
      %312 = arith.mulf %309, %310 : f32
      linalg.yield %312 : f32
    } -> tensor<1x16x32xf32>
    %313 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0280993301 : f32
    %314 = tensor.splat %313 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %315 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %316 = tensor.splat %315 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %317 = "quant_ext.quantize_per_tensor"(%308, %314, %316) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_3", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %318 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0280993301 : f32
    %319 = tensor.splat %318 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %320 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %321 = tensor.splat %320 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %322 = "quant_ext.dequantize_per_tensor"(%317, %319, %321) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_25", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %323 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0280993301 : f32
    %324 = tensor.splat %323 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %325 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %326 = tensor.splat %325 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %327 = "quant_ext.dequantize_per_tensor"(%317, %324, %326) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_26", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %328 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0280993301 : f32
    %329 = tensor.splat %328 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %330 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %331 = tensor.splat %330 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %332 = "quant_ext.dequantize_per_tensor"(%317, %329, %331) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_27", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %333 = tensor.empty() : tensor<32x32xf32>
    %334 = linalg.transpose ins(%48:tensor<32x32xf32>) outs(%333:tensor<32x32xf32>) permutation = [1, 0]
    %335 = tensor.collapse_shape %332 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : tensor<1x16x32xf32> into tensor<512xf32>
    %336 = tensor.expand_shape %335 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : tensor<512xf32> into tensor<16x32xf32>
    %337 = tensor.empty() : tensor<16x32xf32>
    %338 = arith.constant {prov.module = "pre_attn"} 0.000000e+00 : f32
    %339 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} ins(%338 : f32) outs(%337 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %340 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q", prov.transposed_b = "true"} ins(%336, %334 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%339 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %341 = tensor.collapse_shape %340 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : tensor<16x32xf32> into tensor<512xf32>
    %342 = tensor.expand_shape %341 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : tensor<512xf32> into tensor<1x16x32xf32>
    %343 = tensor.collapse_shape %342 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x32xf32> into tensor<512xf32>
    %344 = tensor.expand_shape %343 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %345 = tensor.empty() : tensor<1x2x16x16xf32>
    %346 = linalg.transpose ins(%344:tensor<1x16x2x16xf32>) outs(%345:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %347 = tensor.empty() : tensor<32x32xf32>
    %348 = linalg.transpose ins(%53:tensor<32x32xf32>) outs(%347:tensor<32x32xf32>) permutation = [1, 0]
    %349 = tensor.collapse_shape %327 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : tensor<1x16x32xf32> into tensor<512xf32>
    %350 = tensor.expand_shape %349 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : tensor<512xf32> into tensor<16x32xf32>
    %351 = tensor.empty() : tensor<16x32xf32>
    %352 = arith.constant {prov.module = "pre_attn"} 0.000000e+00 : f32
    %353 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} ins(%352 : f32) outs(%351 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %354 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k", prov.transposed_b = "true"} ins(%350, %348 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%353 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %355 = tensor.collapse_shape %354 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : tensor<16x32xf32> into tensor<512xf32>
    %356 = tensor.expand_shape %355 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : tensor<512xf32> into tensor<1x16x32xf32>
    %357 = tensor.collapse_shape %356 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x32xf32> into tensor<512xf32>
    %358 = tensor.expand_shape %357 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %359 = tensor.empty() : tensor<1x2x16x16xf32>
    %360 = linalg.transpose ins(%358:tensor<1x16x2x16xf32>) outs(%359:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %361 = tensor.empty() : tensor<32x32xf32>
    %362 = linalg.transpose ins(%58:tensor<32x32xf32>) outs(%361:tensor<32x32xf32>) permutation = [1, 0]
    %363 = tensor.collapse_shape %322 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : tensor<1x16x32xf32> into tensor<512xf32>
    %364 = tensor.expand_shape %363 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : tensor<512xf32> into tensor<16x32xf32>
    %365 = tensor.empty() : tensor<16x32xf32>
    %366 = arith.constant {prov.module = "pre_attn"} 0.000000e+00 : f32
    %367 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} ins(%366 : f32) outs(%365 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %368 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v", prov.transposed_b = "true"} ins(%364, %362 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%367 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %369 = tensor.collapse_shape %368 [[0 : i64, 1 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : tensor<16x32xf32> into tensor<512xf32>
    %370 = tensor.expand_shape %369 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : tensor<512xf32> into tensor<1x16x32xf32>
    %371 = tensor.collapse_shape %370 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x32xf32> into tensor<512xf32>
    %372 = tensor.expand_shape %371 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %373 = tensor.empty() : tensor<1x2x16x16xf32>
    %374 = linalg.transpose ins(%372:tensor<1x16x2x16xf32>) outs(%373:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %375 = tensor.empty() : tensor<8xf32>
    %376 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%375 : tensor<8xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb15(%377: f32):
      %378 = linalg.index 0 : index
      %379 = arith.index_cast %378 : index to i64
      %380 = arith.sitofp %379 : i64 to f32
      %381 = arith.constant 1.000000e+00 : f32
      %382 = arith.mulf %380, %381 : f32
      %383 = arith.constant 0.000000e+00 : f32
      %384 = arith.addf %383, %382 : f32
      linalg.yield %384 : f32
    } -> tensor<8xf32>
    %385 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 8.000000e+00 : f32
    %386 = tensor.splat %385 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32>
    %387 = tensor.empty() : tensor<8xf32>
    %388 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%376, %386 : tensor<8xf32>, tensor<8xf32>) outs(%387 : tensor<8xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb16(%389: f32, %390: f32, %391: f32):
      %392 = arith.divf %389, %390 : f32
      linalg.yield %392 : f32
    } -> tensor<8xf32>
    %393 = tensor.empty() : tensor<8xf32>
    %394 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%388 : tensor<8xf32>) outs(%393 : tensor<8xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb17(%395: f32, %396: f32):
      %397 = arith.constant 1.000000e+04 : f32
      %398 = math.powf %397, %395 : f32
      linalg.yield %398 : f32
    } -> tensor<8xf32>
    %399 = tensor.empty() : tensor<8xf32>
    %400 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%394 : tensor<8xf32>) outs(%399 : tensor<8xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb18(%401: f32, %402: f32):
      %403 = arith.constant 1.000000e+00 : f32
      %404 = arith.divf %403, %401 : f32
      linalg.yield %404 : f32
    } -> tensor<8xf32>
    %405 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 1.000000e+00 : f32
    %406 = tensor.splat %405 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32>
    %407 = tensor.empty() : tensor<8xf32>
    %408 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%400, %406 : tensor<8xf32>, tensor<8xf32>) outs(%407 : tensor<8xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb19(%409: f32, %410: f32, %411: f32):
      %412 = arith.mulf %409, %410 : f32
      linalg.yield %412 : f32
    } -> tensor<8xf32>
    %413 = tensor.expand_shape %258 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16xi64> into tensor<16x1xi64>
    %414 = tensor.empty() : tensor<16x1xf32>
    %415 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%413 : tensor<16x1xi64>) outs(%414 : tensor<16x1xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb20(%416: i64, %417: f32):
      %418 = arith.sitofp %416 : i64 to f32
      linalg.yield %418 : f32
    } -> tensor<16x1xf32>
    %419 = tensor.expand_shape %408 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32> into tensor<1x8xf32>
    %420 = tensor.empty() : tensor<16x8xf32>
    %421 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%415, %419 : tensor<16x1xf32>, tensor<1x8xf32>) outs(%420 : tensor<16x8xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb21(%422: f32, %423: f32, %424: f32):
      %425 = arith.mulf %422, %423 : f32
      linalg.yield %425 : f32
    } -> tensor<16x8xf32>
    %426 = tensor.empty() : tensor<16x8xf32>
    %427 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%421 : tensor<16x8xf32>) outs(%426 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb22(%428: f32, %429: f32):
      %430 = math.cos %428 : f32
      linalg.yield %430 : f32
    } -> tensor<16x8xf32>
    %431 = tensor.empty() : tensor<16x8xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%421 : tensor<16x8xf32>) outs(%431 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb23(%433: f32, %434: f32):
      %435 = math.cos %433 : f32
      linalg.yield %435 : f32
    } -> tensor<16x8xf32>
    %436 = tensor.concat dim(1) %427, %432 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %437 = tensor.collapse_shape %436 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16x16xf32> into tensor<256xf32>
    %438 = tensor.expand_shape %437 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x16x16xf32>
    %439 = tensor.collapse_shape %438 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x16xf32> into tensor<256xf32>
    %440 = tensor.expand_shape %439 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %441 = tensor.empty() : tensor<16x8xf32>
    %442 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%421 : tensor<16x8xf32>) outs(%441 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb24(%443: f32, %444: f32):
      %445 = math.sin %443 : f32
      linalg.yield %445 : f32
    } -> tensor<16x8xf32>
    %446 = tensor.empty() : tensor<16x8xf32>
    %447 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%421 : tensor<16x8xf32>) outs(%446 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb25(%448: f32, %449: f32):
      %450 = math.sin %448 : f32
      linalg.yield %450 : f32
    } -> tensor<16x8xf32>
    %451 = tensor.concat dim(1) %442, %447 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %452 = tensor.collapse_shape %451 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16x16xf32> into tensor<256xf32>
    %453 = tensor.expand_shape %452 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x16x16xf32>
    %454 = tensor.collapse_shape %453 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x16xf32> into tensor<256xf32>
    %455 = tensor.expand_shape %454 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %456 = "tensor.extract_slice"(%346) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %457 = "tensor.extract_slice"(%346) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %458 = tensor.empty() : tensor<1x2x16x16xf32>
    %459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346, %440 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%458 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb26(%460: f32, %461: f32, %462: f32):
      %463 = arith.mulf %460, %461 : f32
      linalg.yield %463 : f32
    } -> tensor<1x2x16x16xf32>
    %464 = tensor.empty() : tensor<1x2x16x8xf32>
    %465 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%457 : tensor<1x2x16x8xf32>) outs(%464 : tensor<1x2x16x8xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb27(%466: f32, %467: f32):
      %468 = arith.negf %466 : f32
      linalg.yield %468 : f32
    } -> tensor<1x2x16x8xf32>
    %469 = tensor.concat dim(3) %465, %456 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %470 = tensor.empty() : tensor<1x2x16x16xf32>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%469, %455 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%470 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb28(%472: f32, %473: f32, %474: f32):
      %475 = arith.mulf %472, %473 : f32
      linalg.yield %475 : f32
    } -> tensor<1x2x16x16xf32>
    %476 = tensor.empty() : tensor<1x2x16x16xf32>
    %477 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459, %471 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%476 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb29(%478: f32, %479: f32, %480: f32):
      %481 = arith.addf %478, %479 : f32
      linalg.yield %481 : f32
    } -> tensor<1x2x16x16xf32>
    %482 = tensor.empty() : tensor<8xf32>
    %483 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%482 : tensor<8xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb30(%484: f32):
      %485 = linalg.index 0 : index
      %486 = arith.index_cast %485 : index to i64
      %487 = arith.sitofp %486 : i64 to f32
      %488 = arith.constant 1.000000e+00 : f32
      %489 = arith.mulf %487, %488 : f32
      %490 = arith.constant 0.000000e+00 : f32
      %491 = arith.addf %490, %489 : f32
      linalg.yield %491 : f32
    } -> tensor<8xf32>
    %492 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 8.000000e+00 : f32
    %493 = tensor.splat %492 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32>
    %494 = tensor.empty() : tensor<8xf32>
    %495 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%483, %493 : tensor<8xf32>, tensor<8xf32>) outs(%494 : tensor<8xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb31(%496: f32, %497: f32, %498: f32):
      %499 = arith.divf %496, %497 : f32
      linalg.yield %499 : f32
    } -> tensor<8xf32>
    %500 = tensor.empty() : tensor<8xf32>
    %501 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%495 : tensor<8xf32>) outs(%500 : tensor<8xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb32(%502: f32, %503: f32):
      %504 = arith.constant 1.000000e+04 : f32
      %505 = math.powf %504, %502 : f32
      linalg.yield %505 : f32
    } -> tensor<8xf32>
    %506 = tensor.empty() : tensor<8xf32>
    %507 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%501 : tensor<8xf32>) outs(%506 : tensor<8xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb33(%508: f32, %509: f32):
      %510 = arith.constant 1.000000e+00 : f32
      %511 = arith.divf %510, %508 : f32
      linalg.yield %511 : f32
    } -> tensor<8xf32>
    %512 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 1.000000e+00 : f32
    %513 = tensor.splat %512 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32>
    %514 = tensor.empty() : tensor<8xf32>
    %515 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%507, %513 : tensor<8xf32>, tensor<8xf32>) outs(%514 : tensor<8xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb34(%516: f32, %517: f32, %518: f32):
      %519 = arith.mulf %516, %517 : f32
      linalg.yield %519 : f32
    } -> tensor<8xf32>
    %520 = tensor.expand_shape %258 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16xi64> into tensor<16x1xi64>
    %521 = tensor.empty() : tensor<16x1xf32>
    %522 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%520 : tensor<16x1xi64>) outs(%521 : tensor<16x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb35(%523: i64, %524: f32):
      %525 = arith.sitofp %523 : i64 to f32
      linalg.yield %525 : f32
    } -> tensor<16x1xf32>
    %526 = tensor.expand_shape %515 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<8xf32> into tensor<1x8xf32>
    %527 = tensor.empty() : tensor<16x8xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%522, %526 : tensor<16x1xf32>, tensor<1x8xf32>) outs(%527 : tensor<16x8xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb36(%529: f32, %530: f32, %531: f32):
      %532 = arith.mulf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<16x8xf32>
    %533 = tensor.empty() : tensor<16x8xf32>
    %534 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%528 : tensor<16x8xf32>) outs(%533 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb37(%535: f32, %536: f32):
      %537 = math.cos %535 : f32
      linalg.yield %537 : f32
    } -> tensor<16x8xf32>
    %538 = tensor.empty() : tensor<16x8xf32>
    %539 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%528 : tensor<16x8xf32>) outs(%538 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb38(%540: f32, %541: f32):
      %542 = math.cos %540 : f32
      linalg.yield %542 : f32
    } -> tensor<16x8xf32>
    %543 = tensor.concat dim(1) %534, %539 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %544 = tensor.collapse_shape %543 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16x16xf32> into tensor<256xf32>
    %545 = tensor.expand_shape %544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x16x16xf32>
    %546 = tensor.collapse_shape %545 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x16xf32> into tensor<256xf32>
    %547 = tensor.expand_shape %546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %548 = tensor.empty() : tensor<16x8xf32>
    %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%528 : tensor<16x8xf32>) outs(%548 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb39(%550: f32, %551: f32):
      %552 = math.sin %550 : f32
      linalg.yield %552 : f32
    } -> tensor<16x8xf32>
    %553 = tensor.empty() : tensor<16x8xf32>
    %554 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%528 : tensor<16x8xf32>) outs(%553 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb40(%555: f32, %556: f32):
      %557 = math.sin %555 : f32
      linalg.yield %557 : f32
    } -> tensor<16x8xf32>
    %558 = tensor.concat dim(1) %549, %554 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %559 = tensor.collapse_shape %558 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<16x16xf32> into tensor<256xf32>
    %560 = tensor.expand_shape %559 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x16x16xf32>
    %561 = tensor.collapse_shape %560 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x16xf32> into tensor<256xf32>
    %562 = tensor.expand_shape %561 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %563 = "tensor.extract_slice"(%360) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %564 = "tensor.extract_slice"(%360) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %565 = tensor.empty() : tensor<1x2x16x16xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %547 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%565 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb41(%567: f32, %568: f32, %569: f32):
      %570 = arith.mulf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x2x16x16xf32>
    %571 = tensor.empty() : tensor<1x2x16x8xf32>
    %572 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%564 : tensor<1x2x16x8xf32>) outs(%571 : tensor<1x2x16x8xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb42(%573: f32, %574: f32):
      %575 = arith.negf %573 : f32
      linalg.yield %575 : f32
    } -> tensor<1x2x16x8xf32>
    %576 = tensor.concat dim(3) %572, %563 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %577 = tensor.empty() : tensor<1x2x16x16xf32>
    %578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%576, %562 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%577 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb43(%579: f32, %580: f32, %581: f32):
      %582 = arith.mulf %579, %580 : f32
      linalg.yield %582 : f32
    } -> tensor<1x2x16x16xf32>
    %583 = tensor.empty() : tensor<1x2x16x16xf32>
    %584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %578 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%583 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb44(%585: f32, %586: f32, %587: f32):
      %588 = arith.addf %585, %586 : f32
      linalg.yield %588 : f32
    } -> tensor<1x2x16x16xf32>
    %589 = tensor.empty() : tensor<1x2x16x16xf32>
    %590 = linalg.transpose ins(%584:tensor<1x2x16x16xf32>) outs(%589:tensor<1x2x16x16xf32>) permutation = [0, 1, 3, 2]
    %591 = tensor.empty() : tensor<1x2x16x16xf32>
    %592 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%477 : tensor<1x2x16x16xf32>) outs(%591 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb45(%593: f32, %594: f32):
      linalg.yield %593 : f32
    } -> tensor<1x2x16x16xf32>
    %595 = tensor.collapse_shape %592 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %596 = tensor.expand_shape %595 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<2x16x16xf32>
    %597 = tensor.empty() : tensor<1x2x16x16xf32>
    %598 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%590 : tensor<1x2x16x16xf32>) outs(%597 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb46(%599: f32, %600: f32):
      linalg.yield %599 : f32
    } -> tensor<1x2x16x16xf32>
    %601 = tensor.collapse_shape %598 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %602 = tensor.expand_shape %601 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<2x16x16xf32>
    %603 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 0.000000e+00 : f32
    %604 = tensor.splat %603 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<2x16x16xf32>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%596, %602 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%604 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb47(%606: f32, %607: f32, %608: f32):
      %609 = arith.mulf %606, %607 : f32
      %610 = arith.addf %608, %609 : f32
      linalg.yield %610 : f32
    } -> tensor<2x16x16xf32>
    %611 = tensor.collapse_shape %605 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<2x16x16xf32> into tensor<512xf32>
    %612 = tensor.expand_shape %611 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %613 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 2.500000e-01 : f32
    %614 = tensor.splat %613 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16x16xf32>
    %615 = tensor.empty() : tensor<1x2x16x16xf32>
    %616 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%612, %614 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%615 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb48(%617: f32, %618: f32, %619: f32):
      %620 = arith.mulf %617, %618 : f32
      linalg.yield %620 : f32
    } -> tensor<1x2x16x16xf32>
    %621 = tensor.empty() : tensor<1x2x16x16xf32>
    %622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%616, %256 : tensor<1x2x16x16xf32>, tensor<1x1x1x16xf32>) outs(%621 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb49(%623: f32, %624: f32, %625: f32):
      %626 = arith.addf %623, %624 : f32
      linalg.yield %626 : f32
    } -> tensor<1x2x16x16xf32>
    %627 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 0xff800000 : f32
    %628 = tensor.splat %627 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16xf32>
    %629 = linalg.reduce ins(%622:tensor<1x2x16x16xf32>) outs(%628:tensor<1x2x16xf32>) dimensions = [3]
    (%630: f32, %631: f32) {
      %632 = arith.maximumf %630, %631 : f32
      linalg.yield %632 : f32
    }
    %633 = tensor.collapse_shape %629 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16xf32> into tensor<32xf32>
    %634 = tensor.expand_shape %633 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %635 = tensor.empty() : tensor<1x2x16x16xf32>
    %636 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%622, %634 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%635 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb50(%637: f32, %638: f32, %639: f32):
      %640 = arith.subf %637, %638 : f32
      linalg.yield %640 : f32
    } -> tensor<1x2x16x16xf32>
    %641 = tensor.empty() : tensor<1x2x16x16xf32>
    %642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%636 : tensor<1x2x16x16xf32>) outs(%641 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb51(%643: f32, %644: f32):
      %645 = math.exp %643 : f32
      linalg.yield %645 : f32
    } -> tensor<1x2x16x16xf32>
    %646 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 0.000000e+00 : f32
    %647 = tensor.splat %646 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16xf32>
    %648 = linalg.reduce ins(%642:tensor<1x2x16x16xf32>) outs(%647:tensor<1x2x16xf32>) dimensions = [3]
    (%649: f32, %650: f32) {
      %651 = arith.addf %649, %650 : f32
      linalg.yield %651 : f32
    }
    %652 = tensor.collapse_shape %648 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16xf32> into tensor<32xf32>
    %653 = tensor.expand_shape %652 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %654 = tensor.empty() : tensor<1x2x16x16xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%642, %653 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%654 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb52(%656: f32, %657: f32, %658: f32):
      %659 = arith.divf %656, %657 : f32
      linalg.yield %659 : f32
    } -> tensor<1x2x16x16xf32>
    %660 = tensor.empty() : tensor<1x2x16x16xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655 : tensor<1x2x16x16xf32>) outs(%660 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb53(%662: f32, %663: f32):
      linalg.yield %662 : f32
    } -> tensor<1x2x16x16xf32>
    %664 = tensor.collapse_shape %661 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %665 = tensor.expand_shape %664 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<2x16x16xf32>
    %666 = tensor.empty() : tensor<1x2x16x16xf32>
    %667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374 : tensor<1x2x16x16xf32>) outs(%666 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb54(%668: f32, %669: f32):
      linalg.yield %668 : f32
    } -> tensor<1x2x16x16xf32>
    %670 = tensor.collapse_shape %667 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %671 = tensor.expand_shape %670 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<2x16x16xf32>
    %672 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} 0.000000e+00 : f32
    %673 = tensor.splat %672 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<2x16x16xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%665, %671 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%673 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} {
    ^bb55(%675: f32, %676: f32, %677: f32):
      %678 = arith.mulf %675, %676 : f32
      %679 = arith.addf %677, %678 : f32
      linalg.yield %679 : f32
    } -> tensor<2x16x16xf32>
    %680 = tensor.collapse_shape %674 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<2x16x16xf32> into tensor<512xf32>
    %681 = tensor.expand_shape %680 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %682 = tensor.empty() : tensor<1x16x2x16xf32>
    %683 = linalg.transpose ins(%681:tensor<1x2x16x16xf32>) outs(%682:tensor<1x16x2x16xf32>) permutation = [0, 2, 1, 3]
    %684 = tensor.collapse_shape %683 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<1x16x2x16xf32> into tensor<512xf32>
    %685 = tensor.expand_shape %684 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : tensor<512xf32> into tensor<1x16x32xf32>
    %686 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00384216523 : f32
    %687 = tensor.splat %686 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %688 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %689 = tensor.splat %688 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %690 = "quant_ext.quantize_per_tensor"(%685, %687, %689) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_4", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %691 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00384216523 : f32
    %692 = tensor.splat %691 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %693 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %694 = tensor.splat %693 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %695 = "quant_ext.dequantize_per_tensor"(%690, %692, %694) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_28", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %696 = tensor.empty() : tensor<32x32xf32>
    %697 = linalg.transpose ins(%63:tensor<32x32xf32>) outs(%696:tensor<32x32xf32>) permutation = [1, 0]
    %698 = tensor.collapse_shape %695 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : tensor<1x16x32xf32> into tensor<512xf32>
    %699 = tensor.expand_shape %698 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : tensor<512xf32> into tensor<16x32xf32>
    %700 = tensor.empty() : tensor<16x32xf32>
    %701 = arith.constant {prov.module = "pre_attn"} 0.000000e+00 : f32
    %702 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} ins(%701 : f32) outs(%700 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %703 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o", prov.transposed_b = "true"} ins(%699, %697 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%702 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %704 = tensor.collapse_shape %703 [[0 : i64, 1 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : tensor<16x32xf32> into tensor<512xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : tensor<512xf32> into tensor<1x16x32xf32>
    %706 = tensor.empty() : tensor<1x16x32xf32>
    %707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %705 : tensor<1x16x32xf32>, tensor<1x16x32xf32>) outs(%706 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb56(%708: f32, %709: f32, %710: f32):
      %711 = arith.addf %708, %709 : f32
      linalg.yield %711 : f32
    } -> tensor<1x16x32xf32>
    %712 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0192604251 : f32
    %713 = tensor.splat %712 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %714 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %715 = tensor.splat %714 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %716 = "quant_ext.quantize_per_tensor"(%707, %713, %715) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_5", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %717 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0192604251 : f32
    %718 = tensor.splat %717 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %719 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %720 = tensor.splat %719 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %721 = "quant_ext.dequantize_per_tensor"(%716, %718, %720) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_29", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %722 = tensor.empty() : tensor<1x16x32xf32>
    %723 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707 : tensor<1x16x32xf32>) outs(%722 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb57(%724: f32, %725: f32):
      %726 = arith.constant 2.000000e+00 : f32
      %727 = math.powf %724, %726 : f32
      linalg.yield %727 : f32
    } -> tensor<1x16x32xf32>
    %728 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} 0.000000e+00 : f32
    %729 = tensor.splat %728 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : tensor<1x16xf32>
    %730 = linalg.reduce ins(%723:tensor<1x16x32xf32>) outs(%729:tensor<1x16xf32>) dimensions = [2]
    (%731: f32, %732: f32) {
      %733 = arith.addf %731, %732 : f32
      linalg.yield %733 : f32
    }
    %734 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} 3.200000e+01 : f32
    %735 = tensor.splat %734 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : tensor<1x16xf32>
    %736 = tensor.empty() : tensor<1x16xf32>
    %737 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%730, %735 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%736 : tensor<1x16xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb58(%738: f32, %739: f32, %740: f32):
      %741 = arith.divf %738, %739 : f32
      linalg.yield %741 : f32
    } -> tensor<1x16xf32>
    %742 = tensor.collapse_shape %737 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : tensor<1x16xf32> into tensor<16xf32>
    %743 = tensor.expand_shape %742 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : tensor<16xf32> into tensor<1x16x1xf32>
    %744 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} 1.000000e-05 : f32
    %745 = tensor.splat %744 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : tensor<1x16x1xf32>
    %746 = tensor.empty() : tensor<1x16x1xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%743, %745 : tensor<1x16x1xf32>, tensor<1x16x1xf32>) outs(%746 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb59(%748: f32, %749: f32, %750: f32):
      %751 = arith.addf %748, %749 : f32
      linalg.yield %751 : f32
    } -> tensor<1x16x1xf32>
    %752 = tensor.empty() : tensor<1x16x1xf32>
    %753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%747 : tensor<1x16x1xf32>) outs(%752 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb60(%754: f32, %755: f32):
      %756 = math.rsqrt %754 : f32
      linalg.yield %756 : f32
    } -> tensor<1x16x1xf32>
    %757 = tensor.empty() : tensor<1x16x32xf32>
    %758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %753 : tensor<1x16x32xf32>, tensor<1x16x1xf32>) outs(%757 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb61(%759: f32, %760: f32, %761: f32):
      %762 = arith.mulf %759, %760 : f32
      linalg.yield %762 : f32
    } -> tensor<1x16x32xf32>
    %763 = tensor.empty() : tensor<1x16x32xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%758, %2 : tensor<1x16x32xf32>, tensor<32xf32>) outs(%763 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} {
    ^bb62(%765: f32, %766: f32, %767: f32):
      %768 = arith.mulf %765, %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x16x32xf32>
    %769 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0270536188 : f32
    %770 = tensor.splat %769 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %771 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %772 = tensor.splat %771 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %773 = "quant_ext.quantize_per_tensor"(%764, %770, %772) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_6", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %774 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0270536188 : f32
    %775 = tensor.splat %774 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %776 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %777 = tensor.splat %776 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %778 = "quant_ext.dequantize_per_tensor"(%773, %775, %777) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_30", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %779 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0270536188 : f32
    %780 = tensor.splat %779 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %781 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %782 = tensor.splat %781 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %783 = "quant_ext.dequantize_per_tensor"(%773, %780, %782) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_31", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %784 = tensor.empty() : tensor<32x64xf32>
    %785 = linalg.transpose ins(%68:tensor<64x32xf32>) outs(%784:tensor<32x64xf32>) permutation = [1, 0]
    %786 = tensor.collapse_shape %783 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : tensor<1x16x32xf32> into tensor<512xf32>
    %787 = tensor.expand_shape %786 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : tensor<512xf32> into tensor<16x32xf32>
    %788 = tensor.empty() : tensor<16x64xf32>
    %789 = arith.constant {prov.module = "pre_ffn"} 0.000000e+00 : f32
    %790 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} ins(%789 : f32) outs(%788 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %791 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g", prov.transposed_b = "true"} ins(%787, %785 : tensor<16x32xf32>, tensor<32x64xf32>) outs(%790 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %792 = tensor.collapse_shape %791 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : tensor<16x64xf32> into tensor<1024xf32>
    %793 = tensor.expand_shape %792 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 64] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : tensor<1024xf32> into tensor<1x16x64xf32>
    %794 = tensor.empty() : tensor<1x16x64xf32>
    %795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793 : tensor<1x16x64xf32>) outs(%794 : tensor<1x16x64xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} {
    ^bb63(%796: f32, %797: f32):
      %798 = arith.constant 1.000000e+00 : f32
      %799 = arith.negf %796 : f32
      %800 = math.exp %799 : f32
      %801 = arith.addf %798, %800 : f32
      %802 = arith.divf %798, %801 : f32
      linalg.yield %802 : f32
    } -> tensor<1x16x64xf32>
    %803 = tensor.empty() : tensor<1x16x64xf32>
    %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793, %795 : tensor<1x16x64xf32>, tensor<1x16x64xf32>) outs(%803 : tensor<1x16x64xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} {
    ^bb64(%805: f32, %806: f32, %807: f32):
      %808 = arith.mulf %805, %806 : f32
      linalg.yield %808 : f32
    } -> tensor<1x16x64xf32>
    %809 = tensor.empty() : tensor<32x64xf32>
    %810 = linalg.transpose ins(%73:tensor<64x32xf32>) outs(%809:tensor<32x64xf32>) permutation = [1, 0]
    %811 = tensor.collapse_shape %778 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : tensor<1x16x32xf32> into tensor<512xf32>
    %812 = tensor.expand_shape %811 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : tensor<512xf32> into tensor<16x32xf32>
    %813 = tensor.empty() : tensor<16x64xf32>
    %814 = arith.constant {prov.module = "pre_ffn"} 0.000000e+00 : f32
    %815 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} ins(%814 : f32) outs(%813 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %816 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u", prov.transposed_b = "true"} ins(%812, %810 : tensor<16x32xf32>, tensor<32x64xf32>) outs(%815 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %817 = tensor.collapse_shape %816 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : tensor<16x64xf32> into tensor<1024xf32>
    %818 = tensor.expand_shape %817 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 64] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : tensor<1024xf32> into tensor<1x16x64xf32>
    %819 = tensor.empty() : tensor<1x16x64xf32>
    %820 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804, %818 : tensor<1x16x64xf32>, tensor<1x16x64xf32>) outs(%819 : tensor<1x16x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} {
    ^bb65(%821: f32, %822: f32, %823: f32):
      %824 = arith.mulf %821, %822 : f32
      linalg.yield %824 : f32
    } -> tensor<1x16x64xf32>
    %825 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.013225005 : f32
    %826 = tensor.splat %825 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %827 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %828 = tensor.splat %827 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %829 = "quant_ext.quantize_per_tensor"(%820, %826, %828) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_7", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xi8>
    %830 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.013225005 : f32
    %831 = tensor.splat %830 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %832 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %833 = tensor.splat %832 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %834 = "quant_ext.dequantize_per_tensor"(%829, %831, %833) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_32", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xf32>
    %835 = tensor.empty() : tensor<64x32xf32>
    %836 = linalg.transpose ins(%78:tensor<32x64xf32>) outs(%835:tensor<64x32xf32>) permutation = [1, 0]
    %837 = tensor.collapse_shape %834 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : tensor<1x16x64xf32> into tensor<1024xf32>
    %838 = tensor.expand_shape %837 [[0 : i64, 1 : i64]] output_shape [16, 64] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : tensor<1024xf32> into tensor<16x64xf32>
    %839 = tensor.empty() : tensor<16x32xf32>
    %840 = arith.constant {prov.module = "pre_ffn"} 0.000000e+00 : f32
    %841 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} ins(%840 : f32) outs(%839 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %842 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d", prov.transposed_b = "true"} ins(%838, %836 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%841 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %843 = tensor.collapse_shape %842 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : tensor<16x32xf32> into tensor<512xf32>
    %844 = tensor.expand_shape %843 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : tensor<512xf32> into tensor<1x16x32xf32>
    %845 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 4.187350e-03 : f32
    %846 = tensor.splat %845 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %847 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %848 = tensor.splat %847 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %849 = "quant_ext.quantize_per_tensor"(%844, %846, %848) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_8", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %850 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 4.187350e-03 : f32
    %851 = tensor.splat %850 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %852 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %853 = tensor.splat %852 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %854 = "quant_ext.dequantize_per_tensor"(%849, %851, %853) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_33", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %855 = tensor.empty() : tensor<1x16x32xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721, %854 : tensor<1x16x32xf32>, tensor<1x16x32xf32>) outs(%855 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb66(%857: f32, %858: f32, %859: f32):
      %860 = arith.addf %857, %858 : f32
      linalg.yield %860 : f32
    } -> tensor<1x16x32xf32>
    %861 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0192180183 : f32
    %862 = tensor.splat %861 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %863 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %864 = tensor.splat %863 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %865 = "quant_ext.quantize_per_tensor"(%856, %862, %864) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_9", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %866 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0192180183 : f32
    %867 = tensor.splat %866 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %868 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %869 = tensor.splat %868 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %870 = "quant_ext.dequantize_per_tensor"(%865, %867, %869) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_34", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %871 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0192180183 : f32
    %872 = tensor.splat %871 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %873 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %874 = tensor.splat %873 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %875 = "quant_ext.dequantize_per_tensor"(%865, %872, %874) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_35", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %876 = tensor.empty() : tensor<32x32xf32>
    %877 = linalg.transpose ins(%83:tensor<32x32xf32>) outs(%876:tensor<32x32xf32>) permutation = [1, 0]
    %878 = tensor.collapse_shape %173 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : tensor<1x16x32xf32> into tensor<512xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : tensor<512xf32> into tensor<16x32xf32>
    %880 = tensor.empty() : tensor<16x32xf32>
    %881 = arith.constant {prov.module = "act_in"} 0.000000e+00 : f32
    %882 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "act_in"} ins(%881 : f32) outs(%880 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %883 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in", prov.transposed_b = "true"} ins(%879, %877 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%882 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %884 = tensor.collapse_shape %883 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : tensor<16x32xf32> into tensor<512xf32>
    %885 = tensor.expand_shape %884 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : tensor<512xf32> into tensor<1x16x32xf32>
    %886 = tensor.empty() : tensor<16xf32>
    %887 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%886 : tensor<16xf32>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32"} {
    ^bb67(%888: f32):
      %889 = linalg.index 0 : index
      %890 = arith.index_cast %889 : index to i64
      %891 = arith.sitofp %890 : i64 to f32
      %892 = arith.constant 1.000000e+00 : f32
      %893 = arith.mulf %891, %892 : f32
      %894 = arith.constant 0.000000e+00 : f32
      %895 = arith.addf %894, %893 : f32
      linalg.yield %895 : f32
    } -> tensor<16xf32>
    %896 = arith.constant {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -9.2103405 : f32
    %897 = tensor.splat %896 {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16xf32>
    %898 = tensor.empty() : tensor<16xf32>
    %899 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%887, %897 : tensor<16xf32>, tensor<16xf32>) outs(%898 : tensor<16xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb68(%900: f32, %901: f32, %902: f32):
      %903 = arith.mulf %900, %901 : f32
      linalg.yield %903 : f32
    } -> tensor<16xf32>
    %904 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.600000e+01 : f32
    %905 = tensor.splat %904 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<16xf32>
    %906 = tensor.empty() : tensor<16xf32>
    %907 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%899, %905 : tensor<16xf32>, tensor<16xf32>) outs(%906 : tensor<16xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb69(%908: f32, %909: f32, %910: f32):
      %911 = arith.divf %908, %909 : f32
      linalg.yield %911 : f32
    } -> tensor<16xf32>
    %912 = tensor.empty() : tensor<16xf32>
    %913 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%907 : tensor<16xf32>) outs(%912 : tensor<16xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} {
    ^bb70(%914: f32, %915: f32):
      %916 = math.exp %914 : f32
      linalg.yield %916 : f32
    } -> tensor<16xf32>
    %917 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 5.000000e-01 : f32
    %918 = tensor.splat %917 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16xf32>
    %919 = tensor.empty() : tensor<16xf32>
    %920 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%913, %918 : tensor<16xf32>, tensor<16xf32>) outs(%919 : tensor<16xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb71(%921: f32, %922: f32, %923: f32):
      %924 = arith.mulf %921, %922 : f32
      linalg.yield %924 : f32
    } -> tensor<16xf32>
    %925 = tensor.empty() : tensor<16xf32>
    %926 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%920 : tensor<16xf32>) outs(%925 : tensor<16xf32>) attrs =  {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb72(%927: f32, %928: f32):
      %929 = math.sin %927 : f32
      linalg.yield %929 : f32
    } -> tensor<16xf32>
    %930 = tensor.empty() : tensor<16xf32>
    %931 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%920 : tensor<16xf32>) outs(%930 : tensor<16xf32>) attrs =  {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb73(%932: f32, %933: f32):
      %934 = math.cos %932 : f32
      linalg.yield %934 : f32
    } -> tensor<16xf32>
    %935 = tensor.concat dim(0) %926, %931 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>) -> tensor<32xf32>
    %936 = tensor.expand_shape %935 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32xf32>
    %937 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00783930812 : f32
    %938 = tensor.splat %937 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %939 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %940 = tensor.splat %939 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %941 = "quant_ext.quantize_per_tensor"(%936, %938, %940) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_10", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32xi8>
    %942 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00783930812 : f32
    %943 = tensor.splat %942 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %944 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %945 = tensor.splat %944 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %946 = "quant_ext.dequantize_per_tensor"(%941, %943, %945) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_36", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32xf32>
    %947 = tensor.empty() : tensor<32x32xf32>
    %948 = linalg.transpose ins(%88:tensor<32x32xf32>) outs(%947:tensor<32x32xf32>) permutation = [1, 0]
    %949 = tensor.empty() : tensor<1x32xf32>
    %950 = arith.constant {prov.module = "t_in"} 0.000000e+00 : f32
    %951 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "t_in"} ins(%950 : f32) outs(%949 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %952 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "t_in", prov.fqn = "t_in", prov.transposed_b = "true"} ins(%946, %948 : tensor<1x32xf32>, tensor<32x32xf32>) outs(%951 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %953 = tensor.collapse_shape %952 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %954 = tensor.expand_shape %953 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
    %955 = tensor.empty() : tensor<1x16x32xf32>
    %956 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%885, %954 : tensor<1x16x32xf32>, tensor<1x1x32xf32>) outs(%955 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb74(%957: f32, %958: f32, %959: f32):
      %960 = arith.addf %957, %958 : f32
      linalg.yield %960 : f32
    } -> tensor<1x16x32xf32>
    %961 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} -1.000000e+04 : f32
    %962 = tensor.splat %961 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : tensor<16x16xf32>
    %963 = tensor.empty() : tensor<16xi64>
    %964 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%963 : tensor<16xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb75(%965: i64):
      %966 = linalg.index 0 : index
      %967 = arith.index_cast %966 : index to i64
      %968 = arith.constant 1 : i64
      %969 = arith.muli %967, %968 : i64
      %970 = arith.constant 0 : i64
      %971 = arith.addi %970, %969 : i64
      linalg.yield %971 : i64
    } -> tensor<16xi64>
    %972 = tensor.expand_shape %964 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<1x16xi64>
    %973 = tensor.empty() : tensor<16xi64>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%973 : tensor<16xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb76(%975: i64):
      %976 = linalg.index 0 : index
      %977 = arith.index_cast %976 : index to i64
      %978 = arith.constant 1 : i64
      %979 = arith.muli %977, %978 : i64
      %980 = arith.constant 0 : i64
      %981 = arith.addi %980, %979 : i64
      linalg.yield %981 : i64
    } -> tensor<16xi64>
    %982 = tensor.expand_shape %974 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<16x1xi64>
    %983 = tensor.empty() : tensor<16x16xi64>
    %984 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%972, %982 : tensor<1x16xi64>, tensor<16x1xi64>) outs(%983 : tensor<16x16xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} {
    ^bb77(%985: i64, %986: i64, %987: i64):
      %988 = arith.subi %985, %986 : i64
      linalg.yield %988 : i64
    } -> tensor<16x16xi64>
    %989 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} 1 : i64
    %990 = tensor.splat %989 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} : tensor<16x16xi64>
    %991 = tensor.empty() : tensor<16x16xi1>
    %992 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%984, %990 : tensor<16x16xi64>, tensor<16x16xi64>) outs(%991 : tensor<16x16xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} {
    ^bb78(%993: i64, %994: i64, %995: i1):
      %996 = arith.cmpi sge, %993, %994 : i64
      linalg.yield %996 : i1
    } -> tensor<16x16xi1>
    %997 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %998 = tensor.splat %997 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %999 = tensor.empty() : tensor<16x16xf32>
    %1000 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%992, %962, %998 : tensor<16x16xi1>, tensor<16x16xf32>, tensor<f32>) outs(%999 : tensor<16x16xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb79(%1001: i1, %1002: f32, %1003: f32, %1004: f32):
      %1005 = arith.select %1001, %1002, %1003 : f32
      linalg.yield %1005 : f32
    } -> tensor<16x16xf32>
    %1006 = tensor.collapse_shape %1000 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<16x16xf32> into tensor<256xf32>
    %1007 = tensor.expand_shape %1006 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x16x16xf32>
    %1008 = tensor.collapse_shape %1007 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x16x16xf32> into tensor<256xf32>
    %1009 = tensor.expand_shape %1008 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %1010 = tensor.empty() : tensor<1x16x32xf32>
    %1011 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%956 : tensor<1x16x32xf32>) outs(%1010 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb80(%1012: f32, %1013: f32):
      %1014 = arith.constant 2.000000e+00 : f32
      %1015 = math.powf %1012, %1014 : f32
      linalg.yield %1015 : f32
    } -> tensor<1x16x32xf32>
    %1016 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} 0.000000e+00 : f32
    %1017 = tensor.splat %1016 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : tensor<1x16xf32>
    %1018 = linalg.reduce ins(%1011:tensor<1x16x32xf32>) outs(%1017:tensor<1x16xf32>) dimensions = [2]
    (%1019: f32, %1020: f32) {
      %1021 = arith.addf %1019, %1020 : f32
      linalg.yield %1021 : f32
    }
    %1022 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} 3.200000e+01 : f32
    %1023 = tensor.splat %1022 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : tensor<1x16xf32>
    %1024 = tensor.empty() : tensor<1x16xf32>
    %1025 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1018, %1023 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%1024 : tensor<1x16xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb81(%1026: f32, %1027: f32, %1028: f32):
      %1029 = arith.divf %1026, %1027 : f32
      linalg.yield %1029 : f32
    } -> tensor<1x16xf32>
    %1030 = tensor.collapse_shape %1025 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : tensor<1x16xf32> into tensor<16xf32>
    %1031 = tensor.expand_shape %1030 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : tensor<16xf32> into tensor<1x16x1xf32>
    %1032 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} 1.000000e-05 : f32
    %1033 = tensor.splat %1032 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : tensor<1x16x1xf32>
    %1034 = tensor.empty() : tensor<1x16x1xf32>
    %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1031, %1033 : tensor<1x16x1xf32>, tensor<1x16x1xf32>) outs(%1034 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb82(%1036: f32, %1037: f32, %1038: f32):
      %1039 = arith.addf %1036, %1037 : f32
      linalg.yield %1039 : f32
    } -> tensor<1x16x1xf32>
    %1040 = tensor.empty() : tensor<1x16x1xf32>
    %1041 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1035 : tensor<1x16x1xf32>) outs(%1040 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb83(%1042: f32, %1043: f32):
      %1044 = math.rsqrt %1042 : f32
      linalg.yield %1044 : f32
    } -> tensor<1x16x1xf32>
    %1045 = tensor.empty() : tensor<1x16x32xf32>
    %1046 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%956, %1041 : tensor<1x16x32xf32>, tensor<1x16x1xf32>) outs(%1045 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb84(%1047: f32, %1048: f32, %1049: f32):
      %1050 = arith.mulf %1047, %1048 : f32
      linalg.yield %1050 : f32
    } -> tensor<1x16x32xf32>
    %1051 = tensor.empty() : tensor<1x16x32xf32>
    %1052 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1046, %3 : tensor<1x16x32xf32>, tensor<32xf32>) outs(%1051 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} {
    ^bb85(%1053: f32, %1054: f32, %1055: f32):
      %1056 = arith.mulf %1053, %1054 : f32
      linalg.yield %1056 : f32
    } -> tensor<1x16x32xf32>
    %1057 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.024379164 : f32
    %1058 = tensor.splat %1057 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1059 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1060 = tensor.splat %1059 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1061 = "quant_ext.quantize_per_tensor"(%1052, %1058, %1060) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_11", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1062 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.024379164 : f32
    %1063 = tensor.splat %1062 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1064 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1065 = tensor.splat %1064 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1066 = "quant_ext.dequantize_per_tensor"(%1061, %1063, %1065) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_37", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1067 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.024379164 : f32
    %1068 = tensor.splat %1067 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1069 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1070 = tensor.splat %1069 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1071 = "quant_ext.dequantize_per_tensor"(%1061, %1068, %1070) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_38", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1072 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.024379164 : f32
    %1073 = tensor.splat %1072 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1074 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1075 = tensor.splat %1074 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1076 = "quant_ext.dequantize_per_tensor"(%1061, %1073, %1075) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_39", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1077 = tensor.empty() : tensor<16xi64>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1077 : tensor<16xi64>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb86(%1079: i64):
      %1080 = linalg.index 0 : index
      %1081 = arith.index_cast %1080 : index to i64
      %1082 = arith.constant 1 : i64
      %1083 = arith.muli %1081, %1082 : i64
      %1084 = arith.constant 0 : i64
      %1085 = arith.addi %1084, %1083 : i64
      linalg.yield %1085 : i64
    } -> tensor<16xi64>
    %1086 = tensor.empty() : tensor<32x32xf32>
    %1087 = linalg.transpose ins(%93:tensor<32x32xf32>) outs(%1086:tensor<32x32xf32>) permutation = [1, 0]
    %1088 = tensor.collapse_shape %1076 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1089 = tensor.expand_shape %1088 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : tensor<512xf32> into tensor<16x32xf32>
    %1090 = tensor.empty() : tensor<16x32xf32>
    %1091 = arith.constant {prov.module = "a_self"} 0.000000e+00 : f32
    %1092 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} ins(%1091 : f32) outs(%1090 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1093 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q", prov.transposed_b = "true"} ins(%1089, %1087 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1092 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1094 = tensor.collapse_shape %1093 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : tensor<16x32xf32> into tensor<512xf32>
    %1095 = tensor.expand_shape %1094 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1096 = tensor.collapse_shape %1095 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1097 = tensor.expand_shape %1096 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1098 = tensor.empty() : tensor<1x2x16x16xf32>
    %1099 = linalg.transpose ins(%1097:tensor<1x16x2x16xf32>) outs(%1098:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1100 = tensor.empty() : tensor<32x32xf32>
    %1101 = linalg.transpose ins(%98:tensor<32x32xf32>) outs(%1100:tensor<32x32xf32>) permutation = [1, 0]
    %1102 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1103 = tensor.expand_shape %1102 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : tensor<512xf32> into tensor<16x32xf32>
    %1104 = tensor.empty() : tensor<16x32xf32>
    %1105 = arith.constant {prov.module = "a_self"} 0.000000e+00 : f32
    %1106 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} ins(%1105 : f32) outs(%1104 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1107 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k", prov.transposed_b = "true"} ins(%1103, %1101 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1106 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1108 = tensor.collapse_shape %1107 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : tensor<16x32xf32> into tensor<512xf32>
    %1109 = tensor.expand_shape %1108 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1110 = tensor.collapse_shape %1109 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1111 = tensor.expand_shape %1110 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1112 = tensor.empty() : tensor<1x2x16x16xf32>
    %1113 = linalg.transpose ins(%1111:tensor<1x16x2x16xf32>) outs(%1112:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1114 = tensor.empty() : tensor<32x32xf32>
    %1115 = linalg.transpose ins(%103:tensor<32x32xf32>) outs(%1114:tensor<32x32xf32>) permutation = [1, 0]
    %1116 = tensor.collapse_shape %1066 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1117 = tensor.expand_shape %1116 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : tensor<512xf32> into tensor<16x32xf32>
    %1118 = tensor.empty() : tensor<16x32xf32>
    %1119 = arith.constant {prov.module = "a_self"} 0.000000e+00 : f32
    %1120 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} ins(%1119 : f32) outs(%1118 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1121 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v", prov.transposed_b = "true"} ins(%1117, %1115 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1120 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1122 = tensor.collapse_shape %1121 [[0 : i64, 1 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : tensor<16x32xf32> into tensor<512xf32>
    %1123 = tensor.expand_shape %1122 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1124 = tensor.collapse_shape %1123 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1125 = tensor.expand_shape %1124 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1126 = tensor.empty() : tensor<1x2x16x16xf32>
    %1127 = linalg.transpose ins(%1125:tensor<1x16x2x16xf32>) outs(%1126:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1128 = tensor.empty() : tensor<8xf32>
    %1129 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1128 : tensor<8xf32>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb87(%1130: f32):
      %1131 = linalg.index 0 : index
      %1132 = arith.index_cast %1131 : index to i64
      %1133 = arith.sitofp %1132 : i64 to f32
      %1134 = arith.constant 1.000000e+00 : f32
      %1135 = arith.mulf %1133, %1134 : f32
      %1136 = arith.constant 0.000000e+00 : f32
      %1137 = arith.addf %1136, %1135 : f32
      linalg.yield %1137 : f32
    } -> tensor<8xf32>
    %1138 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 8.000000e+00 : f32
    %1139 = tensor.splat %1138 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32>
    %1140 = tensor.empty() : tensor<8xf32>
    %1141 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1129, %1139 : tensor<8xf32>, tensor<8xf32>) outs(%1140 : tensor<8xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb88(%1142: f32, %1143: f32, %1144: f32):
      %1145 = arith.divf %1142, %1143 : f32
      linalg.yield %1145 : f32
    } -> tensor<8xf32>
    %1146 = tensor.empty() : tensor<8xf32>
    %1147 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1141 : tensor<8xf32>) outs(%1146 : tensor<8xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb89(%1148: f32, %1149: f32):
      %1150 = arith.constant 1.000000e+04 : f32
      %1151 = math.powf %1150, %1148 : f32
      linalg.yield %1151 : f32
    } -> tensor<8xf32>
    %1152 = tensor.empty() : tensor<8xf32>
    %1153 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1147 : tensor<8xf32>) outs(%1152 : tensor<8xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb90(%1154: f32, %1155: f32):
      %1156 = arith.constant 1.000000e+00 : f32
      %1157 = arith.divf %1156, %1154 : f32
      linalg.yield %1157 : f32
    } -> tensor<8xf32>
    %1158 = arith.constant {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 1.000000e+00 : f32
    %1159 = tensor.splat %1158 {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32>
    %1160 = tensor.empty() : tensor<8xf32>
    %1161 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1153, %1159 : tensor<8xf32>, tensor<8xf32>) outs(%1160 : tensor<8xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb91(%1162: f32, %1163: f32, %1164: f32):
      %1165 = arith.mulf %1162, %1163 : f32
      linalg.yield %1165 : f32
    } -> tensor<8xf32>
    %1166 = tensor.expand_shape %1078 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16xi64> into tensor<16x1xi64>
    %1167 = tensor.empty() : tensor<16x1xf32>
    %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1166 : tensor<16x1xi64>) outs(%1167 : tensor<16x1xf32>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb92(%1169: i64, %1170: f32):
      %1171 = arith.sitofp %1169 : i64 to f32
      linalg.yield %1171 : f32
    } -> tensor<16x1xf32>
    %1172 = tensor.expand_shape %1161 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32> into tensor<1x8xf32>
    %1173 = tensor.empty() : tensor<16x8xf32>
    %1174 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1168, %1172 : tensor<16x1xf32>, tensor<1x8xf32>) outs(%1173 : tensor<16x8xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb93(%1175: f32, %1176: f32, %1177: f32):
      %1178 = arith.mulf %1175, %1176 : f32
      linalg.yield %1178 : f32
    } -> tensor<16x8xf32>
    %1179 = tensor.empty() : tensor<16x8xf32>
    %1180 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1174 : tensor<16x8xf32>) outs(%1179 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb94(%1181: f32, %1182: f32):
      %1183 = math.cos %1181 : f32
      linalg.yield %1183 : f32
    } -> tensor<16x8xf32>
    %1184 = tensor.empty() : tensor<16x8xf32>
    %1185 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1174 : tensor<16x8xf32>) outs(%1184 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb95(%1186: f32, %1187: f32):
      %1188 = math.cos %1186 : f32
      linalg.yield %1188 : f32
    } -> tensor<16x8xf32>
    %1189 = tensor.concat dim(1) %1180, %1185 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1190 = tensor.collapse_shape %1189 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16x16xf32> into tensor<256xf32>
    %1191 = tensor.expand_shape %1190 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x16x16xf32>
    %1192 = tensor.collapse_shape %1191 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x16xf32> into tensor<256xf32>
    %1193 = tensor.expand_shape %1192 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %1194 = tensor.empty() : tensor<16x8xf32>
    %1195 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1174 : tensor<16x8xf32>) outs(%1194 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb96(%1196: f32, %1197: f32):
      %1198 = math.sin %1196 : f32
      linalg.yield %1198 : f32
    } -> tensor<16x8xf32>
    %1199 = tensor.empty() : tensor<16x8xf32>
    %1200 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1174 : tensor<16x8xf32>) outs(%1199 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb97(%1201: f32, %1202: f32):
      %1203 = math.sin %1201 : f32
      linalg.yield %1203 : f32
    } -> tensor<16x8xf32>
    %1204 = tensor.concat dim(1) %1195, %1200 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1205 = tensor.collapse_shape %1204 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16x16xf32> into tensor<256xf32>
    %1206 = tensor.expand_shape %1205 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x16x16xf32>
    %1207 = tensor.collapse_shape %1206 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x16xf32> into tensor<256xf32>
    %1208 = tensor.expand_shape %1207 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %1209 = "tensor.extract_slice"(%1099) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1210 = "tensor.extract_slice"(%1099) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1211 = tensor.empty() : tensor<1x2x16x16xf32>
    %1212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1099, %1193 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%1211 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb98(%1213: f32, %1214: f32, %1215: f32):
      %1216 = arith.mulf %1213, %1214 : f32
      linalg.yield %1216 : f32
    } -> tensor<1x2x16x16xf32>
    %1217 = tensor.empty() : tensor<1x2x16x8xf32>
    %1218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1210 : tensor<1x2x16x8xf32>) outs(%1217 : tensor<1x2x16x8xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb99(%1219: f32, %1220: f32):
      %1221 = arith.negf %1219 : f32
      linalg.yield %1221 : f32
    } -> tensor<1x2x16x8xf32>
    %1222 = tensor.concat dim(3) %1218, %1209 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %1223 = tensor.empty() : tensor<1x2x16x16xf32>
    %1224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1222, %1208 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%1223 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb100(%1225: f32, %1226: f32, %1227: f32):
      %1228 = arith.mulf %1225, %1226 : f32
      linalg.yield %1228 : f32
    } -> tensor<1x2x16x16xf32>
    %1229 = tensor.empty() : tensor<1x2x16x16xf32>
    %1230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1212, %1224 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%1229 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb101(%1231: f32, %1232: f32, %1233: f32):
      %1234 = arith.addf %1231, %1232 : f32
      linalg.yield %1234 : f32
    } -> tensor<1x2x16x16xf32>
    %1235 = tensor.empty() : tensor<8xf32>
    %1236 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1235 : tensor<8xf32>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb102(%1237: f32):
      %1238 = linalg.index 0 : index
      %1239 = arith.index_cast %1238 : index to i64
      %1240 = arith.sitofp %1239 : i64 to f32
      %1241 = arith.constant 1.000000e+00 : f32
      %1242 = arith.mulf %1240, %1241 : f32
      %1243 = arith.constant 0.000000e+00 : f32
      %1244 = arith.addf %1243, %1242 : f32
      linalg.yield %1244 : f32
    } -> tensor<8xf32>
    %1245 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 8.000000e+00 : f32
    %1246 = tensor.splat %1245 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32>
    %1247 = tensor.empty() : tensor<8xf32>
    %1248 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1236, %1246 : tensor<8xf32>, tensor<8xf32>) outs(%1247 : tensor<8xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb103(%1249: f32, %1250: f32, %1251: f32):
      %1252 = arith.divf %1249, %1250 : f32
      linalg.yield %1252 : f32
    } -> tensor<8xf32>
    %1253 = tensor.empty() : tensor<8xf32>
    %1254 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1248 : tensor<8xf32>) outs(%1253 : tensor<8xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb104(%1255: f32, %1256: f32):
      %1257 = arith.constant 1.000000e+04 : f32
      %1258 = math.powf %1257, %1255 : f32
      linalg.yield %1258 : f32
    } -> tensor<8xf32>
    %1259 = tensor.empty() : tensor<8xf32>
    %1260 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1254 : tensor<8xf32>) outs(%1259 : tensor<8xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb105(%1261: f32, %1262: f32):
      %1263 = arith.constant 1.000000e+00 : f32
      %1264 = arith.divf %1263, %1261 : f32
      linalg.yield %1264 : f32
    } -> tensor<8xf32>
    %1265 = arith.constant {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 1.000000e+00 : f32
    %1266 = tensor.splat %1265 {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32>
    %1267 = tensor.empty() : tensor<8xf32>
    %1268 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1260, %1266 : tensor<8xf32>, tensor<8xf32>) outs(%1267 : tensor<8xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb106(%1269: f32, %1270: f32, %1271: f32):
      %1272 = arith.mulf %1269, %1270 : f32
      linalg.yield %1272 : f32
    } -> tensor<8xf32>
    %1273 = tensor.expand_shape %1078 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_31", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16xi64> into tensor<16x1xi64>
    %1274 = tensor.empty() : tensor<16x1xf32>
    %1275 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1273 : tensor<16x1xi64>) outs(%1274 : tensor<16x1xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb107(%1276: i64, %1277: f32):
      %1278 = arith.sitofp %1276 : i64 to f32
      linalg.yield %1278 : f32
    } -> tensor<16x1xf32>
    %1279 = tensor.expand_shape %1268 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_32", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<8xf32> into tensor<1x8xf32>
    %1280 = tensor.empty() : tensor<16x8xf32>
    %1281 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1275, %1279 : tensor<16x1xf32>, tensor<1x8xf32>) outs(%1280 : tensor<16x8xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb108(%1282: f32, %1283: f32, %1284: f32):
      %1285 = arith.mulf %1282, %1283 : f32
      linalg.yield %1285 : f32
    } -> tensor<16x8xf32>
    %1286 = tensor.empty() : tensor<16x8xf32>
    %1287 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1281 : tensor<16x8xf32>) outs(%1286 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb109(%1288: f32, %1289: f32):
      %1290 = math.cos %1288 : f32
      linalg.yield %1290 : f32
    } -> tensor<16x8xf32>
    %1291 = tensor.empty() : tensor<16x8xf32>
    %1292 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1281 : tensor<16x8xf32>) outs(%1291 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_8", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb110(%1293: f32, %1294: f32):
      %1295 = math.cos %1293 : f32
      linalg.yield %1295 : f32
    } -> tensor<16x8xf32>
    %1296 = tensor.concat dim(1) %1287, %1292 {prov.region_id = "cat_12", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1297 = tensor.collapse_shape %1296 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_33", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16x16xf32> into tensor<256xf32>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_33", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x16x16xf32>
    %1299 = tensor.collapse_shape %1298 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_34", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x16xf32> into tensor<256xf32>
    %1300 = tensor.expand_shape %1299 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_34", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %1301 = tensor.empty() : tensor<16x8xf32>
    %1302 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1281 : tensor<16x8xf32>) outs(%1301 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb111(%1303: f32, %1304: f32):
      %1305 = math.sin %1303 : f32
      linalg.yield %1305 : f32
    } -> tensor<16x8xf32>
    %1306 = tensor.empty() : tensor<16x8xf32>
    %1307 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1281 : tensor<16x8xf32>) outs(%1306 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_8", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb112(%1308: f32, %1309: f32):
      %1310 = math.sin %1308 : f32
      linalg.yield %1310 : f32
    } -> tensor<16x8xf32>
    %1311 = tensor.concat dim(1) %1302, %1307 {prov.region_id = "cat_13", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1312 = tensor.collapse_shape %1311 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_35", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<16x16xf32> into tensor<256xf32>
    %1313 = tensor.expand_shape %1312 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 16] {prov.region_id = "unsqueeze_35", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x16x16xf32>
    %1314 = tensor.collapse_shape %1313 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_36", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x16xf32> into tensor<256xf32>
    %1315 = tensor.expand_shape %1314 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 16, 16] {prov.region_id = "unsqueeze_36", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<256xf32> into tensor<1x1x16x16xf32>
    %1316 = "tensor.extract_slice"(%1113) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1317 = "tensor.extract_slice"(%1113) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1318 = tensor.empty() : tensor<1x2x16x16xf32>
    %1319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1113, %1300 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%1318 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb113(%1320: f32, %1321: f32, %1322: f32):
      %1323 = arith.mulf %1320, %1321 : f32
      linalg.yield %1323 : f32
    } -> tensor<1x2x16x16xf32>
    %1324 = tensor.empty() : tensor<1x2x16x8xf32>
    %1325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1317 : tensor<1x2x16x8xf32>) outs(%1324 : tensor<1x2x16x8xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb114(%1326: f32, %1327: f32):
      %1328 = arith.negf %1326 : f32
      linalg.yield %1328 : f32
    } -> tensor<1x2x16x8xf32>
    %1329 = tensor.concat dim(3) %1325, %1316 {prov.region_id = "cat_14", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %1330 = tensor.empty() : tensor<1x2x16x16xf32>
    %1331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1329, %1315 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%1330 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb115(%1332: f32, %1333: f32, %1334: f32):
      %1335 = arith.mulf %1332, %1333 : f32
      linalg.yield %1335 : f32
    } -> tensor<1x2x16x16xf32>
    %1336 = tensor.empty() : tensor<1x2x16x16xf32>
    %1337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1319, %1331 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%1336 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb116(%1338: f32, %1339: f32, %1340: f32):
      %1341 = arith.addf %1338, %1339 : f32
      linalg.yield %1341 : f32
    } -> tensor<1x2x16x16xf32>
    %1342 = tensor.empty() : tensor<1x2x16x16xf32>
    %1343 = linalg.transpose ins(%1337:tensor<1x2x16x16xf32>) outs(%1342:tensor<1x2x16x16xf32>) permutation = [0, 1, 3, 2]
    %1344 = tensor.empty() : tensor<1x2x16x16xf32>
    %1345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1230 : tensor<1x2x16x16xf32>) outs(%1344 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb117(%1346: f32, %1347: f32):
      linalg.yield %1346 : f32
    } -> tensor<1x2x16x16xf32>
    %1348 = tensor.collapse_shape %1345 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1349 = tensor.expand_shape %1348 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1350 = tensor.empty() : tensor<1x2x16x16xf32>
    %1351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1343 : tensor<1x2x16x16xf32>) outs(%1350 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb118(%1352: f32, %1353: f32):
      linalg.yield %1352 : f32
    } -> tensor<1x2x16x16xf32>
    %1354 = tensor.collapse_shape %1351 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1355 = tensor.expand_shape %1354 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1356 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 0.000000e+00 : f32
    %1357 = tensor.splat %1356 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<2x16x16xf32>
    %1358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1349, %1355 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%1357 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb119(%1359: f32, %1360: f32, %1361: f32):
      %1362 = arith.mulf %1359, %1360 : f32
      %1363 = arith.addf %1361, %1362 : f32
      linalg.yield %1363 : f32
    } -> tensor<2x16x16xf32>
    %1364 = tensor.collapse_shape %1358 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<2x16x16xf32> into tensor<512xf32>
    %1365 = tensor.expand_shape %1364 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %1366 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 4.000000e+00 : f32
    %1367 = tensor.splat %1366 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16x16xf32>
    %1368 = tensor.empty() : tensor<1x2x16x16xf32>
    %1369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1365, %1367 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%1368 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb120(%1370: f32, %1371: f32, %1372: f32):
      %1373 = arith.divf %1370, %1371 : f32
      linalg.yield %1373 : f32
    } -> tensor<1x2x16x16xf32>
    %1374 = tensor.empty() : tensor<1x2x16x16xf32>
    %1375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1369, %1009 : tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>) outs(%1374 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb121(%1376: f32, %1377: f32, %1378: f32):
      %1379 = arith.addf %1376, %1377 : f32
      linalg.yield %1379 : f32
    } -> tensor<1x2x16x16xf32>
    %1380 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 0xff800000 : f32
    %1381 = tensor.splat %1380 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16xf32>
    %1382 = linalg.reduce ins(%1375:tensor<1x2x16x16xf32>) outs(%1381:tensor<1x2x16xf32>) dimensions = [3]
    (%1383: f32, %1384: f32) {
      %1385 = arith.maximumf %1383, %1384 : f32
      linalg.yield %1385 : f32
    }
    %1386 = tensor.collapse_shape %1382 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16xf32> into tensor<32xf32>
    %1387 = tensor.expand_shape %1386 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %1388 = tensor.empty() : tensor<1x2x16x16xf32>
    %1389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1375, %1387 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%1388 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb122(%1390: f32, %1391: f32, %1392: f32):
      %1393 = arith.subf %1390, %1391 : f32
      linalg.yield %1393 : f32
    } -> tensor<1x2x16x16xf32>
    %1394 = tensor.empty() : tensor<1x2x16x16xf32>
    %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1389 : tensor<1x2x16x16xf32>) outs(%1394 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb123(%1396: f32, %1397: f32):
      %1398 = math.exp %1396 : f32
      linalg.yield %1398 : f32
    } -> tensor<1x2x16x16xf32>
    %1399 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 0.000000e+00 : f32
    %1400 = tensor.splat %1399 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16xf32>
    %1401 = linalg.reduce ins(%1395:tensor<1x2x16x16xf32>) outs(%1400:tensor<1x2x16xf32>) dimensions = [3]
    (%1402: f32, %1403: f32) {
      %1404 = arith.addf %1402, %1403 : f32
      linalg.yield %1404 : f32
    }
    %1405 = tensor.collapse_shape %1401 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16xf32> into tensor<32xf32>
    %1406 = tensor.expand_shape %1405 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %1407 = tensor.empty() : tensor<1x2x16x16xf32>
    %1408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1395, %1406 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%1407 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb124(%1409: f32, %1410: f32, %1411: f32):
      %1412 = arith.divf %1409, %1410 : f32
      linalg.yield %1412 : f32
    } -> tensor<1x2x16x16xf32>
    %1413 = tensor.empty() : tensor<1x2x16x16xf32>
    %1414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1408 : tensor<1x2x16x16xf32>) outs(%1413 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb125(%1415: f32, %1416: f32):
      linalg.yield %1415 : f32
    } -> tensor<1x2x16x16xf32>
    %1417 = tensor.collapse_shape %1414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1418 = tensor.expand_shape %1417 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1419 = tensor.empty() : tensor<1x2x16x16xf32>
    %1420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1127 : tensor<1x2x16x16xf32>) outs(%1419 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb126(%1421: f32, %1422: f32):
      linalg.yield %1421 : f32
    } -> tensor<1x2x16x16xf32>
    %1423 = tensor.collapse_shape %1420 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1424 = tensor.expand_shape %1423 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1425 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} 0.000000e+00 : f32
    %1426 = tensor.splat %1425 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<2x16x16xf32>
    %1427 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1418, %1424 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%1426 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} {
    ^bb127(%1428: f32, %1429: f32, %1430: f32):
      %1431 = arith.mulf %1428, %1429 : f32
      %1432 = arith.addf %1430, %1431 : f32
      linalg.yield %1432 : f32
    } -> tensor<2x16x16xf32>
    %1433 = tensor.collapse_shape %1427 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<2x16x16xf32> into tensor<512xf32>
    %1434 = tensor.expand_shape %1433 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %1435 = tensor.empty() : tensor<1x16x2x16xf32>
    %1436 = linalg.transpose ins(%1434:tensor<1x2x16x16xf32>) outs(%1435:tensor<1x16x2x16xf32>) permutation = [0, 2, 1, 3]
    %1437 = tensor.collapse_shape %1436 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<1x16x2x16xf32> into tensor<512xf32>
    %1438 = tensor.expand_shape %1437 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1439 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0112701422 : f32
    %1440 = tensor.splat %1439 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1441 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1442 = tensor.splat %1441 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1443 = "quant_ext.quantize_per_tensor"(%1438, %1440, %1442) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_12", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1444 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0112701422 : f32
    %1445 = tensor.splat %1444 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1446 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1447 = tensor.splat %1446 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1448 = "quant_ext.dequantize_per_tensor"(%1443, %1445, %1447) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_40", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1449 = tensor.empty() : tensor<32x32xf32>
    %1450 = linalg.transpose ins(%108:tensor<32x32xf32>) outs(%1449:tensor<32x32xf32>) permutation = [1, 0]
    %1451 = tensor.collapse_shape %1448 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1452 = tensor.expand_shape %1451 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : tensor<512xf32> into tensor<16x32xf32>
    %1453 = tensor.empty() : tensor<16x32xf32>
    %1454 = arith.constant {prov.module = "a_self"} 0.000000e+00 : f32
    %1455 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} ins(%1454 : f32) outs(%1453 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1456 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o", prov.transposed_b = "true"} ins(%1452, %1450 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1455 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1457 = tensor.collapse_shape %1456 [[0 : i64, 1 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : tensor<16x32xf32> into tensor<512xf32>
    %1458 = tensor.expand_shape %1457 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1459 = tensor.empty() : tensor<1x16x32xf32>
    %1460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%956, %1458 : tensor<1x16x32xf32>, tensor<1x16x32xf32>) outs(%1459 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb128(%1461: f32, %1462: f32, %1463: f32):
      %1464 = arith.addf %1461, %1462 : f32
      linalg.yield %1464 : f32
    } -> tensor<1x16x32xf32>
    %1465 = tensor.empty() : tensor<1x16x32xf32>
    %1466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1460 : tensor<1x16x32xf32>) outs(%1465 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb129(%1467: f32, %1468: f32):
      %1469 = arith.constant 2.000000e+00 : f32
      %1470 = math.powf %1467, %1469 : f32
      linalg.yield %1470 : f32
    } -> tensor<1x16x32xf32>
    %1471 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} 0.000000e+00 : f32
    %1472 = tensor.splat %1471 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : tensor<1x16xf32>
    %1473 = linalg.reduce ins(%1466:tensor<1x16x32xf32>) outs(%1472:tensor<1x16xf32>) dimensions = [2]
    (%1474: f32, %1475: f32) {
      %1476 = arith.addf %1474, %1475 : f32
      linalg.yield %1476 : f32
    }
    %1477 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} 3.200000e+01 : f32
    %1478 = tensor.splat %1477 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : tensor<1x16xf32>
    %1479 = tensor.empty() : tensor<1x16xf32>
    %1480 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1473, %1478 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%1479 : tensor<1x16xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb130(%1481: f32, %1482: f32, %1483: f32):
      %1484 = arith.divf %1481, %1482 : f32
      linalg.yield %1484 : f32
    } -> tensor<1x16xf32>
    %1485 = tensor.collapse_shape %1480 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : tensor<1x16xf32> into tensor<16xf32>
    %1486 = tensor.expand_shape %1485 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : tensor<16xf32> into tensor<1x16x1xf32>
    %1487 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} 1.000000e-05 : f32
    %1488 = tensor.splat %1487 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : tensor<1x16x1xf32>
    %1489 = tensor.empty() : tensor<1x16x1xf32>
    %1490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1486, %1488 : tensor<1x16x1xf32>, tensor<1x16x1xf32>) outs(%1489 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb131(%1491: f32, %1492: f32, %1493: f32):
      %1494 = arith.addf %1491, %1492 : f32
      linalg.yield %1494 : f32
    } -> tensor<1x16x1xf32>
    %1495 = tensor.empty() : tensor<1x16x1xf32>
    %1496 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1490 : tensor<1x16x1xf32>) outs(%1495 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb132(%1497: f32, %1498: f32):
      %1499 = math.rsqrt %1497 : f32
      linalg.yield %1499 : f32
    } -> tensor<1x16x1xf32>
    %1500 = tensor.empty() : tensor<1x16x32xf32>
    %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1460, %1496 : tensor<1x16x32xf32>, tensor<1x16x1xf32>) outs(%1500 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb133(%1502: f32, %1503: f32, %1504: f32):
      %1505 = arith.mulf %1502, %1503 : f32
      linalg.yield %1505 : f32
    } -> tensor<1x16x32xf32>
    %1506 = tensor.empty() : tensor<1x16x32xf32>
    %1507 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1501, %4 : tensor<1x16x32xf32>, tensor<32xf32>) outs(%1506 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} {
    ^bb134(%1508: f32, %1509: f32, %1510: f32):
      %1511 = arith.mulf %1508, %1509 : f32
      linalg.yield %1511 : f32
    } -> tensor<1x16x32xf32>
    %1512 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0251007639 : f32
    %1513 = tensor.splat %1512 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1514 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1515 = tensor.splat %1514 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1516 = "quant_ext.quantize_per_tensor"(%1507, %1513, %1515) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_13", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1517 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0251007639 : f32
    %1518 = tensor.splat %1517 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1519 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1520 = tensor.splat %1519 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1521 = "quant_ext.dequantize_per_tensor"(%1516, %1518, %1520) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_41", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1522 = tensor.empty() : tensor<32x32xf32>
    %1523 = linalg.transpose ins(%113:tensor<32x32xf32>) outs(%1522:tensor<32x32xf32>) permutation = [1, 0]
    %1524 = tensor.collapse_shape %1521 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1525 = tensor.expand_shape %1524 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : tensor<512xf32> into tensor<16x32xf32>
    %1526 = tensor.empty() : tensor<16x32xf32>
    %1527 = arith.constant {prov.module = "a_cross"} 0.000000e+00 : f32
    %1528 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} ins(%1527 : f32) outs(%1526 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1529 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q", prov.transposed_b = "true"} ins(%1525, %1523 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1528 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1530 = tensor.collapse_shape %1529 [[0 : i64, 1 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : tensor<16x32xf32> into tensor<512xf32>
    %1531 = tensor.expand_shape %1530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1532 = tensor.collapse_shape %1531 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1533 = tensor.expand_shape %1532 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1534 = tensor.empty() : tensor<1x2x16x16xf32>
    %1535 = linalg.transpose ins(%1533:tensor<1x16x2x16xf32>) outs(%1534:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1536 = tensor.empty() : tensor<32x32xf32>
    %1537 = linalg.transpose ins(%118:tensor<32x32xf32>) outs(%1536:tensor<32x32xf32>) permutation = [1, 0]
    %1538 = tensor.collapse_shape %875 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1539 = tensor.expand_shape %1538 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : tensor<512xf32> into tensor<16x32xf32>
    %1540 = tensor.empty() : tensor<16x32xf32>
    %1541 = arith.constant {prov.module = "a_cross"} 0.000000e+00 : f32
    %1542 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} ins(%1541 : f32) outs(%1540 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1543 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k", prov.transposed_b = "true"} ins(%1539, %1537 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1542 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1544 = tensor.collapse_shape %1543 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : tensor<16x32xf32> into tensor<512xf32>
    %1545 = tensor.expand_shape %1544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1546 = tensor.collapse_shape %1545 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1547 = tensor.expand_shape %1546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1548 = tensor.empty() : tensor<1x2x16x16xf32>
    %1549 = linalg.transpose ins(%1547:tensor<1x16x2x16xf32>) outs(%1548:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1550 = tensor.empty() : tensor<32x32xf32>
    %1551 = linalg.transpose ins(%123:tensor<32x32xf32>) outs(%1550:tensor<32x32xf32>) permutation = [1, 0]
    %1552 = tensor.collapse_shape %870 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1553 = tensor.expand_shape %1552 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : tensor<512xf32> into tensor<16x32xf32>
    %1554 = tensor.empty() : tensor<16x32xf32>
    %1555 = arith.constant {prov.module = "a_cross"} 0.000000e+00 : f32
    %1556 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} ins(%1555 : f32) outs(%1554 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1557 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v", prov.transposed_b = "true"} ins(%1553, %1551 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1556 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1558 = tensor.collapse_shape %1557 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : tensor<16x32xf32> into tensor<512xf32>
    %1559 = tensor.expand_shape %1558 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1560 = tensor.collapse_shape %1559 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 2, 16] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x16x2x16xf32>
    %1562 = tensor.empty() : tensor<1x2x16x16xf32>
    %1563 = linalg.transpose ins(%1561:tensor<1x16x2x16xf32>) outs(%1562:tensor<1x2x16x16xf32>) permutation = [0, 2, 1, 3]
    %1564 = tensor.empty() : tensor<1x2x16x16xf32>
    %1565 = linalg.transpose ins(%1549:tensor<1x2x16x16xf32>) outs(%1564:tensor<1x2x16x16xf32>) permutation = [0, 1, 3, 2]
    %1566 = tensor.empty() : tensor<1x2x16x16xf32>
    %1567 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1535 : tensor<1x2x16x16xf32>) outs(%1566 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb135(%1568: f32, %1569: f32):
      linalg.yield %1568 : f32
    } -> tensor<1x2x16x16xf32>
    %1570 = tensor.collapse_shape %1567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1571 = tensor.expand_shape %1570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1572 = tensor.empty() : tensor<1x2x16x16xf32>
    %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1565 : tensor<1x2x16x16xf32>) outs(%1572 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb136(%1574: f32, %1575: f32):
      linalg.yield %1574 : f32
    } -> tensor<1x2x16x16xf32>
    %1576 = tensor.collapse_shape %1573 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1577 = tensor.expand_shape %1576 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1578 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} 0.000000e+00 : f32
    %1579 = tensor.splat %1578 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<2x16x16xf32>
    %1580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1571, %1577 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%1579 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb137(%1581: f32, %1582: f32, %1583: f32):
      %1584 = arith.mulf %1581, %1582 : f32
      %1585 = arith.addf %1583, %1584 : f32
      linalg.yield %1585 : f32
    } -> tensor<2x16x16xf32>
    %1586 = tensor.collapse_shape %1580 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<2x16x16xf32> into tensor<512xf32>
    %1587 = tensor.expand_shape %1586 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %1588 = arith.constant {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} 4.000000e+00 : f32
    %1589 = tensor.splat %1588 {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16x16xf32>
    %1590 = tensor.empty() : tensor<1x2x16x16xf32>
    %1591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1587, %1589 : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) outs(%1590 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb138(%1592: f32, %1593: f32, %1594: f32):
      %1595 = arith.divf %1592, %1593 : f32
      linalg.yield %1595 : f32
    } -> tensor<1x2x16x16xf32>
    %1596 = tensor.empty() : tensor<1x2x16x16xf32>
    %1597 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1591, %256 : tensor<1x2x16x16xf32>, tensor<1x1x1x16xf32>) outs(%1596 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb139(%1598: f32, %1599: f32, %1600: f32):
      %1601 = arith.addf %1598, %1599 : f32
      linalg.yield %1601 : f32
    } -> tensor<1x2x16x16xf32>
    %1602 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} 0xff800000 : f32
    %1603 = tensor.splat %1602 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16xf32>
    %1604 = linalg.reduce ins(%1597:tensor<1x2x16x16xf32>) outs(%1603:tensor<1x2x16xf32>) dimensions = [3]
    (%1605: f32, %1606: f32) {
      %1607 = arith.maximumf %1605, %1606 : f32
      linalg.yield %1607 : f32
    }
    %1608 = tensor.collapse_shape %1604 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16xf32> into tensor<32xf32>
    %1609 = tensor.expand_shape %1608 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %1610 = tensor.empty() : tensor<1x2x16x16xf32>
    %1611 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1597, %1609 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%1610 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb140(%1612: f32, %1613: f32, %1614: f32):
      %1615 = arith.subf %1612, %1613 : f32
      linalg.yield %1615 : f32
    } -> tensor<1x2x16x16xf32>
    %1616 = tensor.empty() : tensor<1x2x16x16xf32>
    %1617 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1611 : tensor<1x2x16x16xf32>) outs(%1616 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb141(%1618: f32, %1619: f32):
      %1620 = math.exp %1618 : f32
      linalg.yield %1620 : f32
    } -> tensor<1x2x16x16xf32>
    %1621 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} 0.000000e+00 : f32
    %1622 = tensor.splat %1621 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16xf32>
    %1623 = linalg.reduce ins(%1617:tensor<1x2x16x16xf32>) outs(%1622:tensor<1x2x16xf32>) dimensions = [3]
    (%1624: f32, %1625: f32) {
      %1626 = arith.addf %1624, %1625 : f32
      linalg.yield %1626 : f32
    }
    %1627 = tensor.collapse_shape %1623 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16xf32> into tensor<32xf32>
    %1628 = tensor.expand_shape %1627 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<32xf32> into tensor<1x2x16x1xf32>
    %1629 = tensor.empty() : tensor<1x2x16x16xf32>
    %1630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1617, %1628 : tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>) outs(%1629 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb142(%1631: f32, %1632: f32, %1633: f32):
      %1634 = arith.divf %1631, %1632 : f32
      linalg.yield %1634 : f32
    } -> tensor<1x2x16x16xf32>
    %1635 = tensor.empty() : tensor<1x2x16x16xf32>
    %1636 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1630 : tensor<1x2x16x16xf32>) outs(%1635 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb143(%1637: f32, %1638: f32):
      linalg.yield %1637 : f32
    } -> tensor<1x2x16x16xf32>
    %1639 = tensor.collapse_shape %1636 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1640 = tensor.expand_shape %1639 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1641 = tensor.empty() : tensor<1x2x16x16xf32>
    %1642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1563 : tensor<1x2x16x16xf32>) outs(%1641 : tensor<1x2x16x16xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb144(%1643: f32, %1644: f32):
      linalg.yield %1643 : f32
    } -> tensor<1x2x16x16xf32>
    %1645 = tensor.collapse_shape %1642 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x2x16x16xf32> into tensor<512xf32>
    %1646 = tensor.expand_shape %1645 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 16, 16] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<2x16x16xf32>
    %1647 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} 0.000000e+00 : f32
    %1648 = tensor.splat %1647 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<2x16x16xf32>
    %1649 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1640, %1646 : tensor<2x16x16xf32>, tensor<2x16x16xf32>) outs(%1648 : tensor<2x16x16xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} {
    ^bb145(%1650: f32, %1651: f32, %1652: f32):
      %1653 = arith.mulf %1650, %1651 : f32
      %1654 = arith.addf %1652, %1653 : f32
      linalg.yield %1654 : f32
    } -> tensor<2x16x16xf32>
    %1655 = tensor.collapse_shape %1649 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<2x16x16xf32> into tensor<512xf32>
    %1656 = tensor.expand_shape %1655 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 16, 16] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x2x16x16xf32>
    %1657 = tensor.empty() : tensor<1x16x2x16xf32>
    %1658 = linalg.transpose ins(%1656:tensor<1x2x16x16xf32>) outs(%1657:tensor<1x16x2x16xf32>) permutation = [0, 2, 1, 3]
    %1659 = tensor.collapse_shape %1658 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<1x16x2x16xf32> into tensor<512xf32>
    %1660 = tensor.expand_shape %1659 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1661 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00221519568 : f32
    %1662 = tensor.splat %1661 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1663 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1664 = tensor.splat %1663 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1665 = "quant_ext.quantize_per_tensor"(%1660, %1662, %1664) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_14", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1666 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00221519568 : f32
    %1667 = tensor.splat %1666 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1668 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1669 = tensor.splat %1668 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1670 = "quant_ext.dequantize_per_tensor"(%1665, %1667, %1669) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_42", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1671 = tensor.empty() : tensor<32x32xf32>
    %1672 = linalg.transpose ins(%128:tensor<32x32xf32>) outs(%1671:tensor<32x32xf32>) permutation = [1, 0]
    %1673 = tensor.collapse_shape %1670 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1674 = tensor.expand_shape %1673 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : tensor<512xf32> into tensor<16x32xf32>
    %1675 = tensor.empty() : tensor<16x32xf32>
    %1676 = arith.constant {prov.module = "a_cross"} 0.000000e+00 : f32
    %1677 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} ins(%1676 : f32) outs(%1675 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1678 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o", prov.transposed_b = "true"} ins(%1674, %1672 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1677 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1679 = tensor.collapse_shape %1678 [[0 : i64, 1 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : tensor<16x32xf32> into tensor<512xf32>
    %1680 = tensor.expand_shape %1679 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1681 = tensor.empty() : tensor<1x16x32xf32>
    %1682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1460, %1680 : tensor<1x16x32xf32>, tensor<1x16x32xf32>) outs(%1681 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb146(%1683: f32, %1684: f32, %1685: f32):
      %1686 = arith.addf %1683, %1684 : f32
      linalg.yield %1686 : f32
    } -> tensor<1x16x32xf32>
    %1687 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0179154743 : f32
    %1688 = tensor.splat %1687 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1689 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1690 = tensor.splat %1689 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1691 = "quant_ext.quantize_per_tensor"(%1682, %1688, %1690) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_15", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1692 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0179154743 : f32
    %1693 = tensor.splat %1692 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1694 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1695 = tensor.splat %1694 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1696 = "quant_ext.dequantize_per_tensor"(%1691, %1693, %1695) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_43", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1697 = tensor.empty() : tensor<1x16x32xf32>
    %1698 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1682 : tensor<1x16x32xf32>) outs(%1697 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb147(%1699: f32, %1700: f32):
      %1701 = arith.constant 2.000000e+00 : f32
      %1702 = math.powf %1699, %1701 : f32
      linalg.yield %1702 : f32
    } -> tensor<1x16x32xf32>
    %1703 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} 0.000000e+00 : f32
    %1704 = tensor.splat %1703 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : tensor<1x16xf32>
    %1705 = linalg.reduce ins(%1698:tensor<1x16x32xf32>) outs(%1704:tensor<1x16xf32>) dimensions = [2]
    (%1706: f32, %1707: f32) {
      %1708 = arith.addf %1706, %1707 : f32
      linalg.yield %1708 : f32
    }
    %1709 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} 3.200000e+01 : f32
    %1710 = tensor.splat %1709 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : tensor<1x16xf32>
    %1711 = tensor.empty() : tensor<1x16xf32>
    %1712 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1705, %1710 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%1711 : tensor<1x16xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb148(%1713: f32, %1714: f32, %1715: f32):
      %1716 = arith.divf %1713, %1714 : f32
      linalg.yield %1716 : f32
    } -> tensor<1x16xf32>
    %1717 = tensor.collapse_shape %1712 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : tensor<1x16xf32> into tensor<16xf32>
    %1718 = tensor.expand_shape %1717 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : tensor<16xf32> into tensor<1x16x1xf32>
    %1719 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} 1.000000e-05 : f32
    %1720 = tensor.splat %1719 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : tensor<1x16x1xf32>
    %1721 = tensor.empty() : tensor<1x16x1xf32>
    %1722 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1718, %1720 : tensor<1x16x1xf32>, tensor<1x16x1xf32>) outs(%1721 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb149(%1723: f32, %1724: f32, %1725: f32):
      %1726 = arith.addf %1723, %1724 : f32
      linalg.yield %1726 : f32
    } -> tensor<1x16x1xf32>
    %1727 = tensor.empty() : tensor<1x16x1xf32>
    %1728 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1722 : tensor<1x16x1xf32>) outs(%1727 : tensor<1x16x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb150(%1729: f32, %1730: f32):
      %1731 = math.rsqrt %1729 : f32
      linalg.yield %1731 : f32
    } -> tensor<1x16x1xf32>
    %1732 = tensor.empty() : tensor<1x16x32xf32>
    %1733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1682, %1728 : tensor<1x16x32xf32>, tensor<1x16x1xf32>) outs(%1732 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb151(%1734: f32, %1735: f32, %1736: f32):
      %1737 = arith.mulf %1734, %1735 : f32
      linalg.yield %1737 : f32
    } -> tensor<1x16x32xf32>
    %1738 = tensor.empty() : tensor<1x16x32xf32>
    %1739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1733, %5 : tensor<1x16x32xf32>, tensor<32xf32>) outs(%1738 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} {
    ^bb152(%1740: f32, %1741: f32, %1742: f32):
      %1743 = arith.mulf %1740, %1741 : f32
      linalg.yield %1743 : f32
    } -> tensor<1x16x32xf32>
    %1744 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0251181908 : f32
    %1745 = tensor.splat %1744 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1746 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1747 = tensor.splat %1746 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1748 = "quant_ext.quantize_per_tensor"(%1739, %1745, %1747) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_16", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1749 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0251181908 : f32
    %1750 = tensor.splat %1749 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1751 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1752 = tensor.splat %1751 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1753 = "quant_ext.dequantize_per_tensor"(%1748, %1750, %1752) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_44", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1754 = tensor.empty() : tensor<32x64xf32>
    %1755 = linalg.transpose ins(%133:tensor<64x32xf32>) outs(%1754:tensor<32x64xf32>) permutation = [1, 0]
    %1756 = tensor.collapse_shape %1753 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1757 = tensor.expand_shape %1756 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : tensor<512xf32> into tensor<16x32xf32>
    %1758 = tensor.empty() : tensor<16x64xf32>
    %1759 = arith.constant {prov.module = "a_ffn"} 0.000000e+00 : f32
    %1760 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_ffn"} ins(%1759 : f32) outs(%1758 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %1761 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up", prov.transposed_b = "true"} ins(%1757, %1755 : tensor<16x32xf32>, tensor<32x64xf32>) outs(%1760 : tensor<16x64xf32>) -> tensor<16x64xf32>
    %1762 = tensor.collapse_shape %1761 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : tensor<16x64xf32> into tensor<1024xf32>
    %1763 = tensor.expand_shape %1762 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 64] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : tensor<1024xf32> into tensor<1x16x64xf32>
    %1764 = tensor.empty() : tensor<1x16x64xf32>
    %1765 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763 : tensor<1x16x64xf32>) outs(%1764 : tensor<1x16x64xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn"} {
    ^bb153(%1766: f32, %1767: f32):
      %1768 = arith.constant 5.000000e-01 : f32
      %1769 = arith.constant 1.000000e+00 : f32
      %1770 = arith.constant 0.707106769 : f32
      %1771 = arith.mulf %1766, %1770 : f32
      %1772 = math.erf %1771 : f32
      %1773 = arith.addf %1769, %1772 : f32
      %1774 = arith.mulf %1768, %1766 : f32
      %1775 = arith.mulf %1774, %1773 : f32
      linalg.yield %1775 : f32
    } -> tensor<1x16x64xf32>
    %1776 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0134872682 : f32
    %1777 = tensor.splat %1776 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1778 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1779 = tensor.splat %1778 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1780 = "quant_ext.quantize_per_tensor"(%1765, %1777, %1779) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_17", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xi8>
    %1781 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0134872682 : f32
    %1782 = tensor.splat %1781 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1783 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1784 = tensor.splat %1783 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1785 = "quant_ext.dequantize_per_tensor"(%1780, %1782, %1784) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_45", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xf32>
    %1786 = tensor.empty() : tensor<64x32xf32>
    %1787 = linalg.transpose ins(%138:tensor<32x64xf32>) outs(%1786:tensor<64x32xf32>) permutation = [1, 0]
    %1788 = tensor.collapse_shape %1785 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : tensor<1x16x64xf32> into tensor<1024xf32>
    %1789 = tensor.expand_shape %1788 [[0 : i64, 1 : i64]] output_shape [16, 64] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : tensor<1024xf32> into tensor<16x64xf32>
    %1790 = tensor.empty() : tensor<16x32xf32>
    %1791 = arith.constant {prov.module = "a_ffn"} 0.000000e+00 : f32
    %1792 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "a_ffn"} ins(%1791 : f32) outs(%1790 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1793 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn", prov.transposed_b = "true"} ins(%1789, %1787 : tensor<16x64xf32>, tensor<64x32xf32>) outs(%1792 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1794 = tensor.collapse_shape %1793 [[0 : i64, 1 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : tensor<16x32xf32> into tensor<512xf32>
    %1795 = tensor.expand_shape %1794 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : tensor<512xf32> into tensor<1x16x32xf32>
    %1796 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00395684736 : f32
    %1797 = tensor.splat %1796 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1798 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1799 = tensor.splat %1798 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1800 = "quant_ext.quantize_per_tensor"(%1795, %1797, %1799) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_18", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1801 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00395684736 : f32
    %1802 = tensor.splat %1801 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1803 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1804 = tensor.splat %1803 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1805 = "quant_ext.dequantize_per_tensor"(%1800, %1802, %1804) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_46", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1806 = tensor.empty() : tensor<1x16x32xf32>
    %1807 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1696, %1805 : tensor<1x16x32xf32>, tensor<1x16x32xf32>) outs(%1806 : tensor<1x16x32xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb154(%1808: f32, %1809: f32, %1810: f32):
      %1811 = arith.addf %1808, %1809 : f32
      linalg.yield %1811 : f32
    } -> tensor<1x16x32xf32>
    %1812 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0190579668 : f32
    %1813 = tensor.splat %1812 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1814 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1815 = tensor.splat %1814 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1816 = "quant_ext.quantize_per_tensor"(%1807, %1813, %1815) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_19", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1817 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0190579668 : f32
    %1818 = tensor.splat %1817 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1819 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1820 = tensor.splat %1819 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1821 = "quant_ext.dequantize_per_tensor"(%1816, %1818, %1820) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_47", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1822 = tensor.empty() : tensor<32x32xf32>
    %1823 = linalg.transpose ins(%143:tensor<32x32xf32>) outs(%1822:tensor<32x32xf32>) permutation = [1, 0]
    %1824 = tensor.collapse_shape %1821 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : tensor<1x16x32xf32> into tensor<512xf32>
    %1825 = tensor.expand_shape %1824 [[0 : i64, 1 : i64]] output_shape [16, 32] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : tensor<512xf32> into tensor<16x32xf32>
    %1826 = tensor.empty() : tensor<16x32xf32>
    %1827 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1828 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1827 : f32) outs(%1826 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1829 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head", prov.transposed_b = "true"} ins(%1825, %1823 : tensor<16x32xf32>, tensor<32x32xf32>) outs(%1828 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %1830 = tensor.collapse_shape %1829 [[0 : i64, 1 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : tensor<16x32xf32> into tensor<512xf32>
    %1831 = tensor.expand_shape %1830 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 32] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : tensor<512xf32> into tensor<1x16x32xf32>
    func.return %1831 : tensor<1x16x32xf32>
  }
}
