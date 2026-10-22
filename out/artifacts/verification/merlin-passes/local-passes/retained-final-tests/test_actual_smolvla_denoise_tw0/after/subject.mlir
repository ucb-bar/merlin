"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<256x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32x3x16x16xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<64x32xi8>, tensor<64x32xi8>, tensor<32x64xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<32x32xi8>, tensor<64x32xi8>, tensor<32x64xi8>, tensor<32x32xi8>, tensor<1x3x32x32xf32>, tensor<1xi1>, tensor<1x11xi64>, tensor<1x11xi1>, tensor<1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>}> ({
  ^bb0(%0: tensor<256x32xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32xf32>, %5: tensor<32xf32>, %6: tensor<32x3x16x16xi8>, %7: tensor<32x32xi8>, %8: tensor<32x32xi8>, %9: tensor<32x32xi8>, %10: tensor<32x32xi8>, %11: tensor<32x32xi8>, %12: tensor<64x32xi8>, %13: tensor<64x32xi8>, %14: tensor<32x64xi8>, %15: tensor<32x32xi8>, %16: tensor<32x32xi8>, %17: tensor<32x32xi8>, %18: tensor<32x32xi8>, %19: tensor<32x32xi8>, %20: tensor<32x32xi8>, %21: tensor<32x32xi8>, %22: tensor<32x32xi8>, %23: tensor<32x32xi8>, %24: tensor<32x32xi8>, %25: tensor<64x32xi8>, %26: tensor<32x64xi8>, %27: tensor<32x32xi8>, %28: tensor<1x3x32x32xf32>, %29: tensor<1xi1>, %30: tensor<1x11xi64>, %31: tensor<1x11xi1>, %32: tensor<1x32xf32>, %33: tensor<1x16x32xf32>):
    %34 = "arith.constant"() <{value = 0.000284120732 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %35 = "tensor.splat"(%34) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %36 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %37 = "tensor.splat"(%36) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %38 = "quant_ext.dequantize_per_tensor"(%6, %35, %37) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x3x16x16xi8>, tensor<f32>, tensor<i64>) -> tensor<32x3x16x16xf32>
    %39 = "arith.constant"() <{value = 0.00139179907 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %40 = "tensor.splat"(%39) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %41 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %42 = "tensor.splat"(%41) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %43 = "quant_ext.dequantize_per_tensor"(%7, %40, %42) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %44 = "arith.constant"() <{value = 0.00138446689 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %45 = "tensor.splat"(%44) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %46 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %47 = "tensor.splat"(%46) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %48 = "quant_ext.dequantize_per_tensor"(%8, %45, %47) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %49 = "arith.constant"() <{value = 0.00138657703 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %50 = "tensor.splat"(%49) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %51 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %52 = "tensor.splat"(%51) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %53 = "quant_ext.dequantize_per_tensor"(%9, %50, %52) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_3", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %54 = "arith.constant"() <{value = 0.00139146182 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %55 = "tensor.splat"(%54) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %56 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %57 = "tensor.splat"(%56) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %58 = "quant_ext.dequantize_per_tensor"(%10, %55, %57) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_4", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %59 = "arith.constant"() <{value = 0.00139190163 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %60 = "tensor.splat"(%59) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %61 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %62 = "tensor.splat"(%61) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %63 = "quant_ext.dequantize_per_tensor"(%11, %60, %62) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_5", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %64 = "arith.constant"() <{value = 0.00139121036 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %65 = "tensor.splat"(%64) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %66 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %67 = "tensor.splat"(%66) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %68 = "quant_ext.dequantize_per_tensor"(%12, %65, %67) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_6", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %69 = "arith.constant"() <{value = 0.00139179872 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %70 = "tensor.splat"(%69) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %71 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %72 = "tensor.splat"(%71) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %73 = "quant_ext.dequantize_per_tensor"(%13, %70, %72) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_7", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %74 = "arith.constant"() <{value = 0.000984238577 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %75 = "tensor.splat"(%74) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %76 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %77 = "tensor.splat"(%76) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %78 = "quant_ext.dequantize_per_tensor"(%14, %75, %77) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_8", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x64xi8>, tensor<f32>, tensor<i64>) -> tensor<32x64xf32>
    %79 = "arith.constant"() <{value = 0.00138849823 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %80 = "tensor.splat"(%79) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %81 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %82 = "tensor.splat"(%81) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %83 = "quant_ext.dequantize_per_tensor"(%15, %80, %82) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_9", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %84 = "arith.constant"() <{value = 0.00139160384 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %85 = "tensor.splat"(%84) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %86 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %87 = "tensor.splat"(%86) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %88 = "quant_ext.dequantize_per_tensor"(%16, %85, %87) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_10", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %89 = "arith.constant"() <{value = 0.00139066449 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %90 = "tensor.splat"(%89) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %91 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %92 = "tensor.splat"(%91) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %93 = "quant_ext.dequantize_per_tensor"(%17, %90, %92) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_11", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %94 = "arith.constant"() <{value = 0.00138944248 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %95 = "tensor.splat"(%94) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %96 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %97 = "tensor.splat"(%96) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %98 = "quant_ext.dequantize_per_tensor"(%18, %95, %97) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_12", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %99 = "arith.constant"() <{value = 0.0013912085 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %100 = "tensor.splat"(%99) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %101 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %102 = "tensor.splat"(%101) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %103 = "quant_ext.dequantize_per_tensor"(%19, %100, %102) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_13", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %104 = "arith.constant"() <{value = 0.00138616748 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %105 = "tensor.splat"(%104) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %106 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %107 = "tensor.splat"(%106) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %108 = "quant_ext.dequantize_per_tensor"(%20, %105, %107) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_14", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %109 = "arith.constant"() <{value = 0.00139072898 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %110 = "tensor.splat"(%109) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %111 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %112 = "tensor.splat"(%111) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %113 = "quant_ext.dequantize_per_tensor"(%21, %110, %112) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_15", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %114 = "arith.constant"() <{value = 0.0013903965 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %115 = "tensor.splat"(%114) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %116 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %117 = "tensor.splat"(%116) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %118 = "quant_ext.dequantize_per_tensor"(%22, %115, %117) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_16", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %119 = "arith.constant"() <{value = 0.00138967822 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %120 = "tensor.splat"(%119) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %121 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %122 = "tensor.splat"(%121) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %123 = "quant_ext.dequantize_per_tensor"(%23, %120, %122) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_17", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %124 = "arith.constant"() <{value = 0.0013885845 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %125 = "tensor.splat"(%124) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %126 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %127 = "tensor.splat"(%126) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %128 = "quant_ext.dequantize_per_tensor"(%24, %125, %127) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_18", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %129 = "arith.constant"() <{value = 0.00139177206 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %130 = "tensor.splat"(%129) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %131 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %132 = "tensor.splat"(%131) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %133 = "quant_ext.dequantize_per_tensor"(%25, %130, %132) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_19", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %134 = "arith.constant"() <{value = 9.841940e-04 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %135 = "tensor.splat"(%134) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %136 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %137 = "tensor.splat"(%136) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %138 = "quant_ext.dequantize_per_tensor"(%26, %135, %137) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_20", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x64xi8>, tensor<f32>, tensor<i64>) -> tensor<32x64xf32>
    %139 = "arith.constant"() <{value = 0.00138796284 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %140 = "tensor.splat"(%139) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %141 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %142 = "tensor.splat"(%141) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %143 = "quant_ext.dequantize_per_tensor"(%27, %140, %142) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_21", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %144 = "arith.constant"() <{value = 0.0143915368 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %145 = "tensor.splat"(%144) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %146 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %147 = "tensor.splat"(%146) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %148 = "quant_ext.quantize_per_tensor"(%28, %145, %147) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x3x32x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x3x32x32xi8>
    %149 = "arith.constant"() <{value = 0.0143915368 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %150 = "tensor.splat"(%149) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %151 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %152 = "tensor.splat"(%151) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %153 = "quant_ext.dequantize_per_tensor"(%148, %150, %152) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_22", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x3x32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x3x32x32xf32>
    %154 = "arith.constant"() <{value = 0.0103853745 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %155 = "tensor.splat"(%154) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %156 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %157 = "tensor.splat"(%156) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %158 = "quant_ext.quantize_per_tensor"(%32, %155, %157) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32xi8>
    %159 = "arith.constant"() <{value = 0.0103853745 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %160 = "tensor.splat"(%159) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %161 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %162 = "tensor.splat"(%161) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %163 = "quant_ext.dequantize_per_tensor"(%158, %160, %162) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_23", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32xf32>
    %164 = "arith.constant"() <{value = 0.0307341404 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %165 = "tensor.splat"(%164) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %166 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %167 = "tensor.splat"(%166) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %168 = "quant_ext.quantize_per_tensor"(%33, %165, %167) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_2", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %169 = "arith.constant"() <{value = 0.0307341404 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %170 = "tensor.splat"(%169) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %171 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %172 = "tensor.splat"(%171) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %173 = "quant_ext.dequantize_per_tensor"(%168, %170, %172) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_24", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %174 = "tensor.empty"() : () -> tensor<3x16x16x1x2x2xf32>
    %175 = "linalg.generic"(%153, %174) <{indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 16) + d1), ((d5 * 16) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%176: f32, %177: f32):
      "linalg.yield"(%176) : (f32) -> ()
    }) {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<1x3x32x32xf32>, tensor<3x16x16x1x2x2xf32>) -> tensor<3x16x16x1x2x2xf32>
    %178 = "tensor.collapse_shape"(%175) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]]}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<3x16x16x1x2x2xf32>) -> tensor<3072xf32>
    %179 = "tensor.expand_shape"(%178) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 768, 4>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<3072xf32>) -> tensor<768x4xf32>
    %180 = "tensor.collapse_shape"(%38) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<32x3x16x16xf32>) -> tensor<24576xf32>
    %181 = "tensor.expand_shape"(%180) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 32, 768>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<24576xf32>) -> tensor<32x768xf32>
    %182 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : () -> f32
    %183 = "tensor.splat"(%182) {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (f32) -> tensor<32x4xf32>
    %184 = "linalg.matmul"(%181, %179, %183) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%185: f32, %186: f32, %187: f32):
      %188 = "arith.mulf"(%185, %186) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %189 = "arith.addf"(%188, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%189) : (f32) -> ()
    }) {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<32x768xf32>, tensor<768x4xf32>, tensor<32x4xf32>) -> tensor<32x4xf32>
    %190 = "tensor.collapse_shape"(%184) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<32x4xf32>) -> tensor<128xf32>
    %191 = "tensor.expand_shape"(%190) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 32, 1, 2, 2>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<128xf32>) -> tensor<32x1x2x2xf32>
    %192 = "tensor.collapse_shape"(%191) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<32x1x2x2xf32>) -> tensor<128xf32>
    %193 = "tensor.expand_shape"(%192) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 32, 2, 2>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch", prov.fqn = "patch"} : (tensor<128xf32>) -> tensor<1x32x2x2xf32>
    %194 = "tensor.collapse_shape"(%193) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : (tensor<1x32x2x2xf32>) -> tensor<128xf32>
    %195 = "tensor.expand_shape"(%194) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 32, 4>}> {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : (tensor<128xf32>) -> tensor<1x32x4xf32>
    %196 = "tensor.empty"() : () -> tensor<1x4x32xf32>
    %197 = "linalg.transpose"(%195, %196) <{permutation = array<i64: 0, 2, 1>}> ({
    ^bb0(%198: f32, %199: f32):
      "linalg.yield"(%198) : (f32) -> ()
    }) : (tensor<1x32x4xf32>, tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %200 = "tensor.empty"() : () -> tensor<1xf32>
    %201 = "linalg.generic"(%29, %200) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%202: i1, %203: f32):
      %204 = "arith.uitofp"(%202) {prov.transforms = "bool_sitofp_uitofp"} : (i1) -> f32
      "linalg.yield"(%204) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} : (tensor<1xi1>, tensor<1xf32>) -> tensor<1xf32>
    %205 = "tensor.expand_shape"(%201) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1>}> {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1xf32>) -> tensor<1x1xf32>
    %206 = "tensor.collapse_shape"(%205) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x1xf32>) -> tensor<1xf32>
    %207 = "tensor.expand_shape"(%206) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 1, 1>}> {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %208 = "tensor.empty"() : () -> tensor<1x4x32xf32>
    %209 = "linalg.generic"(%197, %207, %208) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%210: f32, %211: f32, %212: f32):
      %213 = "arith.mulf"(%210, %211) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%213) : (f32) -> ()
    }) {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x32xf32>, tensor<1x1x1xf32>, tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %214 = "tensor.empty"() : () -> tensor<1x11x32xf32>
    %215 = "linalg.generic"(%30, %214) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%216: i64, %217: f32):
      %218 = "arith.index_cast"(%216) : (i64) -> index
      %219 = "linalg.index"() <{dim = 2 : i64}> : () -> index
      %220 = "tensor.extract"(%0, %218, %219) : (tensor<256x32xf32>, index, index) -> f32
      "linalg.yield"(%220) : (f32) -> ()
    }) {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "tok", prov.fqn = "tok"} : (tensor<1x11xi64>, tensor<1x11x32xf32>) -> tensor<1x11x32xf32>
    %221 = "tensor.empty"() : () -> tensor<32x32xf32>
    %222 = "linalg.transpose"(%43, %221) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%223: f32, %224: f32):
      "linalg.yield"(%223) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %225 = "tensor.empty"() : () -> tensor<1x32xf32>
    %226 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "state_in"} : () -> f32
    %227 = "linalg.fill"(%226, %225) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%228: f32, %229: f32):
      "linalg.yield"(%228) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "state_in"} : (f32, tensor<1x32xf32>) -> tensor<1x32xf32>
    %230 = "linalg.matmul"(%163, %222, %227) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%231: f32, %232: f32, %233: f32):
      %234 = "arith.mulf"(%231, %232) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %235 = "arith.addf"(%234, %233) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%235) : (f32) -> ()
    }) {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "state_in", prov.fqn = "state_in", prov.transposed_b = "true"} : (tensor<1x32xf32>, tensor<32x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
    %236 = "tensor.collapse_shape"(%230) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x32xf32>) -> tensor<32xf32>
    %237 = "tensor.expand_shape"(%236) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 1, 32>}> {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %238 = "tensor.concat"(%209, %215, %237) <{dim = 1 : i64}> {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x32xf32>, tensor<1x11x32xf32>, tensor<1x1x32xf32>) -> tensor<1x16x32xf32>
    %239 = "tensor.expand_shape"(%29) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1>}> {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : (tensor<1xi1>) -> tensor<1x1xi1>
    %240 = "tensor.empty"() : () -> tensor<1x4xi1>
    %241 = "linalg.generic"(%239, %240) <{indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%242: i1, %243: i1):
      "linalg.yield"(%242) : (i1) -> ()
    }) {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool"} : (tensor<1x1xi1>, tensor<1x4xi1>) -> tensor<1x4xi1>
    %244 = "arith.constant"() <{value = true}> {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bool"} : () -> i1
    %245 = "tensor.splat"(%244) {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bool"} : (i1) -> tensor<1xi1>
    %246 = "tensor.expand_shape"(%245) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1>}> {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : (tensor<1xi1>) -> tensor<1x1xi1>
    %247 = "tensor.concat"(%241, %31, %246) <{dim = 1 : i64}> {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "bool"} : (tensor<1x4xi1>, tensor<1x11xi1>, tensor<1x1xi1>) -> tensor<1x16xi1>
    %248 = "tensor.empty"() : () -> tensor<1x16xf32>
    %249 = "linalg.generic"(%247, %248) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%250: i1, %251: f32):
      %252 = "arith.uitofp"(%250) {prov.transforms = "bool_sitofp_uitofp"} : (i1) -> f32
      "linalg.yield"(%252) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} : (tensor<1x16xi1>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %253 = "arith.constant"() <{value = 1.000000e+00 : f32}> {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %254 = "tensor.splat"(%253) {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} : (f32) -> tensor<1x16xf32>
    %255 = "tensor.empty"() : () -> tensor<1x16xf32>
    %256 = "linalg.generic"(%254, %249, %255) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%257: f32, %258: f32, %259: f32):
      %260 = "arith.subf"(%257, %258) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%260) : (f32) -> ()
    }) {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %261 = "arith.constant"() <{value = -1.000000e+04 : f32}> {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %262 = "tensor.splat"(%261) {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (f32) -> tensor<1x16xf32>
    %263 = "tensor.empty"() : () -> tensor<1x16xf32>
    %264 = "linalg.generic"(%256, %262, %263) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%265: f32, %266: f32, %267: f32):
      %268 = "arith.mulf"(%265, %266) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%268) : (f32) -> ()
    }) {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %269 = "tensor.collapse_shape"(%264) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %270 = "tensor.expand_shape"(%269) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 1, 16>}> {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<16xf32>) -> tensor<1x1x16xf32>
    %271 = "tensor.collapse_shape"(%270) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x1x16xf32>) -> tensor<16xf32>
    %272 = "tensor.expand_shape"(%271) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 1, 16>}> {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %273 = "tensor.empty"() : () -> tensor<16xi64>
    %274 = "linalg.generic"(%273) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%275: i64):
      %276 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %277 = "arith.index_cast"(%276) : (index) -> i64
      %278 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %279 = "arith.muli"(%277, %278) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %280 = "arith.constant"() <{value = 0 : i64}> : () -> i64
      %281 = "arith.addi"(%280, %279) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%281) : (i64) -> ()
    }) {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<16xi64>
    %282 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %283 = "linalg.generic"(%238, %282) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%284: f32, %285: f32):
      %286 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %287 = "math.powf"(%284, %286) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%287) : (f32) -> ()
    }) {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %288 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : () -> f32
    %289 = "tensor.splat"(%288) {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (f32) -> tensor<1x16xf32>
    %290 = "linalg.reduce"(%283, %289) <{dimensions = array<i64: 2>}> ({
    ^bb0(%291: f32, %292: f32):
      %293 = "arith.addf"(%291, %292) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%293) : (f32) -> ()
    }) : (tensor<1x16x32xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %294 = "arith.constant"() <{value = 3.200000e+01 : f32}> {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : () -> f32
    %295 = "tensor.splat"(%294) {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (f32) -> tensor<1x16xf32>
    %296 = "tensor.empty"() : () -> tensor<1x16xf32>
    %297 = "linalg.generic"(%290, %295, %296) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%298: f32, %299: f32, %300: f32):
      %301 = "arith.divf"(%298, %299) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%301) : (f32) -> ()
    }) {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %302 = "tensor.collapse_shape"(%297) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %303 = "tensor.expand_shape"(%302) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 1>}> {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<16xf32>) -> tensor<1x16x1xf32>
    %304 = "arith.constant"() <{value = 1.000000e-05 : f32}> {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : () -> f32
    %305 = "tensor.splat"(%304) {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (f32) -> tensor<1x16x1xf32>
    %306 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %307 = "linalg.generic"(%303, %305, %306) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%308: f32, %309: f32, %310: f32):
      %311 = "arith.addf"(%308, %309) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%311) : (f32) -> ()
    }) {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %312 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %313 = "linalg.generic"(%307, %312) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%314: f32, %315: f32):
      %316 = "math.rsqrt"(%314) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%316) : (f32) -> ()
    }) {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %317 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %318 = "linalg.generic"(%238, %313, %317) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%319: f32, %320: f32, %321: f32):
      %322 = "arith.mulf"(%319, %320) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%322) : (f32) -> ()
    }) {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16x32xf32>, tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %323 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %324 = "linalg.generic"(%318, %1, %323) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%325: f32, %326: f32, %327: f32):
      %328 = "arith.mulf"(%325, %326) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%328) : (f32) -> ()
    }) {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre1", prov.fqn = "n_pre1"} : (tensor<1x16x32xf32>, tensor<32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %329 = "arith.constant"() <{value = 0.0280993301 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %330 = "tensor.splat"(%329) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %331 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %332 = "tensor.splat"(%331) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %333 = "quant_ext.quantize_per_tensor"(%324, %330, %332) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_3", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %334 = "arith.constant"() <{value = 0.0280993301 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %335 = "tensor.splat"(%334) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %336 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %337 = "tensor.splat"(%336) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %338 = "quant_ext.dequantize_per_tensor"(%333, %335, %337) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_25", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %339 = "arith.constant"() <{value = 0.0280993301 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %340 = "tensor.splat"(%339) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %341 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %342 = "tensor.splat"(%341) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %343 = "quant_ext.dequantize_per_tensor"(%333, %340, %342) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_26", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %344 = "arith.constant"() <{value = 0.0280993301 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %345 = "tensor.splat"(%344) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %346 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %347 = "tensor.splat"(%346) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %348 = "quant_ext.dequantize_per_tensor"(%333, %345, %347) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_27", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %349 = "tensor.empty"() : () -> tensor<32x32xf32>
    %350 = "linalg.transpose"(%48, %349) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%351: f32, %352: f32):
      "linalg.yield"(%351) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %353 = "tensor.collapse_shape"(%348) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %354 = "tensor.expand_shape"(%353) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %355 = "tensor.empty"() : () -> tensor<16x32xf32>
    %356 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_attn"} : () -> f32
    %357 = "linalg.fill"(%356, %355) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%358: f32, %359: f32):
      "linalg.yield"(%358) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %360 = "linalg.matmul"(%354, %350, %357) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%361: f32, %362: f32, %363: f32):
      %364 = "arith.mulf"(%361, %362) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %365 = "arith.addf"(%364, %363) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%365) : (f32) -> ()
    }) {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %366 = "tensor.collapse_shape"(%360) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %367 = "tensor.expand_shape"(%366) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.q"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %368 = "tensor.collapse_shape"(%367) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %369 = "tensor.expand_shape"(%368) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %370 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %371 = "linalg.transpose"(%369, %370) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%372: f32, %373: f32):
      "linalg.yield"(%372) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %374 = "tensor.empty"() : () -> tensor<32x32xf32>
    %375 = "linalg.transpose"(%53, %374) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%376: f32, %377: f32):
      "linalg.yield"(%376) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %378 = "tensor.collapse_shape"(%343) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %379 = "tensor.expand_shape"(%378) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %380 = "tensor.empty"() : () -> tensor<16x32xf32>
    %381 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_attn"} : () -> f32
    %382 = "linalg.fill"(%381, %380) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%383: f32, %384: f32):
      "linalg.yield"(%383) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %385 = "linalg.matmul"(%379, %375, %382) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%386: f32, %387: f32, %388: f32):
      %389 = "arith.mulf"(%386, %387) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %390 = "arith.addf"(%389, %388) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%390) : (f32) -> ()
    }) {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %391 = "tensor.collapse_shape"(%385) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %392 = "tensor.expand_shape"(%391) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.k"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %393 = "tensor.collapse_shape"(%392) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %394 = "tensor.expand_shape"(%393) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %395 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %396 = "linalg.transpose"(%394, %395) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%397: f32, %398: f32):
      "linalg.yield"(%397) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %399 = "tensor.empty"() : () -> tensor<32x32xf32>
    %400 = "linalg.transpose"(%58, %399) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%401: f32, %402: f32):
      "linalg.yield"(%401) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %403 = "tensor.collapse_shape"(%338) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %404 = "tensor.expand_shape"(%403) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %405 = "tensor.empty"() : () -> tensor<16x32xf32>
    %406 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_attn"} : () -> f32
    %407 = "linalg.fill"(%406, %405) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%408: f32, %409: f32):
      "linalg.yield"(%408) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %410 = "linalg.matmul"(%404, %400, %407) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%411: f32, %412: f32, %413: f32):
      %414 = "arith.mulf"(%411, %412) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %415 = "arith.addf"(%414, %413) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%415) : (f32) -> ()
    }) {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %416 = "tensor.collapse_shape"(%410) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %417 = "tensor.expand_shape"(%416) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.v"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %418 = "tensor.collapse_shape"(%417) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %419 = "tensor.expand_shape"(%418) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %420 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %421 = "linalg.transpose"(%419, %420) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%422: f32, %423: f32):
      "linalg.yield"(%422) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %424 = "tensor.empty"() : () -> tensor<8xf32>
    %425 = "linalg.generic"(%424) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%426: f32):
      %427 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %428 = "arith.index_cast"(%427) : (index) -> i64
      %429 = "arith.sitofp"(%428) : (i64) -> f32
      %430 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %431 = "arith.mulf"(%429, %430) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %432 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %433 = "arith.addf"(%432, %431) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%433) : (f32) -> ()
    }) {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>) -> tensor<8xf32>
    %434 = "arith.constant"() <{value = 8.000000e+00 : f32}> {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %435 = "tensor.splat"(%434) {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<8xf32>
    %436 = "tensor.empty"() : () -> tensor<8xf32>
    %437 = "linalg.generic"(%425, %435, %436) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%438: f32, %439: f32, %440: f32):
      %441 = "arith.divf"(%438, %439) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%441) : (f32) -> ()
    }) {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %442 = "tensor.empty"() : () -> tensor<8xf32>
    %443 = "linalg.generic"(%437, %442) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%444: f32, %445: f32):
      %446 = "arith.constant"() <{value = 1.000000e+04 : f32}> : () -> f32
      %447 = "math.powf"(%446, %444) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%447) : (f32) -> ()
    }) {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %448 = "tensor.empty"() : () -> tensor<8xf32>
    %449 = "linalg.generic"(%443, %448) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%450: f32, %451: f32):
      %452 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %453 = "arith.divf"(%452, %450) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%453) : (f32) -> ()
    }) {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %454 = "arith.constant"() <{value = 1.000000e+00 : f32}> {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %455 = "tensor.splat"(%454) {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<8xf32>
    %456 = "tensor.empty"() : () -> tensor<8xf32>
    %457 = "linalg.generic"(%449, %455, %456) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%458: f32, %459: f32, %460: f32):
      %461 = "arith.mulf"(%458, %459) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%461) : (f32) -> ()
    }) {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %462 = "tensor.expand_shape"(%274) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 1>}> {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16xi64>) -> tensor<16x1xi64>
    %463 = "tensor.empty"() : () -> tensor<16x1xf32>
    %464 = "linalg.generic"(%462, %463) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%465: i64, %466: f32):
      %467 = "arith.sitofp"(%465) : (i64) -> f32
      "linalg.yield"(%467) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x1xi64>, tensor<16x1xf32>) -> tensor<16x1xf32>
    %468 = "tensor.expand_shape"(%457) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 8>}> {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>) -> tensor<1x8xf32>
    %469 = "tensor.empty"() : () -> tensor<16x8xf32>
    %470 = "linalg.generic"(%464, %468, %469) <{indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%471: f32, %472: f32, %473: f32):
      %474 = "arith.mulf"(%471, %472) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%474) : (f32) -> ()
    }) {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x1xf32>, tensor<1x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %475 = "tensor.empty"() : () -> tensor<16x8xf32>
    %476 = "linalg.generic"(%470, %475) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%477: f32, %478: f32):
      %479 = "math.cos"(%477) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%479) : (f32) -> ()
    }) {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %480 = "tensor.empty"() : () -> tensor<16x8xf32>
    %481 = "linalg.generic"(%470, %480) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%482: f32, %483: f32):
      %484 = "math.cos"(%482) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%484) : (f32) -> ()
    }) {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %485 = "tensor.concat"(%476, %481) <{dim = 1 : i64}> {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %486 = "tensor.collapse_shape"(%485) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %487 = "tensor.expand_shape"(%486) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %488 = "tensor.collapse_shape"(%487) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %489 = "tensor.expand_shape"(%488) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %490 = "tensor.empty"() : () -> tensor<16x8xf32>
    %491 = "linalg.generic"(%470, %490) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%492: f32, %493: f32):
      %494 = "math.sin"(%492) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%494) : (f32) -> ()
    }) {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %495 = "tensor.empty"() : () -> tensor<16x8xf32>
    %496 = "linalg.generic"(%470, %495) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%497: f32, %498: f32):
      %499 = "math.sin"(%497) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%499) : (f32) -> ()
    }) {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %500 = "tensor.concat"(%491, %496) <{dim = 1 : i64}> {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %501 = "tensor.collapse_shape"(%500) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %502 = "tensor.expand_shape"(%501) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %503 = "tensor.collapse_shape"(%502) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %504 = "tensor.expand_shape"(%503) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %505 = "tensor.extract_slice"(%371) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %506 = "tensor.extract_slice"(%371) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %507 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %508 = "linalg.generic"(%371, %489, %507) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%509: f32, %510: f32, %511: f32):
      %512 = "arith.mulf"(%509, %510) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%512) : (f32) -> ()
    }) {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %513 = "tensor.empty"() : () -> tensor<1x2x16x8xf32>
    %514 = "linalg.generic"(%506, %513) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%515: f32, %516: f32):
      %517 = "arith.negf"(%515) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%517) : (f32) -> ()
    }) {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32>
    %518 = "tensor.concat"(%514, %505) <{dim = 3 : i64}> {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %519 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %520 = "linalg.generic"(%518, %504, %519) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%521: f32, %522: f32, %523: f32):
      %524 = "arith.mulf"(%521, %522) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%524) : (f32) -> ()
    }) {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %525 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %526 = "linalg.generic"(%508, %520, %525) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%527: f32, %528: f32, %529: f32):
      %530 = "arith.addf"(%527, %528) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%530) : (f32) -> ()
    }) {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %531 = "tensor.empty"() : () -> tensor<8xf32>
    %532 = "linalg.generic"(%531) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%533: f32):
      %534 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %535 = "arith.index_cast"(%534) : (index) -> i64
      %536 = "arith.sitofp"(%535) : (i64) -> f32
      %537 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %538 = "arith.mulf"(%536, %537) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %539 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %540 = "arith.addf"(%539, %538) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%540) : (f32) -> ()
    }) {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>) -> tensor<8xf32>
    %541 = "arith.constant"() <{value = 8.000000e+00 : f32}> {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %542 = "tensor.splat"(%541) {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<8xf32>
    %543 = "tensor.empty"() : () -> tensor<8xf32>
    %544 = "linalg.generic"(%532, %542, %543) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%545: f32, %546: f32, %547: f32):
      %548 = "arith.divf"(%545, %546) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%548) : (f32) -> ()
    }) {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %549 = "tensor.empty"() : () -> tensor<8xf32>
    %550 = "linalg.generic"(%544, %549) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%551: f32, %552: f32):
      %553 = "arith.constant"() <{value = 1.000000e+04 : f32}> : () -> f32
      %554 = "math.powf"(%553, %551) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%554) : (f32) -> ()
    }) {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %555 = "tensor.empty"() : () -> tensor<8xf32>
    %556 = "linalg.generic"(%550, %555) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%557: f32, %558: f32):
      %559 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %560 = "arith.divf"(%559, %557) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%560) : (f32) -> ()
    }) {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %561 = "arith.constant"() <{value = 1.000000e+00 : f32}> {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %562 = "tensor.splat"(%561) {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<8xf32>
    %563 = "tensor.empty"() : () -> tensor<8xf32>
    %564 = "linalg.generic"(%556, %562, %563) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%565: f32, %566: f32, %567: f32):
      %568 = "arith.mulf"(%565, %566) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%568) : (f32) -> ()
    }) {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %569 = "tensor.expand_shape"(%274) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 1>}> {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16xi64>) -> tensor<16x1xi64>
    %570 = "tensor.empty"() : () -> tensor<16x1xf32>
    %571 = "linalg.generic"(%569, %570) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%572: i64, %573: f32):
      %574 = "arith.sitofp"(%572) : (i64) -> f32
      "linalg.yield"(%574) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x1xi64>, tensor<16x1xf32>) -> tensor<16x1xf32>
    %575 = "tensor.expand_shape"(%564) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 8>}> {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<8xf32>) -> tensor<1x8xf32>
    %576 = "tensor.empty"() : () -> tensor<16x8xf32>
    %577 = "linalg.generic"(%571, %575, %576) <{indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%578: f32, %579: f32, %580: f32):
      %581 = "arith.mulf"(%578, %579) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%581) : (f32) -> ()
    }) {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x1xf32>, tensor<1x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %582 = "tensor.empty"() : () -> tensor<16x8xf32>
    %583 = "linalg.generic"(%577, %582) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%584: f32, %585: f32):
      %586 = "math.cos"(%584) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%586) : (f32) -> ()
    }) {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %587 = "tensor.empty"() : () -> tensor<16x8xf32>
    %588 = "linalg.generic"(%577, %587) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%589: f32, %590: f32):
      %591 = "math.cos"(%589) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%591) : (f32) -> ()
    }) {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %592 = "tensor.concat"(%583, %588) <{dim = 1 : i64}> {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %593 = "tensor.collapse_shape"(%592) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %594 = "tensor.expand_shape"(%593) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %595 = "tensor.collapse_shape"(%594) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %596 = "tensor.expand_shape"(%595) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %597 = "tensor.empty"() : () -> tensor<16x8xf32>
    %598 = "linalg.generic"(%577, %597) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%599: f32, %600: f32):
      %601 = "math.sin"(%599) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%601) : (f32) -> ()
    }) {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %602 = "tensor.empty"() : () -> tensor<16x8xf32>
    %603 = "linalg.generic"(%577, %602) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%604: f32, %605: f32):
      %606 = "math.sin"(%604) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%606) : (f32) -> ()
    }) {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %607 = "tensor.concat"(%598, %603) <{dim = 1 : i64}> {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %608 = "tensor.collapse_shape"(%607) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %609 = "tensor.expand_shape"(%608) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %610 = "tensor.collapse_shape"(%609) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %611 = "tensor.expand_shape"(%610) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %612 = "tensor.extract_slice"(%396) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %613 = "tensor.extract_slice"(%396) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %614 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %615 = "linalg.generic"(%396, %596, %614) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%616: f32, %617: f32, %618: f32):
      %619 = "arith.mulf"(%616, %617) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%619) : (f32) -> ()
    }) {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %620 = "tensor.empty"() : () -> tensor<1x2x16x8xf32>
    %621 = "linalg.generic"(%613, %620) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%622: f32, %623: f32):
      %624 = "arith.negf"(%622) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%624) : (f32) -> ()
    }) {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32>
    %625 = "tensor.concat"(%621, %612) <{dim = 3 : i64}> {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %626 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %627 = "linalg.generic"(%625, %611, %626) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%628: f32, %629: f32, %630: f32):
      %631 = "arith.mulf"(%628, %629) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%631) : (f32) -> ()
    }) {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %632 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %633 = "linalg.generic"(%615, %627, %632) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%634: f32, %635: f32, %636: f32):
      %637 = "arith.addf"(%634, %635) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%637) : (f32) -> ()
    }) {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %638 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %639 = "linalg.transpose"(%633, %638) <{permutation = array<i64: 0, 1, 3, 2>}> ({
    ^bb0(%640: f32, %641: f32):
      "linalg.yield"(%640) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %642 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %643 = "linalg.generic"(%526, %642) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%644: f32, %645: f32):
      "linalg.yield"(%644) : (f32) -> ()
    }) {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %646 = "tensor.collapse_shape"(%643) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %647 = "tensor.expand_shape"(%646) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %648 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %649 = "linalg.generic"(%639, %648) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%650: f32, %651: f32):
      "linalg.yield"(%650) : (f32) -> ()
    }) {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %652 = "tensor.collapse_shape"(%649) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %653 = "tensor.expand_shape"(%652) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %654 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %655 = "tensor.splat"(%654) {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<2x16x16xf32>
    %656 = "linalg.generic"(%647, %653, %655) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%657: f32, %658: f32, %659: f32):
      %660 = "arith.mulf"(%657, %658) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %661 = "arith.addf"(%659, %660) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%661) : (f32) -> ()
    }) {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %662 = "tensor.collapse_shape"(%656) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %663 = "tensor.expand_shape"(%662) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %664 = "arith.constant"() <{value = 2.500000e-01 : f32}> {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %665 = "tensor.splat"(%664) {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<1x2x16x16xf32>
    %666 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %667 = "linalg.generic"(%663, %665, %666) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%668: f32, %669: f32, %670: f32):
      %671 = "arith.mulf"(%668, %669) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%671) : (f32) -> ()
    }) {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %672 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %673 = "linalg.generic"(%667, %272, %672) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%674: f32, %675: f32, %676: f32):
      %677 = "arith.addf"(%674, %675) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%677) : (f32) -> ()
    }) {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x1x1x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %678 = "arith.constant"() <{value = 0xff800000 : f32}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %679 = "tensor.splat"(%678) {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<1x2x16xf32>
    %680 = "linalg.reduce"(%673, %679) <{dimensions = array<i64: 3>}> ({
    ^bb0(%681: f32, %682: f32):
      %683 = "arith.maximumf"(%681, %682) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%683) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %684 = "tensor.collapse_shape"(%680) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %685 = "tensor.expand_shape"(%684) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %686 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %687 = "linalg.generic"(%673, %685, %686) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%688: f32, %689: f32, %690: f32):
      %691 = "arith.subf"(%688, %689) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%691) : (f32) -> ()
    }) {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %692 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %693 = "linalg.generic"(%687, %692) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%694: f32, %695: f32):
      %696 = "math.exp"(%694) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%696) : (f32) -> ()
    }) {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %697 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %698 = "tensor.splat"(%697) {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<1x2x16xf32>
    %699 = "linalg.reduce"(%693, %698) <{dimensions = array<i64: 3>}> ({
    ^bb0(%700: f32, %701: f32):
      %702 = "arith.addf"(%700, %701) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%702) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %703 = "tensor.collapse_shape"(%699) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %704 = "tensor.expand_shape"(%703) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %705 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %706 = "linalg.generic"(%693, %704, %705) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%707: f32, %708: f32, %709: f32):
      %710 = "arith.divf"(%707, %708) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%710) : (f32) -> ()
    }) {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %711 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %712 = "linalg.generic"(%706, %711) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%713: f32, %714: f32):
      "linalg.yield"(%713) : (f32) -> ()
    }) {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %715 = "tensor.collapse_shape"(%712) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %716 = "tensor.expand_shape"(%715) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %717 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %718 = "linalg.generic"(%421, %717) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%719: f32, %720: f32):
      "linalg.yield"(%719) : (f32) -> ()
    }) {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %721 = "tensor.collapse_shape"(%718) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %722 = "tensor.expand_shape"(%721) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %723 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : () -> f32
    %724 = "tensor.splat"(%723) {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (f32) -> tensor<2x16x16xf32>
    %725 = "linalg.generic"(%716, %722, %724) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%726: f32, %727: f32, %728: f32):
      %729 = "arith.mulf"(%726, %727) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %730 = "arith.addf"(%728, %729) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%730) : (f32) -> ()
    }) {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %731 = "tensor.collapse_shape"(%725) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %732 = "tensor.expand_shape"(%731) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %733 = "tensor.empty"() : () -> tensor<1x16x2x16xf32>
    %734 = "linalg.transpose"(%732, %733) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%735: f32, %736: f32):
      "linalg.yield"(%735) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x16x2x16xf32>) -> tensor<1x16x2x16xf32>
    %737 = "tensor.collapse_shape"(%734) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<1x16x2x16xf32>) -> tensor<512xf32>
    %738 = "tensor.expand_shape"(%737) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %739 = "arith.constant"() <{value = 0.00384216523 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %740 = "tensor.splat"(%739) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %741 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %742 = "tensor.splat"(%741) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %743 = "quant_ext.quantize_per_tensor"(%738, %740, %742) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_4", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %744 = "arith.constant"() <{value = 0.00384216523 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %745 = "tensor.splat"(%744) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %746 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %747 = "tensor.splat"(%746) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %748 = "quant_ext.dequantize_per_tensor"(%743, %745, %747) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_28", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %749 = "tensor.empty"() : () -> tensor<32x32xf32>
    %750 = "linalg.transpose"(%63, %749) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%751: f32, %752: f32):
      "linalg.yield"(%751) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %753 = "tensor.collapse_shape"(%748) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %754 = "tensor.expand_shape"(%753) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %755 = "tensor.empty"() : () -> tensor<16x32xf32>
    %756 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_attn"} : () -> f32
    %757 = "linalg.fill"(%756, %755) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%758: f32, %759: f32):
      "linalg.yield"(%758) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_attn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %760 = "linalg.matmul"(%754, %750, %757) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%761: f32, %762: f32, %763: f32):
      %764 = "arith.mulf"(%761, %762) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %765 = "arith.addf"(%764, %763) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%765) : (f32) -> ()
    }) {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %766 = "tensor.collapse_shape"(%760) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %767 = "tensor.expand_shape"(%766) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_attn", prov.fqn = "pre_attn.o"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %768 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %769 = "linalg.generic"(%238, %767, %768) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%770: f32, %771: f32, %772: f32):
      %773 = "arith.addf"(%770, %771) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%773) : (f32) -> ()
    }) {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %774 = "arith.constant"() <{value = 0.0192604251 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %775 = "tensor.splat"(%774) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %776 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %777 = "tensor.splat"(%776) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %778 = "quant_ext.quantize_per_tensor"(%769, %775, %777) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_5", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %779 = "arith.constant"() <{value = 0.0192604251 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %780 = "tensor.splat"(%779) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %781 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %782 = "tensor.splat"(%781) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %783 = "quant_ext.dequantize_per_tensor"(%778, %780, %782) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_29", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %784 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %785 = "linalg.generic"(%769, %784) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%786: f32, %787: f32):
      %788 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %789 = "math.powf"(%786, %788) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%789) : (f32) -> ()
    }) {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %790 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : () -> f32
    %791 = "tensor.splat"(%790) {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (f32) -> tensor<1x16xf32>
    %792 = "linalg.reduce"(%785, %791) <{dimensions = array<i64: 2>}> ({
    ^bb0(%793: f32, %794: f32):
      %795 = "arith.addf"(%793, %794) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%795) : (f32) -> ()
    }) : (tensor<1x16x32xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %796 = "arith.constant"() <{value = 3.200000e+01 : f32}> {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : () -> f32
    %797 = "tensor.splat"(%796) {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (f32) -> tensor<1x16xf32>
    %798 = "tensor.empty"() : () -> tensor<1x16xf32>
    %799 = "linalg.generic"(%792, %797, %798) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%800: f32, %801: f32, %802: f32):
      %803 = "arith.divf"(%800, %801) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%803) : (f32) -> ()
    }) {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %804 = "tensor.collapse_shape"(%799) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %805 = "tensor.expand_shape"(%804) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 1>}> {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<16xf32>) -> tensor<1x16x1xf32>
    %806 = "arith.constant"() <{value = 1.000000e-05 : f32}> {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : () -> f32
    %807 = "tensor.splat"(%806) {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (f32) -> tensor<1x16x1xf32>
    %808 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %809 = "linalg.generic"(%805, %807, %808) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%810: f32, %811: f32, %812: f32):
      %813 = "arith.addf"(%810, %811) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%813) : (f32) -> ()
    }) {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %814 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %815 = "linalg.generic"(%809, %814) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%816: f32, %817: f32):
      %818 = "math.rsqrt"(%816) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%818) : (f32) -> ()
    }) {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %819 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %820 = "linalg.generic"(%769, %815, %819) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%821: f32, %822: f32, %823: f32):
      %824 = "arith.mulf"(%821, %822) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%824) : (f32) -> ()
    }) {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16x32xf32>, tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %825 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %826 = "linalg.generic"(%820, %2, %825) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%827: f32, %828: f32, %829: f32):
      %830 = "arith.mulf"(%827, %828) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%830) : (f32) -> ()
    }) {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_pre2", prov.fqn = "n_pre2"} : (tensor<1x16x32xf32>, tensor<32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %831 = "arith.constant"() <{value = 0.0270536188 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %832 = "tensor.splat"(%831) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %833 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %834 = "tensor.splat"(%833) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %835 = "quant_ext.quantize_per_tensor"(%826, %832, %834) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_6", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %836 = "arith.constant"() <{value = 0.0270536188 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %837 = "tensor.splat"(%836) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %838 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %839 = "tensor.splat"(%838) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %840 = "quant_ext.dequantize_per_tensor"(%835, %837, %839) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_30", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %841 = "arith.constant"() <{value = 0.0270536188 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %842 = "tensor.splat"(%841) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %843 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %844 = "tensor.splat"(%843) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %845 = "quant_ext.dequantize_per_tensor"(%835, %842, %844) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_31", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %846 = "tensor.empty"() : () -> tensor<32x64xf32>
    %847 = "linalg.transpose"(%68, %846) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%848: f32, %849: f32):
      "linalg.yield"(%848) : (f32) -> ()
    }) : (tensor<64x32xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %850 = "tensor.collapse_shape"(%845) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %851 = "tensor.expand_shape"(%850) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %852 = "tensor.empty"() : () -> tensor<16x64xf32>
    %853 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_ffn"} : () -> f32
    %854 = "linalg.fill"(%853, %852) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%855: f32, %856: f32):
      "linalg.yield"(%855) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} : (f32, tensor<16x64xf32>) -> tensor<16x64xf32>
    %857 = "linalg.matmul"(%851, %847, %854) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%858: f32, %859: f32, %860: f32):
      %861 = "arith.mulf"(%858, %859) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %862 = "arith.addf"(%861, %860) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%862) : (f32) -> ()
    }) {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %863 = "tensor.collapse_shape"(%857) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : (tensor<16x64xf32>) -> tensor<1024xf32>
    %864 = "tensor.expand_shape"(%863) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 64>}> {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.g"} : (tensor<1024xf32>) -> tensor<1x16x64xf32>
    %865 = "tensor.empty"() : () -> tensor<1x16x64xf32>
    %866 = "linalg.generic"(%864, %865) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%867: f32, %868: f32):
      %869 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %870 = "arith.negf"(%867) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %871 = "math.exp"(%870) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %872 = "arith.addf"(%869, %871) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %873 = "arith.divf"(%869, %872) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%873) : (f32) -> ()
    }) {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} : (tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %874 = "tensor.empty"() : () -> tensor<1x16x64xf32>
    %875 = "linalg.generic"(%864, %866, %874) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%876: f32, %877: f32, %878: f32):
      %879 = "arith.mulf"(%876, %877) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%879) : (f32) -> ()
    }) {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} : (tensor<1x16x64xf32>, tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %880 = "tensor.empty"() : () -> tensor<32x64xf32>
    %881 = "linalg.transpose"(%73, %880) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%882: f32, %883: f32):
      "linalg.yield"(%882) : (f32) -> ()
    }) : (tensor<64x32xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %884 = "tensor.collapse_shape"(%840) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %885 = "tensor.expand_shape"(%884) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %886 = "tensor.empty"() : () -> tensor<16x64xf32>
    %887 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_ffn"} : () -> f32
    %888 = "linalg.fill"(%887, %886) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%889: f32, %890: f32):
      "linalg.yield"(%889) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} : (f32, tensor<16x64xf32>) -> tensor<16x64xf32>
    %891 = "linalg.matmul"(%885, %881, %888) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%892: f32, %893: f32, %894: f32):
      %895 = "arith.mulf"(%892, %893) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %896 = "arith.addf"(%895, %894) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%896) : (f32) -> ()
    }) {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %897 = "tensor.collapse_shape"(%891) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : (tensor<16x64xf32>) -> tensor<1024xf32>
    %898 = "tensor.expand_shape"(%897) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 64>}> {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.u"} : (tensor<1024xf32>) -> tensor<1x16x64xf32>
    %899 = "tensor.empty"() : () -> tensor<1x16x64xf32>
    %900 = "linalg.generic"(%875, %898, %899) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%901: f32, %902: f32, %903: f32):
      %904 = "arith.mulf"(%901, %902) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%904) : (f32) -> ()
    }) {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn"} : (tensor<1x16x64xf32>, tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %905 = "arith.constant"() <{value = 0.013225005 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %906 = "tensor.splat"(%905) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %907 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %908 = "tensor.splat"(%907) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %909 = "quant_ext.quantize_per_tensor"(%900, %906, %908) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_7", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xi8>
    %910 = "arith.constant"() <{value = 0.013225005 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %911 = "tensor.splat"(%910) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %912 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %913 = "tensor.splat"(%912) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %914 = "quant_ext.dequantize_per_tensor"(%909, %911, %913) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_32", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xf32>
    %915 = "tensor.empty"() : () -> tensor<64x32xf32>
    %916 = "linalg.transpose"(%78, %915) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%917: f32, %918: f32):
      "linalg.yield"(%917) : (f32) -> ()
    }) : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    %919 = "tensor.collapse_shape"(%914) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : (tensor<1x16x64xf32>) -> tensor<1024xf32>
    %920 = "tensor.expand_shape"(%919) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 64>}> {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : (tensor<1024xf32>) -> tensor<16x64xf32>
    %921 = "tensor.empty"() : () -> tensor<16x32xf32>
    %922 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "pre_ffn"} : () -> f32
    %923 = "linalg.fill"(%922, %921) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%924: f32, %925: f32):
      "linalg.yield"(%924) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "pre_ffn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %926 = "linalg.matmul"(%920, %916, %923) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%927: f32, %928: f32, %929: f32):
      %930 = "arith.mulf"(%927, %928) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %931 = "arith.addf"(%930, %929) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%931) : (f32) -> ()
    }) {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d", prov.transposed_b = "true"} : (tensor<16x64xf32>, tensor<64x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %932 = "tensor.collapse_shape"(%926) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %933 = "tensor.expand_shape"(%932) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "pre_ffn", prov.fqn = "pre_ffn.d"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %934 = "arith.constant"() <{value = 4.187350e-03 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %935 = "tensor.splat"(%934) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %936 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %937 = "tensor.splat"(%936) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %938 = "quant_ext.quantize_per_tensor"(%933, %935, %937) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_8", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %939 = "arith.constant"() <{value = 4.187350e-03 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %940 = "tensor.splat"(%939) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %941 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %942 = "tensor.splat"(%941) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %943 = "quant_ext.dequantize_per_tensor"(%938, %940, %942) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_33", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %944 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %945 = "linalg.generic"(%783, %943, %944) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%946: f32, %947: f32, %948: f32):
      %949 = "arith.addf"(%946, %947) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%949) : (f32) -> ()
    }) {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %950 = "arith.constant"() <{value = 0.0192180183 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %951 = "tensor.splat"(%950) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %952 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %953 = "tensor.splat"(%952) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %954 = "quant_ext.quantize_per_tensor"(%945, %951, %953) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_9", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %955 = "arith.constant"() <{value = 0.0192180183 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %956 = "tensor.splat"(%955) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %957 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %958 = "tensor.splat"(%957) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %959 = "quant_ext.dequantize_per_tensor"(%954, %956, %958) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_34", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %960 = "arith.constant"() <{value = 0.0192180183 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %961 = "tensor.splat"(%960) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %962 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %963 = "tensor.splat"(%962) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %964 = "quant_ext.dequantize_per_tensor"(%954, %961, %963) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_35", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %965 = "tensor.empty"() : () -> tensor<32x32xf32>
    %966 = "linalg.transpose"(%83, %965) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%967: f32, %968: f32):
      "linalg.yield"(%967) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %969 = "tensor.collapse_shape"(%173) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %970 = "tensor.expand_shape"(%969) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %971 = "tensor.empty"() : () -> tensor<16x32xf32>
    %972 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "act_in"} : () -> f32
    %973 = "linalg.fill"(%972, %971) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%974: f32, %975: f32):
      "linalg.yield"(%974) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "act_in"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %976 = "linalg.matmul"(%970, %966, %973) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%977: f32, %978: f32, %979: f32):
      %980 = "arith.mulf"(%977, %978) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %981 = "arith.addf"(%980, %979) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%981) : (f32) -> ()
    }) {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %982 = "tensor.collapse_shape"(%976) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %983 = "tensor.expand_shape"(%982) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "act_in", prov.fqn = "act_in"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %984 = "tensor.empty"() : () -> tensor<16xf32>
    %985 = "linalg.generic"(%984) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%986: f32):
      %987 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %988 = "arith.index_cast"(%987) : (index) -> i64
      %989 = "arith.sitofp"(%988) : (i64) -> f32
      %990 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %991 = "arith.mulf"(%989, %990) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %992 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %993 = "arith.addf"(%992, %991) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%993) : (f32) -> ()
    }) {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32"} : (tensor<16xf32>) -> tensor<16xf32>
    %994 = "arith.constant"() <{value = -9.2103405 : f32}> {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %995 = "tensor.splat"(%994) {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (f32) -> tensor<16xf32>
    %996 = "tensor.empty"() : () -> tensor<16xf32>
    %997 = "linalg.generic"(%985, %995, %996) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%998: f32, %999: f32, %1000: f32):
      %1001 = "arith.mulf"(%998, %999) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1001) : (f32) -> ()
    }) {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1002 = "arith.constant"() <{value = 1.600000e+01 : f32}> {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %1003 = "tensor.splat"(%1002) {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : (f32) -> tensor<16xf32>
    %1004 = "tensor.empty"() : () -> tensor<16xf32>
    %1005 = "linalg.generic"(%997, %1003, %1004) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1006: f32, %1007: f32, %1008: f32):
      %1009 = "arith.divf"(%1006, %1007) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1009) : (f32) -> ()
    }) {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1010 = "tensor.empty"() : () -> tensor<16xf32>
    %1011 = "linalg.generic"(%1005, %1010) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1012: f32, %1013: f32):
      %1014 = "math.exp"(%1012) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1014) : (f32) -> ()
    }) {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1015 = "arith.constant"() <{value = 5.000000e-01 : f32}> {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : () -> f32
    %1016 = "tensor.splat"(%1015) {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (f32) -> tensor<16xf32>
    %1017 = "tensor.empty"() : () -> tensor<16xf32>
    %1018 = "linalg.generic"(%1011, %1016, %1017) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1019: f32, %1020: f32, %1021: f32):
      %1022 = "arith.mulf"(%1019, %1020) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1022) : (f32) -> ()
    }) {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1023 = "tensor.empty"() : () -> tensor<16xf32>
    %1024 = "linalg.generic"(%1018, %1023) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1025: f32, %1026: f32):
      %1027 = "math.sin"(%1025) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1027) : (f32) -> ()
    }) {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1028 = "tensor.empty"() : () -> tensor<16xf32>
    %1029 = "linalg.generic"(%1018, %1028) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1030: f32, %1031: f32):
      %1032 = "math.cos"(%1030) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1032) : (f32) -> ()
    }) {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1033 = "tensor.concat"(%1024, %1029) <{dim = 0 : i64}> {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<16xf32>, tensor<16xf32>) -> tensor<32xf32>
    %1034 = "tensor.expand_shape"(%1033) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 32>}> {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<32xf32>) -> tensor<1x32xf32>
    %1035 = "arith.constant"() <{value = 0.00783930812 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1036 = "tensor.splat"(%1035) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1037 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1038 = "tensor.splat"(%1037) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1039 = "quant_ext.quantize_per_tensor"(%1034, %1036, %1038) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_10", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32xi8>
    %1040 = "arith.constant"() <{value = 0.00783930812 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1041 = "tensor.splat"(%1040) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1042 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1043 = "tensor.splat"(%1042) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1044 = "quant_ext.dequantize_per_tensor"(%1039, %1041, %1043) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_36", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32xf32>
    %1045 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1046 = "linalg.transpose"(%88, %1045) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1047: f32, %1048: f32):
      "linalg.yield"(%1047) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1049 = "tensor.empty"() : () -> tensor<1x32xf32>
    %1050 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "t_in"} : () -> f32
    %1051 = "linalg.fill"(%1050, %1049) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1052: f32, %1053: f32):
      "linalg.yield"(%1052) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "t_in"} : (f32, tensor<1x32xf32>) -> tensor<1x32xf32>
    %1054 = "linalg.matmul"(%1044, %1046, %1051) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1055: f32, %1056: f32, %1057: f32):
      %1058 = "arith.mulf"(%1055, %1056) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1059 = "arith.addf"(%1058, %1057) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1059) : (f32) -> ()
    }) {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "t_in", prov.fqn = "t_in", prov.transposed_b = "true"} : (tensor<1x32xf32>, tensor<32x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
    %1060 = "tensor.collapse_shape"(%1054) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x32xf32>) -> tensor<32xf32>
    %1061 = "tensor.expand_shape"(%1060) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 1, 32>}> {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %1062 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1063 = "linalg.generic"(%983, %1061, %1062) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1064: f32, %1065: f32, %1066: f32):
      %1067 = "arith.addf"(%1064, %1065) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1067) : (f32) -> ()
    }) {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1068 = "arith.constant"() <{value = -1.000000e+04 : f32}> {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : () -> f32
    %1069 = "tensor.splat"(%1068) {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : (f32) -> tensor<16x16xf32>
    %1070 = "tensor.empty"() : () -> tensor<16xi64>
    %1071 = "linalg.generic"(%1070) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%1072: i64):
      %1073 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %1074 = "arith.index_cast"(%1073) : (index) -> i64
      %1075 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %1076 = "arith.muli"(%1074, %1075) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %1077 = "arith.constant"() <{value = 0 : i64}> : () -> i64
      %1078 = "arith.addi"(%1077, %1076) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%1078) : (i64) -> ()
    }) {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<16xi64>
    %1079 = "tensor.expand_shape"(%1071) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 16>}> {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<1x16xi64>
    %1080 = "tensor.empty"() : () -> tensor<16xi64>
    %1081 = "linalg.generic"(%1080) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%1082: i64):
      %1083 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %1084 = "arith.index_cast"(%1083) : (index) -> i64
      %1085 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %1086 = "arith.muli"(%1084, %1085) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %1087 = "arith.constant"() <{value = 0 : i64}> : () -> i64
      %1088 = "arith.addi"(%1087, %1086) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%1088) : (i64) -> ()
    }) {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<16xi64>
    %1089 = "tensor.expand_shape"(%1081) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 1>}> {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<16x1xi64>
    %1090 = "tensor.empty"() : () -> tensor<16x16xi64>
    %1091 = "linalg.generic"(%1079, %1089, %1090) <{indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1092: i64, %1093: i64, %1094: i64):
      %1095 = "arith.subi"(%1092, %1093) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%1095) : (i64) -> ()
    }) {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} : (tensor<1x16xi64>, tensor<16x1xi64>, tensor<16x16xi64>) -> tensor<16x16xi64>
    %1096 = "arith.constant"() <{value = 1 : i64}> {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} : () -> i64
    %1097 = "tensor.splat"(%1096) {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} : (i64) -> tensor<16x16xi64>
    %1098 = "tensor.empty"() : () -> tensor<16x16xi1>
    %1099 = "linalg.generic"(%1091, %1097, %1098) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1100: i64, %1101: i64, %1102: i1):
      %1103 = "arith.cmpi"(%1100, %1101) <{predicate = 5 : i64}> : (i64, i64) -> i1
      "linalg.yield"(%1103) : (i1) -> ()
    }) {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} : (tensor<16x16xi64>, tensor<16x16xi64>, tensor<16x16xi1>) -> tensor<16x16xi1>
    %1104 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1105 = "tensor.splat"(%1104) {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1106 = "tensor.empty"() : () -> tensor<16x16xf32>
    %1107 = "linalg.generic"(%1099, %1069, %1105, %1106) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%1108: i1, %1109: f32, %1110: f32, %1111: f32):
      %1112 = "arith.select"(%1108, %1109, %1110) : (i1, f32, f32) -> f32
      "linalg.yield"(%1112) : (f32) -> ()
    }) {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} : (tensor<16x16xi1>, tensor<16x16xf32>, tensor<f32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %1113 = "tensor.collapse_shape"(%1107) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %1114 = "tensor.expand_shape"(%1113) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %1115 = "tensor.collapse_shape"(%1114) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %1116 = "tensor.expand_shape"(%1115) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %1117 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1118 = "linalg.generic"(%1063, %1117) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1119: f32, %1120: f32):
      %1121 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %1122 = "math.powf"(%1119, %1121) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1122) : (f32) -> ()
    }) {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1123 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : () -> f32
    %1124 = "tensor.splat"(%1123) {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (f32) -> tensor<1x16xf32>
    %1125 = "linalg.reduce"(%1118, %1124) <{dimensions = array<i64: 2>}> ({
    ^bb0(%1126: f32, %1127: f32):
      %1128 = "arith.addf"(%1126, %1127) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1128) : (f32) -> ()
    }) : (tensor<1x16x32xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1129 = "arith.constant"() <{value = 3.200000e+01 : f32}> {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : () -> f32
    %1130 = "tensor.splat"(%1129) {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (f32) -> tensor<1x16xf32>
    %1131 = "tensor.empty"() : () -> tensor<1x16xf32>
    %1132 = "linalg.generic"(%1125, %1130, %1131) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1133: f32, %1134: f32, %1135: f32):
      %1136 = "arith.divf"(%1133, %1134) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1136) : (f32) -> ()
    }) {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1137 = "tensor.collapse_shape"(%1132) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %1138 = "tensor.expand_shape"(%1137) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 1>}> {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<16xf32>) -> tensor<1x16x1xf32>
    %1139 = "arith.constant"() <{value = 1.000000e-05 : f32}> {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : () -> f32
    %1140 = "tensor.splat"(%1139) {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (f32) -> tensor<1x16x1xf32>
    %1141 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1142 = "linalg.generic"(%1138, %1140, %1141) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1143: f32, %1144: f32, %1145: f32):
      %1146 = "arith.addf"(%1143, %1144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1146) : (f32) -> ()
    }) {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1147 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1148 = "linalg.generic"(%1142, %1147) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1149: f32, %1150: f32):
      %1151 = "math.rsqrt"(%1149) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1151) : (f32) -> ()
    }) {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1152 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1153 = "linalg.generic"(%1063, %1148, %1152) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1154: f32, %1155: f32, %1156: f32):
      %1157 = "arith.mulf"(%1154, %1155) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1157) : (f32) -> ()
    }) {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16x32xf32>, tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1158 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1159 = "linalg.generic"(%1153, %3, %1158) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1160: f32, %1161: f32, %1162: f32):
      %1163 = "arith.mulf"(%1160, %1161) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1163) : (f32) -> ()
    }) {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a1", prov.fqn = "n_a1"} : (tensor<1x16x32xf32>, tensor<32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1164 = "arith.constant"() <{value = 0.024379164 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1165 = "tensor.splat"(%1164) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1166 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1167 = "tensor.splat"(%1166) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1168 = "quant_ext.quantize_per_tensor"(%1159, %1165, %1167) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_11", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1169 = "arith.constant"() <{value = 0.024379164 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1170 = "tensor.splat"(%1169) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1171 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1172 = "tensor.splat"(%1171) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1173 = "quant_ext.dequantize_per_tensor"(%1168, %1170, %1172) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_37", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1174 = "arith.constant"() <{value = 0.024379164 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1175 = "tensor.splat"(%1174) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1176 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1177 = "tensor.splat"(%1176) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1178 = "quant_ext.dequantize_per_tensor"(%1168, %1175, %1177) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_38", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1179 = "arith.constant"() <{value = 0.024379164 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1180 = "tensor.splat"(%1179) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1181 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1182 = "tensor.splat"(%1181) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1183 = "quant_ext.dequantize_per_tensor"(%1168, %1180, %1182) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_39", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1184 = "tensor.empty"() : () -> tensor<16xi64>
    %1185 = "linalg.generic"(%1184) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%1186: i64):
      %1187 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %1188 = "arith.index_cast"(%1187) : (index) -> i64
      %1189 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %1190 = "arith.muli"(%1188, %1189) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %1191 = "arith.constant"() <{value = 0 : i64}> : () -> i64
      %1192 = "arith.addi"(%1191, %1190) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%1192) : (i64) -> ()
    }) {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} : (tensor<16xi64>) -> tensor<16xi64>
    %1193 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1194 = "linalg.transpose"(%93, %1193) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1195: f32, %1196: f32):
      "linalg.yield"(%1195) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1197 = "tensor.collapse_shape"(%1183) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1198 = "tensor.expand_shape"(%1197) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1199 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1200 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_self"} : () -> f32
    %1201 = "linalg.fill"(%1200, %1199) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1202: f32, %1203: f32):
      "linalg.yield"(%1202) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1204 = "linalg.matmul"(%1198, %1194, %1201) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1205: f32, %1206: f32, %1207: f32):
      %1208 = "arith.mulf"(%1205, %1206) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1209 = "arith.addf"(%1208, %1207) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1209) : (f32) -> ()
    }) {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1210 = "tensor.collapse_shape"(%1204) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1211 = "tensor.expand_shape"(%1210) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.q"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1212 = "tensor.collapse_shape"(%1211) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1213 = "tensor.expand_shape"(%1212) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1214 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1215 = "linalg.transpose"(%1213, %1214) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1216: f32, %1217: f32):
      "linalg.yield"(%1216) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1218 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1219 = "linalg.transpose"(%98, %1218) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1220: f32, %1221: f32):
      "linalg.yield"(%1220) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1222 = "tensor.collapse_shape"(%1178) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1223 = "tensor.expand_shape"(%1222) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1224 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1225 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_self"} : () -> f32
    %1226 = "linalg.fill"(%1225, %1224) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1227: f32, %1228: f32):
      "linalg.yield"(%1227) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1229 = "linalg.matmul"(%1223, %1219, %1226) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1230: f32, %1231: f32, %1232: f32):
      %1233 = "arith.mulf"(%1230, %1231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1234 = "arith.addf"(%1233, %1232) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1234) : (f32) -> ()
    }) {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1235 = "tensor.collapse_shape"(%1229) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1236 = "tensor.expand_shape"(%1235) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.k"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1237 = "tensor.collapse_shape"(%1236) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1238 = "tensor.expand_shape"(%1237) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1239 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1240 = "linalg.transpose"(%1238, %1239) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1241: f32, %1242: f32):
      "linalg.yield"(%1241) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1243 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1244 = "linalg.transpose"(%103, %1243) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1245: f32, %1246: f32):
      "linalg.yield"(%1245) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1247 = "tensor.collapse_shape"(%1173) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1248 = "tensor.expand_shape"(%1247) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1249 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1250 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_self"} : () -> f32
    %1251 = "linalg.fill"(%1250, %1249) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1252: f32, %1253: f32):
      "linalg.yield"(%1252) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1254 = "linalg.matmul"(%1248, %1244, %1251) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1255: f32, %1256: f32, %1257: f32):
      %1258 = "arith.mulf"(%1255, %1256) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1259 = "arith.addf"(%1258, %1257) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1259) : (f32) -> ()
    }) {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1260 = "tensor.collapse_shape"(%1254) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1261 = "tensor.expand_shape"(%1260) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.v"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1262 = "tensor.collapse_shape"(%1261) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1263 = "tensor.expand_shape"(%1262) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1264 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1265 = "linalg.transpose"(%1263, %1264) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1266: f32, %1267: f32):
      "linalg.yield"(%1266) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1268 = "tensor.empty"() : () -> tensor<8xf32>
    %1269 = "linalg.generic"(%1268) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%1270: f32):
      %1271 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %1272 = "arith.index_cast"(%1271) : (index) -> i64
      %1273 = "arith.sitofp"(%1272) : (i64) -> f32
      %1274 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %1275 = "arith.mulf"(%1273, %1274) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1276 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %1277 = "arith.addf"(%1276, %1275) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1277) : (f32) -> ()
    }) {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>) -> tensor<8xf32>
    %1278 = "arith.constant"() <{value = 8.000000e+00 : f32}> {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1279 = "tensor.splat"(%1278) {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<8xf32>
    %1280 = "tensor.empty"() : () -> tensor<8xf32>
    %1281 = "linalg.generic"(%1269, %1279, %1280) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1282: f32, %1283: f32, %1284: f32):
      %1285 = "arith.divf"(%1282, %1283) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1285) : (f32) -> ()
    }) {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1286 = "tensor.empty"() : () -> tensor<8xf32>
    %1287 = "linalg.generic"(%1281, %1286) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1288: f32, %1289: f32):
      %1290 = "arith.constant"() <{value = 1.000000e+04 : f32}> : () -> f32
      %1291 = "math.powf"(%1290, %1288) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1291) : (f32) -> ()
    }) {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1292 = "tensor.empty"() : () -> tensor<8xf32>
    %1293 = "linalg.generic"(%1287, %1292) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1294: f32, %1295: f32):
      %1296 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %1297 = "arith.divf"(%1296, %1294) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1297) : (f32) -> ()
    }) {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1298 = "arith.constant"() <{value = 1.000000e+00 : f32}> {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1299 = "tensor.splat"(%1298) {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<8xf32>
    %1300 = "tensor.empty"() : () -> tensor<8xf32>
    %1301 = "linalg.generic"(%1293, %1299, %1300) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1302: f32, %1303: f32, %1304: f32):
      %1305 = "arith.mulf"(%1302, %1303) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1305) : (f32) -> ()
    }) {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1306 = "tensor.expand_shape"(%1185) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 1>}> {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16xi64>) -> tensor<16x1xi64>
    %1307 = "tensor.empty"() : () -> tensor<16x1xf32>
    %1308 = "linalg.generic"(%1306, %1307) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1309: i64, %1310: f32):
      %1311 = "arith.sitofp"(%1309) : (i64) -> f32
      "linalg.yield"(%1311) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x1xi64>, tensor<16x1xf32>) -> tensor<16x1xf32>
    %1312 = "tensor.expand_shape"(%1301) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 8>}> {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>) -> tensor<1x8xf32>
    %1313 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1314 = "linalg.generic"(%1308, %1312, %1313) <{indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1315: f32, %1316: f32, %1317: f32):
      %1318 = "arith.mulf"(%1315, %1316) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1318) : (f32) -> ()
    }) {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x1xf32>, tensor<1x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1319 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1320 = "linalg.generic"(%1314, %1319) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1321: f32, %1322: f32):
      %1323 = "math.cos"(%1321) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1323) : (f32) -> ()
    }) {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1324 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1325 = "linalg.generic"(%1314, %1324) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1326: f32, %1327: f32):
      %1328 = "math.cos"(%1326) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1328) : (f32) -> ()
    }) {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1329 = "tensor.concat"(%1320, %1325) <{dim = 1 : i64}> {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1330 = "tensor.collapse_shape"(%1329) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %1331 = "tensor.expand_shape"(%1330) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %1332 = "tensor.collapse_shape"(%1331) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %1333 = "tensor.expand_shape"(%1332) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %1334 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1335 = "linalg.generic"(%1314, %1334) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1336: f32, %1337: f32):
      %1338 = "math.sin"(%1336) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1338) : (f32) -> ()
    }) {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1339 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1340 = "linalg.generic"(%1314, %1339) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1341: f32, %1342: f32):
      %1343 = "math.sin"(%1341) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1343) : (f32) -> ()
    }) {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1344 = "tensor.concat"(%1335, %1340) <{dim = 1 : i64}> {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1345 = "tensor.collapse_shape"(%1344) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %1346 = "tensor.expand_shape"(%1345) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %1347 = "tensor.collapse_shape"(%1346) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %1348 = "tensor.expand_shape"(%1347) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %1349 = "tensor.extract_slice"(%1215) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1350 = "tensor.extract_slice"(%1215) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1351 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1352 = "linalg.generic"(%1215, %1333, %1351) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1353: f32, %1354: f32, %1355: f32):
      %1356 = "arith.mulf"(%1353, %1354) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1356) : (f32) -> ()
    }) {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1357 = "tensor.empty"() : () -> tensor<1x2x16x8xf32>
    %1358 = "linalg.generic"(%1350, %1357) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1359: f32, %1360: f32):
      %1361 = "arith.negf"(%1359) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1361) : (f32) -> ()
    }) {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32>
    %1362 = "tensor.concat"(%1358, %1349) <{dim = 3 : i64}> {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %1363 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1364 = "linalg.generic"(%1362, %1348, %1363) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1365: f32, %1366: f32, %1367: f32):
      %1368 = "arith.mulf"(%1365, %1366) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1368) : (f32) -> ()
    }) {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1369 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1370 = "linalg.generic"(%1352, %1364, %1369) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1371: f32, %1372: f32, %1373: f32):
      %1374 = "arith.addf"(%1371, %1372) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1374) : (f32) -> ()
    }) {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1375 = "tensor.empty"() : () -> tensor<8xf32>
    %1376 = "linalg.generic"(%1375) <{indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%1377: f32):
      %1378 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %1379 = "arith.index_cast"(%1378) : (index) -> i64
      %1380 = "arith.sitofp"(%1379) : (i64) -> f32
      %1381 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %1382 = "arith.mulf"(%1380, %1381) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1383 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %1384 = "arith.addf"(%1383, %1382) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1384) : (f32) -> ()
    }) {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>) -> tensor<8xf32>
    %1385 = "arith.constant"() <{value = 8.000000e+00 : f32}> {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1386 = "tensor.splat"(%1385) {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<8xf32>
    %1387 = "tensor.empty"() : () -> tensor<8xf32>
    %1388 = "linalg.generic"(%1376, %1386, %1387) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1389: f32, %1390: f32, %1391: f32):
      %1392 = "arith.divf"(%1389, %1390) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1392) : (f32) -> ()
    }) {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1393 = "tensor.empty"() : () -> tensor<8xf32>
    %1394 = "linalg.generic"(%1388, %1393) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1395: f32, %1396: f32):
      %1397 = "arith.constant"() <{value = 1.000000e+04 : f32}> : () -> f32
      %1398 = "math.powf"(%1397, %1395) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1398) : (f32) -> ()
    }) {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1399 = "tensor.empty"() : () -> tensor<8xf32>
    %1400 = "linalg.generic"(%1394, %1399) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1401: f32, %1402: f32):
      %1403 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %1404 = "arith.divf"(%1403, %1401) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1404) : (f32) -> ()
    }) {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1405 = "arith.constant"() <{value = 1.000000e+00 : f32}> {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1406 = "tensor.splat"(%1405) {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<8xf32>
    %1407 = "tensor.empty"() : () -> tensor<8xf32>
    %1408 = "linalg.generic"(%1400, %1406, %1407) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1409: f32, %1410: f32, %1411: f32):
      %1412 = "arith.mulf"(%1409, %1410) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1412) : (f32) -> ()
    }) {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %1413 = "tensor.expand_shape"(%1185) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 1>}> {prov.region_id = "unsqueeze_31", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16xi64>) -> tensor<16x1xi64>
    %1414 = "tensor.empty"() : () -> tensor<16x1xf32>
    %1415 = "linalg.generic"(%1413, %1414) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1416: i64, %1417: f32):
      %1418 = "arith.sitofp"(%1416) : (i64) -> f32
      "linalg.yield"(%1418) : (f32) -> ()
    }) {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x1xi64>, tensor<16x1xf32>) -> tensor<16x1xf32>
    %1419 = "tensor.expand_shape"(%1408) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 8>}> {prov.region_id = "unsqueeze_32", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<8xf32>) -> tensor<1x8xf32>
    %1420 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1421 = "linalg.generic"(%1415, %1419, %1420) <{indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1422: f32, %1423: f32, %1424: f32):
      %1425 = "arith.mulf"(%1422, %1423) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1425) : (f32) -> ()
    }) {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x1xf32>, tensor<1x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1426 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1427 = "linalg.generic"(%1421, %1426) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1428: f32, %1429: f32):
      %1430 = "math.cos"(%1428) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1430) : (f32) -> ()
    }) {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1431 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1432 = "linalg.generic"(%1421, %1431) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1433: f32, %1434: f32):
      %1435 = "math.cos"(%1433) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1435) : (f32) -> ()
    }) {prov.region_id = "cos_8", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1436 = "tensor.concat"(%1427, %1432) <{dim = 1 : i64}> {prov.region_id = "cat_12", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1437 = "tensor.collapse_shape"(%1436) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_33", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %1438 = "tensor.expand_shape"(%1437) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_33", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %1439 = "tensor.collapse_shape"(%1438) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_34", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %1440 = "tensor.expand_shape"(%1439) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_34", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %1441 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1442 = "linalg.generic"(%1421, %1441) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1443: f32, %1444: f32):
      %1445 = "math.sin"(%1443) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1445) : (f32) -> ()
    }) {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1446 = "tensor.empty"() : () -> tensor<16x8xf32>
    %1447 = "linalg.generic"(%1421, %1446) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1448: f32, %1449: f32):
      %1450 = "math.sin"(%1448) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1450) : (f32) -> ()
    }) {prov.region_id = "sin_8", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %1451 = "tensor.concat"(%1442, %1447) <{dim = 1 : i64}> {prov.region_id = "cat_13", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %1452 = "tensor.collapse_shape"(%1451) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "unsqueeze_35", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<16x16xf32>) -> tensor<256xf32>
    %1453 = "tensor.expand_shape"(%1452) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 16>}> {prov.region_id = "unsqueeze_35", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x16x16xf32>
    %1454 = "tensor.collapse_shape"(%1453) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "unsqueeze_36", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x16xf32>) -> tensor<256xf32>
    %1455 = "tensor.expand_shape"(%1454) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 1, 16, 16>}> {prov.region_id = "unsqueeze_36", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<256xf32>) -> tensor<1x1x16x16xf32>
    %1456 = "tensor.extract_slice"(%1240) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1457 = "tensor.extract_slice"(%1240) <{static_offsets = array<i64: 0, 0, 0, 8>, static_sizes = array<i64: 1, 2, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<1x2x16x8xf32>
    %1458 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1459 = "linalg.generic"(%1240, %1440, %1458) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1460: f32, %1461: f32, %1462: f32):
      %1463 = "arith.mulf"(%1460, %1461) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1463) : (f32) -> ()
    }) {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1464 = "tensor.empty"() : () -> tensor<1x2x16x8xf32>
    %1465 = "linalg.generic"(%1457, %1464) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1466: f32, %1467: f32):
      %1468 = "arith.negf"(%1466) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1468) : (f32) -> ()
    }) {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32>
    %1469 = "tensor.concat"(%1465, %1456) <{dim = 3 : i64}> {prov.region_id = "cat_14", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x16xf32>
    %1470 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1471 = "linalg.generic"(%1469, %1455, %1470) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1472: f32, %1473: f32, %1474: f32):
      %1475 = "arith.mulf"(%1472, %1473) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1475) : (f32) -> ()
    }) {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1476 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1477 = "linalg.generic"(%1459, %1471, %1476) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1478: f32, %1479: f32, %1480: f32):
      %1481 = "arith.addf"(%1478, %1479) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1481) : (f32) -> ()
    }) {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1482 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1483 = "linalg.transpose"(%1477, %1482) <{permutation = array<i64: 0, 1, 3, 2>}> ({
    ^bb0(%1484: f32, %1485: f32):
      "linalg.yield"(%1484) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1486 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1487 = "linalg.generic"(%1370, %1486) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1488: f32, %1489: f32):
      "linalg.yield"(%1488) : (f32) -> ()
    }) {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1490 = "tensor.collapse_shape"(%1487) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1491 = "tensor.expand_shape"(%1490) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1492 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1493 = "linalg.generic"(%1483, %1492) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1494: f32, %1495: f32):
      "linalg.yield"(%1494) : (f32) -> ()
    }) {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1496 = "tensor.collapse_shape"(%1493) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1497 = "tensor.expand_shape"(%1496) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1498 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1499 = "tensor.splat"(%1498) {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<2x16x16xf32>
    %1500 = "linalg.generic"(%1491, %1497, %1499) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1501: f32, %1502: f32, %1503: f32):
      %1504 = "arith.mulf"(%1501, %1502) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1505 = "arith.addf"(%1503, %1504) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1505) : (f32) -> ()
    }) {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %1506 = "tensor.collapse_shape"(%1500) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %1507 = "tensor.expand_shape"(%1506) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %1508 = "arith.constant"() <{value = 4.000000e+00 : f32}> {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1509 = "tensor.splat"(%1508) {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<1x2x16x16xf32>
    %1510 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1511 = "linalg.generic"(%1507, %1509, %1510) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1512: f32, %1513: f32, %1514: f32):
      %1515 = "arith.divf"(%1512, %1513) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1515) : (f32) -> ()
    }) {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1516 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1517 = "linalg.generic"(%1511, %1116, %1516) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1518: f32, %1519: f32, %1520: f32):
      %1521 = "arith.addf"(%1518, %1519) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1521) : (f32) -> ()
    }) {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x1x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1522 = "arith.constant"() <{value = 0xff800000 : f32}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1523 = "tensor.splat"(%1522) {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<1x2x16xf32>
    %1524 = "linalg.reduce"(%1517, %1523) <{dimensions = array<i64: 3>}> ({
    ^bb0(%1525: f32, %1526: f32):
      %1527 = "arith.maximumf"(%1525, %1526) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1527) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %1528 = "tensor.collapse_shape"(%1524) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %1529 = "tensor.expand_shape"(%1528) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %1530 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1531 = "linalg.generic"(%1517, %1529, %1530) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1532: f32, %1533: f32, %1534: f32):
      %1535 = "arith.subf"(%1532, %1533) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1535) : (f32) -> ()
    }) {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1536 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1537 = "linalg.generic"(%1531, %1536) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1538: f32, %1539: f32):
      %1540 = "math.exp"(%1538) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1540) : (f32) -> ()
    }) {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1541 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1542 = "tensor.splat"(%1541) {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<1x2x16xf32>
    %1543 = "linalg.reduce"(%1537, %1542) <{dimensions = array<i64: 3>}> ({
    ^bb0(%1544: f32, %1545: f32):
      %1546 = "arith.addf"(%1544, %1545) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1546) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %1547 = "tensor.collapse_shape"(%1543) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %1548 = "tensor.expand_shape"(%1547) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %1549 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1550 = "linalg.generic"(%1537, %1548, %1549) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1551: f32, %1552: f32, %1553: f32):
      %1554 = "arith.divf"(%1551, %1552) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1554) : (f32) -> ()
    }) {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1555 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1556 = "linalg.generic"(%1550, %1555) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1557: f32, %1558: f32):
      "linalg.yield"(%1557) : (f32) -> ()
    }) {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1559 = "tensor.collapse_shape"(%1556) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1560 = "tensor.expand_shape"(%1559) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1561 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1562 = "linalg.generic"(%1265, %1561) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1563: f32, %1564: f32):
      "linalg.yield"(%1563) : (f32) -> ()
    }) {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1565 = "tensor.collapse_shape"(%1562) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1566 = "tensor.expand_shape"(%1565) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1567 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : () -> f32
    %1568 = "tensor.splat"(%1567) {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (f32) -> tensor<2x16x16xf32>
    %1569 = "linalg.generic"(%1560, %1566, %1568) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1570: f32, %1571: f32, %1572: f32):
      %1573 = "arith.mulf"(%1570, %1571) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1574 = "arith.addf"(%1572, %1573) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1574) : (f32) -> ()
    }) {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %1575 = "tensor.collapse_shape"(%1569) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %1576 = "tensor.expand_shape"(%1575) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %1577 = "tensor.empty"() : () -> tensor<1x16x2x16xf32>
    %1578 = "linalg.transpose"(%1576, %1577) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1579: f32, %1580: f32):
      "linalg.yield"(%1579) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x16x2x16xf32>) -> tensor<1x16x2x16xf32>
    %1581 = "tensor.collapse_shape"(%1578) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<1x16x2x16xf32>) -> tensor<512xf32>
    %1582 = "tensor.expand_shape"(%1581) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1583 = "arith.constant"() <{value = 0.0112701422 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1584 = "tensor.splat"(%1583) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1585 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1586 = "tensor.splat"(%1585) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1587 = "quant_ext.quantize_per_tensor"(%1582, %1584, %1586) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_12", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1588 = "arith.constant"() <{value = 0.0112701422 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1589 = "tensor.splat"(%1588) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1590 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1591 = "tensor.splat"(%1590) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1592 = "quant_ext.dequantize_per_tensor"(%1587, %1589, %1591) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_40", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1593 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1594 = "linalg.transpose"(%108, %1593) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1595: f32, %1596: f32):
      "linalg.yield"(%1595) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1597 = "tensor.collapse_shape"(%1592) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1598 = "tensor.expand_shape"(%1597) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1599 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1600 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_self"} : () -> f32
    %1601 = "linalg.fill"(%1600, %1599) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1602: f32, %1603: f32):
      "linalg.yield"(%1602) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_self"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1604 = "linalg.matmul"(%1598, %1594, %1601) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1605: f32, %1606: f32, %1607: f32):
      %1608 = "arith.mulf"(%1605, %1606) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1609 = "arith.addf"(%1608, %1607) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1609) : (f32) -> ()
    }) {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1610 = "tensor.collapse_shape"(%1604) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1611 = "tensor.expand_shape"(%1610) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_self", prov.fqn = "a_self.o"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1612 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1613 = "linalg.generic"(%1063, %1611, %1612) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1614: f32, %1615: f32, %1616: f32):
      %1617 = "arith.addf"(%1614, %1615) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1617) : (f32) -> ()
    }) {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1618 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1619 = "linalg.generic"(%1613, %1618) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1620: f32, %1621: f32):
      %1622 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %1623 = "math.powf"(%1620, %1622) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1623) : (f32) -> ()
    }) {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1624 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : () -> f32
    %1625 = "tensor.splat"(%1624) {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (f32) -> tensor<1x16xf32>
    %1626 = "linalg.reduce"(%1619, %1625) <{dimensions = array<i64: 2>}> ({
    ^bb0(%1627: f32, %1628: f32):
      %1629 = "arith.addf"(%1627, %1628) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1629) : (f32) -> ()
    }) : (tensor<1x16x32xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1630 = "arith.constant"() <{value = 3.200000e+01 : f32}> {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : () -> f32
    %1631 = "tensor.splat"(%1630) {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (f32) -> tensor<1x16xf32>
    %1632 = "tensor.empty"() : () -> tensor<1x16xf32>
    %1633 = "linalg.generic"(%1626, %1631, %1632) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1634: f32, %1635: f32, %1636: f32):
      %1637 = "arith.divf"(%1634, %1635) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1637) : (f32) -> ()
    }) {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1638 = "tensor.collapse_shape"(%1633) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %1639 = "tensor.expand_shape"(%1638) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 1>}> {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<16xf32>) -> tensor<1x16x1xf32>
    %1640 = "arith.constant"() <{value = 1.000000e-05 : f32}> {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : () -> f32
    %1641 = "tensor.splat"(%1640) {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (f32) -> tensor<1x16x1xf32>
    %1642 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1643 = "linalg.generic"(%1639, %1641, %1642) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1644: f32, %1645: f32, %1646: f32):
      %1647 = "arith.addf"(%1644, %1645) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1647) : (f32) -> ()
    }) {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1648 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1649 = "linalg.generic"(%1643, %1648) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1650: f32, %1651: f32):
      %1652 = "math.rsqrt"(%1650) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1652) : (f32) -> ()
    }) {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1653 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1654 = "linalg.generic"(%1613, %1649, %1653) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1655: f32, %1656: f32, %1657: f32):
      %1658 = "arith.mulf"(%1655, %1656) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1658) : (f32) -> ()
    }) {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16x32xf32>, tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1659 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1660 = "linalg.generic"(%1654, %4, %1659) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1661: f32, %1662: f32, %1663: f32):
      %1664 = "arith.mulf"(%1661, %1662) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1664) : (f32) -> ()
    }) {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a2", prov.fqn = "n_a2"} : (tensor<1x16x32xf32>, tensor<32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1665 = "arith.constant"() <{value = 0.0251007639 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1666 = "tensor.splat"(%1665) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1667 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1668 = "tensor.splat"(%1667) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1669 = "quant_ext.quantize_per_tensor"(%1660, %1666, %1668) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_13", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1670 = "arith.constant"() <{value = 0.0251007639 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1671 = "tensor.splat"(%1670) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1672 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1673 = "tensor.splat"(%1672) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1674 = "quant_ext.dequantize_per_tensor"(%1669, %1671, %1673) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_41", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1675 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1676 = "linalg.transpose"(%113, %1675) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1677: f32, %1678: f32):
      "linalg.yield"(%1677) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1679 = "tensor.collapse_shape"(%1674) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1680 = "tensor.expand_shape"(%1679) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1681 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1682 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_cross"} : () -> f32
    %1683 = "linalg.fill"(%1682, %1681) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1684: f32, %1685: f32):
      "linalg.yield"(%1684) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1686 = "linalg.matmul"(%1680, %1676, %1683) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1687: f32, %1688: f32, %1689: f32):
      %1690 = "arith.mulf"(%1687, %1688) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1691 = "arith.addf"(%1690, %1689) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1691) : (f32) -> ()
    }) {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1692 = "tensor.collapse_shape"(%1686) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1693 = "tensor.expand_shape"(%1692) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.q"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1694 = "tensor.collapse_shape"(%1693) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1695 = "tensor.expand_shape"(%1694) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1696 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1697 = "linalg.transpose"(%1695, %1696) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1698: f32, %1699: f32):
      "linalg.yield"(%1698) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1700 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1701 = "linalg.transpose"(%118, %1700) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1702: f32, %1703: f32):
      "linalg.yield"(%1702) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1704 = "tensor.collapse_shape"(%964) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1705 = "tensor.expand_shape"(%1704) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1706 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1707 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_cross"} : () -> f32
    %1708 = "linalg.fill"(%1707, %1706) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1709: f32, %1710: f32):
      "linalg.yield"(%1709) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1711 = "linalg.matmul"(%1705, %1701, %1708) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1712: f32, %1713: f32, %1714: f32):
      %1715 = "arith.mulf"(%1712, %1713) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1716 = "arith.addf"(%1715, %1714) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1716) : (f32) -> ()
    }) {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1717 = "tensor.collapse_shape"(%1711) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1718 = "tensor.expand_shape"(%1717) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.k"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1719 = "tensor.collapse_shape"(%1718) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1720 = "tensor.expand_shape"(%1719) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1721 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1722 = "linalg.transpose"(%1720, %1721) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1723: f32, %1724: f32):
      "linalg.yield"(%1723) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1725 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1726 = "linalg.transpose"(%123, %1725) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1727: f32, %1728: f32):
      "linalg.yield"(%1727) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1729 = "tensor.collapse_shape"(%959) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1730 = "tensor.expand_shape"(%1729) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1731 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1732 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_cross"} : () -> f32
    %1733 = "linalg.fill"(%1732, %1731) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1734: f32, %1735: f32):
      "linalg.yield"(%1734) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1736 = "linalg.matmul"(%1730, %1726, %1733) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1737: f32, %1738: f32, %1739: f32):
      %1740 = "arith.mulf"(%1737, %1738) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1741 = "arith.addf"(%1740, %1739) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1741) : (f32) -> ()
    }) {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1742 = "tensor.collapse_shape"(%1736) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1743 = "tensor.expand_shape"(%1742) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.v"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1744 = "tensor.collapse_shape"(%1743) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1745 = "tensor.expand_shape"(%1744) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 16, 2, 16>}> {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x16x2x16xf32>
    %1746 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1747 = "linalg.transpose"(%1745, %1746) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1748: f32, %1749: f32):
      "linalg.yield"(%1748) : (f32) -> ()
    }) : (tensor<1x16x2x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1750 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1751 = "linalg.transpose"(%1722, %1750) <{permutation = array<i64: 0, 1, 3, 2>}> ({
    ^bb0(%1752: f32, %1753: f32):
      "linalg.yield"(%1752) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1754 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1755 = "linalg.generic"(%1697, %1754) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1756: f32, %1757: f32):
      "linalg.yield"(%1756) : (f32) -> ()
    }) {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1758 = "tensor.collapse_shape"(%1755) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1759 = "tensor.expand_shape"(%1758) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1760 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1761 = "linalg.generic"(%1751, %1760) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1762: f32, %1763: f32):
      "linalg.yield"(%1762) : (f32) -> ()
    }) {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1764 = "tensor.collapse_shape"(%1761) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1765 = "tensor.expand_shape"(%1764) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1766 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : () -> f32
    %1767 = "tensor.splat"(%1766) {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (f32) -> tensor<2x16x16xf32>
    %1768 = "linalg.generic"(%1759, %1765, %1767) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1769: f32, %1770: f32, %1771: f32):
      %1772 = "arith.mulf"(%1769, %1770) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1773 = "arith.addf"(%1771, %1772) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1773) : (f32) -> ()
    }) {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %1774 = "tensor.collapse_shape"(%1768) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %1775 = "tensor.expand_shape"(%1774) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %1776 = "arith.constant"() <{value = 4.000000e+00 : f32}> {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : () -> f32
    %1777 = "tensor.splat"(%1776) {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (f32) -> tensor<1x2x16x16xf32>
    %1778 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1779 = "linalg.generic"(%1775, %1777, %1778) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1780: f32, %1781: f32, %1782: f32):
      %1783 = "arith.divf"(%1780, %1781) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1783) : (f32) -> ()
    }) {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1784 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1785 = "linalg.generic"(%1779, %272, %1784) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1786: f32, %1787: f32, %1788: f32):
      %1789 = "arith.addf"(%1786, %1787) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1789) : (f32) -> ()
    }) {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x1x1x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1790 = "arith.constant"() <{value = 0xff800000 : f32}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : () -> f32
    %1791 = "tensor.splat"(%1790) {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (f32) -> tensor<1x2x16xf32>
    %1792 = "linalg.reduce"(%1785, %1791) <{dimensions = array<i64: 3>}> ({
    ^bb0(%1793: f32, %1794: f32):
      %1795 = "arith.maximumf"(%1793, %1794) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1795) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %1796 = "tensor.collapse_shape"(%1792) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %1797 = "tensor.expand_shape"(%1796) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %1798 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1799 = "linalg.generic"(%1785, %1797, %1798) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1800: f32, %1801: f32, %1802: f32):
      %1803 = "arith.subf"(%1800, %1801) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1803) : (f32) -> ()
    }) {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1804 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1805 = "linalg.generic"(%1799, %1804) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1806: f32, %1807: f32):
      %1808 = "math.exp"(%1806) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1808) : (f32) -> ()
    }) {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1809 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : () -> f32
    %1810 = "tensor.splat"(%1809) {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (f32) -> tensor<1x2x16xf32>
    %1811 = "linalg.reduce"(%1805, %1810) <{dimensions = array<i64: 3>}> ({
    ^bb0(%1812: f32, %1813: f32):
      %1814 = "arith.addf"(%1812, %1813) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1814) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
    %1815 = "tensor.collapse_shape"(%1811) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16xf32>) -> tensor<32xf32>
    %1816 = "tensor.expand_shape"(%1815) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 1>}> {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<32xf32>) -> tensor<1x2x16x1xf32>
    %1817 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1818 = "linalg.generic"(%1805, %1816, %1817) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1819: f32, %1820: f32, %1821: f32):
      %1822 = "arith.divf"(%1819, %1820) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1822) : (f32) -> ()
    }) {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x1xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1823 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1824 = "linalg.generic"(%1818, %1823) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1825: f32, %1826: f32):
      "linalg.yield"(%1825) : (f32) -> ()
    }) {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1827 = "tensor.collapse_shape"(%1824) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1828 = "tensor.expand_shape"(%1827) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1829 = "tensor.empty"() : () -> tensor<1x2x16x16xf32>
    %1830 = "linalg.generic"(%1747, %1829) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1831: f32, %1832: f32):
      "linalg.yield"(%1831) : (f32) -> ()
    }) {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
    %1833 = "tensor.collapse_shape"(%1830) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x2x16x16xf32>) -> tensor<512xf32>
    %1834 = "tensor.expand_shape"(%1833) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 2, 16, 16>}> {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<2x16x16xf32>
    %1835 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : () -> f32
    %1836 = "tensor.splat"(%1835) {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (f32) -> tensor<2x16x16xf32>
    %1837 = "linalg.generic"(%1828, %1834, %1836) <{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1838: f32, %1839: f32, %1840: f32):
      %1841 = "arith.mulf"(%1838, %1839) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1842 = "arith.addf"(%1840, %1841) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1842) : (f32) -> ()
    }) {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<2x16x16xf32>, tensor<2x16x16xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    %1843 = "tensor.collapse_shape"(%1837) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<2x16x16xf32>) -> tensor<512xf32>
    %1844 = "tensor.expand_shape"(%1843) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]], static_output_shape = array<i64: 1, 2, 16, 16>}> {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x2x16x16xf32>
    %1845 = "tensor.empty"() : () -> tensor<1x16x2x16xf32>
    %1846 = "linalg.transpose"(%1844, %1845) <{permutation = array<i64: 0, 2, 1, 3>}> ({
    ^bb0(%1847: f32, %1848: f32):
      "linalg.yield"(%1847) : (f32) -> ()
    }) : (tensor<1x2x16x16xf32>, tensor<1x16x2x16xf32>) -> tensor<1x16x2x16xf32>
    %1849 = "tensor.collapse_shape"(%1846) <{reassociation = [[0 : i64, 1 : i64, 2 : i64, 3 : i64]]}> {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<1x16x2x16xf32>) -> tensor<512xf32>
    %1850 = "tensor.expand_shape"(%1849) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1851 = "arith.constant"() <{value = 0.00221519568 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1852 = "tensor.splat"(%1851) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1853 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1854 = "tensor.splat"(%1853) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1855 = "quant_ext.quantize_per_tensor"(%1850, %1852, %1854) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_14", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1856 = "arith.constant"() <{value = 0.00221519568 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1857 = "tensor.splat"(%1856) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1858 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1859 = "tensor.splat"(%1858) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1860 = "quant_ext.dequantize_per_tensor"(%1855, %1857, %1859) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_42", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1861 = "tensor.empty"() : () -> tensor<32x32xf32>
    %1862 = "linalg.transpose"(%128, %1861) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1863: f32, %1864: f32):
      "linalg.yield"(%1863) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1865 = "tensor.collapse_shape"(%1860) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1866 = "tensor.expand_shape"(%1865) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1867 = "tensor.empty"() : () -> tensor<16x32xf32>
    %1868 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_cross"} : () -> f32
    %1869 = "linalg.fill"(%1868, %1867) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1870: f32, %1871: f32):
      "linalg.yield"(%1870) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_cross"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1872 = "linalg.matmul"(%1866, %1862, %1869) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1873: f32, %1874: f32, %1875: f32):
      %1876 = "arith.mulf"(%1873, %1874) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1877 = "arith.addf"(%1876, %1875) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1877) : (f32) -> ()
    }) {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %1878 = "tensor.collapse_shape"(%1872) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %1879 = "tensor.expand_shape"(%1878) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_cross", prov.fqn = "a_cross.o"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %1880 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1881 = "linalg.generic"(%1613, %1879, %1880) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1882: f32, %1883: f32, %1884: f32):
      %1885 = "arith.addf"(%1882, %1883) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1885) : (f32) -> ()
    }) {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1886 = "arith.constant"() <{value = 0.0179154743 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1887 = "tensor.splat"(%1886) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1888 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1889 = "tensor.splat"(%1888) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1890 = "quant_ext.quantize_per_tensor"(%1881, %1887, %1889) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_15", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1891 = "arith.constant"() <{value = 0.0179154743 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1892 = "tensor.splat"(%1891) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1893 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1894 = "tensor.splat"(%1893) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1895 = "quant_ext.dequantize_per_tensor"(%1890, %1892, %1894) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_43", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1896 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1897 = "linalg.generic"(%1881, %1896) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1898: f32, %1899: f32):
      %1900 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %1901 = "math.powf"(%1898, %1900) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1901) : (f32) -> ()
    }) {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1902 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : () -> f32
    %1903 = "tensor.splat"(%1902) {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (f32) -> tensor<1x16xf32>
    %1904 = "linalg.reduce"(%1897, %1903) <{dimensions = array<i64: 2>}> ({
    ^bb0(%1905: f32, %1906: f32):
      %1907 = "arith.addf"(%1905, %1906) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1907) : (f32) -> ()
    }) : (tensor<1x16x32xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1908 = "arith.constant"() <{value = 3.200000e+01 : f32}> {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : () -> f32
    %1909 = "tensor.splat"(%1908) {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (f32) -> tensor<1x16xf32>
    %1910 = "tensor.empty"() : () -> tensor<1x16xf32>
    %1911 = "linalg.generic"(%1904, %1909, %1910) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1912: f32, %1913: f32, %1914: f32):
      %1915 = "arith.divf"(%1912, %1913) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1915) : (f32) -> ()
    }) {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16xf32>, tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    %1916 = "tensor.collapse_shape"(%1911) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16xf32>) -> tensor<16xf32>
    %1917 = "tensor.expand_shape"(%1916) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 1>}> {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<16xf32>) -> tensor<1x16x1xf32>
    %1918 = "arith.constant"() <{value = 1.000000e-05 : f32}> {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : () -> f32
    %1919 = "tensor.splat"(%1918) {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (f32) -> tensor<1x16x1xf32>
    %1920 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1921 = "linalg.generic"(%1917, %1919, %1920) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1922: f32, %1923: f32, %1924: f32):
      %1925 = "arith.addf"(%1922, %1923) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1925) : (f32) -> ()
    }) {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1926 = "tensor.empty"() : () -> tensor<1x16x1xf32>
    %1927 = "linalg.generic"(%1921, %1926) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1928: f32, %1929: f32):
      %1930 = "math.rsqrt"(%1928) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "linalg.yield"(%1930) : (f32) -> ()
    }) {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %1931 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1932 = "linalg.generic"(%1881, %1927, %1931) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1933: f32, %1934: f32, %1935: f32):
      %1936 = "arith.mulf"(%1933, %1934) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1936) : (f32) -> ()
    }) {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16x32xf32>, tensor<1x16x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1937 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %1938 = "linalg.generic"(%1932, %5, %1937) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%1939: f32, %1940: f32, %1941: f32):
      %1942 = "arith.mulf"(%1939, %1940) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1942) : (f32) -> ()
    }) {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "n_a3", prov.fqn = "n_a3"} : (tensor<1x16x32xf32>, tensor<32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %1943 = "arith.constant"() <{value = 0.0251181908 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1944 = "tensor.splat"(%1943) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1945 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1946 = "tensor.splat"(%1945) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1947 = "quant_ext.quantize_per_tensor"(%1938, %1944, %1946) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_16", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %1948 = "arith.constant"() <{value = 0.0251181908 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1949 = "tensor.splat"(%1948) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1950 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1951 = "tensor.splat"(%1950) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1952 = "quant_ext.dequantize_per_tensor"(%1947, %1949, %1951) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_44", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %1953 = "tensor.empty"() : () -> tensor<32x64xf32>
    %1954 = "linalg.transpose"(%133, %1953) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1955: f32, %1956: f32):
      "linalg.yield"(%1955) : (f32) -> ()
    }) : (tensor<64x32xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    %1957 = "tensor.collapse_shape"(%1952) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %1958 = "tensor.expand_shape"(%1957) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %1959 = "tensor.empty"() : () -> tensor<16x64xf32>
    %1960 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_ffn"} : () -> f32
    %1961 = "linalg.fill"(%1960, %1959) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1962: f32, %1963: f32):
      "linalg.yield"(%1962) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_ffn"} : (f32, tensor<16x64xf32>) -> tensor<16x64xf32>
    %1964 = "linalg.matmul"(%1958, %1954, %1961) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%1965: f32, %1966: f32, %1967: f32):
      %1968 = "arith.mulf"(%1965, %1966) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1969 = "arith.addf"(%1968, %1967) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1969) : (f32) -> ()
    }) {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x64xf32>, tensor<16x64xf32>) -> tensor<16x64xf32>
    %1970 = "tensor.collapse_shape"(%1964) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : (tensor<16x64xf32>) -> tensor<1024xf32>
    %1971 = "tensor.expand_shape"(%1970) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 64>}> {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.up"} : (tensor<1024xf32>) -> tensor<1x16x64xf32>
    %1972 = "tensor.empty"() : () -> tensor<1x16x64xf32>
    %1973 = "linalg.generic"(%1971, %1972) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%1974: f32, %1975: f32):
      %1976 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
      %1977 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %1978 = "arith.constant"() <{value = 0.707106769 : f32}> : () -> f32
      %1979 = "arith.mulf"(%1974, %1978) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1980 = "math.erf"(%1979) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      %1981 = "arith.addf"(%1977, %1980) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1982 = "arith.mulf"(%1976, %1974) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1983 = "arith.mulf"(%1982, %1981) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1983) : (f32) -> ()
    }) {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn"} : (tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
    %1984 = "arith.constant"() <{value = 0.0134872682 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %1985 = "tensor.splat"(%1984) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %1986 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %1987 = "tensor.splat"(%1986) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %1988 = "quant_ext.quantize_per_tensor"(%1973, %1985, %1987) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_17", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xi8>
    %1989 = "arith.constant"() <{value = 0.0134872682 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %1990 = "tensor.splat"(%1989) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %1991 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %1992 = "tensor.splat"(%1991) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %1993 = "quant_ext.dequantize_per_tensor"(%1988, %1990, %1992) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_45", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x64xf32>
    %1994 = "tensor.empty"() : () -> tensor<64x32xf32>
    %1995 = "linalg.transpose"(%138, %1994) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%1996: f32, %1997: f32):
      "linalg.yield"(%1996) : (f32) -> ()
    }) : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    %1998 = "tensor.collapse_shape"(%1993) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : (tensor<1x16x64xf32>) -> tensor<1024xf32>
    %1999 = "tensor.expand_shape"(%1998) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 64>}> {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : (tensor<1024xf32>) -> tensor<16x64xf32>
    %2000 = "tensor.empty"() : () -> tensor<16x32xf32>
    %2001 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "a_ffn"} : () -> f32
    %2002 = "linalg.fill"(%2001, %2000) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%2003: f32, %2004: f32):
      "linalg.yield"(%2003) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "a_ffn"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %2005 = "linalg.matmul"(%1999, %1995, %2002) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%2006: f32, %2007: f32, %2008: f32):
      %2009 = "arith.mulf"(%2006, %2007) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %2010 = "arith.addf"(%2009, %2008) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%2010) : (f32) -> ()
    }) {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn", prov.transposed_b = "true"} : (tensor<16x64xf32>, tensor<64x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %2011 = "tensor.collapse_shape"(%2005) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %2012 = "tensor.expand_shape"(%2011) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "a_ffn", prov.fqn = "a_ffn.dn"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    %2013 = "arith.constant"() <{value = 0.00395684736 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %2014 = "tensor.splat"(%2013) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %2015 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %2016 = "tensor.splat"(%2015) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %2017 = "quant_ext.quantize_per_tensor"(%2012, %2014, %2016) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_18", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %2018 = "arith.constant"() <{value = 0.00395684736 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %2019 = "tensor.splat"(%2018) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %2020 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %2021 = "tensor.splat"(%2020) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %2022 = "quant_ext.dequantize_per_tensor"(%2017, %2019, %2021) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_46", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %2023 = "tensor.empty"() : () -> tensor<1x16x32xf32>
    %2024 = "linalg.generic"(%1895, %2022, %2023) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%2025: f32, %2026: f32, %2027: f32):
      %2028 = "arith.addf"(%2025, %2026) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%2028) : (f32) -> ()
    }) {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2029 = "arith.constant"() <{value = 0.0190579668 : f32}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> f32
    %2030 = "tensor.splat"(%2029) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (f32) -> tensor<f32>
    %2031 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : () -> i64
    %2032 = "tensor.splat"(%2031) {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (i64) -> tensor<i64>
    %2033 = "quant_ext.quantize_per_tensor"(%2024, %2030, %2032) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_19", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xi8>
    %2034 = "arith.constant"() <{value = 0.0190579668 : f32}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> f32
    %2035 = "tensor.splat"(%2034) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (f32) -> tensor<f32>
    %2036 = "arith.constant"() <{value = 0 : i64}> {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : () -> i64
    %2037 = "tensor.splat"(%2036) {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (i64) -> tensor<i64>
    %2038 = "quant_ext.dequantize_per_tensor"(%2033, %2035, %2037) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_47", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x32xf32>
    %2039 = "tensor.empty"() : () -> tensor<32x32xf32>
    %2040 = "linalg.transpose"(%143, %2039) <{permutation = array<i64: 1, 0>}> ({
    ^bb0(%2041: f32, %2042: f32):
      "linalg.yield"(%2041) : (f32) -> ()
    }) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %2043 = "tensor.collapse_shape"(%2038) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : (tensor<1x16x32xf32>) -> tensor<512xf32>
    %2044 = "tensor.expand_shape"(%2043) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 16, 32>}> {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : (tensor<512xf32>) -> tensor<16x32xf32>
    %2045 = "tensor.empty"() : () -> tensor<16x32xf32>
    %2046 = "arith.constant"() <{value = 0.000000e+00 : f32}> {prov.module = "head"} : () -> f32
    %2047 = "linalg.fill"(%2046, %2045) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%2048: f32, %2049: f32):
      "linalg.yield"(%2048) : (f32) -> ()
    }) {prov.op = "fill", prov.family = "fill", prov.module = "head"} : (f32, tensor<16x32xf32>) -> tensor<16x32xf32>
    %2050 = "linalg.matmul"(%2044, %2040, %2047) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
    ^bb0(%2051: f32, %2052: f32, %2053: f32):
      %2054 = "arith.mulf"(%2051, %2052) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %2055 = "arith.addf"(%2054, %2053) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%2055) : (f32) -> ()
    }) {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head", prov.transposed_b = "true"} : (tensor<16x32xf32>, tensor<32x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    %2056 = "tensor.collapse_shape"(%2050) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : (tensor<16x32xf32>) -> tensor<512xf32>
    %2057 = "tensor.expand_shape"(%2056) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 16, 32>}> {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} : (tensor<512xf32>) -> tensor<1x16x32xf32>
    "func.return"(%2057) : (tensor<1x16x32xf32>) -> ()
  }) : () -> ()
}) {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} : () -> ()