builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} {
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32xf32>, %5: tensor<32xf32>, %6: tensor<64xf32>, %7: tensor<32xf32>, %8: tensor<32xf32>, %9: tensor<32xf32>, %10: tensor<32xf32>, %11: tensor<32xf32>, %12: tensor<64xf32>, %13: tensor<32xf32>, %14: tensor<32xf32>, %15: tensor<256xf32>, %16: tensor<256x8x3x3xf32>, %17: tensor<256xf32>, %18: tensor<32xf32>, %19: tensor<256xf32>, %20: tensor<256x8x3x3xf32>, %21: tensor<256xf32>, %22: tensor<32xf32>, %23: tensor<32xf32>, %24: tensor<32xf32>, %25: tensor<32xf32>, %26: tensor<32xf32>, %27: tensor<64xf32>, %28: tensor<64xf32>, %29: tensor<64xf32>, %30: tensor<64xf32>, %31: tensor<64xf32>, %32: tensor<64xf32>, %33: tensor<128xf32>, %34: tensor<64xf32>, %35: tensor<64xf32>, %36: tensor<64xf32>, %37: tensor<64xf32>, %38: tensor<64xf32>, %39: tensor<128xf32>, %40: tensor<64xf32>, %41: tensor<64xf32>, %42: tensor<512xf32>, %43: tensor<512x8x3x3xf32>, %44: tensor<512xf32>, %45: tensor<64xf32>, %46: tensor<512xf32>, %47: tensor<512x8x3x3xf32>, %48: tensor<512xf32>, %49: tensor<64xf32>, %50: tensor<64xf32>, %51: tensor<64xf32>, %52: tensor<64xf32>, %53: tensor<64xf32>, %54: tensor<512xf32>, %55: tensor<512x517xf32>, %56: tensor<512x128xf32>, %57: tensor<512xf32>, %58: tensor<512xf32>, %59: tensor<512x128xf32>, %60: tensor<512x128xf32>, %61: tensor<512xf32>, %62: tensor<512xf32>, %63: tensor<512x128xf32>, %64: tensor<512x128xf32>, %65: tensor<512xf32>, %66: tensor<512xf32>, %67: tensor<3xf32>, %68: tensor<12xf32>, %69: tensor<32x1x7x7xi8>, %70: tensor<32x32x8x8xi8>, %71: tensor<64x32xi8>, %72: tensor<32x32xi8>, %73: tensor<32x32xi8>, %74: tensor<32x32x8x8xi8>, %75: tensor<64x32xi8>, %76: tensor<32x32xi8>, %77: tensor<32x32xi8>, %78: tensor<256x32xi8>, %79: tensor<32x256xi8>, %80: tensor<256x32xi8>, %81: tensor<32x256xi8>, %82: tensor<64x32x3x3xi8>, %83: tensor<64x64x4x4xi8>, %84: tensor<128x64xi8>, %85: tensor<64x64xi8>, %86: tensor<64x64xi8>, %87: tensor<64x64x4x4xi8>, %88: tensor<128x64xi8>, %89: tensor<64x64xi8>, %90: tensor<64x64xi8>, %91: tensor<512x64xi8>, %92: tensor<64x512xi8>, %93: tensor<512x64xi8>, %94: tensor<64x512xi8>, %95: tensor<512x4608xi8>, %96: tensor<3x128xi8>, %97: tensor<12x48x3x3xi8>, %98: tensor<1x1x60x90xf32>, %99: tensor<1x1xf32>, %100: tensor<1x4xf32>, %101: tensor<3x128xf32>, %102: tensor<3x128xf32>) -> (tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>) {
    %103 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00112468016 : f32
    %104 = tensor.splat %103 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %105 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %106 = tensor.splat %105 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %107 = "quant_ext.dequantize_per_tensor"(%69, %104, %106) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x1x7x7xi8>, tensor<f32>, tensor<i64>) -> tensor<32x1x7x7xf32>
    %108 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000244140625 : f32
    %109 = tensor.splat %108 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %110 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %111 = tensor.splat %110 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %112 = "quant_ext.dequantize_per_tensor"(%70, %109, %111) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32x8x8xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32x8x8xf32>
    %113 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139062968 : f32
    %114 = tensor.splat %113 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %115 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %116 = tensor.splat %115 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %117 = "quant_ext.dequantize_per_tensor"(%71, %114, %116) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %118 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139154948 : f32
    %119 = tensor.splat %118 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %120 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %121 = tensor.splat %120 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %122 = "quant_ext.dequantize_per_tensor"(%72, %119, %121) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_3", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %123 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00138846459 : f32
    %124 = tensor.splat %123 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %125 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %126 = tensor.splat %125 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %127 = "quant_ext.dequantize_per_tensor"(%73, %124, %126) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_4", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %128 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000244140625 : f32
    %129 = tensor.splat %128 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %130 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %131 = tensor.splat %130 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %132 = "quant_ext.dequantize_per_tensor"(%74, %129, %131) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_5", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32x8x8xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32x8x8xf32>
    %133 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139131595 : f32
    %134 = tensor.splat %133 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %135 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %136 = tensor.splat %135 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %137 = "quant_ext.dequantize_per_tensor"(%75, %134, %136) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_6", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32xf32>
    %138 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139002153 : f32
    %139 = tensor.splat %138 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %140 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %141 = tensor.splat %140 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %142 = "quant_ext.dequantize_per_tensor"(%76, %139, %141) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_7", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %143 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139186834 : f32
    %144 = tensor.splat %143 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %145 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %146 = tensor.splat %145 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %147 = "quant_ext.dequantize_per_tensor"(%77, %144, %146) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_8", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x32xi8>, tensor<f32>, tensor<i64>) -> tensor<32x32xf32>
    %148 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139182049 : f32
    %149 = tensor.splat %148 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %150 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %151 = tensor.splat %150 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %152 = "quant_ext.dequantize_per_tensor"(%78, %149, %151) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_9", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<256x32xi8>, tensor<f32>, tensor<i64>) -> tensor<256x32xf32>
    %153 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000492113177 : f32
    %154 = tensor.splat %153 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %155 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %156 = tensor.splat %155 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %157 = "quant_ext.dequantize_per_tensor"(%79, %154, %156) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_10", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x256xi8>, tensor<f32>, tensor<i64>) -> tensor<32x256xf32>
    %158 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00139192014 : f32
    %159 = tensor.splat %158 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %160 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %161 = tensor.splat %160 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %162 = "quant_ext.dequantize_per_tensor"(%80, %159, %161) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_11", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<256x32xi8>, tensor<f32>, tensor<i64>) -> tensor<256x32xf32>
    %163 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000492075691 : f32
    %164 = tensor.splat %163 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %165 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %166 = tensor.splat %165 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %167 = "quant_ext.dequantize_per_tensor"(%81, %164, %166) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_12", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<32x256xi8>, tensor<f32>, tensor<i64>) -> tensor<32x256xf32>
    %168 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000463979406 : f32
    %169 = tensor.splat %168 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %170 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %171 = tensor.splat %170 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %172 = "quant_ext.dequantize_per_tensor"(%82, %169, %171) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_13", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x32x3x3xi8>, tensor<f32>, tensor<i64>) -> tensor<64x32x3x3xf32>
    %173 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000246051524 : f32
    %174 = tensor.splat %173 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %175 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %176 = tensor.splat %175 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %177 = "quant_ext.dequantize_per_tensor"(%83, %174, %176) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_14", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64x4x4xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64x4x4xf32>
    %178 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 9.842500e-04 : f32
    %179 = tensor.splat %178 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %180 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %181 = tensor.splat %180 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %182 = "quant_ext.dequantize_per_tensor"(%84, %179, %181) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_15", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x64xi8>, tensor<f32>, tensor<i64>) -> tensor<128x64xf32>
    %183 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 9.836900e-04 : f32
    %184 = tensor.splat %183 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %185 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %186 = tensor.splat %185 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %187 = "quant_ext.dequantize_per_tensor"(%85, %184, %186) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_16", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64xf32>
    %188 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000983244739 : f32
    %189 = tensor.splat %188 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %190 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %191 = tensor.splat %190 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %192 = "quant_ext.dequantize_per_tensor"(%86, %189, %191) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_17", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64xf32>
    %193 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000246051903 : f32
    %194 = tensor.splat %193 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %195 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %196 = tensor.splat %195 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %197 = "quant_ext.dequantize_per_tensor"(%87, %194, %196) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_18", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64x4x4xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64x4x4xf32>
    %198 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000984198414 : f32
    %199 = tensor.splat %198 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %200 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %201 = tensor.splat %200 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %202 = "quant_ext.dequantize_per_tensor"(%88, %199, %201) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_19", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<128x64xi8>, tensor<f32>, tensor<i64>) -> tensor<128x64xf32>
    %203 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000984080718 : f32
    %204 = tensor.splat %203 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %205 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %206 = tensor.splat %205 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %207 = "quant_ext.dequantize_per_tensor"(%89, %204, %206) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_20", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64xf32>
    %208 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000983910286 : f32
    %209 = tensor.splat %208 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %210 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %211 = tensor.splat %210 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %212 = "quant_ext.dequantize_per_tensor"(%90, %209, %211) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_21", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x64xi8>, tensor<f32>, tensor<i64>) -> tensor<64x64xf32>
    %213 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000984251616 : f32
    %214 = tensor.splat %213 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %215 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %216 = tensor.splat %215 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %217 = "quant_ext.dequantize_per_tensor"(%91, %214, %216) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_22", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<512x64xi8>, tensor<f32>, tensor<i64>) -> tensor<512x64xf32>
    %218 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000347956375 : f32
    %219 = tensor.splat %218 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %220 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %221 = tensor.splat %220 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %222 = "quant_ext.dequantize_per_tensor"(%92, %219, %221) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_23", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x512xi8>, tensor<f32>, tensor<i64>) -> tensor<64x512xf32>
    %223 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00098424952 : f32
    %224 = tensor.splat %223 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %225 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %226 = tensor.splat %225 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %227 = "quant_ext.dequantize_per_tensor"(%93, %224, %226) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_24", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<512x64xi8>, tensor<f32>, tensor<i64>) -> tensor<512x64xf32>
    %228 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 3.479670e-04 : f32
    %229 = tensor.splat %228 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %230 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %231 = tensor.splat %230 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %232 = "quant_ext.dequantize_per_tensor"(%94, %229, %231) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_25", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<64x512xi8>, tensor<f32>, tensor<i64>) -> tensor<64x512xf32>
    %233 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.188752547 : f32
    %234 = tensor.splat %233 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %235 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %236 = tensor.splat %235 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %237 = "quant_ext.dequantize_per_tensor"(%95, %234, %236) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_26", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<512x4608xi8>, tensor<f32>, tensor<i64>) -> tensor<512x4608xf32>
    %238 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.021068031 : f32
    %239 = tensor.splat %238 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %240 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %241 = tensor.splat %240 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %242 = "quant_ext.dequantize_per_tensor"(%96, %239, %241) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_27", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<3x128xi8>, tensor<f32>, tensor<i64>) -> tensor<3x128xf32>
    %243 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000378802099 : f32
    %244 = tensor.splat %243 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %245 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %246 = tensor.splat %245 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %247 = "quant_ext.dequantize_per_tensor"(%97, %244, %246) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_28", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<12x48x3x3xi8>, tensor<f32>, tensor<i64>) -> tensor<12x48x3x3xf32>
    %248 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0291588064 : f32
    %249 = tensor.splat %248 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %250 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %251 = tensor.splat %250 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %252 = "quant_ext.quantize_per_tensor"(%98, %249, %251) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x1x60x90xf32>, tensor<f32>, tensor<i64>) -> tensor<1x1x60x90xi8>
    %253 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0291588064 : f32
    %254 = tensor.splat %253 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %255 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %256 = tensor.splat %255 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %257 = "quant_ext.dequantize_per_tensor"(%252, %254, %256) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_29", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x1x60x90xi8>, tensor<f32>, tensor<i64>) -> tensor<1x1x60x90xf32>
    %258 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %259 = tensor.splat %258 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %260 = "tensor.insert_slice"(%257, %259) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %261 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%260 : tensor<1x1x66x96xf32>) outs(%261 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%263: f32, %264: f32):
      linalg.yield %263 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %265 = tensor.collapse_shape %262 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %266 = tensor.expand_shape %265 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %267 = tensor.collapse_shape %107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %268 = tensor.expand_shape %267 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %269 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %270 = tensor.splat %269 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %271 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%268, %266 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%270 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %272 = tensor.collapse_shape %271 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %273 = tensor.expand_shape %272 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %274 = tensor.collapse_shape %273 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %275 = tensor.expand_shape %274 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %276 = tensor.empty() : tensor<1x32x15x23xf32>
    %277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%275, %0 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%276 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%278: f32, %279: f32, %280: f32):
      %281 = arith.addf %278, %279 : f32
      linalg.yield %281 : f32
    } -> tensor<1x32x15x23xf32>
    %282 = tensor.collapse_shape %277 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %283 = tensor.expand_shape %282 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %284 = tensor.empty() : tensor<1x345x32xf32>
    %285 = linalg.transpose ins(%283:tensor<1x32x345xf32>) outs(%284:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %286 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %287 = tensor.splat %286 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %288 = linalg.reduce ins(%285:tensor<1x345x32xf32>) outs(%287:tensor<1x345xf32>) dimensions = [2]
    (%289: f32, %290: f32) {
      %291 = arith.addf %289, %290 : f32
      linalg.yield %291 : f32
    }
    %292 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %293 = tensor.splat %292 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %294 = tensor.empty() : tensor<1x345xf32>
    %295 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%288, %293 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%294 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%296: f32, %297: f32, %298: f32):
      %299 = arith.divf %296, %297 : f32
      linalg.yield %299 : f32
    } -> tensor<1x345xf32>
    %300 = tensor.collapse_shape %295 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %301 = tensor.expand_shape %300 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %302 = tensor.empty() : tensor<1x345x32xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%285, %301 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%302 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%304: f32, %305: f32, %306: f32):
      %307 = arith.subf %304, %305 : f32
      linalg.yield %307 : f32
    } -> tensor<1x345x32xf32>
    %308 = tensor.empty() : tensor<1x345x32xf32>
    %309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%303, %303 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%308 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%310: f32, %311: f32, %312: f32):
      %313 = arith.mulf %310, %311 : f32
      linalg.yield %313 : f32
    } -> tensor<1x345x32xf32>
    %314 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %315 = tensor.splat %314 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %316 = linalg.reduce ins(%309:tensor<1x345x32xf32>) outs(%315:tensor<1x345xf32>) dimensions = [2]
    (%317: f32, %318: f32) {
      %319 = arith.addf %317, %318 : f32
      linalg.yield %319 : f32
    }
    %320 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %321 = tensor.splat %320 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %322 = tensor.empty() : tensor<1x345xf32>
    %323 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%316, %321 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%322 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%324: f32, %325: f32, %326: f32):
      %327 = arith.divf %324, %325 : f32
      linalg.yield %327 : f32
    } -> tensor<1x345xf32>
    %328 = tensor.collapse_shape %323 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %329 = tensor.expand_shape %328 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %330 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %331 = tensor.splat %330 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %332 = tensor.empty() : tensor<1x345x1xf32>
    %333 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%329, %331 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%332 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%334: f32, %335: f32, %336: f32):
      %337 = arith.addf %334, %335 : f32
      linalg.yield %337 : f32
    } -> tensor<1x345x1xf32>
    %338 = tensor.empty() : tensor<1x345x1xf32>
    %339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%333 : tensor<1x345x1xf32>) outs(%338 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%340: f32, %341: f32):
      %342 = math.rsqrt %340 : f32
      linalg.yield %342 : f32
    } -> tensor<1x345x1xf32>
    %343 = tensor.empty() : tensor<1x345x32xf32>
    %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%303, %339 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%343 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%345: f32, %346: f32, %347: f32):
      %348 = arith.mulf %345, %346 : f32
      linalg.yield %348 : f32
    } -> tensor<1x345x32xf32>
    %349 = tensor.empty() : tensor<1x345x32xf32>
    %350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%344, %1 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%349 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%351: f32, %352: f32, %353: f32):
      %354 = arith.mulf %351, %352 : f32
      linalg.yield %354 : f32
    } -> tensor<1x345x32xf32>
    %355 = tensor.empty() : tensor<1x345x32xf32>
    %356 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%350, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%355 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%357: f32, %358: f32, %359: f32):
      %360 = arith.addf %357, %358 : f32
      linalg.yield %360 : f32
    } -> tensor<1x345x32xf32>
    %361 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0285296869 : f32
    %362 = tensor.splat %361 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %363 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %364 = tensor.splat %363 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %365 = "quant_ext.quantize_per_tensor"(%356, %362, %364) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %366 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0285296869 : f32
    %367 = tensor.splat %366 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %368 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %369 = tensor.splat %368 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %370 = "quant_ext.dequantize_per_tensor"(%365, %367, %369) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_30", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %371 = tensor.empty() : tensor<1x32x345xf32>
    %372 = linalg.transpose ins(%356:tensor<1x345x32xf32>) outs(%371:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %373 = tensor.collapse_shape %372 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %374 = tensor.expand_shape %373 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %375 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0285296869 : f32
    %376 = tensor.splat %375 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %377 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %378 = tensor.splat %377 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %379 = "quant_ext.quantize_per_tensor"(%374, %376, %378) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_2", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32x15x23xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xi8>
    %380 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0285296869 : f32
    %381 = tensor.splat %380 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %382 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %383 = tensor.splat %382 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %384 = "quant_ext.dequantize_per_tensor"(%379, %381, %383) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_31", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32x15x23xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xf32>
    %385 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%384 : tensor<1x32x15x23xf32>) outs(%385 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%387: f32, %388: f32):
      linalg.yield %387 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %389 = tensor.collapse_shape %386 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %390 = tensor.expand_shape %389 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %391 = tensor.collapse_shape %112 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %392 = tensor.expand_shape %391 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %393 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %394 = tensor.splat %393 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %395 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%392, %390 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%394 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %396 = tensor.collapse_shape %395 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %397 = tensor.expand_shape %396 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %398 = tensor.collapse_shape %397 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %399 = tensor.expand_shape %398 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %400 = tensor.empty() : tensor<1x32x1x2xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%399, %3 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%400 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%402: f32, %403: f32, %404: f32):
      %405 = arith.addf %402, %403 : f32
      linalg.yield %405 : f32
    } -> tensor<1x32x1x2xf32>
    %406 = tensor.collapse_shape %401 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %407 = tensor.expand_shape %406 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %408 = tensor.empty() : tensor<1x2x32xf32>
    %409 = linalg.transpose ins(%407:tensor<1x32x2xf32>) outs(%408:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %410 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %411 = tensor.splat %410 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %412 = linalg.reduce ins(%409:tensor<1x2x32xf32>) outs(%411:tensor<1x2xf32>) dimensions = [2]
    (%413: f32, %414: f32) {
      %415 = arith.addf %413, %414 : f32
      linalg.yield %415 : f32
    }
    %416 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %417 = tensor.splat %416 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %418 = tensor.empty() : tensor<1x2xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%412, %417 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%418 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%420: f32, %421: f32, %422: f32):
      %423 = arith.divf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x2xf32>
    %424 = tensor.collapse_shape %419 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %425 = tensor.expand_shape %424 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %426 = tensor.empty() : tensor<1x2x32xf32>
    %427 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%409, %425 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%426 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%428: f32, %429: f32, %430: f32):
      %431 = arith.subf %428, %429 : f32
      linalg.yield %431 : f32
    } -> tensor<1x2x32xf32>
    %432 = tensor.empty() : tensor<1x2x32xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427, %427 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%432 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%434: f32, %435: f32, %436: f32):
      %437 = arith.mulf %434, %435 : f32
      linalg.yield %437 : f32
    } -> tensor<1x2x32xf32>
    %438 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %439 = tensor.splat %438 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %440 = linalg.reduce ins(%433:tensor<1x2x32xf32>) outs(%439:tensor<1x2xf32>) dimensions = [2]
    (%441: f32, %442: f32) {
      %443 = arith.addf %441, %442 : f32
      linalg.yield %443 : f32
    }
    %444 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %445 = tensor.splat %444 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %446 = tensor.empty() : tensor<1x2xf32>
    %447 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%440, %445 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%446 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%448: f32, %449: f32, %450: f32):
      %451 = arith.divf %448, %449 : f32
      linalg.yield %451 : f32
    } -> tensor<1x2xf32>
    %452 = tensor.collapse_shape %447 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %453 = tensor.expand_shape %452 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %454 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %455 = tensor.splat %454 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %456 = tensor.empty() : tensor<1x2x1xf32>
    %457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%453, %455 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%456 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%458: f32, %459: f32, %460: f32):
      %461 = arith.addf %458, %459 : f32
      linalg.yield %461 : f32
    } -> tensor<1x2x1xf32>
    %462 = tensor.empty() : tensor<1x2x1xf32>
    %463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%457 : tensor<1x2x1xf32>) outs(%462 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%464: f32, %465: f32):
      %466 = math.rsqrt %464 : f32
      linalg.yield %466 : f32
    } -> tensor<1x2x1xf32>
    %467 = tensor.empty() : tensor<1x2x32xf32>
    %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427, %463 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%467 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%469: f32, %470: f32, %471: f32):
      %472 = arith.mulf %469, %470 : f32
      linalg.yield %472 : f32
    } -> tensor<1x2x32xf32>
    %473 = tensor.empty() : tensor<1x2x32xf32>
    %474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%468, %4 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%473 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%475: f32, %476: f32, %477: f32):
      %478 = arith.mulf %475, %476 : f32
      linalg.yield %478 : f32
    } -> tensor<1x2x32xf32>
    %479 = tensor.empty() : tensor<1x2x32xf32>
    %480 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%474, %5 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%479 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%481: f32, %482: f32, %483: f32):
      %484 = arith.addf %481, %482 : f32
      linalg.yield %484 : f32
    } -> tensor<1x2x32xf32>
    %485 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0189710855 : f32
    %486 = tensor.splat %485 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %487 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %488 = tensor.splat %487 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %489 = "quant_ext.quantize_per_tensor"(%480, %486, %488) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_3", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x2x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x2x32xi8>
    %490 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0189710855 : f32
    %491 = tensor.splat %490 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %492 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %493 = tensor.splat %492 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %494 = "quant_ext.dequantize_per_tensor"(%489, %491, %493) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_32", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x2x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x2x32xf32>
    %495 = tensor.empty() : tensor<32x64xf32>
    %496 = linalg.transpose ins(%117:tensor<64x32xf32>) outs(%495:tensor<32x64xf32>) permutation = [1, 0]
    %497 = tensor.empty() : tensor<1x2x64xf32>
    %498 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %499 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%498 : f32) outs(%497 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %500 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} ins(%494, %496 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%499 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %501 = tensor.empty() : tensor<1x2x64xf32>
    %502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%500, %6 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%501 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%503: f32, %504: f32, %505: f32):
      %506 = arith.addf %503, %504 : f32
      linalg.yield %506 : f32
    } -> tensor<1x2x64xf32>
    %507 = tensor.collapse_shape %502 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %508 = tensor.expand_shape %507 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %509 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %510 = linalg.transpose ins(%508:tensor<1x2x2x1x32xf32>) outs(%509:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %511 = "tensor.extract_slice"(%510) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %512 = tensor.collapse_shape %511 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %513 = tensor.expand_shape %512 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %514 = "tensor.extract_slice"(%510) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %515 = tensor.collapse_shape %514 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %516 = tensor.expand_shape %515 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %517 = tensor.empty() : tensor<32x32xf32>
    %518 = linalg.transpose ins(%122:tensor<32x32xf32>) outs(%517:tensor<32x32xf32>) permutation = [1, 0]
    %519 = tensor.empty() : tensor<1x345x32xf32>
    %520 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %521 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%520 : f32) outs(%519 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %522 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} ins(%370, %518 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%521 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %523 = tensor.empty() : tensor<1x345x32xf32>
    %524 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%522, %7 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%523 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb23(%525: f32, %526: f32, %527: f32):
      %528 = arith.addf %525, %526 : f32
      linalg.yield %528 : f32
    } -> tensor<1x345x32xf32>
    %529 = tensor.collapse_shape %524 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %530 = tensor.expand_shape %529 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %531 = tensor.empty() : tensor<1x1x345x32xf32>
    %532 = linalg.transpose ins(%530:tensor<1x345x1x32xf32>) outs(%531:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %533 = tensor.empty() : tensor<1x1x32x2xf32>
    %534 = linalg.transpose ins(%513:tensor<1x1x2x32xf32>) outs(%533:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %535 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %536 = tensor.splat %535 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%532, %534 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%536 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb24(%538: f32, %539: f32, %540: f32):
      %541 = arith.mulf %538, %539 : f32
      %542 = arith.addf %540, %541 : f32
      linalg.yield %542 : f32
    } -> tensor<1x1x345x2xf32>
    %543 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %544 = tensor.splat %543 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %545 = tensor.empty() : tensor<1x1x345x2xf32>
    %546 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%537, %544 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%545 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb25(%547: f32, %548: f32, %549: f32):
      %550 = arith.divf %547, %548 : f32
      linalg.yield %550 : f32
    } -> tensor<1x1x345x2xf32>
    %551 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %552 = tensor.splat %551 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %553 = linalg.reduce ins(%546:tensor<1x1x345x2xf32>) outs(%552:tensor<1x1x345xf32>) dimensions = [3]
    (%554: f32, %555: f32) {
      %556 = arith.maximumf %554, %555 : f32
      linalg.yield %556 : f32
    }
    %557 = tensor.collapse_shape %553 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %558 = tensor.expand_shape %557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %559 = tensor.empty() : tensor<1x1x345x2xf32>
    %560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%546, %558 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%559 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb26(%561: f32, %562: f32, %563: f32):
      %564 = arith.subf %561, %562 : f32
      linalg.yield %564 : f32
    } -> tensor<1x1x345x2xf32>
    %565 = tensor.empty() : tensor<1x1x345x2xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%560 : tensor<1x1x345x2xf32>) outs(%565 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb27(%567: f32, %568: f32):
      %569 = math.exp %567 : f32
      linalg.yield %569 : f32
    } -> tensor<1x1x345x2xf32>
    %570 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %571 = tensor.splat %570 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %572 = linalg.reduce ins(%566:tensor<1x1x345x2xf32>) outs(%571:tensor<1x1x345xf32>) dimensions = [3]
    (%573: f32, %574: f32) {
      %575 = arith.addf %573, %574 : f32
      linalg.yield %575 : f32
    }
    %576 = tensor.collapse_shape %572 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %577 = tensor.expand_shape %576 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %578 = tensor.empty() : tensor<1x1x345x2xf32>
    %579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %577 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%578 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb28(%580: f32, %581: f32, %582: f32):
      %583 = arith.divf %580, %581 : f32
      linalg.yield %583 : f32
    } -> tensor<1x1x345x2xf32>
    %584 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %585 = tensor.splat %584 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %586 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%579, %516 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%585 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb29(%587: f32, %588: f32, %589: f32):
      %590 = arith.mulf %587, %588 : f32
      %591 = arith.addf %589, %590 : f32
      linalg.yield %591 : f32
    } -> tensor<1x1x345x32xf32>
    %592 = tensor.empty() : tensor<1x345x1x32xf32>
    %593 = linalg.transpose ins(%586:tensor<1x1x345x32xf32>) outs(%592:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %594 = tensor.collapse_shape %593 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %595 = tensor.expand_shape %594 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %596 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00992619526 : f32
    %597 = tensor.splat %596 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %598 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %599 = tensor.splat %598 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %600 = "quant_ext.quantize_per_tensor"(%595, %597, %599) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_4", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %601 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00992619526 : f32
    %602 = tensor.splat %601 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %603 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %604 = tensor.splat %603 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %605 = "quant_ext.dequantize_per_tensor"(%600, %602, %604) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_33", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %606 = tensor.empty() : tensor<32x32xf32>
    %607 = linalg.transpose ins(%127:tensor<32x32xf32>) outs(%606:tensor<32x32xf32>) permutation = [1, 0]
    %608 = tensor.empty() : tensor<1x345x32xf32>
    %609 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %610 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%609 : f32) outs(%608 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %611 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} ins(%605, %607 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%610 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %612 = tensor.empty() : tensor<1x345x32xf32>
    %613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%611, %8 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%612 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb30(%614: f32, %615: f32, %616: f32):
      %617 = arith.addf %614, %615 : f32
      linalg.yield %617 : f32
    } -> tensor<1x345x32xf32>
    %618 = tensor.empty() : tensor<1x345x32xf32>
    %619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%356, %613 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%618 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb31(%620: f32, %621: f32, %622: f32):
      %623 = arith.addf %620, %621 : f32
      linalg.yield %623 : f32
    } -> tensor<1x345x32xf32>
    %624 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0288852453 : f32
    %625 = tensor.splat %624 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %626 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %627 = tensor.splat %626 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %628 = "quant_ext.quantize_per_tensor"(%619, %625, %627) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_5", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %629 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0288852453 : f32
    %630 = tensor.splat %629 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %631 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %632 = tensor.splat %631 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %633 = "quant_ext.dequantize_per_tensor"(%628, %630, %632) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_34", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %634 = tensor.empty() : tensor<32x256xf32>
    %635 = linalg.transpose ins(%152:tensor<256x32xf32>) outs(%634:tensor<32x256xf32>) permutation = [1, 0]
    %636 = tensor.empty() : tensor<1x345x256xf32>
    %637 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %638 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%637 : f32) outs(%636 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %639 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} ins(%633, %635 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%638 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %640 = tensor.empty() : tensor<1x345x256xf32>
    %641 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639, %15 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%640 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb32(%642: f32, %643: f32, %644: f32):
      %645 = arith.addf %642, %643 : f32
      linalg.yield %645 : f32
    } -> tensor<1x345x256xf32>
    %646 = tensor.empty() : tensor<1x256x345xf32>
    %647 = linalg.transpose ins(%641:tensor<1x345x256xf32>) outs(%646:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %648 = tensor.collapse_shape %647 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %649 = tensor.expand_shape %648 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %650 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %651 = tensor.splat %650 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %652 = "tensor.insert_slice"(%649, %651) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %653 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%652 : tensor<1x256x17x25xf32>) outs(%653 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb33(%655: f32, %656: f32):
      linalg.yield %655 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %657 = tensor.collapse_shape %654 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %658 = tensor.expand_shape %657 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %659 = tensor.collapse_shape %16 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %660 = tensor.expand_shape %659 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %661 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %662 = tensor.splat %661 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%660, %658 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%662 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb34(%664: f32, %665: f32, %666: f32):
      %667 = arith.mulf %664, %665 : f32
      %668 = arith.addf %666, %667 : f32
      linalg.yield %668 : f32
    } -> tensor<32x8x345xf32>
    %669 = tensor.collapse_shape %663 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %670 = tensor.expand_shape %669 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %671 = tensor.collapse_shape %670 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %672 = tensor.expand_shape %671 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %673 = tensor.empty() : tensor<1x256x15x23xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %17 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%673 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb35(%675: f32, %676: f32, %677: f32):
      %678 = arith.addf %675, %676 : f32
      linalg.yield %678 : f32
    } -> tensor<1x256x15x23xf32>
    %679 = tensor.collapse_shape %674 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %680 = tensor.expand_shape %679 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %681 = tensor.empty() : tensor<1x345x256xf32>
    %682 = linalg.transpose ins(%680:tensor<1x256x345xf32>) outs(%681:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %683 = tensor.empty() : tensor<1x345x256xf32>
    %684 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%682 : tensor<1x345x256xf32>) outs(%683 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb36(%685: f32, %686: f32):
      %687 = arith.constant 5.000000e-01 : f32
      %688 = arith.constant 1.000000e+00 : f32
      %689 = arith.constant 0.707106769 : f32
      %690 = arith.mulf %685, %689 : f32
      %691 = math.erf %690 : f32
      %692 = arith.addf %688, %691 : f32
      %693 = arith.mulf %687, %685 : f32
      %694 = arith.mulf %693, %692 : f32
      linalg.yield %694 : f32
    } -> tensor<1x345x256xf32>
    %695 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0103000095 : f32
    %696 = tensor.splat %695 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %697 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %698 = tensor.splat %697 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %699 = "quant_ext.quantize_per_tensor"(%684, %696, %698) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_6", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x256xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x256xi8>
    %700 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0103000095 : f32
    %701 = tensor.splat %700 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %702 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %703 = tensor.splat %702 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %704 = "quant_ext.dequantize_per_tensor"(%699, %701, %703) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_35", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x256xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x256xf32>
    %705 = tensor.empty() : tensor<256x32xf32>
    %706 = linalg.transpose ins(%157:tensor<32x256xf32>) outs(%705:tensor<256x32xf32>) permutation = [1, 0]
    %707 = tensor.empty() : tensor<1x345x32xf32>
    %708 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %709 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%708 : f32) outs(%707 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %710 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} ins(%704, %706 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%709 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %711 = tensor.empty() : tensor<1x345x32xf32>
    %712 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%710, %18 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%711 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb37(%713: f32, %714: f32, %715: f32):
      %716 = arith.addf %713, %714 : f32
      linalg.yield %716 : f32
    } -> tensor<1x345x32xf32>
    %717 = tensor.empty() : tensor<1x345x32xf32>
    %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%619, %712 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%717 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb38(%719: f32, %720: f32, %721: f32):
      %722 = arith.addf %719, %720 : f32
      linalg.yield %722 : f32
    } -> tensor<1x345x32xf32>
    %723 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %724 = tensor.splat %723 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %725 = linalg.reduce ins(%718:tensor<1x345x32xf32>) outs(%724:tensor<1x345xf32>) dimensions = [2]
    (%726: f32, %727: f32) {
      %728 = arith.addf %726, %727 : f32
      linalg.yield %728 : f32
    }
    %729 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %730 = tensor.splat %729 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %731 = tensor.empty() : tensor<1x345xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%725, %730 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%731 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb39(%733: f32, %734: f32, %735: f32):
      %736 = arith.divf %733, %734 : f32
      linalg.yield %736 : f32
    } -> tensor<1x345xf32>
    %737 = tensor.collapse_shape %732 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %738 = tensor.expand_shape %737 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %739 = tensor.empty() : tensor<1x345x32xf32>
    %740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%718, %738 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%739 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb40(%741: f32, %742: f32, %743: f32):
      %744 = arith.subf %741, %742 : f32
      linalg.yield %744 : f32
    } -> tensor<1x345x32xf32>
    %745 = tensor.empty() : tensor<1x345x32xf32>
    %746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%740, %740 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%745 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb41(%747: f32, %748: f32, %749: f32):
      %750 = arith.mulf %747, %748 : f32
      linalg.yield %750 : f32
    } -> tensor<1x345x32xf32>
    %751 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %752 = tensor.splat %751 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %753 = linalg.reduce ins(%746:tensor<1x345x32xf32>) outs(%752:tensor<1x345xf32>) dimensions = [2]
    (%754: f32, %755: f32) {
      %756 = arith.addf %754, %755 : f32
      linalg.yield %756 : f32
    }
    %757 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %758 = tensor.splat %757 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %759 = tensor.empty() : tensor<1x345xf32>
    %760 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%753, %758 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%759 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb42(%761: f32, %762: f32, %763: f32):
      %764 = arith.divf %761, %762 : f32
      linalg.yield %764 : f32
    } -> tensor<1x345xf32>
    %765 = tensor.collapse_shape %760 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %766 = tensor.expand_shape %765 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %767 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %768 = tensor.splat %767 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %769 = tensor.empty() : tensor<1x345x1xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%766, %768 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%769 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb43(%771: f32, %772: f32, %773: f32):
      %774 = arith.addf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x345x1xf32>
    %775 = tensor.empty() : tensor<1x345x1xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%770 : tensor<1x345x1xf32>) outs(%775 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb44(%777: f32, %778: f32):
      %779 = math.rsqrt %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x345x1xf32>
    %780 = tensor.empty() : tensor<1x345x32xf32>
    %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%740, %776 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%780 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb45(%782: f32, %783: f32, %784: f32):
      %785 = arith.mulf %782, %783 : f32
      linalg.yield %785 : f32
    } -> tensor<1x345x32xf32>
    %786 = tensor.empty() : tensor<1x345x32xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781, %23 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%786 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb46(%788: f32, %789: f32, %790: f32):
      %791 = arith.mulf %788, %789 : f32
      linalg.yield %791 : f32
    } -> tensor<1x345x32xf32>
    %792 = tensor.empty() : tensor<1x345x32xf32>
    %793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%787, %24 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%792 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%794: f32, %795: f32, %796: f32):
      %797 = arith.addf %794, %795 : f32
      linalg.yield %797 : f32
    } -> tensor<1x345x32xf32>
    %798 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0264037196 : f32
    %799 = tensor.splat %798 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %800 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %801 = tensor.splat %800 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %802 = "quant_ext.quantize_per_tensor"(%793, %799, %801) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_7", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %803 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0264037196 : f32
    %804 = tensor.splat %803 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %805 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %806 = tensor.splat %805 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %807 = "quant_ext.dequantize_per_tensor"(%802, %804, %806) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_36", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %808 = tensor.empty() : tensor<1x32x345xf32>
    %809 = linalg.transpose ins(%793:tensor<1x345x32xf32>) outs(%808:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %810 = tensor.collapse_shape %809 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %811 = tensor.expand_shape %810 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %812 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0264037196 : f32
    %813 = tensor.splat %812 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %814 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %815 = tensor.splat %814 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %816 = "quant_ext.quantize_per_tensor"(%811, %813, %815) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_8", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32x15x23xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xi8>
    %817 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0264037196 : f32
    %818 = tensor.splat %817 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %819 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %820 = tensor.splat %819 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %821 = "quant_ext.dequantize_per_tensor"(%816, %818, %820) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_37", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32x15x23xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xf32>
    %822 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %823 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%821 : tensor<1x32x15x23xf32>) outs(%822 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb48(%824: f32, %825: f32):
      linalg.yield %824 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %826 = tensor.collapse_shape %823 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %827 = tensor.expand_shape %826 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %828 = tensor.collapse_shape %132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %829 = tensor.expand_shape %828 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %830 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %831 = tensor.splat %830 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %832 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%829, %827 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%831 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %833 = tensor.collapse_shape %832 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %834 = tensor.expand_shape %833 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %835 = tensor.collapse_shape %834 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %836 = tensor.expand_shape %835 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %837 = tensor.empty() : tensor<1x32x1x2xf32>
    %838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%836, %9 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%837 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb49(%839: f32, %840: f32, %841: f32):
      %842 = arith.addf %839, %840 : f32
      linalg.yield %842 : f32
    } -> tensor<1x32x1x2xf32>
    %843 = tensor.collapse_shape %838 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %844 = tensor.expand_shape %843 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %845 = tensor.empty() : tensor<1x2x32xf32>
    %846 = linalg.transpose ins(%844:tensor<1x32x2xf32>) outs(%845:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %847 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %848 = tensor.splat %847 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %849 = linalg.reduce ins(%846:tensor<1x2x32xf32>) outs(%848:tensor<1x2xf32>) dimensions = [2]
    (%850: f32, %851: f32) {
      %852 = arith.addf %850, %851 : f32
      linalg.yield %852 : f32
    }
    %853 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %854 = tensor.splat %853 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %855 = tensor.empty() : tensor<1x2xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%849, %854 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%855 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb50(%857: f32, %858: f32, %859: f32):
      %860 = arith.divf %857, %858 : f32
      linalg.yield %860 : f32
    } -> tensor<1x2xf32>
    %861 = tensor.collapse_shape %856 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %862 = tensor.expand_shape %861 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %863 = tensor.empty() : tensor<1x2x32xf32>
    %864 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%846, %862 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%863 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb51(%865: f32, %866: f32, %867: f32):
      %868 = arith.subf %865, %866 : f32
      linalg.yield %868 : f32
    } -> tensor<1x2x32xf32>
    %869 = tensor.empty() : tensor<1x2x32xf32>
    %870 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%864, %864 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%869 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb52(%871: f32, %872: f32, %873: f32):
      %874 = arith.mulf %871, %872 : f32
      linalg.yield %874 : f32
    } -> tensor<1x2x32xf32>
    %875 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %876 = tensor.splat %875 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %877 = linalg.reduce ins(%870:tensor<1x2x32xf32>) outs(%876:tensor<1x2xf32>) dimensions = [2]
    (%878: f32, %879: f32) {
      %880 = arith.addf %878, %879 : f32
      linalg.yield %880 : f32
    }
    %881 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %882 = tensor.splat %881 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %883 = tensor.empty() : tensor<1x2xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%877, %882 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%883 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb53(%885: f32, %886: f32, %887: f32):
      %888 = arith.divf %885, %886 : f32
      linalg.yield %888 : f32
    } -> tensor<1x2xf32>
    %889 = tensor.collapse_shape %884 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %890 = tensor.expand_shape %889 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %891 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %892 = tensor.splat %891 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %893 = tensor.empty() : tensor<1x2x1xf32>
    %894 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%890, %892 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%893 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb54(%895: f32, %896: f32, %897: f32):
      %898 = arith.addf %895, %896 : f32
      linalg.yield %898 : f32
    } -> tensor<1x2x1xf32>
    %899 = tensor.empty() : tensor<1x2x1xf32>
    %900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894 : tensor<1x2x1xf32>) outs(%899 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb55(%901: f32, %902: f32):
      %903 = math.rsqrt %901 : f32
      linalg.yield %903 : f32
    } -> tensor<1x2x1xf32>
    %904 = tensor.empty() : tensor<1x2x32xf32>
    %905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%864, %900 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%904 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb56(%906: f32, %907: f32, %908: f32):
      %909 = arith.mulf %906, %907 : f32
      linalg.yield %909 : f32
    } -> tensor<1x2x32xf32>
    %910 = tensor.empty() : tensor<1x2x32xf32>
    %911 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%905, %10 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%910 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb57(%912: f32, %913: f32, %914: f32):
      %915 = arith.mulf %912, %913 : f32
      linalg.yield %915 : f32
    } -> tensor<1x2x32xf32>
    %916 = tensor.empty() : tensor<1x2x32xf32>
    %917 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%911, %11 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%916 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb58(%918: f32, %919: f32, %920: f32):
      %921 = arith.addf %918, %919 : f32
      linalg.yield %921 : f32
    } -> tensor<1x2x32xf32>
    %922 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0222785845 : f32
    %923 = tensor.splat %922 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %924 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %925 = tensor.splat %924 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %926 = "quant_ext.quantize_per_tensor"(%917, %923, %925) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_9", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x2x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x2x32xi8>
    %927 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0222785845 : f32
    %928 = tensor.splat %927 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %929 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %930 = tensor.splat %929 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %931 = "quant_ext.dequantize_per_tensor"(%926, %928, %930) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_38", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x2x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x2x32xf32>
    %932 = tensor.empty() : tensor<32x64xf32>
    %933 = linalg.transpose ins(%137:tensor<64x32xf32>) outs(%932:tensor<32x64xf32>) permutation = [1, 0]
    %934 = tensor.empty() : tensor<1x2x64xf32>
    %935 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %936 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%935 : f32) outs(%934 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %937 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} ins(%931, %933 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%936 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %938 = tensor.empty() : tensor<1x2x64xf32>
    %939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%937, %12 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%938 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb59(%940: f32, %941: f32, %942: f32):
      %943 = arith.addf %940, %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x2x64xf32>
    %944 = tensor.collapse_shape %939 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %945 = tensor.expand_shape %944 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %946 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %947 = linalg.transpose ins(%945:tensor<1x2x2x1x32xf32>) outs(%946:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %948 = "tensor.extract_slice"(%947) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %949 = tensor.collapse_shape %948 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %950 = tensor.expand_shape %949 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %951 = "tensor.extract_slice"(%947) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %952 = tensor.collapse_shape %951 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %953 = tensor.expand_shape %952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %954 = tensor.empty() : tensor<32x32xf32>
    %955 = linalg.transpose ins(%142:tensor<32x32xf32>) outs(%954:tensor<32x32xf32>) permutation = [1, 0]
    %956 = tensor.empty() : tensor<1x345x32xf32>
    %957 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %958 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%957 : f32) outs(%956 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %959 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} ins(%807, %955 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%958 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %960 = tensor.empty() : tensor<1x345x32xf32>
    %961 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%959, %13 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%960 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb60(%962: f32, %963: f32, %964: f32):
      %965 = arith.addf %962, %963 : f32
      linalg.yield %965 : f32
    } -> tensor<1x345x32xf32>
    %966 = tensor.collapse_shape %961 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %967 = tensor.expand_shape %966 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %968 = tensor.empty() : tensor<1x1x345x32xf32>
    %969 = linalg.transpose ins(%967:tensor<1x345x1x32xf32>) outs(%968:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %970 = tensor.empty() : tensor<1x1x32x2xf32>
    %971 = linalg.transpose ins(%950:tensor<1x1x2x32xf32>) outs(%970:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %972 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %973 = tensor.splat %972 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%969, %971 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%973 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb61(%975: f32, %976: f32, %977: f32):
      %978 = arith.mulf %975, %976 : f32
      %979 = arith.addf %977, %978 : f32
      linalg.yield %979 : f32
    } -> tensor<1x1x345x2xf32>
    %980 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %981 = tensor.splat %980 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %982 = tensor.empty() : tensor<1x1x345x2xf32>
    %983 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%974, %981 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%982 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb62(%984: f32, %985: f32, %986: f32):
      %987 = arith.divf %984, %985 : f32
      linalg.yield %987 : f32
    } -> tensor<1x1x345x2xf32>
    %988 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %989 = tensor.splat %988 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %990 = linalg.reduce ins(%983:tensor<1x1x345x2xf32>) outs(%989:tensor<1x1x345xf32>) dimensions = [3]
    (%991: f32, %992: f32) {
      %993 = arith.maximumf %991, %992 : f32
      linalg.yield %993 : f32
    }
    %994 = tensor.collapse_shape %990 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %995 = tensor.expand_shape %994 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %996 = tensor.empty() : tensor<1x1x345x2xf32>
    %997 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%983, %995 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%996 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb63(%998: f32, %999: f32, %1000: f32):
      %1001 = arith.subf %998, %999 : f32
      linalg.yield %1001 : f32
    } -> tensor<1x1x345x2xf32>
    %1002 = tensor.empty() : tensor<1x1x345x2xf32>
    %1003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%997 : tensor<1x1x345x2xf32>) outs(%1002 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb64(%1004: f32, %1005: f32):
      %1006 = math.exp %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x1x345x2xf32>
    %1007 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %1008 = tensor.splat %1007 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %1009 = linalg.reduce ins(%1003:tensor<1x1x345x2xf32>) outs(%1008:tensor<1x1x345xf32>) dimensions = [3]
    (%1010: f32, %1011: f32) {
      %1012 = arith.addf %1010, %1011 : f32
      linalg.yield %1012 : f32
    }
    %1013 = tensor.collapse_shape %1009 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %1014 = tensor.expand_shape %1013 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %1015 = tensor.empty() : tensor<1x1x345x2xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1003, %1014 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%1015 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb65(%1017: f32, %1018: f32, %1019: f32):
      %1020 = arith.divf %1017, %1018 : f32
      linalg.yield %1020 : f32
    } -> tensor<1x1x345x2xf32>
    %1021 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1022 = tensor.splat %1021 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %1023 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1016, %953 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%1022 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb66(%1024: f32, %1025: f32, %1026: f32):
      %1027 = arith.mulf %1024, %1025 : f32
      %1028 = arith.addf %1026, %1027 : f32
      linalg.yield %1028 : f32
    } -> tensor<1x1x345x32xf32>
    %1029 = tensor.empty() : tensor<1x345x1x32xf32>
    %1030 = linalg.transpose ins(%1023:tensor<1x1x345x32xf32>) outs(%1029:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %1031 = tensor.collapse_shape %1030 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %1032 = tensor.expand_shape %1031 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1033 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0139026213 : f32
    %1034 = tensor.splat %1033 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1035 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1036 = tensor.splat %1035 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1037 = "quant_ext.quantize_per_tensor"(%1032, %1034, %1036) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_10", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %1038 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0139026213 : f32
    %1039 = tensor.splat %1038 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1040 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1041 = tensor.splat %1040 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1042 = "quant_ext.dequantize_per_tensor"(%1037, %1039, %1041) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_39", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %1043 = tensor.empty() : tensor<32x32xf32>
    %1044 = linalg.transpose ins(%147:tensor<32x32xf32>) outs(%1043:tensor<32x32xf32>) permutation = [1, 0]
    %1045 = tensor.empty() : tensor<1x345x32xf32>
    %1046 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1047 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1046 : f32) outs(%1045 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %1048 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} ins(%1042, %1044 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%1047 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %1049 = tensor.empty() : tensor<1x345x32xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1048, %14 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1049 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb67(%1051: f32, %1052: f32, %1053: f32):
      %1054 = arith.addf %1051, %1052 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x345x32xf32>
    %1055 = tensor.empty() : tensor<1x345x32xf32>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793, %1050 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1055 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb68(%1057: f32, %1058: f32, %1059: f32):
      %1060 = arith.addf %1057, %1058 : f32
      linalg.yield %1060 : f32
    } -> tensor<1x345x32xf32>
    %1061 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0276758987 : f32
    %1062 = tensor.splat %1061 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1063 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1064 = tensor.splat %1063 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1065 = "quant_ext.quantize_per_tensor"(%1056, %1062, %1064) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_11", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x32xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xi8>
    %1066 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0276758987 : f32
    %1067 = tensor.splat %1066 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1068 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1069 = tensor.splat %1068 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1070 = "quant_ext.dequantize_per_tensor"(%1065, %1067, %1069) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_40", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x32xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x32xf32>
    %1071 = tensor.empty() : tensor<32x256xf32>
    %1072 = linalg.transpose ins(%162:tensor<256x32xf32>) outs(%1071:tensor<32x256xf32>) permutation = [1, 0]
    %1073 = tensor.empty() : tensor<1x345x256xf32>
    %1074 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1075 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1074 : f32) outs(%1073 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %1076 = linalg.matmul {prov.region_id = "matmul_12", prov.dispatch_id = "matmul_12", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} ins(%1070, %1072 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%1075 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %1077 = tensor.empty() : tensor<1x345x256xf32>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1076, %19 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%1077 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb69(%1079: f32, %1080: f32, %1081: f32):
      %1082 = arith.addf %1079, %1080 : f32
      linalg.yield %1082 : f32
    } -> tensor<1x345x256xf32>
    %1083 = tensor.empty() : tensor<1x256x345xf32>
    %1084 = linalg.transpose ins(%1078:tensor<1x345x256xf32>) outs(%1083:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %1085 = tensor.collapse_shape %1084 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %1086 = tensor.expand_shape %1085 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1087 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1088 = tensor.splat %1087 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %1089 = "tensor.insert_slice"(%1086, %1088) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %1090 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %1091 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x256x17x25xf32>) outs(%1090 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb70(%1092: f32, %1093: f32):
      linalg.yield %1092 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %1094 = tensor.collapse_shape %1091 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %1095 = tensor.expand_shape %1094 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %1096 = tensor.collapse_shape %20 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %1097 = tensor.expand_shape %1096 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %1098 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1099 = tensor.splat %1098 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1097, %1095 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%1099 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb71(%1101: f32, %1102: f32, %1103: f32):
      %1104 = arith.mulf %1101, %1102 : f32
      %1105 = arith.addf %1103, %1104 : f32
      linalg.yield %1105 : f32
    } -> tensor<32x8x345xf32>
    %1106 = tensor.collapse_shape %1100 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %1107 = tensor.expand_shape %1106 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %1108 = tensor.collapse_shape %1107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %1109 = tensor.expand_shape %1108 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1110 = tensor.empty() : tensor<1x256x15x23xf32>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1109, %21 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%1110 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb72(%1112: f32, %1113: f32, %1114: f32):
      %1115 = arith.addf %1112, %1113 : f32
      linalg.yield %1115 : f32
    } -> tensor<1x256x15x23xf32>
    %1116 = tensor.collapse_shape %1111 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %1117 = tensor.expand_shape %1116 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %1118 = tensor.empty() : tensor<1x345x256xf32>
    %1119 = linalg.transpose ins(%1117:tensor<1x256x345xf32>) outs(%1118:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %1120 = tensor.empty() : tensor<1x345x256xf32>
    %1121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1119 : tensor<1x345x256xf32>) outs(%1120 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb73(%1122: f32, %1123: f32):
      %1124 = arith.constant 5.000000e-01 : f32
      %1125 = arith.constant 1.000000e+00 : f32
      %1126 = arith.constant 0.707106769 : f32
      %1127 = arith.mulf %1122, %1126 : f32
      %1128 = math.erf %1127 : f32
      %1129 = arith.addf %1125, %1128 : f32
      %1130 = arith.mulf %1124, %1122 : f32
      %1131 = arith.mulf %1130, %1129 : f32
      linalg.yield %1131 : f32
    } -> tensor<1x345x256xf32>
    %1132 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0101203388 : f32
    %1133 = tensor.splat %1132 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1134 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1135 = tensor.splat %1134 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1136 = "quant_ext.quantize_per_tensor"(%1121, %1133, %1135) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_12", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x345x256xf32>, tensor<f32>, tensor<i64>) -> tensor<1x345x256xi8>
    %1137 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0101203388 : f32
    %1138 = tensor.splat %1137 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1139 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1140 = tensor.splat %1139 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1141 = "quant_ext.dequantize_per_tensor"(%1136, %1138, %1140) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_41", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x345x256xi8>, tensor<f32>, tensor<i64>) -> tensor<1x345x256xf32>
    %1142 = tensor.empty() : tensor<256x32xf32>
    %1143 = linalg.transpose ins(%167:tensor<32x256xf32>) outs(%1142:tensor<256x32xf32>) permutation = [1, 0]
    %1144 = tensor.empty() : tensor<1x345x32xf32>
    %1145 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1146 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1145 : f32) outs(%1144 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %1147 = linalg.matmul {prov.region_id = "matmul_13", prov.dispatch_id = "matmul_13", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} ins(%1141, %1143 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%1146 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %1148 = tensor.empty() : tensor<1x345x32xf32>
    %1149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %22 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1148 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb74(%1150: f32, %1151: f32, %1152: f32):
      %1153 = arith.addf %1150, %1151 : f32
      linalg.yield %1153 : f32
    } -> tensor<1x345x32xf32>
    %1154 = tensor.empty() : tensor<1x345x32xf32>
    %1155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1056, %1149 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1154 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb75(%1156: f32, %1157: f32, %1158: f32):
      %1159 = arith.addf %1156, %1157 : f32
      linalg.yield %1159 : f32
    } -> tensor<1x345x32xf32>
    %1160 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1161 = tensor.splat %1160 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1162 = linalg.reduce ins(%1155:tensor<1x345x32xf32>) outs(%1161:tensor<1x345xf32>) dimensions = [2]
    (%1163: f32, %1164: f32) {
      %1165 = arith.addf %1163, %1164 : f32
      linalg.yield %1165 : f32
    }
    %1166 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1167 = tensor.splat %1166 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1168 = tensor.empty() : tensor<1x345xf32>
    %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1162, %1167 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1168 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%1170: f32, %1171: f32, %1172: f32):
      %1173 = arith.divf %1170, %1171 : f32
      linalg.yield %1173 : f32
    } -> tensor<1x345xf32>
    %1174 = tensor.collapse_shape %1169 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1175 = tensor.expand_shape %1174 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1176 = tensor.empty() : tensor<1x345x32xf32>
    %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1155, %1175 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1176 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb77(%1178: f32, %1179: f32, %1180: f32):
      %1181 = arith.subf %1178, %1179 : f32
      linalg.yield %1181 : f32
    } -> tensor<1x345x32xf32>
    %1182 = tensor.empty() : tensor<1x345x32xf32>
    %1183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %1177 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1182 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb78(%1184: f32, %1185: f32, %1186: f32):
      %1187 = arith.mulf %1184, %1185 : f32
      linalg.yield %1187 : f32
    } -> tensor<1x345x32xf32>
    %1188 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1189 = tensor.splat %1188 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1190 = linalg.reduce ins(%1183:tensor<1x345x32xf32>) outs(%1189:tensor<1x345xf32>) dimensions = [2]
    (%1191: f32, %1192: f32) {
      %1193 = arith.addf %1191, %1192 : f32
      linalg.yield %1193 : f32
    }
    %1194 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1195 = tensor.splat %1194 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1196 = tensor.empty() : tensor<1x345xf32>
    %1197 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1190, %1195 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1196 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb79(%1198: f32, %1199: f32, %1200: f32):
      %1201 = arith.divf %1198, %1199 : f32
      linalg.yield %1201 : f32
    } -> tensor<1x345xf32>
    %1202 = tensor.collapse_shape %1197 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1203 = tensor.expand_shape %1202 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1204 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1205 = tensor.splat %1204 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1206 = tensor.empty() : tensor<1x345x1xf32>
    %1207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1203, %1205 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1206 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb80(%1208: f32, %1209: f32, %1210: f32):
      %1211 = arith.addf %1208, %1209 : f32
      linalg.yield %1211 : f32
    } -> tensor<1x345x1xf32>
    %1212 = tensor.empty() : tensor<1x345x1xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207 : tensor<1x345x1xf32>) outs(%1212 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb81(%1214: f32, %1215: f32):
      %1216 = math.rsqrt %1214 : f32
      linalg.yield %1216 : f32
    } -> tensor<1x345x1xf32>
    %1217 = tensor.empty() : tensor<1x345x32xf32>
    %1218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %1213 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1217 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb82(%1219: f32, %1220: f32, %1221: f32):
      %1222 = arith.mulf %1219, %1220 : f32
      linalg.yield %1222 : f32
    } -> tensor<1x345x32xf32>
    %1223 = tensor.empty() : tensor<1x345x32xf32>
    %1224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1218, %25 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1223 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb83(%1225: f32, %1226: f32, %1227: f32):
      %1228 = arith.mulf %1225, %1226 : f32
      linalg.yield %1228 : f32
    } -> tensor<1x345x32xf32>
    %1229 = tensor.empty() : tensor<1x345x32xf32>
    %1230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1224, %26 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1229 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%1231: f32, %1232: f32, %1233: f32):
      %1234 = arith.addf %1231, %1232 : f32
      linalg.yield %1234 : f32
    } -> tensor<1x345x32xf32>
    %1235 = tensor.collapse_shape %1230 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1236 = tensor.expand_shape %1235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1237 = tensor.empty() : tensor<1x32x15x23xf32>
    %1238 = linalg.transpose ins(%1236:tensor<1x15x23x32xf32>) outs(%1237:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1239 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0266988203 : f32
    %1240 = tensor.splat %1239 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1241 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1242 = tensor.splat %1241 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1243 = "quant_ext.quantize_per_tensor"(%1238, %1240, %1242) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_13", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x32x15x23xf32>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xi8>
    %1244 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0266988203 : f32
    %1245 = tensor.splat %1244 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1246 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1247 = tensor.splat %1246 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1248 = "quant_ext.dequantize_per_tensor"(%1243, %1245, %1247) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_42", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x32x15x23xi8>, tensor<f32>, tensor<i64>) -> tensor<1x32x15x23xf32>
    %1249 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1250 = tensor.splat %1249 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1251 = "tensor.insert_slice"(%1248, %1250) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1252 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1251 : tensor<1x32x17x25xf32>) outs(%1252 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb85(%1254: f32, %1255: f32):
      linalg.yield %1254 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1256 = tensor.collapse_shape %1253 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1257 = tensor.expand_shape %1256 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1258 = tensor.collapse_shape %172 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1259 = tensor.expand_shape %1258 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1260 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1261 = tensor.splat %1260 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1262 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1259, %1257 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1261 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1263 = tensor.collapse_shape %1262 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1264 = tensor.expand_shape %1263 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1265 = tensor.collapse_shape %1264 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1266 = tensor.expand_shape %1265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1267 = tensor.empty() : tensor<1x64x8x12xf32>
    %1268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1266, %27 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1267 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb86(%1269: f32, %1270: f32, %1271: f32):
      %1272 = arith.addf %1269, %1270 : f32
      linalg.yield %1272 : f32
    } -> tensor<1x64x8x12xf32>
    %1273 = tensor.collapse_shape %1268 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1274 = tensor.expand_shape %1273 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1275 = tensor.empty() : tensor<1x96x64xf32>
    %1276 = linalg.transpose ins(%1274:tensor<1x64x96xf32>) outs(%1275:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1277 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1278 = tensor.splat %1277 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1279 = linalg.reduce ins(%1276:tensor<1x96x64xf32>) outs(%1278:tensor<1x96xf32>) dimensions = [2]
    (%1280: f32, %1281: f32) {
      %1282 = arith.addf %1280, %1281 : f32
      linalg.yield %1282 : f32
    }
    %1283 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1284 = tensor.splat %1283 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1285 = tensor.empty() : tensor<1x96xf32>
    %1286 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1279, %1284 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1285 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb87(%1287: f32, %1288: f32, %1289: f32):
      %1290 = arith.divf %1287, %1288 : f32
      linalg.yield %1290 : f32
    } -> tensor<1x96xf32>
    %1291 = tensor.collapse_shape %1286 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1292 = tensor.expand_shape %1291 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1293 = tensor.empty() : tensor<1x96x64xf32>
    %1294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1276, %1292 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1293 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb88(%1295: f32, %1296: f32, %1297: f32):
      %1298 = arith.subf %1295, %1296 : f32
      linalg.yield %1298 : f32
    } -> tensor<1x96x64xf32>
    %1299 = tensor.empty() : tensor<1x96x64xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294, %1294 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1299 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb89(%1301: f32, %1302: f32, %1303: f32):
      %1304 = arith.mulf %1301, %1302 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x96x64xf32>
    %1305 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1306 = tensor.splat %1305 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1307 = linalg.reduce ins(%1300:tensor<1x96x64xf32>) outs(%1306:tensor<1x96xf32>) dimensions = [2]
    (%1308: f32, %1309: f32) {
      %1310 = arith.addf %1308, %1309 : f32
      linalg.yield %1310 : f32
    }
    %1311 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1312 = tensor.splat %1311 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1313 = tensor.empty() : tensor<1x96xf32>
    %1314 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1307, %1312 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1313 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb90(%1315: f32, %1316: f32, %1317: f32):
      %1318 = arith.divf %1315, %1316 : f32
      linalg.yield %1318 : f32
    } -> tensor<1x96xf32>
    %1319 = tensor.collapse_shape %1314 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1320 = tensor.expand_shape %1319 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1321 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1322 = tensor.splat %1321 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1323 = tensor.empty() : tensor<1x96x1xf32>
    %1324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1320, %1322 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1323 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb91(%1325: f32, %1326: f32, %1327: f32):
      %1328 = arith.addf %1325, %1326 : f32
      linalg.yield %1328 : f32
    } -> tensor<1x96x1xf32>
    %1329 = tensor.empty() : tensor<1x96x1xf32>
    %1330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1324 : tensor<1x96x1xf32>) outs(%1329 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb92(%1331: f32, %1332: f32):
      %1333 = math.rsqrt %1331 : f32
      linalg.yield %1333 : f32
    } -> tensor<1x96x1xf32>
    %1334 = tensor.empty() : tensor<1x96x64xf32>
    %1335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294, %1330 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1334 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb93(%1336: f32, %1337: f32, %1338: f32):
      %1339 = arith.mulf %1336, %1337 : f32
      linalg.yield %1339 : f32
    } -> tensor<1x96x64xf32>
    %1340 = tensor.empty() : tensor<1x96x64xf32>
    %1341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1335, %28 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1340 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb94(%1342: f32, %1343: f32, %1344: f32):
      %1345 = arith.mulf %1342, %1343 : f32
      linalg.yield %1345 : f32
    } -> tensor<1x96x64xf32>
    %1346 = tensor.empty() : tensor<1x96x64xf32>
    %1347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1341, %29 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1346 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb95(%1348: f32, %1349: f32, %1350: f32):
      %1351 = arith.addf %1348, %1349 : f32
      linalg.yield %1351 : f32
    } -> tensor<1x96x64xf32>
    %1352 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0310119204 : f32
    %1353 = tensor.splat %1352 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1354 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1355 = tensor.splat %1354 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1356 = "quant_ext.quantize_per_tensor"(%1347, %1353, %1355) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_14", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %1357 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0310119204 : f32
    %1358 = tensor.splat %1357 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1359 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1360 = tensor.splat %1359 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1361 = "quant_ext.dequantize_per_tensor"(%1356, %1358, %1360) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_43", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %1362 = tensor.empty() : tensor<1x64x96xf32>
    %1363 = linalg.transpose ins(%1347:tensor<1x96x64xf32>) outs(%1362:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1364 = tensor.collapse_shape %1363 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1365 = tensor.expand_shape %1364 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1366 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0310119204 : f32
    %1367 = tensor.splat %1366 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1368 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1369 = tensor.splat %1368 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1370 = "quant_ext.quantize_per_tensor"(%1365, %1367, %1369) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_15", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x64x8x12xf32>, tensor<f32>, tensor<i64>) -> tensor<1x64x8x12xi8>
    %1371 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0310119204 : f32
    %1372 = tensor.splat %1371 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1373 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1374 = tensor.splat %1373 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1375 = "quant_ext.dequantize_per_tensor"(%1370, %1372, %1374) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_44", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x64x8x12xi8>, tensor<f32>, tensor<i64>) -> tensor<1x64x8x12xf32>
    %1376 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1375 : tensor<1x64x8x12xf32>) outs(%1376 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb96(%1378: f32, %1379: f32):
      linalg.yield %1378 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1380 = tensor.collapse_shape %1377 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1381 = tensor.expand_shape %1380 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1382 = tensor.collapse_shape %177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1383 = tensor.expand_shape %1382 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1384 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1385 = tensor.splat %1384 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1386 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1383, %1381 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1385 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1387 = tensor.collapse_shape %1386 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1388 = tensor.expand_shape %1387 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1389 = tensor.collapse_shape %1388 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1390 = tensor.expand_shape %1389 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1391 = tensor.empty() : tensor<1x64x2x3xf32>
    %1392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1390, %30 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1391 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb97(%1393: f32, %1394: f32, %1395: f32):
      %1396 = arith.addf %1393, %1394 : f32
      linalg.yield %1396 : f32
    } -> tensor<1x64x2x3xf32>
    %1397 = tensor.collapse_shape %1392 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1398 = tensor.expand_shape %1397 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1399 = tensor.empty() : tensor<1x6x64xf32>
    %1400 = linalg.transpose ins(%1398:tensor<1x64x6xf32>) outs(%1399:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1401 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1402 = tensor.splat %1401 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1403 = linalg.reduce ins(%1400:tensor<1x6x64xf32>) outs(%1402:tensor<1x6xf32>) dimensions = [2]
    (%1404: f32, %1405: f32) {
      %1406 = arith.addf %1404, %1405 : f32
      linalg.yield %1406 : f32
    }
    %1407 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1408 = tensor.splat %1407 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1409 = tensor.empty() : tensor<1x6xf32>
    %1410 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1403, %1408 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1409 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb98(%1411: f32, %1412: f32, %1413: f32):
      %1414 = arith.divf %1411, %1412 : f32
      linalg.yield %1414 : f32
    } -> tensor<1x6xf32>
    %1415 = tensor.collapse_shape %1410 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1416 = tensor.expand_shape %1415 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1417 = tensor.empty() : tensor<1x6x64xf32>
    %1418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1400, %1416 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1417 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb99(%1419: f32, %1420: f32, %1421: f32):
      %1422 = arith.subf %1419, %1420 : f32
      linalg.yield %1422 : f32
    } -> tensor<1x6x64xf32>
    %1423 = tensor.empty() : tensor<1x6x64xf32>
    %1424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1418, %1418 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1423 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb100(%1425: f32, %1426: f32, %1427: f32):
      %1428 = arith.mulf %1425, %1426 : f32
      linalg.yield %1428 : f32
    } -> tensor<1x6x64xf32>
    %1429 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1430 = tensor.splat %1429 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1431 = linalg.reduce ins(%1424:tensor<1x6x64xf32>) outs(%1430:tensor<1x6xf32>) dimensions = [2]
    (%1432: f32, %1433: f32) {
      %1434 = arith.addf %1432, %1433 : f32
      linalg.yield %1434 : f32
    }
    %1435 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1436 = tensor.splat %1435 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1437 = tensor.empty() : tensor<1x6xf32>
    %1438 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1431, %1436 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1437 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb101(%1439: f32, %1440: f32, %1441: f32):
      %1442 = arith.divf %1439, %1440 : f32
      linalg.yield %1442 : f32
    } -> tensor<1x6xf32>
    %1443 = tensor.collapse_shape %1438 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1444 = tensor.expand_shape %1443 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1445 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1446 = tensor.splat %1445 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1447 = tensor.empty() : tensor<1x6x1xf32>
    %1448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1444, %1446 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1447 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb102(%1449: f32, %1450: f32, %1451: f32):
      %1452 = arith.addf %1449, %1450 : f32
      linalg.yield %1452 : f32
    } -> tensor<1x6x1xf32>
    %1453 = tensor.empty() : tensor<1x6x1xf32>
    %1454 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1448 : tensor<1x6x1xf32>) outs(%1453 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb103(%1455: f32, %1456: f32):
      %1457 = math.rsqrt %1455 : f32
      linalg.yield %1457 : f32
    } -> tensor<1x6x1xf32>
    %1458 = tensor.empty() : tensor<1x6x64xf32>
    %1459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1418, %1454 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1458 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb104(%1460: f32, %1461: f32, %1462: f32):
      %1463 = arith.mulf %1460, %1461 : f32
      linalg.yield %1463 : f32
    } -> tensor<1x6x64xf32>
    %1464 = tensor.empty() : tensor<1x6x64xf32>
    %1465 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1459, %31 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1464 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb105(%1466: f32, %1467: f32, %1468: f32):
      %1469 = arith.mulf %1466, %1467 : f32
      linalg.yield %1469 : f32
    } -> tensor<1x6x64xf32>
    %1470 = tensor.empty() : tensor<1x6x64xf32>
    %1471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1465, %32 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1470 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb106(%1472: f32, %1473: f32, %1474: f32):
      %1475 = arith.addf %1472, %1473 : f32
      linalg.yield %1475 : f32
    } -> tensor<1x6x64xf32>
    %1476 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0280865766 : f32
    %1477 = tensor.splat %1476 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1478 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1479 = tensor.splat %1478 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1480 = "quant_ext.quantize_per_tensor"(%1471, %1477, %1479) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_16", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x6x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x6x64xi8>
    %1481 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0280865766 : f32
    %1482 = tensor.splat %1481 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1483 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1484 = tensor.splat %1483 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1485 = "quant_ext.dequantize_per_tensor"(%1480, %1482, %1484) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_45", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x6x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x6x64xf32>
    %1486 = tensor.empty() : tensor<64x128xf32>
    %1487 = linalg.transpose ins(%182:tensor<128x64xf32>) outs(%1486:tensor<64x128xf32>) permutation = [1, 0]
    %1488 = tensor.empty() : tensor<1x6x128xf32>
    %1489 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1490 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1489 : f32) outs(%1488 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1491 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} ins(%1485, %1487 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1490 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1492 = tensor.empty() : tensor<1x6x128xf32>
    %1493 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1491, %33 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1492 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb107(%1494: f32, %1495: f32, %1496: f32):
      %1497 = arith.addf %1494, %1495 : f32
      linalg.yield %1497 : f32
    } -> tensor<1x6x128xf32>
    %1498 = tensor.collapse_shape %1493 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1499 = tensor.expand_shape %1498 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1500 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1501 = linalg.transpose ins(%1499:tensor<1x6x2x2x32xf32>) outs(%1500:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1502 = "tensor.extract_slice"(%1501) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1503 = tensor.collapse_shape %1502 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1504 = tensor.expand_shape %1503 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1505 = "tensor.extract_slice"(%1501) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1506 = tensor.collapse_shape %1505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1507 = tensor.expand_shape %1506 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1508 = tensor.empty() : tensor<64x64xf32>
    %1509 = linalg.transpose ins(%187:tensor<64x64xf32>) outs(%1508:tensor<64x64xf32>) permutation = [1, 0]
    %1510 = tensor.empty() : tensor<1x96x64xf32>
    %1511 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1512 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1511 : f32) outs(%1510 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1513 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} ins(%1361, %1509 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1512 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1514 = tensor.empty() : tensor<1x96x64xf32>
    %1515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1513, %34 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1514 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb108(%1516: f32, %1517: f32, %1518: f32):
      %1519 = arith.addf %1516, %1517 : f32
      linalg.yield %1519 : f32
    } -> tensor<1x96x64xf32>
    %1520 = tensor.collapse_shape %1515 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1521 = tensor.expand_shape %1520 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1522 = tensor.empty() : tensor<1x2x96x32xf32>
    %1523 = linalg.transpose ins(%1521:tensor<1x96x2x32xf32>) outs(%1522:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1524 = tensor.empty() : tensor<1x2x32x6xf32>
    %1525 = linalg.transpose ins(%1504:tensor<1x2x6x32xf32>) outs(%1524:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1526 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1527 = tensor.splat %1526 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1523, %1525 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1527 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb109(%1529: f32, %1530: f32, %1531: f32):
      %1532 = arith.mulf %1529, %1530 : f32
      %1533 = arith.addf %1531, %1532 : f32
      linalg.yield %1533 : f32
    } -> tensor<1x2x96x6xf32>
    %1534 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1535 = tensor.splat %1534 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1536 = tensor.empty() : tensor<1x2x96x6xf32>
    %1537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1528, %1535 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1536 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb110(%1538: f32, %1539: f32, %1540: f32):
      %1541 = arith.divf %1538, %1539 : f32
      linalg.yield %1541 : f32
    } -> tensor<1x2x96x6xf32>
    %1542 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1543 = tensor.splat %1542 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1544 = linalg.reduce ins(%1537:tensor<1x2x96x6xf32>) outs(%1543:tensor<1x2x96xf32>) dimensions = [3]
    (%1545: f32, %1546: f32) {
      %1547 = arith.maximumf %1545, %1546 : f32
      linalg.yield %1547 : f32
    }
    %1548 = tensor.collapse_shape %1544 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1549 = tensor.expand_shape %1548 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1550 = tensor.empty() : tensor<1x2x96x6xf32>
    %1551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1537, %1549 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1550 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb111(%1552: f32, %1553: f32, %1554: f32):
      %1555 = arith.subf %1552, %1553 : f32
      linalg.yield %1555 : f32
    } -> tensor<1x2x96x6xf32>
    %1556 = tensor.empty() : tensor<1x2x96x6xf32>
    %1557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1551 : tensor<1x2x96x6xf32>) outs(%1556 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb112(%1558: f32, %1559: f32):
      %1560 = math.exp %1558 : f32
      linalg.yield %1560 : f32
    } -> tensor<1x2x96x6xf32>
    %1561 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1562 = tensor.splat %1561 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1563 = linalg.reduce ins(%1557:tensor<1x2x96x6xf32>) outs(%1562:tensor<1x2x96xf32>) dimensions = [3]
    (%1564: f32, %1565: f32) {
      %1566 = arith.addf %1564, %1565 : f32
      linalg.yield %1566 : f32
    }
    %1567 = tensor.collapse_shape %1563 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1568 = tensor.expand_shape %1567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1569 = tensor.empty() : tensor<1x2x96x6xf32>
    %1570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1557, %1568 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1569 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb113(%1571: f32, %1572: f32, %1573: f32):
      %1574 = arith.divf %1571, %1572 : f32
      linalg.yield %1574 : f32
    } -> tensor<1x2x96x6xf32>
    %1575 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1576 = tensor.splat %1575 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1570, %1507 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1576 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb114(%1578: f32, %1579: f32, %1580: f32):
      %1581 = arith.mulf %1578, %1579 : f32
      %1582 = arith.addf %1580, %1581 : f32
      linalg.yield %1582 : f32
    } -> tensor<1x2x96x32xf32>
    %1583 = tensor.empty() : tensor<1x96x2x32xf32>
    %1584 = linalg.transpose ins(%1577:tensor<1x2x96x32xf32>) outs(%1583:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1585 = tensor.collapse_shape %1584 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1586 = tensor.expand_shape %1585 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1587 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00808702782 : f32
    %1588 = tensor.splat %1587 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1589 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1590 = tensor.splat %1589 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1591 = "quant_ext.quantize_per_tensor"(%1586, %1588, %1590) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_17", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %1592 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00808702782 : f32
    %1593 = tensor.splat %1592 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1594 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1595 = tensor.splat %1594 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1596 = "quant_ext.dequantize_per_tensor"(%1591, %1593, %1595) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_46", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %1597 = tensor.empty() : tensor<64x64xf32>
    %1598 = linalg.transpose ins(%192:tensor<64x64xf32>) outs(%1597:tensor<64x64xf32>) permutation = [1, 0]
    %1599 = tensor.empty() : tensor<1x96x64xf32>
    %1600 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1601 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1600 : f32) outs(%1599 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1602 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} ins(%1596, %1598 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1601 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1603 = tensor.empty() : tensor<1x96x64xf32>
    %1604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1602, %35 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1603 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb115(%1605: f32, %1606: f32, %1607: f32):
      %1608 = arith.addf %1605, %1606 : f32
      linalg.yield %1608 : f32
    } -> tensor<1x96x64xf32>
    %1609 = tensor.empty() : tensor<1x96x64xf32>
    %1610 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1347, %1604 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1609 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb116(%1611: f32, %1612: f32, %1613: f32):
      %1614 = arith.addf %1611, %1612 : f32
      linalg.yield %1614 : f32
    } -> tensor<1x96x64xf32>
    %1615 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0315502733 : f32
    %1616 = tensor.splat %1615 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1617 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1618 = tensor.splat %1617 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1619 = "quant_ext.quantize_per_tensor"(%1610, %1616, %1618) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_18", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %1620 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0315502733 : f32
    %1621 = tensor.splat %1620 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1622 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1623 = tensor.splat %1622 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1624 = "quant_ext.dequantize_per_tensor"(%1619, %1621, %1623) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_47", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %1625 = tensor.empty() : tensor<64x512xf32>
    %1626 = linalg.transpose ins(%217:tensor<512x64xf32>) outs(%1625:tensor<64x512xf32>) permutation = [1, 0]
    %1627 = tensor.empty() : tensor<1x96x512xf32>
    %1628 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1629 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1628 : f32) outs(%1627 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1630 = linalg.matmul {prov.region_id = "matmul_19", prov.dispatch_id = "matmul_19", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} ins(%1624, %1626 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%1629 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1631 = tensor.empty() : tensor<1x96x512xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1630, %42 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1631 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb117(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.addf %1633, %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<1x96x512xf32>
    %1637 = tensor.empty() : tensor<1x512x96xf32>
    %1638 = linalg.transpose ins(%1632:tensor<1x96x512xf32>) outs(%1637:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1639 = tensor.collapse_shape %1638 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1640 = tensor.expand_shape %1639 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1641 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1642 = tensor.splat %1641 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1643 = "tensor.insert_slice"(%1640, %1642) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1644 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1643 : tensor<1x512x10x14xf32>) outs(%1644 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb118(%1646: f32, %1647: f32):
      linalg.yield %1646 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1648 = tensor.collapse_shape %1645 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1649 = tensor.expand_shape %1648 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1650 = tensor.collapse_shape %43 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1651 = tensor.expand_shape %1650 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1652 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1653 = tensor.splat %1652 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1651, %1649 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1653 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb119(%1655: f32, %1656: f32, %1657: f32):
      %1658 = arith.mulf %1655, %1656 : f32
      %1659 = arith.addf %1657, %1658 : f32
      linalg.yield %1659 : f32
    } -> tensor<64x8x96xf32>
    %1660 = tensor.collapse_shape %1654 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1661 = tensor.expand_shape %1660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1662 = tensor.collapse_shape %1661 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1663 = tensor.expand_shape %1662 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1664 = tensor.empty() : tensor<1x512x8x12xf32>
    %1665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1663, %44 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1664 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb120(%1666: f32, %1667: f32, %1668: f32):
      %1669 = arith.addf %1666, %1667 : f32
      linalg.yield %1669 : f32
    } -> tensor<1x512x8x12xf32>
    %1670 = tensor.collapse_shape %1665 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1671 = tensor.expand_shape %1670 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1672 = tensor.empty() : tensor<1x96x512xf32>
    %1673 = linalg.transpose ins(%1671:tensor<1x512x96xf32>) outs(%1672:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1674 = tensor.empty() : tensor<1x96x512xf32>
    %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673 : tensor<1x96x512xf32>) outs(%1674 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb121(%1676: f32, %1677: f32):
      %1678 = arith.constant 5.000000e-01 : f32
      %1679 = arith.constant 1.000000e+00 : f32
      %1680 = arith.constant 0.707106769 : f32
      %1681 = arith.mulf %1676, %1680 : f32
      %1682 = math.erf %1681 : f32
      %1683 = arith.addf %1679, %1682 : f32
      %1684 = arith.mulf %1678, %1676 : f32
      %1685 = arith.mulf %1684, %1683 : f32
      linalg.yield %1685 : f32
    } -> tensor<1x96x512xf32>
    %1686 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00960381236 : f32
    %1687 = tensor.splat %1686 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1688 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1689 = tensor.splat %1688 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1690 = "quant_ext.quantize_per_tensor"(%1675, %1687, %1689) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_19", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x512xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x512xi8>
    %1691 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00960381236 : f32
    %1692 = tensor.splat %1691 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1693 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1694 = tensor.splat %1693 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1695 = "quant_ext.dequantize_per_tensor"(%1690, %1692, %1694) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_48", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x512xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x512xf32>
    %1696 = tensor.empty() : tensor<512x64xf32>
    %1697 = linalg.transpose ins(%222:tensor<64x512xf32>) outs(%1696:tensor<512x64xf32>) permutation = [1, 0]
    %1698 = tensor.empty() : tensor<1x96x64xf32>
    %1699 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1700 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1699 : f32) outs(%1698 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1701 = linalg.matmul {prov.region_id = "matmul_20", prov.dispatch_id = "matmul_20", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} ins(%1695, %1697 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%1700 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1702 = tensor.empty() : tensor<1x96x64xf32>
    %1703 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1701, %45 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1702 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb122(%1704: f32, %1705: f32, %1706: f32):
      %1707 = arith.addf %1704, %1705 : f32
      linalg.yield %1707 : f32
    } -> tensor<1x96x64xf32>
    %1708 = tensor.empty() : tensor<1x96x64xf32>
    %1709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1610, %1703 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1708 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb123(%1710: f32, %1711: f32, %1712: f32):
      %1713 = arith.addf %1710, %1711 : f32
      linalg.yield %1713 : f32
    } -> tensor<1x96x64xf32>
    %1714 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1715 = tensor.splat %1714 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1716 = linalg.reduce ins(%1709:tensor<1x96x64xf32>) outs(%1715:tensor<1x96xf32>) dimensions = [2]
    (%1717: f32, %1718: f32) {
      %1719 = arith.addf %1717, %1718 : f32
      linalg.yield %1719 : f32
    }
    %1720 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1721 = tensor.splat %1720 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1722 = tensor.empty() : tensor<1x96xf32>
    %1723 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1716, %1721 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1722 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb124(%1724: f32, %1725: f32, %1726: f32):
      %1727 = arith.divf %1724, %1725 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x96xf32>
    %1728 = tensor.collapse_shape %1723 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1729 = tensor.expand_shape %1728 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1730 = tensor.empty() : tensor<1x96x64xf32>
    %1731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1709, %1729 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1730 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb125(%1732: f32, %1733: f32, %1734: f32):
      %1735 = arith.subf %1732, %1733 : f32
      linalg.yield %1735 : f32
    } -> tensor<1x96x64xf32>
    %1736 = tensor.empty() : tensor<1x96x64xf32>
    %1737 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1731, %1731 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1736 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb126(%1738: f32, %1739: f32, %1740: f32):
      %1741 = arith.mulf %1738, %1739 : f32
      linalg.yield %1741 : f32
    } -> tensor<1x96x64xf32>
    %1742 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1743 = tensor.splat %1742 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1744 = linalg.reduce ins(%1737:tensor<1x96x64xf32>) outs(%1743:tensor<1x96xf32>) dimensions = [2]
    (%1745: f32, %1746: f32) {
      %1747 = arith.addf %1745, %1746 : f32
      linalg.yield %1747 : f32
    }
    %1748 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1749 = tensor.splat %1748 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1750 = tensor.empty() : tensor<1x96xf32>
    %1751 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1744, %1749 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1750 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb127(%1752: f32, %1753: f32, %1754: f32):
      %1755 = arith.divf %1752, %1753 : f32
      linalg.yield %1755 : f32
    } -> tensor<1x96xf32>
    %1756 = tensor.collapse_shape %1751 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1757 = tensor.expand_shape %1756 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1758 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1759 = tensor.splat %1758 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1760 = tensor.empty() : tensor<1x96x1xf32>
    %1761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1757, %1759 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1760 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb128(%1762: f32, %1763: f32, %1764: f32):
      %1765 = arith.addf %1762, %1763 : f32
      linalg.yield %1765 : f32
    } -> tensor<1x96x1xf32>
    %1766 = tensor.empty() : tensor<1x96x1xf32>
    %1767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1761 : tensor<1x96x1xf32>) outs(%1766 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb129(%1768: f32, %1769: f32):
      %1770 = math.rsqrt %1768 : f32
      linalg.yield %1770 : f32
    } -> tensor<1x96x1xf32>
    %1771 = tensor.empty() : tensor<1x96x64xf32>
    %1772 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1731, %1767 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1771 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb130(%1773: f32, %1774: f32, %1775: f32):
      %1776 = arith.mulf %1773, %1774 : f32
      linalg.yield %1776 : f32
    } -> tensor<1x96x64xf32>
    %1777 = tensor.empty() : tensor<1x96x64xf32>
    %1778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1772, %50 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1777 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb131(%1779: f32, %1780: f32, %1781: f32):
      %1782 = arith.mulf %1779, %1780 : f32
      linalg.yield %1782 : f32
    } -> tensor<1x96x64xf32>
    %1783 = tensor.empty() : tensor<1x96x64xf32>
    %1784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1778, %51 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1783 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb132(%1785: f32, %1786: f32, %1787: f32):
      %1788 = arith.addf %1785, %1786 : f32
      linalg.yield %1788 : f32
    } -> tensor<1x96x64xf32>
    %1789 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0298244078 : f32
    %1790 = tensor.splat %1789 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1791 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1792 = tensor.splat %1791 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1793 = "quant_ext.quantize_per_tensor"(%1784, %1790, %1792) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_20", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %1794 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0298244078 : f32
    %1795 = tensor.splat %1794 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1796 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1797 = tensor.splat %1796 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1798 = "quant_ext.dequantize_per_tensor"(%1793, %1795, %1797) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_49", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %1799 = tensor.empty() : tensor<1x64x96xf32>
    %1800 = linalg.transpose ins(%1784:tensor<1x96x64xf32>) outs(%1799:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1801 = tensor.collapse_shape %1800 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1802 = tensor.expand_shape %1801 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1803 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0298244078 : f32
    %1804 = tensor.splat %1803 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1805 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1806 = tensor.splat %1805 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1807 = "quant_ext.quantize_per_tensor"(%1802, %1804, %1806) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_21", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x64x8x12xf32>, tensor<f32>, tensor<i64>) -> tensor<1x64x8x12xi8>
    %1808 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0298244078 : f32
    %1809 = tensor.splat %1808 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1810 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1811 = tensor.splat %1810 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1812 = "quant_ext.dequantize_per_tensor"(%1807, %1809, %1811) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_50", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x64x8x12xi8>, tensor<f32>, tensor<i64>) -> tensor<1x64x8x12xf32>
    %1813 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1814 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1812 : tensor<1x64x8x12xf32>) outs(%1813 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb133(%1815: f32, %1816: f32):
      linalg.yield %1815 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1817 = tensor.collapse_shape %1814 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1818 = tensor.expand_shape %1817 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1819 = tensor.collapse_shape %197 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1820 = tensor.expand_shape %1819 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1821 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1822 = tensor.splat %1821 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1823 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1820, %1818 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1822 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1824 = tensor.collapse_shape %1823 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1825 = tensor.expand_shape %1824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1826 = tensor.collapse_shape %1825 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1827 = tensor.expand_shape %1826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1828 = tensor.empty() : tensor<1x64x2x3xf32>
    %1829 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1827, %36 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1828 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb134(%1830: f32, %1831: f32, %1832: f32):
      %1833 = arith.addf %1830, %1831 : f32
      linalg.yield %1833 : f32
    } -> tensor<1x64x2x3xf32>
    %1834 = tensor.collapse_shape %1829 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1835 = tensor.expand_shape %1834 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1836 = tensor.empty() : tensor<1x6x64xf32>
    %1837 = linalg.transpose ins(%1835:tensor<1x64x6xf32>) outs(%1836:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1838 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1839 = tensor.splat %1838 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1840 = linalg.reduce ins(%1837:tensor<1x6x64xf32>) outs(%1839:tensor<1x6xf32>) dimensions = [2]
    (%1841: f32, %1842: f32) {
      %1843 = arith.addf %1841, %1842 : f32
      linalg.yield %1843 : f32
    }
    %1844 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1845 = tensor.splat %1844 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1846 = tensor.empty() : tensor<1x6xf32>
    %1847 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1840, %1845 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1846 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb135(%1848: f32, %1849: f32, %1850: f32):
      %1851 = arith.divf %1848, %1849 : f32
      linalg.yield %1851 : f32
    } -> tensor<1x6xf32>
    %1852 = tensor.collapse_shape %1847 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1853 = tensor.expand_shape %1852 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1854 = tensor.empty() : tensor<1x6x64xf32>
    %1855 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1837, %1853 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1854 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb136(%1856: f32, %1857: f32, %1858: f32):
      %1859 = arith.subf %1856, %1857 : f32
      linalg.yield %1859 : f32
    } -> tensor<1x6x64xf32>
    %1860 = tensor.empty() : tensor<1x6x64xf32>
    %1861 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855, %1855 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1860 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb137(%1862: f32, %1863: f32, %1864: f32):
      %1865 = arith.mulf %1862, %1863 : f32
      linalg.yield %1865 : f32
    } -> tensor<1x6x64xf32>
    %1866 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1867 = tensor.splat %1866 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1868 = linalg.reduce ins(%1861:tensor<1x6x64xf32>) outs(%1867:tensor<1x6xf32>) dimensions = [2]
    (%1869: f32, %1870: f32) {
      %1871 = arith.addf %1869, %1870 : f32
      linalg.yield %1871 : f32
    }
    %1872 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1873 = tensor.splat %1872 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1874 = tensor.empty() : tensor<1x6xf32>
    %1875 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1868, %1873 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1874 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb138(%1876: f32, %1877: f32, %1878: f32):
      %1879 = arith.divf %1876, %1877 : f32
      linalg.yield %1879 : f32
    } -> tensor<1x6xf32>
    %1880 = tensor.collapse_shape %1875 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1881 = tensor.expand_shape %1880 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1882 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1883 = tensor.splat %1882 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1884 = tensor.empty() : tensor<1x6x1xf32>
    %1885 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1881, %1883 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1884 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb139(%1886: f32, %1887: f32, %1888: f32):
      %1889 = arith.addf %1886, %1887 : f32
      linalg.yield %1889 : f32
    } -> tensor<1x6x1xf32>
    %1890 = tensor.empty() : tensor<1x6x1xf32>
    %1891 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1885 : tensor<1x6x1xf32>) outs(%1890 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb140(%1892: f32, %1893: f32):
      %1894 = math.rsqrt %1892 : f32
      linalg.yield %1894 : f32
    } -> tensor<1x6x1xf32>
    %1895 = tensor.empty() : tensor<1x6x64xf32>
    %1896 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855, %1891 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1895 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb141(%1897: f32, %1898: f32, %1899: f32):
      %1900 = arith.mulf %1897, %1898 : f32
      linalg.yield %1900 : f32
    } -> tensor<1x6x64xf32>
    %1901 = tensor.empty() : tensor<1x6x64xf32>
    %1902 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1896, %37 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1901 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb142(%1903: f32, %1904: f32, %1905: f32):
      %1906 = arith.mulf %1903, %1904 : f32
      linalg.yield %1906 : f32
    } -> tensor<1x6x64xf32>
    %1907 = tensor.empty() : tensor<1x6x64xf32>
    %1908 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1902, %38 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1907 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb143(%1909: f32, %1910: f32, %1911: f32):
      %1912 = arith.addf %1909, %1910 : f32
      linalg.yield %1912 : f32
    } -> tensor<1x6x64xf32>
    %1913 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0251790583 : f32
    %1914 = tensor.splat %1913 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %1915 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %1916 = tensor.splat %1915 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %1917 = "quant_ext.quantize_per_tensor"(%1908, %1914, %1916) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_22", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x6x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x6x64xi8>
    %1918 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0251790583 : f32
    %1919 = tensor.splat %1918 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1920 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %1921 = tensor.splat %1920 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %1922 = "quant_ext.dequantize_per_tensor"(%1917, %1919, %1921) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_51", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x6x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x6x64xf32>
    %1923 = tensor.empty() : tensor<64x128xf32>
    %1924 = linalg.transpose ins(%202:tensor<128x64xf32>) outs(%1923:tensor<64x128xf32>) permutation = [1, 0]
    %1925 = tensor.empty() : tensor<1x6x128xf32>
    %1926 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1927 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1926 : f32) outs(%1925 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1928 = linalg.matmul {prov.region_id = "matmul_21", prov.dispatch_id = "matmul_21", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} ins(%1922, %1924 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1927 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1929 = tensor.empty() : tensor<1x6x128xf32>
    %1930 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1928, %39 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1929 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb144(%1931: f32, %1932: f32, %1933: f32):
      %1934 = arith.addf %1931, %1932 : f32
      linalg.yield %1934 : f32
    } -> tensor<1x6x128xf32>
    %1935 = tensor.collapse_shape %1930 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1936 = tensor.expand_shape %1935 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1937 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1938 = linalg.transpose ins(%1936:tensor<1x6x2x2x32xf32>) outs(%1937:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1939 = "tensor.extract_slice"(%1938) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1940 = tensor.collapse_shape %1939 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1941 = tensor.expand_shape %1940 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1942 = "tensor.extract_slice"(%1938) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1943 = tensor.collapse_shape %1942 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1944 = tensor.expand_shape %1943 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1945 = tensor.empty() : tensor<64x64xf32>
    %1946 = linalg.transpose ins(%207:tensor<64x64xf32>) outs(%1945:tensor<64x64xf32>) permutation = [1, 0]
    %1947 = tensor.empty() : tensor<1x96x64xf32>
    %1948 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1949 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1948 : f32) outs(%1947 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1950 = linalg.matmul {prov.region_id = "matmul_22", prov.dispatch_id = "matmul_22", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} ins(%1798, %1946 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1949 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1951 = tensor.empty() : tensor<1x96x64xf32>
    %1952 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1950, %40 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1951 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb145(%1953: f32, %1954: f32, %1955: f32):
      %1956 = arith.addf %1953, %1954 : f32
      linalg.yield %1956 : f32
    } -> tensor<1x96x64xf32>
    %1957 = tensor.collapse_shape %1952 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1958 = tensor.expand_shape %1957 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1959 = tensor.empty() : tensor<1x2x96x32xf32>
    %1960 = linalg.transpose ins(%1958:tensor<1x96x2x32xf32>) outs(%1959:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1961 = tensor.empty() : tensor<1x2x32x6xf32>
    %1962 = linalg.transpose ins(%1941:tensor<1x2x6x32xf32>) outs(%1961:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1963 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1964 = tensor.splat %1963 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1960, %1962 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1964 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb146(%1966: f32, %1967: f32, %1968: f32):
      %1969 = arith.mulf %1966, %1967 : f32
      %1970 = arith.addf %1968, %1969 : f32
      linalg.yield %1970 : f32
    } -> tensor<1x2x96x6xf32>
    %1971 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1972 = tensor.splat %1971 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1973 = tensor.empty() : tensor<1x2x96x6xf32>
    %1974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1965, %1972 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1973 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb147(%1975: f32, %1976: f32, %1977: f32):
      %1978 = arith.divf %1975, %1976 : f32
      linalg.yield %1978 : f32
    } -> tensor<1x2x96x6xf32>
    %1979 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1980 = tensor.splat %1979 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1981 = linalg.reduce ins(%1974:tensor<1x2x96x6xf32>) outs(%1980:tensor<1x2x96xf32>) dimensions = [3]
    (%1982: f32, %1983: f32) {
      %1984 = arith.maximumf %1982, %1983 : f32
      linalg.yield %1984 : f32
    }
    %1985 = tensor.collapse_shape %1981 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1986 = tensor.expand_shape %1985 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1987 = tensor.empty() : tensor<1x2x96x6xf32>
    %1988 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1974, %1986 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1987 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb148(%1989: f32, %1990: f32, %1991: f32):
      %1992 = arith.subf %1989, %1990 : f32
      linalg.yield %1992 : f32
    } -> tensor<1x2x96x6xf32>
    %1993 = tensor.empty() : tensor<1x2x96x6xf32>
    %1994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1988 : tensor<1x2x96x6xf32>) outs(%1993 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb149(%1995: f32, %1996: f32):
      %1997 = math.exp %1995 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x2x96x6xf32>
    %1998 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1999 = tensor.splat %1998 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %2000 = linalg.reduce ins(%1994:tensor<1x2x96x6xf32>) outs(%1999:tensor<1x2x96xf32>) dimensions = [3]
    (%2001: f32, %2002: f32) {
      %2003 = arith.addf %2001, %2002 : f32
      linalg.yield %2003 : f32
    }
    %2004 = tensor.collapse_shape %2000 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %2005 = tensor.expand_shape %2004 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %2006 = tensor.empty() : tensor<1x2x96x6xf32>
    %2007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1994, %2005 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%2006 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb150(%2008: f32, %2009: f32, %2010: f32):
      %2011 = arith.divf %2008, %2009 : f32
      linalg.yield %2011 : f32
    } -> tensor<1x2x96x6xf32>
    %2012 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2013 = tensor.splat %2012 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %2014 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2007, %1944 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%2013 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb151(%2015: f32, %2016: f32, %2017: f32):
      %2018 = arith.mulf %2015, %2016 : f32
      %2019 = arith.addf %2017, %2018 : f32
      linalg.yield %2019 : f32
    } -> tensor<1x2x96x32xf32>
    %2020 = tensor.empty() : tensor<1x96x2x32xf32>
    %2021 = linalg.transpose ins(%2014:tensor<1x2x96x32xf32>) outs(%2020:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %2022 = tensor.collapse_shape %2021 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %2023 = tensor.expand_shape %2022 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2024 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0102703795 : f32
    %2025 = tensor.splat %2024 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2026 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2027 = tensor.splat %2026 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2028 = "quant_ext.quantize_per_tensor"(%2023, %2025, %2027) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_23", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %2029 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0102703795 : f32
    %2030 = tensor.splat %2029 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2031 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2032 = tensor.splat %2031 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2033 = "quant_ext.dequantize_per_tensor"(%2028, %2030, %2032) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_52", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %2034 = tensor.empty() : tensor<64x64xf32>
    %2035 = linalg.transpose ins(%212:tensor<64x64xf32>) outs(%2034:tensor<64x64xf32>) permutation = [1, 0]
    %2036 = tensor.empty() : tensor<1x96x64xf32>
    %2037 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2038 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2037 : f32) outs(%2036 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %2039 = linalg.matmul {prov.region_id = "matmul_25", prov.dispatch_id = "matmul_25", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} ins(%2033, %2035 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%2038 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %2040 = tensor.empty() : tensor<1x96x64xf32>
    %2041 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2039, %41 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2040 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb152(%2042: f32, %2043: f32, %2044: f32):
      %2045 = arith.addf %2042, %2043 : f32
      linalg.yield %2045 : f32
    } -> tensor<1x96x64xf32>
    %2046 = tensor.empty() : tensor<1x96x64xf32>
    %2047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1784, %2041 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2046 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb153(%2048: f32, %2049: f32, %2050: f32):
      %2051 = arith.addf %2048, %2049 : f32
      linalg.yield %2051 : f32
    } -> tensor<1x96x64xf32>
    %2052 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0287442114 : f32
    %2053 = tensor.splat %2052 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2054 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2055 = tensor.splat %2054 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2056 = "quant_ext.quantize_per_tensor"(%2047, %2053, %2055) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_24", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x64xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xi8>
    %2057 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0287442114 : f32
    %2058 = tensor.splat %2057 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2059 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2060 = tensor.splat %2059 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2061 = "quant_ext.dequantize_per_tensor"(%2056, %2058, %2060) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_53", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x64xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x64xf32>
    %2062 = tensor.empty() : tensor<64x512xf32>
    %2063 = linalg.transpose ins(%227:tensor<512x64xf32>) outs(%2062:tensor<64x512xf32>) permutation = [1, 0]
    %2064 = tensor.empty() : tensor<1x96x512xf32>
    %2065 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2066 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2065 : f32) outs(%2064 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %2067 = linalg.matmul {prov.region_id = "matmul_26", prov.dispatch_id = "matmul_26", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} ins(%2061, %2063 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%2066 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %2068 = tensor.empty() : tensor<1x96x512xf32>
    %2069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2067, %46 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%2068 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb154(%2070: f32, %2071: f32, %2072: f32):
      %2073 = arith.addf %2070, %2071 : f32
      linalg.yield %2073 : f32
    } -> tensor<1x96x512xf32>
    %2074 = tensor.empty() : tensor<1x512x96xf32>
    %2075 = linalg.transpose ins(%2069:tensor<1x96x512xf32>) outs(%2074:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %2076 = tensor.collapse_shape %2075 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %2077 = tensor.expand_shape %2076 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2078 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2079 = tensor.splat %2078 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %2080 = "tensor.insert_slice"(%2077, %2079) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %2081 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %2082 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2080 : tensor<1x512x10x14xf32>) outs(%2081 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb155(%2083: f32, %2084: f32):
      linalg.yield %2083 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %2085 = tensor.collapse_shape %2082 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %2086 = tensor.expand_shape %2085 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %2087 = tensor.collapse_shape %47 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %2088 = tensor.expand_shape %2087 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %2089 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2090 = tensor.splat %2089 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %2091 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2088, %2086 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%2090 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb156(%2092: f32, %2093: f32, %2094: f32):
      %2095 = arith.mulf %2092, %2093 : f32
      %2096 = arith.addf %2094, %2095 : f32
      linalg.yield %2096 : f32
    } -> tensor<64x8x96xf32>
    %2097 = tensor.collapse_shape %2091 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %2098 = tensor.expand_shape %2097 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %2099 = tensor.collapse_shape %2098 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %2100 = tensor.expand_shape %2099 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2101 = tensor.empty() : tensor<1x512x8x12xf32>
    %2102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2100, %48 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%2101 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb157(%2103: f32, %2104: f32, %2105: f32):
      %2106 = arith.addf %2103, %2104 : f32
      linalg.yield %2106 : f32
    } -> tensor<1x512x8x12xf32>
    %2107 = tensor.collapse_shape %2102 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %2108 = tensor.expand_shape %2107 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %2109 = tensor.empty() : tensor<1x96x512xf32>
    %2110 = linalg.transpose ins(%2108:tensor<1x512x96xf32>) outs(%2109:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %2111 = tensor.empty() : tensor<1x96x512xf32>
    %2112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2110 : tensor<1x96x512xf32>) outs(%2111 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb158(%2113: f32, %2114: f32):
      %2115 = arith.constant 5.000000e-01 : f32
      %2116 = arith.constant 1.000000e+00 : f32
      %2117 = arith.constant 0.707106769 : f32
      %2118 = arith.mulf %2113, %2117 : f32
      %2119 = math.erf %2118 : f32
      %2120 = arith.addf %2116, %2119 : f32
      %2121 = arith.mulf %2115, %2113 : f32
      %2122 = arith.mulf %2121, %2120 : f32
      linalg.yield %2122 : f32
    } -> tensor<1x96x512xf32>
    %2123 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.011980596 : f32
    %2124 = tensor.splat %2123 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2125 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2126 = tensor.splat %2125 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2127 = "quant_ext.quantize_per_tensor"(%2112, %2124, %2126) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_25", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x96x512xf32>, tensor<f32>, tensor<i64>) -> tensor<1x96x512xi8>
    %2128 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.011980596 : f32
    %2129 = tensor.splat %2128 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2130 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2131 = tensor.splat %2130 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2132 = "quant_ext.dequantize_per_tensor"(%2127, %2129, %2131) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_54", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x96x512xi8>, tensor<f32>, tensor<i64>) -> tensor<1x96x512xf32>
    %2133 = tensor.empty() : tensor<512x64xf32>
    %2134 = linalg.transpose ins(%232:tensor<64x512xf32>) outs(%2133:tensor<512x64xf32>) permutation = [1, 0]
    %2135 = tensor.empty() : tensor<1x96x64xf32>
    %2136 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2137 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2136 : f32) outs(%2135 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %2138 = linalg.matmul {prov.region_id = "matmul_27", prov.dispatch_id = "matmul_27", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} ins(%2132, %2134 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%2137 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %2139 = tensor.empty() : tensor<1x96x64xf32>
    %2140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2138, %49 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2139 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb159(%2141: f32, %2142: f32, %2143: f32):
      %2144 = arith.addf %2141, %2142 : f32
      linalg.yield %2144 : f32
    } -> tensor<1x96x64xf32>
    %2145 = tensor.empty() : tensor<1x96x64xf32>
    %2146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2047, %2140 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2145 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%2147: f32, %2148: f32, %2149: f32):
      %2150 = arith.addf %2147, %2148 : f32
      linalg.yield %2150 : f32
    } -> tensor<1x96x64xf32>
    %2151 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2152 = tensor.splat %2151 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2153 = linalg.reduce ins(%2146:tensor<1x96x64xf32>) outs(%2152:tensor<1x96xf32>) dimensions = [2]
    (%2154: f32, %2155: f32) {
      %2156 = arith.addf %2154, %2155 : f32
      linalg.yield %2156 : f32
    }
    %2157 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2158 = tensor.splat %2157 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2159 = tensor.empty() : tensor<1x96xf32>
    %2160 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2153, %2158 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2159 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%2161: f32, %2162: f32, %2163: f32):
      %2164 = arith.divf %2161, %2162 : f32
      linalg.yield %2164 : f32
    } -> tensor<1x96xf32>
    %2165 = tensor.collapse_shape %2160 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2166 = tensor.expand_shape %2165 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2167 = tensor.empty() : tensor<1x96x64xf32>
    %2168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2146, %2166 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2167 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb162(%2169: f32, %2170: f32, %2171: f32):
      %2172 = arith.subf %2169, %2170 : f32
      linalg.yield %2172 : f32
    } -> tensor<1x96x64xf32>
    %2173 = tensor.empty() : tensor<1x96x64xf32>
    %2174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2168, %2168 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2173 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb163(%2175: f32, %2176: f32, %2177: f32):
      %2178 = arith.mulf %2175, %2176 : f32
      linalg.yield %2178 : f32
    } -> tensor<1x96x64xf32>
    %2179 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2180 = tensor.splat %2179 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2181 = linalg.reduce ins(%2174:tensor<1x96x64xf32>) outs(%2180:tensor<1x96xf32>) dimensions = [2]
    (%2182: f32, %2183: f32) {
      %2184 = arith.addf %2182, %2183 : f32
      linalg.yield %2184 : f32
    }
    %2185 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2186 = tensor.splat %2185 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2187 = tensor.empty() : tensor<1x96xf32>
    %2188 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2181, %2186 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2187 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb164(%2189: f32, %2190: f32, %2191: f32):
      %2192 = arith.divf %2189, %2190 : f32
      linalg.yield %2192 : f32
    } -> tensor<1x96xf32>
    %2193 = tensor.collapse_shape %2188 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2194 = tensor.expand_shape %2193 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2195 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %2196 = tensor.splat %2195 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %2197 = tensor.empty() : tensor<1x96x1xf32>
    %2198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2194, %2196 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%2197 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb165(%2199: f32, %2200: f32, %2201: f32):
      %2202 = arith.addf %2199, %2200 : f32
      linalg.yield %2202 : f32
    } -> tensor<1x96x1xf32>
    %2203 = tensor.empty() : tensor<1x96x1xf32>
    %2204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2198 : tensor<1x96x1xf32>) outs(%2203 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb166(%2205: f32, %2206: f32):
      %2207 = math.rsqrt %2205 : f32
      linalg.yield %2207 : f32
    } -> tensor<1x96x1xf32>
    %2208 = tensor.empty() : tensor<1x96x64xf32>
    %2209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2168, %2204 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2208 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb167(%2210: f32, %2211: f32, %2212: f32):
      %2213 = arith.mulf %2210, %2211 : f32
      linalg.yield %2213 : f32
    } -> tensor<1x96x64xf32>
    %2214 = tensor.empty() : tensor<1x96x64xf32>
    %2215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2209, %52 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2214 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb168(%2216: f32, %2217: f32, %2218: f32):
      %2219 = arith.mulf %2216, %2217 : f32
      linalg.yield %2219 : f32
    } -> tensor<1x96x64xf32>
    %2220 = tensor.empty() : tensor<1x96x64xf32>
    %2221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2215, %53 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2220 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb169(%2222: f32, %2223: f32, %2224: f32):
      %2225 = arith.addf %2222, %2223 : f32
      linalg.yield %2225 : f32
    } -> tensor<1x96x64xf32>
    %2226 = tensor.collapse_shape %2221 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2227 = tensor.expand_shape %2226 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2228 = tensor.empty() : tensor<1x64x8x12xf32>
    %2229 = linalg.transpose ins(%2227:tensor<1x8x12x64xf32>) outs(%2228:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2230 = tensor.collapse_shape %2229 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2231 = tensor.expand_shape %2230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2232 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2233 = linalg.transpose ins(%2231:tensor<1x16x2x2x8x12xf32>) outs(%2232:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2234 = tensor.collapse_shape %2233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2235 = tensor.expand_shape %2234 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2236 = tensor.empty() : tensor<1x32x23x15xf32>
    %2237 = linalg.transpose ins(%1238:tensor<1x32x15x23xf32>) outs(%2236:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2238 = tensor.collapse_shape %2237 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2239 = tensor.expand_shape %2238 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2240 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2241 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2242 = tensor.splat %2241 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2243 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2239, %2240 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2242 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2244 = tensor.collapse_shape %2243 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2245 = tensor.expand_shape %2244 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2246 = tensor.empty() : tensor<1x32x16x23xf32>
    %2247 = linalg.transpose ins(%2245:tensor<1x32x23x16xf32>) outs(%2246:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2248 = tensor.collapse_shape %2247 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2249 = tensor.expand_shape %2248 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2250 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2251 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2252 = tensor.splat %2251 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2253 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2249, %2250 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2252 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2254 = tensor.collapse_shape %2253 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2255 = tensor.expand_shape %2254 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2256 = tensor.concat dim(1) %2235, %2255 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2257 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0274245348 : f32
    %2258 = tensor.splat %2257 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2259 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2260 = tensor.splat %2259 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2261 = "quant_ext.quantize_per_tensor"(%2256, %2258, %2260) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_26", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x48x16x24xf32>, tensor<f32>, tensor<i64>) -> tensor<1x48x16x24xi8>
    %2262 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0274245348 : f32
    %2263 = tensor.splat %2262 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2264 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2265 = tensor.splat %2264 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2266 = "quant_ext.dequantize_per_tensor"(%2261, %2263, %2265) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_55", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x48x16x24xi8>, tensor<f32>, tensor<i64>) -> tensor<1x48x16x24xf32>
    %2267 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2268 = tensor.splat %2267 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2269 = "tensor.insert_slice"(%2266, %2268) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2270 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2269 : tensor<1x48x18x26xf32>) outs(%2270 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb170(%2272: f32, %2273: f32):
      linalg.yield %2272 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2274 = tensor.collapse_shape %2271 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2275 = tensor.expand_shape %2274 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2276 = tensor.collapse_shape %247 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2277 = tensor.expand_shape %2276 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2278 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2279 = tensor.splat %2278 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2280 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2277, %2275 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2279 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2281 = tensor.collapse_shape %2280 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2282 = tensor.expand_shape %2281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2283 = tensor.collapse_shape %2282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2284 = tensor.expand_shape %2283 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2285 = tensor.empty() : tensor<1x12x16x24xf32>
    %2286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2284, %68 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2285 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb171(%2287: f32, %2288: f32, %2289: f32):
      %2290 = arith.addf %2287, %2288 : f32
      linalg.yield %2290 : f32
    } -> tensor<1x12x16x24xf32>
    %2291 = tensor.collapse_shape %2286 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2292 = tensor.expand_shape %2291 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2293 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0141646797 : f32
    %2294 = tensor.splat %2293 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2295 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2296 = tensor.splat %2295 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2297 = "quant_ext.quantize_per_tensor"(%2292, %2294, %2296) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_27", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x4608xf32>, tensor<f32>, tensor<i64>) -> tensor<1x4608xi8>
    %2298 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0141646797 : f32
    %2299 = tensor.splat %2298 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2300 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2301 = tensor.splat %2300 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2302 = "quant_ext.dequantize_per_tensor"(%2297, %2299, %2301) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_56", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x4608xi8>, tensor<f32>, tensor<i64>) -> tensor<1x4608xf32>
    %2303 = tensor.empty() : tensor<4608x512xf32>
    %2304 = linalg.transpose ins(%237:tensor<512x4608xf32>) outs(%2303:tensor<4608x512xf32>) permutation = [1, 0]
    %2305 = tensor.empty() : tensor<1x512xf32>
    %2306 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2307 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2306 : f32) outs(%2305 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2308 = linalg.matmul {prov.region_id = "matmul_28", prov.dispatch_id = "matmul_28", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%2302, %2304 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2307 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2309 = tensor.empty() : tensor<1x512xf32>
    %2310 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2308, %54 : tensor<1x512xf32>, tensor<512xf32>) outs(%2309 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb172(%2311: f32, %2312: f32, %2313: f32):
      %2314 = arith.addf %2311, %2312 : f32
      linalg.yield %2314 : f32
    } -> tensor<1x512xf32>
    %2315 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2316 = tensor.splat %2315 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2317 = tensor.empty() : tensor<1x1xf32>
    %2318 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %2316 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2317 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb173(%2319: f32, %2320: f32, %2321: f32):
      %2322 = arith.divf %2319, %2320 : f32
      linalg.yield %2322 : f32
    } -> tensor<1x1xf32>
    %2323 = tensor.concat dim(1) %2310, %2318, %100 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2324 = tensor.collapse_shape %2323 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2325 = tensor.expand_shape %2324 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2326 = tensor.collapse_shape %101 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2327 = tensor.expand_shape %2326 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2328 = tensor.collapse_shape %102 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2329 = tensor.expand_shape %2328 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2330 = "tensor.extract_slice"(%2325) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 517>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x517xf32>) -> tensor<1x1x517xf32>
    %2331 = tensor.collapse_shape %2330 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2332 = tensor.expand_shape %2331 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2333 = "tensor.extract_slice"(%2327) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2334 = tensor.collapse_shape %2333 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2335 = tensor.expand_shape %2334 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2336 = "tensor.extract_slice"(%2329) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2337 = tensor.collapse_shape %2336 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2338 = tensor.expand_shape %2337 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2339 = tensor.empty() : tensor<517x512xf32>
    %2340 = linalg.transpose ins(%55:tensor<512x517xf32>) outs(%2339:tensor<517x512xf32>) permutation = [1, 0]
    %2341 = tensor.empty() : tensor<1x512xf32>
    %2342 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2343 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2342 : f32) outs(%2341 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2344 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2332, %2340 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2343 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2345 = tensor.empty() : tensor<128x512xf32>
    %2346 = linalg.transpose ins(%56:tensor<512x128xf32>) outs(%2345:tensor<128x512xf32>) permutation = [1, 0]
    %2347 = tensor.empty() : tensor<1x512xf32>
    %2348 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2349 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2348 : f32) outs(%2347 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2350 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2335, %2346 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2349 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2351 = tensor.empty() : tensor<1x512xf32>
    %2352 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2344, %2350, %57, %58 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2351 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb174(%2353: f32, %2354: f32, %2355: f32, %2356: f32, %2357: f32):
      %2358 = arith.addf %2353, %2354 : f32
      %2359 = arith.addf %2358, %2355 : f32
      %2360 = arith.addf %2359, %2356 : f32
      linalg.yield %2360 : f32
    } -> tensor<1x512xf32>
    %2361 = "tensor.extract_slice"(%2352) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2362 = "tensor.extract_slice"(%2352) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2363 = "tensor.extract_slice"(%2352) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2364 = "tensor.extract_slice"(%2352) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2365 = tensor.empty() : tensor<1x128xf32>
    %2366 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2361, %2362, %2363, %2338 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2365 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb175(%2367: f32, %2368: f32, %2369: f32, %2370: f32, %2371: f32):
      %2372 = arith.constant 1.000000e+00 : f32
      %2373 = arith.negf %2368 : f32
      %2374 = math.exp %2373 : f32
      %2375 = arith.addf %2372, %2374 : f32
      %2376 = arith.divf %2372, %2375 : f32
      %2377 = arith.constant 1.000000e+00 : f32
      %2378 = arith.negf %2367 : f32
      %2379 = math.exp %2378 : f32
      %2380 = arith.addf %2377, %2379 : f32
      %2381 = arith.divf %2377, %2380 : f32
      %2382 = math.tanh %2369 : f32
      %2383 = arith.mulf %2376, %2370 : f32
      %2384 = arith.mulf %2381, %2382 : f32
      %2385 = arith.addf %2383, %2384 : f32
      linalg.yield %2385 : f32
    } -> tensor<1x128xf32>
    %2386 = tensor.empty() : tensor<1x128xf32>
    %2387 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2364, %2366 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2386 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb176(%2388: f32, %2389: f32, %2390: f32):
      %2391 = arith.constant 1.000000e+00 : f32
      %2392 = arith.negf %2388 : f32
      %2393 = math.exp %2392 : f32
      %2394 = arith.addf %2391, %2393 : f32
      %2395 = arith.divf %2391, %2394 : f32
      %2396 = math.tanh %2389 : f32
      %2397 = arith.mulf %2395, %2396 : f32
      linalg.yield %2397 : f32
    } -> tensor<1x128xf32>
    %2398 = "tensor.extract_slice"(%2327) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2399 = tensor.collapse_shape %2398 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2400 = tensor.expand_shape %2399 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2401 = "tensor.extract_slice"(%2329) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2402 = tensor.collapse_shape %2401 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2403 = tensor.expand_shape %2402 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2404 = tensor.empty() : tensor<128x512xf32>
    %2405 = linalg.transpose ins(%59:tensor<512x128xf32>) outs(%2404:tensor<128x512xf32>) permutation = [1, 0]
    %2406 = tensor.empty() : tensor<1x512xf32>
    %2407 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2408 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2407 : f32) outs(%2406 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2409 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2387, %2405 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2408 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2410 = tensor.empty() : tensor<128x512xf32>
    %2411 = linalg.transpose ins(%60:tensor<512x128xf32>) outs(%2410:tensor<128x512xf32>) permutation = [1, 0]
    %2412 = tensor.empty() : tensor<1x512xf32>
    %2413 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2414 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2413 : f32) outs(%2412 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2415 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2400, %2411 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2414 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2416 = tensor.empty() : tensor<1x512xf32>
    %2417 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2409, %2415, %61, %62 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2416 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb177(%2418: f32, %2419: f32, %2420: f32, %2421: f32, %2422: f32):
      %2423 = arith.addf %2418, %2419 : f32
      %2424 = arith.addf %2423, %2420 : f32
      %2425 = arith.addf %2424, %2421 : f32
      linalg.yield %2425 : f32
    } -> tensor<1x512xf32>
    %2426 = "tensor.extract_slice"(%2417) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2427 = "tensor.extract_slice"(%2417) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2428 = "tensor.extract_slice"(%2417) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2429 = "tensor.extract_slice"(%2417) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2430 = tensor.empty() : tensor<1x128xf32>
    %2431 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2426, %2427, %2428, %2403 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2430 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb178(%2432: f32, %2433: f32, %2434: f32, %2435: f32, %2436: f32):
      %2437 = arith.constant 1.000000e+00 : f32
      %2438 = arith.negf %2433 : f32
      %2439 = math.exp %2438 : f32
      %2440 = arith.addf %2437, %2439 : f32
      %2441 = arith.divf %2437, %2440 : f32
      %2442 = arith.constant 1.000000e+00 : f32
      %2443 = arith.negf %2432 : f32
      %2444 = math.exp %2443 : f32
      %2445 = arith.addf %2442, %2444 : f32
      %2446 = arith.divf %2442, %2445 : f32
      %2447 = math.tanh %2434 : f32
      %2448 = arith.mulf %2441, %2435 : f32
      %2449 = arith.mulf %2446, %2447 : f32
      %2450 = arith.addf %2448, %2449 : f32
      linalg.yield %2450 : f32
    } -> tensor<1x128xf32>
    %2451 = tensor.empty() : tensor<1x128xf32>
    %2452 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2429, %2431 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2451 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb179(%2453: f32, %2454: f32, %2455: f32):
      %2456 = arith.constant 1.000000e+00 : f32
      %2457 = arith.negf %2453 : f32
      %2458 = math.exp %2457 : f32
      %2459 = arith.addf %2456, %2458 : f32
      %2460 = arith.divf %2456, %2459 : f32
      %2461 = math.tanh %2454 : f32
      %2462 = arith.mulf %2460, %2461 : f32
      linalg.yield %2462 : f32
    } -> tensor<1x128xf32>
    %2463 = "tensor.extract_slice"(%2327) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2464 = tensor.collapse_shape %2463 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2465 = tensor.expand_shape %2464 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2466 = "tensor.extract_slice"(%2329) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2467 = tensor.collapse_shape %2466 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2468 = tensor.expand_shape %2467 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2469 = tensor.empty() : tensor<128x512xf32>
    %2470 = linalg.transpose ins(%63:tensor<512x128xf32>) outs(%2469:tensor<128x512xf32>) permutation = [1, 0]
    %2471 = tensor.empty() : tensor<1x512xf32>
    %2472 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2473 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2472 : f32) outs(%2471 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2474 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2452, %2470 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2473 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2475 = tensor.empty() : tensor<128x512xf32>
    %2476 = linalg.transpose ins(%64:tensor<512x128xf32>) outs(%2475:tensor<128x512xf32>) permutation = [1, 0]
    %2477 = tensor.empty() : tensor<1x512xf32>
    %2478 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2479 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2478 : f32) outs(%2477 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2480 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2465, %2476 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2479 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2481 = tensor.empty() : tensor<1x512xf32>
    %2482 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2474, %2480, %65, %66 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2481 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb180(%2483: f32, %2484: f32, %2485: f32, %2486: f32, %2487: f32):
      %2488 = arith.addf %2483, %2484 : f32
      %2489 = arith.addf %2488, %2485 : f32
      %2490 = arith.addf %2489, %2486 : f32
      linalg.yield %2490 : f32
    } -> tensor<1x512xf32>
    %2491 = "tensor.extract_slice"(%2482) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2492 = "tensor.extract_slice"(%2482) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2493 = "tensor.extract_slice"(%2482) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2494 = "tensor.extract_slice"(%2482) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2495 = tensor.empty() : tensor<1x128xf32>
    %2496 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2491, %2492, %2493, %2468 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2495 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb181(%2497: f32, %2498: f32, %2499: f32, %2500: f32, %2501: f32):
      %2502 = arith.constant 1.000000e+00 : f32
      %2503 = arith.negf %2498 : f32
      %2504 = math.exp %2503 : f32
      %2505 = arith.addf %2502, %2504 : f32
      %2506 = arith.divf %2502, %2505 : f32
      %2507 = arith.constant 1.000000e+00 : f32
      %2508 = arith.negf %2497 : f32
      %2509 = math.exp %2508 : f32
      %2510 = arith.addf %2507, %2509 : f32
      %2511 = arith.divf %2507, %2510 : f32
      %2512 = math.tanh %2499 : f32
      %2513 = arith.mulf %2506, %2500 : f32
      %2514 = arith.mulf %2511, %2512 : f32
      %2515 = arith.addf %2513, %2514 : f32
      linalg.yield %2515 : f32
    } -> tensor<1x128xf32>
    %2516 = tensor.empty() : tensor<1x128xf32>
    %2517 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2494, %2496 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2516 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb182(%2518: f32, %2519: f32, %2520: f32):
      %2521 = arith.constant 1.000000e+00 : f32
      %2522 = arith.negf %2518 : f32
      %2523 = math.exp %2522 : f32
      %2524 = arith.addf %2521, %2523 : f32
      %2525 = arith.divf %2521, %2524 : f32
      %2526 = math.tanh %2519 : f32
      %2527 = arith.mulf %2525, %2526 : f32
      linalg.yield %2527 : f32
    } -> tensor<1x128xf32>
    %2528 = tensor.empty() : tensor<1x1x128xf32>
    %2529 = tensor.collapse_shape %2517 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2530 = tensor.expand_shape %2529 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2531 = "tensor.insert_slice"(%2530, %2528) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2532 = tensor.empty() : tensor<3x1x128xf32>
    %2533 = tensor.collapse_shape %2387 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2534 = tensor.expand_shape %2533 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2535 = "tensor.insert_slice"(%2534, %2532) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2536 = tensor.collapse_shape %2452 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2537 = tensor.expand_shape %2536 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2538 = "tensor.insert_slice"(%2537, %2535) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2539 = tensor.collapse_shape %2517 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2540 = tensor.expand_shape %2539 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2541 = "tensor.insert_slice"(%2540, %2538) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2542 = tensor.empty() : tensor<3x1x128xf32>
    %2543 = tensor.collapse_shape %2366 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2544 = tensor.expand_shape %2543 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2545 = "tensor.insert_slice"(%2544, %2542) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2546 = tensor.collapse_shape %2431 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2547 = tensor.expand_shape %2546 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2548 = "tensor.insert_slice"(%2547, %2545) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2549 = tensor.collapse_shape %2496 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2550 = tensor.expand_shape %2549 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2551 = "tensor.insert_slice"(%2550, %2548) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2552 = tensor.collapse_shape %2531 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2553 = tensor.expand_shape %2552 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2554 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.000373565505 : f32
    %2555 = tensor.splat %2554 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %2556 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %2557 = tensor.splat %2556 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %2558 = "quant_ext.quantize_per_tensor"(%2553, %2555, %2557) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_28", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x128xf32>, tensor<f32>, tensor<i64>) -> tensor<1x128xi8>
    %2559 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.000373565505 : f32
    %2560 = tensor.splat %2559 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %2561 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %2562 = tensor.splat %2561 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %2563 = "quant_ext.dequantize_per_tensor"(%2558, %2560, %2562) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_57", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x128xi8>, tensor<f32>, tensor<i64>) -> tensor<1x128xf32>
    %2564 = tensor.collapse_shape %2541 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2565 = tensor.expand_shape %2564 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2566 = tensor.collapse_shape %2551 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2567 = tensor.expand_shape %2566 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2568 = tensor.empty() : tensor<128x3xf32>
    %2569 = linalg.transpose ins(%242:tensor<3x128xf32>) outs(%2568:tensor<128x3xf32>) permutation = [1, 0]
    %2570 = tensor.empty() : tensor<1x3xf32>
    %2571 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2572 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2571 : f32) outs(%2570 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2573 = linalg.matmul {prov.region_id = "matmul_29", prov.dispatch_id = "matmul_29", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2563, %2569 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2572 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2574 = tensor.empty() : tensor<1x3xf32>
    %2575 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2573, %67 : tensor<1x3xf32>, tensor<3xf32>) outs(%2574 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_29", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb183(%2576: f32, %2577: f32, %2578: f32):
      %2579 = arith.addf %2576, %2577 : f32
      linalg.yield %2579 : f32
    } -> tensor<1x3xf32>
    func.return %2575, %2565, %2567 : tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>
  }
}
