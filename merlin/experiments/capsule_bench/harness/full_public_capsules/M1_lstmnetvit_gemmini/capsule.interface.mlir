builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64xf32>, %9: tensor<64x32xi8>, %10: tensor<64xf32>, %11: tensor<64xi64>, %12: tensor<32xf32>, %13: tensor<32x32xi8>, %14: tensor<32xf32>, %15: tensor<32xi64>, %16: tensor<32xf32>, %17: tensor<32x32xi8>, %18: tensor<32xf32>, %19: tensor<32xi64>, %20: tensor<32x32x8x8xf32>, %21: tensor<32xf32>, %22: tensor<32xf32>, %23: tensor<32xf32>, %24: tensor<64xf32>, %25: tensor<64x32xi8>, %26: tensor<64xf32>, %27: tensor<64xi64>, %28: tensor<32xf32>, %29: tensor<32x32xi8>, %30: tensor<32xf32>, %31: tensor<32xi64>, %32: tensor<32xf32>, %33: tensor<32x32xi8>, %34: tensor<32xf32>, %35: tensor<32xi64>, %36: tensor<256xf32>, %37: tensor<256x32xi8>, %38: tensor<256xf32>, %39: tensor<256xi64>, %40: tensor<256x8x3x3xf32>, %41: tensor<256xf32>, %42: tensor<32xf32>, %43: tensor<32x256xi8>, %44: tensor<32xf32>, %45: tensor<32xi64>, %46: tensor<256xf32>, %47: tensor<256x32xi8>, %48: tensor<256xf32>, %49: tensor<256xi64>, %50: tensor<256x8x3x3xf32>, %51: tensor<256xf32>, %52: tensor<32xf32>, %53: tensor<32x256xi8>, %54: tensor<32xf32>, %55: tensor<32xi64>, %56: tensor<32xf32>, %57: tensor<32xf32>, %58: tensor<32xf32>, %59: tensor<32xf32>, %60: tensor<64x32x3x3xf32>, %61: tensor<64xf32>, %62: tensor<64xf32>, %63: tensor<64xf32>, %64: tensor<64x64x4x4xf32>, %65: tensor<64xf32>, %66: tensor<64xf32>, %67: tensor<64xf32>, %68: tensor<128xf32>, %69: tensor<128x64xi8>, %70: tensor<128xf32>, %71: tensor<128xi64>, %72: tensor<64xf32>, %73: tensor<64x64xi8>, %74: tensor<64xf32>, %75: tensor<64xi64>, %76: tensor<64xf32>, %77: tensor<64x64xi8>, %78: tensor<64xf32>, %79: tensor<64xi64>, %80: tensor<64x64x4x4xf32>, %81: tensor<64xf32>, %82: tensor<64xf32>, %83: tensor<64xf32>, %84: tensor<128xf32>, %85: tensor<128x64xi8>, %86: tensor<128xf32>, %87: tensor<128xi64>, %88: tensor<64xf32>, %89: tensor<64x64xi8>, %90: tensor<64xf32>, %91: tensor<64xi64>, %92: tensor<64xf32>, %93: tensor<64x64xi8>, %94: tensor<64xf32>, %95: tensor<64xi64>, %96: tensor<512xf32>, %97: tensor<512x64xi8>, %98: tensor<512xf32>, %99: tensor<512xi64>, %100: tensor<512x8x3x3xf32>, %101: tensor<512xf32>, %102: tensor<64xf32>, %103: tensor<64x512xi8>, %104: tensor<64xf32>, %105: tensor<64xi64>, %106: tensor<512xf32>, %107: tensor<512x64xi8>, %108: tensor<512xf32>, %109: tensor<512xi64>, %110: tensor<512x8x3x3xf32>, %111: tensor<512xf32>, %112: tensor<64xf32>, %113: tensor<64x512xi8>, %114: tensor<64xf32>, %115: tensor<64xi64>, %116: tensor<64xf32>, %117: tensor<64xf32>, %118: tensor<64xf32>, %119: tensor<64xf32>, %120: tensor<512xf32>, %121: tensor<512x4608xi8>, %122: tensor<512xf32>, %123: tensor<512xi64>, %124: tensor<512x517xf32>, %125: tensor<512x128xf32>, %126: tensor<512xf32>, %127: tensor<512xf32>, %128: tensor<512x128xf32>, %129: tensor<512x128xf32>, %130: tensor<512xf32>, %131: tensor<512xf32>, %132: tensor<512x128xf32>, %133: tensor<512x128xf32>, %134: tensor<512xf32>, %135: tensor<512xf32>, %136: tensor<3xf32>, %137: tensor<3x128xi8>, %138: tensor<3xf32>, %139: tensor<3xi64>, %140: tensor<12x48x3x3xf32>, %141: tensor<12xf32>, %142: tensor<1x1x60x90xf32>, %143: tensor<1x1xf32>, %144: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %145 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %146 = tensor.splat %145 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %147 = "tensor.insert_slice"(%142, %146) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %148 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%147 : tensor<1x1x66x96xf32>) outs(%148 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%150: f32, %151: f32):
      linalg.yield %150 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %152 = tensor.collapse_shape %149 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %153 = tensor.expand_shape %152 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %154 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %155 = tensor.expand_shape %154 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %156 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %157 = tensor.splat %156 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %158 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%155, %153 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%157 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %159 = tensor.collapse_shape %158 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %160 = tensor.expand_shape %159 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %161 = tensor.collapse_shape %160 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %162 = tensor.expand_shape %161 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %163 = tensor.empty() : tensor<1x32x15x23xf32>
    %164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%163 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%165: f32, %166: f32, %167: f32):
      %168 = arith.addf %165, %166 : f32
      linalg.yield %168 : f32
    } -> tensor<1x32x15x23xf32>
    %169 = tensor.collapse_shape %164 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %170 = tensor.expand_shape %169 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %171 = tensor.empty() : tensor<1x345x32xf32>
    %172 = linalg.transpose ins(%170:tensor<1x32x345xf32>) outs(%171:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %173 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %174 = tensor.splat %173 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %175 = linalg.reduce ins(%172:tensor<1x345x32xf32>) outs(%174:tensor<1x345xf32>) dimensions = [2]
    (%176: f32, %177: f32) {
      %178 = arith.addf %176, %177 : f32
      linalg.yield %178 : f32
    }
    %179 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %180 = tensor.splat %179 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %181 = tensor.empty() : tensor<1x345xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%175, %180 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%181 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%183: f32, %184: f32, %185: f32):
      %186 = arith.divf %183, %184 : f32
      linalg.yield %186 : f32
    } -> tensor<1x345xf32>
    %187 = tensor.collapse_shape %182 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %188 = tensor.expand_shape %187 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %189 = tensor.empty() : tensor<1x345x32xf32>
    %190 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %188 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%189 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%191: f32, %192: f32, %193: f32):
      %194 = arith.subf %191, %192 : f32
      linalg.yield %194 : f32
    } -> tensor<1x345x32xf32>
    %195 = tensor.empty() : tensor<1x345x32xf32>
    %196 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %190 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%195 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%197: f32, %198: f32, %199: f32):
      %200 = arith.mulf %197, %198 : f32
      linalg.yield %200 : f32
    } -> tensor<1x345x32xf32>
    %201 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %202 = tensor.splat %201 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %203 = linalg.reduce ins(%196:tensor<1x345x32xf32>) outs(%202:tensor<1x345xf32>) dimensions = [2]
    (%204: f32, %205: f32) {
      %206 = arith.addf %204, %205 : f32
      linalg.yield %206 : f32
    }
    %207 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %208 = tensor.splat %207 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %209 = tensor.empty() : tensor<1x345xf32>
    %210 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%203, %208 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%209 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%211: f32, %212: f32, %213: f32):
      %214 = arith.divf %211, %212 : f32
      linalg.yield %214 : f32
    } -> tensor<1x345xf32>
    %215 = tensor.collapse_shape %210 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %216 = tensor.expand_shape %215 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %217 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %218 = tensor.splat %217 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %219 = tensor.empty() : tensor<1x345x1xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%216, %218 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%219 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%221: f32, %222: f32, %223: f32):
      %224 = arith.addf %221, %222 : f32
      linalg.yield %224 : f32
    } -> tensor<1x345x1xf32>
    %225 = tensor.empty() : tensor<1x345x1xf32>
    %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%220 : tensor<1x345x1xf32>) outs(%225 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%227: f32, %228: f32):
      %229 = math.rsqrt %227 : f32
      linalg.yield %229 : f32
    } -> tensor<1x345x1xf32>
    %230 = tensor.empty() : tensor<1x345x32xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %226 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%230 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%232: f32, %233: f32, %234: f32):
      %235 = arith.mulf %232, %233 : f32
      linalg.yield %235 : f32
    } -> tensor<1x345x32xf32>
    %236 = tensor.empty() : tensor<1x345x32xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%236 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%238: f32, %239: f32, %240: f32):
      %241 = arith.mulf %238, %239 : f32
      linalg.yield %241 : f32
    } -> tensor<1x345x32xf32>
    %242 = tensor.empty() : tensor<1x345x32xf32>
    %243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%237, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%242 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%244: f32, %245: f32, %246: f32):
      %247 = arith.addf %244, %245 : f32
      linalg.yield %247 : f32
    } -> tensor<1x345x32xf32>
    %248 = tensor.empty() : tensor<1x32x345xf32>
    %249 = linalg.transpose ins(%243:tensor<1x345x32xf32>) outs(%248:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %250 = tensor.collapse_shape %249 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %251 = tensor.expand_shape %250 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %252 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%251 : tensor<1x32x15x23xf32>) outs(%252 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%254: f32, %255: f32):
      linalg.yield %254 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %256 = tensor.collapse_shape %253 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %257 = tensor.expand_shape %256 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %258 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %259 = tensor.expand_shape %258 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %260 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %261 = tensor.splat %260 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %262 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%259, %257 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%261 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %263 = tensor.collapse_shape %262 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %264 = tensor.expand_shape %263 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %265 = tensor.collapse_shape %264 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %266 = tensor.expand_shape %265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %267 = tensor.empty() : tensor<1x32x1x2xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%266, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%267 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%269: f32, %270: f32, %271: f32):
      %272 = arith.addf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<1x32x1x2xf32>
    %273 = tensor.collapse_shape %268 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %274 = tensor.expand_shape %273 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %275 = tensor.empty() : tensor<1x2x32xf32>
    %276 = linalg.transpose ins(%274:tensor<1x32x2xf32>) outs(%275:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %277 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %278 = tensor.splat %277 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %279 = linalg.reduce ins(%276:tensor<1x2x32xf32>) outs(%278:tensor<1x2xf32>) dimensions = [2]
    (%280: f32, %281: f32) {
      %282 = arith.addf %280, %281 : f32
      linalg.yield %282 : f32
    }
    %283 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %284 = tensor.splat %283 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %285 = tensor.empty() : tensor<1x2xf32>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%279, %284 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%285 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%287: f32, %288: f32, %289: f32):
      %290 = arith.divf %287, %288 : f32
      linalg.yield %290 : f32
    } -> tensor<1x2xf32>
    %291 = tensor.collapse_shape %286 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %292 = tensor.expand_shape %291 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %293 = tensor.empty() : tensor<1x2x32xf32>
    %294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%276, %292 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%293 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%295: f32, %296: f32, %297: f32):
      %298 = arith.subf %295, %296 : f32
      linalg.yield %298 : f32
    } -> tensor<1x2x32xf32>
    %299 = tensor.empty() : tensor<1x2x32xf32>
    %300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%294, %294 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%299 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%301: f32, %302: f32, %303: f32):
      %304 = arith.mulf %301, %302 : f32
      linalg.yield %304 : f32
    } -> tensor<1x2x32xf32>
    %305 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %306 = tensor.splat %305 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %307 = linalg.reduce ins(%300:tensor<1x2x32xf32>) outs(%306:tensor<1x2xf32>) dimensions = [2]
    (%308: f32, %309: f32) {
      %310 = arith.addf %308, %309 : f32
      linalg.yield %310 : f32
    }
    %311 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %312 = tensor.splat %311 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %313 = tensor.empty() : tensor<1x2xf32>
    %314 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%307, %312 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%313 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%315: f32, %316: f32, %317: f32):
      %318 = arith.divf %315, %316 : f32
      linalg.yield %318 : f32
    } -> tensor<1x2xf32>
    %319 = tensor.collapse_shape %314 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %320 = tensor.expand_shape %319 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %321 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %322 = tensor.splat %321 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %323 = tensor.empty() : tensor<1x2x1xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%320, %322 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%323 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%325: f32, %326: f32, %327: f32):
      %328 = arith.addf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x2x1xf32>
    %329 = tensor.empty() : tensor<1x2x1xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%324 : tensor<1x2x1xf32>) outs(%329 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%331: f32, %332: f32):
      %333 = math.rsqrt %331 : f32
      linalg.yield %333 : f32
    } -> tensor<1x2x1xf32>
    %334 = tensor.empty() : tensor<1x2x32xf32>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%294, %330 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%334 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%336: f32, %337: f32, %338: f32):
      %339 = arith.mulf %336, %337 : f32
      linalg.yield %339 : f32
    } -> tensor<1x2x32xf32>
    %340 = tensor.empty() : tensor<1x2x32xf32>
    %341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%335, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%340 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%342: f32, %343: f32, %344: f32):
      %345 = arith.mulf %342, %343 : f32
      linalg.yield %345 : f32
    } -> tensor<1x2x32xf32>
    %346 = tensor.empty() : tensor<1x2x32xf32>
    %347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%341, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%346 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%348: f32, %349: f32, %350: f32):
      %351 = arith.addf %348, %349 : f32
      linalg.yield %351 : f32
    } -> tensor<1x2x32xf32>
    %352 = tensor.empty() : tensor<32x64xi8>
    %353 = linalg.transpose ins(%9:tensor<64x32xi8>) outs(%352:tensor<32x64xi8>) permutation = [1, 0]
    %354 = tensor.collapse_shape %347 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %355 = tensor.expand_shape %354 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %356 = tensor.empty() : tensor<32x64xf32>
    %357 = arith.constant 0 : i32
    %358 = tensor.splat %357 : tensor<64xi32>
    %359 = "quant_ext.dequantize_per_channel"(%353, %10, %358) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<32x64xf32>
    %360 = tensor.empty() : tensor<2x64xf32>
    %361 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %362 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%361 : f32) outs(%360 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %363 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} ins(%355, %359 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%362 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %364 = tensor.empty() : tensor<2x64xf32>
    %365 = tensor.collapse_shape %363 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %366 = tensor.expand_shape %365 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %367 = tensor.empty() : tensor<1x2x64xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%366, %8 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%367 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%369: f32, %370: f32, %371: f32):
      %372 = arith.addf %369, %370 : f32
      linalg.yield %372 : f32
    } -> tensor<1x2x64xf32>
    %373 = tensor.collapse_shape %368 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x64xf32> into tensor<128xf32>
    %374 = tensor.expand_shape %373 [[0 : i64, 1 : i64]] output_shape [2, 64] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<128xf32> into tensor<2x64xf32>
    %375 = tensor.collapse_shape %374 [[0 : i64, 1 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<2x64xf32> into tensor<128xf32>
    %376 = tensor.expand_shape %375 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x64xf32>
    %377 = tensor.collapse_shape %376 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %378 = tensor.expand_shape %377 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %379 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %380 = linalg.transpose ins(%378:tensor<1x2x2x1x32xf32>) outs(%379:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %381 = "tensor.extract_slice"(%380) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %382 = tensor.collapse_shape %381 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %383 = tensor.expand_shape %382 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %384 = "tensor.extract_slice"(%380) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %385 = tensor.collapse_shape %384 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %386 = tensor.expand_shape %385 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %387 = tensor.empty() : tensor<32x32xi8>
    %388 = linalg.transpose ins(%13:tensor<32x32xi8>) outs(%387:tensor<32x32xi8>) permutation = [1, 0]
    %389 = tensor.collapse_shape %243 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %390 = tensor.expand_shape %389 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %391 = tensor.empty() : tensor<32x32xf32>
    %392 = arith.constant 0 : i32
    %393 = tensor.splat %392 : tensor<32xi32>
    %394 = "quant_ext.dequantize_per_channel"(%388, %14, %393) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %395 = tensor.empty() : tensor<345x32xf32>
    %396 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %397 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%396 : f32) outs(%395 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %398 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} ins(%390, %394 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%397 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %399 = tensor.empty() : tensor<345x32xf32>
    %400 = tensor.collapse_shape %398 [[0 : i64, 1 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %401 = tensor.expand_shape %400 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %402 = tensor.empty() : tensor<1x345x32xf32>
    %403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401, %12 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%402 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb23(%404: f32, %405: f32, %406: f32):
      %407 = arith.addf %404, %405 : f32
      linalg.yield %407 : f32
    } -> tensor<1x345x32xf32>
    %408 = tensor.collapse_shape %403 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %409 = tensor.expand_shape %408 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %410 = tensor.empty() : tensor<1x1x32x2xf32>
    %411 = linalg.transpose ins(%383:tensor<1x1x2x32xf32>) outs(%410:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %412 = tensor.empty() : tensor<1x1x32x2xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x1x32x2xf32>) outs(%412 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb24(%414: f32, %415: f32):
      linalg.yield %414 : f32
    } -> tensor<1x1x32x2xf32>
    %416 = tensor.collapse_shape %413 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %417 = tensor.expand_shape %416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %418 = tensor.collapse_shape %409 [[0 : i64, 1 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345x32xf32> into tensor<11040xf32>
    %419 = tensor.expand_shape %418 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %420 = tensor.collapse_shape %419 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %421 = tensor.expand_shape %420 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %422 = tensor.empty() : tensor<1x1x345x32xf32>
    %423 = linalg.transpose ins(%421:tensor<1x345x1x32xf32>) outs(%422:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %424 = tensor.empty() : tensor<1x1x345x32xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%423 : tensor<1x1x345x32xf32>) outs(%424 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb25(%426: f32, %427: f32):
      linalg.yield %426 : f32
    } -> tensor<1x1x345x32xf32>
    %428 = tensor.collapse_shape %425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %429 = tensor.expand_shape %428 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %430 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %431 = tensor.splat %430 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%429, %417 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%431 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb26(%433: f32, %434: f32, %435: f32):
      %436 = arith.mulf %433, %434 : f32
      %437 = arith.addf %435, %436 : f32
      linalg.yield %437 : f32
    } -> tensor<1x345x2xf32>
    %438 = tensor.collapse_shape %432 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %439 = tensor.expand_shape %438 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %440 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %441 = tensor.splat %440 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %442 = tensor.empty() : tensor<1x1x345x2xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %441 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%442 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb27(%444: f32, %445: f32, %446: f32):
      %447 = arith.divf %444, %445 : f32
      linalg.yield %447 : f32
    } -> tensor<1x1x345x2xf32>
    %448 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %449 = tensor.splat %448 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %450 = linalg.reduce ins(%443:tensor<1x1x345x2xf32>) outs(%449:tensor<1x1x345xf32>) dimensions = [3]
    (%451: f32, %452: f32) {
      %453 = arith.maximumf %451, %452 : f32
      linalg.yield %453 : f32
    }
    %454 = tensor.collapse_shape %450 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %455 = tensor.expand_shape %454 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %456 = tensor.empty() : tensor<1x1x345x2xf32>
    %457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %455 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%456 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb28(%458: f32, %459: f32, %460: f32):
      %461 = arith.subf %458, %459 : f32
      linalg.yield %461 : f32
    } -> tensor<1x1x345x2xf32>
    %462 = tensor.empty() : tensor<1x1x345x2xf32>
    %463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%457 : tensor<1x1x345x2xf32>) outs(%462 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb29(%464: f32, %465: f32):
      %466 = math.exp %464 : f32
      linalg.yield %466 : f32
    } -> tensor<1x1x345x2xf32>
    %467 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %468 = tensor.splat %467 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %469 = linalg.reduce ins(%463:tensor<1x1x345x2xf32>) outs(%468:tensor<1x1x345xf32>) dimensions = [3]
    (%470: f32, %471: f32) {
      %472 = arith.addf %470, %471 : f32
      linalg.yield %472 : f32
    }
    %473 = tensor.collapse_shape %469 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %474 = tensor.expand_shape %473 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %475 = tensor.empty() : tensor<1x1x345x2xf32>
    %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %474 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%475 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb30(%477: f32, %478: f32, %479: f32):
      %480 = arith.divf %477, %478 : f32
      linalg.yield %480 : f32
    } -> tensor<1x1x345x2xf32>
    %481 = tensor.empty() : tensor<1x1x345x2xf32>
    %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476 : tensor<1x1x345x2xf32>) outs(%481 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb31(%483: f32, %484: f32):
      linalg.yield %483 : f32
    } -> tensor<1x1x345x2xf32>
    %485 = tensor.collapse_shape %482 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %486 = tensor.expand_shape %485 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %487 = tensor.empty() : tensor<1x1x2x32xf32>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386 : tensor<1x1x2x32xf32>) outs(%487 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb32(%489: f32, %490: f32):
      linalg.yield %489 : f32
    } -> tensor<1x1x2x32xf32>
    %491 = tensor.collapse_shape %488 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %492 = tensor.expand_shape %491 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %493 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %494 = tensor.splat %493 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%486, %492 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%494 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb33(%496: f32, %497: f32, %498: f32):
      %499 = arith.mulf %496, %497 : f32
      %500 = arith.addf %498, %499 : f32
      linalg.yield %500 : f32
    } -> tensor<1x345x32xf32>
    %501 = tensor.collapse_shape %495 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %502 = tensor.expand_shape %501 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %503 = tensor.empty() : tensor<1x345x1x32xf32>
    %504 = linalg.transpose ins(%502:tensor<1x1x345x32xf32>) outs(%503:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %505 = tensor.collapse_shape %504 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %506 = tensor.expand_shape %505 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %507 = tensor.empty() : tensor<32x32xi8>
    %508 = linalg.transpose ins(%17:tensor<32x32xi8>) outs(%507:tensor<32x32xi8>) permutation = [1, 0]
    %509 = tensor.collapse_shape %506 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %510 = tensor.expand_shape %509 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %511 = tensor.empty() : tensor<32x32xf32>
    %512 = arith.constant 0 : i32
    %513 = tensor.splat %512 : tensor<32xi32>
    %514 = "quant_ext.dequantize_per_channel"(%508, %18, %513) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %515 = tensor.empty() : tensor<345x32xf32>
    %516 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %517 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%516 : f32) outs(%515 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %518 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} ins(%510, %514 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%517 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %519 = tensor.empty() : tensor<345x32xf32>
    %520 = tensor.collapse_shape %518 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %521 = tensor.expand_shape %520 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %522 = tensor.empty() : tensor<1x345x32xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%521, %16 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%522 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb34(%524: f32, %525: f32, %526: f32):
      %527 = arith.addf %524, %525 : f32
      linalg.yield %527 : f32
    } -> tensor<1x345x32xf32>
    %528 = tensor.collapse_shape %523 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %529 = tensor.expand_shape %528 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %530 = tensor.collapse_shape %529 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %531 = tensor.expand_shape %530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %532 = tensor.empty() : tensor<1x345x32xf32>
    %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%243, %531 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%532 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb35(%534: f32, %535: f32, %536: f32):
      %537 = arith.addf %534, %535 : f32
      linalg.yield %537 : f32
    } -> tensor<1x345x32xf32>
    %538 = tensor.empty() : tensor<32x256xi8>
    %539 = linalg.transpose ins(%37:tensor<256x32xi8>) outs(%538:tensor<32x256xi8>) permutation = [1, 0]
    %540 = tensor.collapse_shape %533 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %541 = tensor.expand_shape %540 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %542 = tensor.empty() : tensor<32x256xf32>
    %543 = arith.constant 0 : i32
    %544 = tensor.splat %543 : tensor<256xi32>
    %545 = "quant_ext.dequantize_per_channel"(%539, %38, %544) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x256xi8>, tensor<256xf32>, tensor<256xi32>) -> tensor<32x256xf32>
    %546 = tensor.empty() : tensor<345x256xf32>
    %547 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %548 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%547 : f32) outs(%546 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %549 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} ins(%541, %545 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%548 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %550 = tensor.empty() : tensor<345x256xf32>
    %551 = tensor.collapse_shape %549 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %552 = tensor.expand_shape %551 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %553 = tensor.empty() : tensor<1x345x256xf32>
    %554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%552, %36 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%553 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb36(%555: f32, %556: f32, %557: f32):
      %558 = arith.addf %555, %556 : f32
      linalg.yield %558 : f32
    } -> tensor<1x345x256xf32>
    %559 = tensor.collapse_shape %554 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %560 = tensor.expand_shape %559 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<88320xf32> into tensor<345x256xf32>
    %561 = tensor.collapse_shape %560 [[0 : i64, 1 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<345x256xf32> into tensor<88320xf32>
    %562 = tensor.expand_shape %561 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %563 = tensor.empty() : tensor<1x256x345xf32>
    %564 = linalg.transpose ins(%562:tensor<1x345x256xf32>) outs(%563:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %565 = tensor.collapse_shape %564 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %566 = tensor.expand_shape %565 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %567 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %568 = tensor.splat %567 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %569 = "tensor.insert_slice"(%566, %568) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %570 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%569 : tensor<1x256x17x25xf32>) outs(%570 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb37(%572: f32, %573: f32):
      linalg.yield %572 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %574 = tensor.collapse_shape %571 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %575 = tensor.expand_shape %574 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %576 = tensor.collapse_shape %40 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %577 = tensor.expand_shape %576 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %578 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %579 = tensor.splat %578 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%577, %575 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%579 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb38(%581: f32, %582: f32, %583: f32):
      %584 = arith.mulf %581, %582 : f32
      %585 = arith.addf %583, %584 : f32
      linalg.yield %585 : f32
    } -> tensor<32x8x345xf32>
    %586 = tensor.collapse_shape %580 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %587 = tensor.expand_shape %586 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %588 = tensor.collapse_shape %587 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %589 = tensor.expand_shape %588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %590 = tensor.empty() : tensor<1x256x15x23xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%589, %41 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%590 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb39(%592: f32, %593: f32, %594: f32):
      %595 = arith.addf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<1x256x15x23xf32>
    %596 = tensor.collapse_shape %591 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %597 = tensor.expand_shape %596 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %598 = tensor.empty() : tensor<1x345x256xf32>
    %599 = linalg.transpose ins(%597:tensor<1x256x345xf32>) outs(%598:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %600 = tensor.empty() : tensor<1x345x256xf32>
    %601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599 : tensor<1x345x256xf32>) outs(%600 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb40(%602: f32, %603: f32):
      %604 = arith.constant 5.000000e-01 : f32
      %605 = arith.constant 1.000000e+00 : f32
      %606 = arith.constant 0.707106769 : f32
      %607 = arith.mulf %602, %606 : f32
      %608 = math.erf %607 : f32
      %609 = arith.addf %605, %608 : f32
      %610 = arith.mulf %604, %602 : f32
      %611 = arith.mulf %610, %609 : f32
      linalg.yield %611 : f32
    } -> tensor<1x345x256xf32>
    %612 = tensor.empty() : tensor<256x32xi8>
    %613 = linalg.transpose ins(%43:tensor<32x256xi8>) outs(%612:tensor<256x32xi8>) permutation = [1, 0]
    %614 = tensor.collapse_shape %601 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %615 = tensor.expand_shape %614 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %616 = tensor.empty() : tensor<256x32xf32>
    %617 = arith.constant 0 : i32
    %618 = tensor.splat %617 : tensor<32xi32>
    %619 = "quant_ext.dequantize_per_channel"(%613, %44, %618) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<256x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<256x32xf32>
    %620 = tensor.empty() : tensor<345x32xf32>
    %621 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %622 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%621 : f32) outs(%620 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %623 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} ins(%615, %619 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%622 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %624 = tensor.empty() : tensor<345x32xf32>
    %625 = tensor.collapse_shape %623 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %626 = tensor.expand_shape %625 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %627 = tensor.empty() : tensor<1x345x32xf32>
    %628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%626, %42 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%627 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb41(%629: f32, %630: f32, %631: f32):
      %632 = arith.addf %629, %630 : f32
      linalg.yield %632 : f32
    } -> tensor<1x345x32xf32>
    %633 = tensor.collapse_shape %628 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %634 = tensor.expand_shape %633 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<345x32xf32>
    %635 = tensor.collapse_shape %634 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %636 = tensor.expand_shape %635 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %637 = tensor.empty() : tensor<1x345x32xf32>
    %638 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%533, %636 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%637 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb42(%639: f32, %640: f32, %641: f32):
      %642 = arith.addf %639, %640 : f32
      linalg.yield %642 : f32
    } -> tensor<1x345x32xf32>
    %643 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %644 = tensor.splat %643 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %645 = linalg.reduce ins(%638:tensor<1x345x32xf32>) outs(%644:tensor<1x345xf32>) dimensions = [2]
    (%646: f32, %647: f32) {
      %648 = arith.addf %646, %647 : f32
      linalg.yield %648 : f32
    }
    %649 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %650 = tensor.splat %649 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %651 = tensor.empty() : tensor<1x345xf32>
    %652 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%645, %650 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%651 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb43(%653: f32, %654: f32, %655: f32):
      %656 = arith.divf %653, %654 : f32
      linalg.yield %656 : f32
    } -> tensor<1x345xf32>
    %657 = tensor.collapse_shape %652 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %658 = tensor.expand_shape %657 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %659 = tensor.empty() : tensor<1x345x32xf32>
    %660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%638, %658 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%659 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb44(%661: f32, %662: f32, %663: f32):
      %664 = arith.subf %661, %662 : f32
      linalg.yield %664 : f32
    } -> tensor<1x345x32xf32>
    %665 = tensor.empty() : tensor<1x345x32xf32>
    %666 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660, %660 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%665 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb45(%667: f32, %668: f32, %669: f32):
      %670 = arith.mulf %667, %668 : f32
      linalg.yield %670 : f32
    } -> tensor<1x345x32xf32>
    %671 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %672 = tensor.splat %671 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %673 = linalg.reduce ins(%666:tensor<1x345x32xf32>) outs(%672:tensor<1x345xf32>) dimensions = [2]
    (%674: f32, %675: f32) {
      %676 = arith.addf %674, %675 : f32
      linalg.yield %676 : f32
    }
    %677 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %678 = tensor.splat %677 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %679 = tensor.empty() : tensor<1x345xf32>
    %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%673, %678 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%679 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb46(%681: f32, %682: f32, %683: f32):
      %684 = arith.divf %681, %682 : f32
      linalg.yield %684 : f32
    } -> tensor<1x345xf32>
    %685 = tensor.collapse_shape %680 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %686 = tensor.expand_shape %685 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %687 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %688 = tensor.splat %687 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %689 = tensor.empty() : tensor<1x345x1xf32>
    %690 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686, %688 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%689 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%691: f32, %692: f32, %693: f32):
      %694 = arith.addf %691, %692 : f32
      linalg.yield %694 : f32
    } -> tensor<1x345x1xf32>
    %695 = tensor.empty() : tensor<1x345x1xf32>
    %696 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%690 : tensor<1x345x1xf32>) outs(%695 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb48(%697: f32, %698: f32):
      %699 = math.rsqrt %697 : f32
      linalg.yield %699 : f32
    } -> tensor<1x345x1xf32>
    %700 = tensor.empty() : tensor<1x345x32xf32>
    %701 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660, %696 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%700 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb49(%702: f32, %703: f32, %704: f32):
      %705 = arith.mulf %702, %703 : f32
      linalg.yield %705 : f32
    } -> tensor<1x345x32xf32>
    %706 = tensor.empty() : tensor<1x345x32xf32>
    %707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%701, %56 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%706 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb50(%708: f32, %709: f32, %710: f32):
      %711 = arith.mulf %708, %709 : f32
      linalg.yield %711 : f32
    } -> tensor<1x345x32xf32>
    %712 = tensor.empty() : tensor<1x345x32xf32>
    %713 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %57 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%712 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb51(%714: f32, %715: f32, %716: f32):
      %717 = arith.addf %714, %715 : f32
      linalg.yield %717 : f32
    } -> tensor<1x345x32xf32>
    %718 = tensor.empty() : tensor<1x32x345xf32>
    %719 = linalg.transpose ins(%713:tensor<1x345x32xf32>) outs(%718:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %720 = tensor.collapse_shape %719 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %721 = tensor.expand_shape %720 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %722 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %723 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%721 : tensor<1x32x15x23xf32>) outs(%722 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb52(%724: f32, %725: f32):
      linalg.yield %724 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %726 = tensor.collapse_shape %723 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %727 = tensor.expand_shape %726 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %728 = tensor.collapse_shape %20 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %729 = tensor.expand_shape %728 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %730 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %731 = tensor.splat %730 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %732 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%729, %727 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%731 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %733 = tensor.collapse_shape %732 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %734 = tensor.expand_shape %733 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %735 = tensor.collapse_shape %734 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %736 = tensor.expand_shape %735 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %737 = tensor.empty() : tensor<1x32x1x2xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736, %21 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%737 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb53(%739: f32, %740: f32, %741: f32):
      %742 = arith.addf %739, %740 : f32
      linalg.yield %742 : f32
    } -> tensor<1x32x1x2xf32>
    %743 = tensor.collapse_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %744 = tensor.expand_shape %743 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %745 = tensor.empty() : tensor<1x2x32xf32>
    %746 = linalg.transpose ins(%744:tensor<1x32x2xf32>) outs(%745:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %747 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %748 = tensor.splat %747 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %749 = linalg.reduce ins(%746:tensor<1x2x32xf32>) outs(%748:tensor<1x2xf32>) dimensions = [2]
    (%750: f32, %751: f32) {
      %752 = arith.addf %750, %751 : f32
      linalg.yield %752 : f32
    }
    %753 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %754 = tensor.splat %753 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %755 = tensor.empty() : tensor<1x2xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%749, %754 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%755 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb54(%757: f32, %758: f32, %759: f32):
      %760 = arith.divf %757, %758 : f32
      linalg.yield %760 : f32
    } -> tensor<1x2xf32>
    %761 = tensor.collapse_shape %756 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %762 = tensor.expand_shape %761 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %763 = tensor.empty() : tensor<1x2x32xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%746, %762 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%763 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb55(%765: f32, %766: f32, %767: f32):
      %768 = arith.subf %765, %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x2x32xf32>
    %769 = tensor.empty() : tensor<1x2x32xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764, %764 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%769 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb56(%771: f32, %772: f32, %773: f32):
      %774 = arith.mulf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x2x32xf32>
    %775 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %776 = tensor.splat %775 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %777 = linalg.reduce ins(%770:tensor<1x2x32xf32>) outs(%776:tensor<1x2xf32>) dimensions = [2]
    (%778: f32, %779: f32) {
      %780 = arith.addf %778, %779 : f32
      linalg.yield %780 : f32
    }
    %781 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %782 = tensor.splat %781 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %783 = tensor.empty() : tensor<1x2xf32>
    %784 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%777, %782 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%783 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb57(%785: f32, %786: f32, %787: f32):
      %788 = arith.divf %785, %786 : f32
      linalg.yield %788 : f32
    } -> tensor<1x2xf32>
    %789 = tensor.collapse_shape %784 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %790 = tensor.expand_shape %789 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %791 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %792 = tensor.splat %791 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %793 = tensor.empty() : tensor<1x2x1xf32>
    %794 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%790, %792 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%793 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb58(%795: f32, %796: f32, %797: f32):
      %798 = arith.addf %795, %796 : f32
      linalg.yield %798 : f32
    } -> tensor<1x2x1xf32>
    %799 = tensor.empty() : tensor<1x2x1xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%794 : tensor<1x2x1xf32>) outs(%799 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb59(%801: f32, %802: f32):
      %803 = math.rsqrt %801 : f32
      linalg.yield %803 : f32
    } -> tensor<1x2x1xf32>
    %804 = tensor.empty() : tensor<1x2x32xf32>
    %805 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764, %800 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%804 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb60(%806: f32, %807: f32, %808: f32):
      %809 = arith.mulf %806, %807 : f32
      linalg.yield %809 : f32
    } -> tensor<1x2x32xf32>
    %810 = tensor.empty() : tensor<1x2x32xf32>
    %811 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%805, %22 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%810 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb61(%812: f32, %813: f32, %814: f32):
      %815 = arith.mulf %812, %813 : f32
      linalg.yield %815 : f32
    } -> tensor<1x2x32xf32>
    %816 = tensor.empty() : tensor<1x2x32xf32>
    %817 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%811, %23 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%816 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb62(%818: f32, %819: f32, %820: f32):
      %821 = arith.addf %818, %819 : f32
      linalg.yield %821 : f32
    } -> tensor<1x2x32xf32>
    %822 = tensor.empty() : tensor<32x64xi8>
    %823 = linalg.transpose ins(%25:tensor<64x32xi8>) outs(%822:tensor<32x64xi8>) permutation = [1, 0]
    %824 = tensor.collapse_shape %817 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %825 = tensor.expand_shape %824 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %826 = tensor.empty() : tensor<32x64xf32>
    %827 = arith.constant 0 : i32
    %828 = tensor.splat %827 : tensor<64xi32>
    %829 = "quant_ext.dequantize_per_channel"(%823, %26, %828) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<32x64xf32>
    %830 = tensor.empty() : tensor<2x64xf32>
    %831 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %832 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%831 : f32) outs(%830 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %833 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} ins(%825, %829 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%832 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %834 = tensor.empty() : tensor<2x64xf32>
    %835 = tensor.collapse_shape %833 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %836 = tensor.expand_shape %835 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %837 = tensor.empty() : tensor<1x2x64xf32>
    %838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%836, %24 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%837 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb63(%839: f32, %840: f32, %841: f32):
      %842 = arith.addf %839, %840 : f32
      linalg.yield %842 : f32
    } -> tensor<1x2x64xf32>
    %843 = tensor.collapse_shape %838 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x64xf32> into tensor<128xf32>
    %844 = tensor.expand_shape %843 [[0 : i64, 1 : i64]] output_shape [2, 64] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<128xf32> into tensor<2x64xf32>
    %845 = tensor.collapse_shape %844 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<2x64xf32> into tensor<128xf32>
    %846 = tensor.expand_shape %845 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x64xf32>
    %847 = tensor.collapse_shape %846 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %848 = tensor.expand_shape %847 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %849 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %850 = linalg.transpose ins(%848:tensor<1x2x2x1x32xf32>) outs(%849:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %851 = "tensor.extract_slice"(%850) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %852 = tensor.collapse_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %853 = tensor.expand_shape %852 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %854 = "tensor.extract_slice"(%850) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %855 = tensor.collapse_shape %854 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %856 = tensor.expand_shape %855 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %857 = tensor.empty() : tensor<32x32xi8>
    %858 = linalg.transpose ins(%29:tensor<32x32xi8>) outs(%857:tensor<32x32xi8>) permutation = [1, 0]
    %859 = tensor.collapse_shape %713 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %860 = tensor.expand_shape %859 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %861 = tensor.empty() : tensor<32x32xf32>
    %862 = arith.constant 0 : i32
    %863 = tensor.splat %862 : tensor<32xi32>
    %864 = "quant_ext.dequantize_per_channel"(%858, %30, %863) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %865 = tensor.empty() : tensor<345x32xf32>
    %866 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %867 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%866 : f32) outs(%865 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %868 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} ins(%860, %864 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%867 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %869 = tensor.empty() : tensor<345x32xf32>
    %870 = tensor.collapse_shape %868 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %871 = tensor.expand_shape %870 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %872 = tensor.empty() : tensor<1x345x32xf32>
    %873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%871, %28 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%872 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb64(%874: f32, %875: f32, %876: f32):
      %877 = arith.addf %874, %875 : f32
      linalg.yield %877 : f32
    } -> tensor<1x345x32xf32>
    %878 = tensor.collapse_shape %873 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %880 = tensor.empty() : tensor<1x1x32x2xf32>
    %881 = linalg.transpose ins(%853:tensor<1x1x2x32xf32>) outs(%880:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %882 = tensor.empty() : tensor<1x1x32x2xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%881 : tensor<1x1x32x2xf32>) outs(%882 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb65(%884: f32, %885: f32):
      linalg.yield %884 : f32
    } -> tensor<1x1x32x2xf32>
    %886 = tensor.collapse_shape %883 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %887 = tensor.expand_shape %886 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %888 = tensor.collapse_shape %879 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345x32xf32> into tensor<11040xf32>
    %889 = tensor.expand_shape %888 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %890 = tensor.collapse_shape %889 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %891 = tensor.expand_shape %890 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %892 = tensor.empty() : tensor<1x1x345x32xf32>
    %893 = linalg.transpose ins(%891:tensor<1x345x1x32xf32>) outs(%892:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %894 = tensor.empty() : tensor<1x1x345x32xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%893 : tensor<1x1x345x32xf32>) outs(%894 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb66(%896: f32, %897: f32):
      linalg.yield %896 : f32
    } -> tensor<1x1x345x32xf32>
    %898 = tensor.collapse_shape %895 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %899 = tensor.expand_shape %898 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %900 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %901 = tensor.splat %900 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %902 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%899, %887 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%901 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb67(%903: f32, %904: f32, %905: f32):
      %906 = arith.mulf %903, %904 : f32
      %907 = arith.addf %905, %906 : f32
      linalg.yield %907 : f32
    } -> tensor<1x345x2xf32>
    %908 = tensor.collapse_shape %902 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %909 = tensor.expand_shape %908 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %910 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %911 = tensor.splat %910 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %912 = tensor.empty() : tensor<1x1x345x2xf32>
    %913 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%909, %911 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%912 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb68(%914: f32, %915: f32, %916: f32):
      %917 = arith.divf %914, %915 : f32
      linalg.yield %917 : f32
    } -> tensor<1x1x345x2xf32>
    %918 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %919 = tensor.splat %918 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %920 = linalg.reduce ins(%913:tensor<1x1x345x2xf32>) outs(%919:tensor<1x1x345xf32>) dimensions = [3]
    (%921: f32, %922: f32) {
      %923 = arith.maximumf %921, %922 : f32
      linalg.yield %923 : f32
    }
    %924 = tensor.collapse_shape %920 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %925 = tensor.expand_shape %924 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %926 = tensor.empty() : tensor<1x1x345x2xf32>
    %927 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%913, %925 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%926 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb69(%928: f32, %929: f32, %930: f32):
      %931 = arith.subf %928, %929 : f32
      linalg.yield %931 : f32
    } -> tensor<1x1x345x2xf32>
    %932 = tensor.empty() : tensor<1x1x345x2xf32>
    %933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%927 : tensor<1x1x345x2xf32>) outs(%932 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb70(%934: f32, %935: f32):
      %936 = math.exp %934 : f32
      linalg.yield %936 : f32
    } -> tensor<1x1x345x2xf32>
    %937 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %938 = tensor.splat %937 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %939 = linalg.reduce ins(%933:tensor<1x1x345x2xf32>) outs(%938:tensor<1x1x345xf32>) dimensions = [3]
    (%940: f32, %941: f32) {
      %942 = arith.addf %940, %941 : f32
      linalg.yield %942 : f32
    }
    %943 = tensor.collapse_shape %939 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %944 = tensor.expand_shape %943 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %945 = tensor.empty() : tensor<1x1x345x2xf32>
    %946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%933, %944 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%945 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb71(%947: f32, %948: f32, %949: f32):
      %950 = arith.divf %947, %948 : f32
      linalg.yield %950 : f32
    } -> tensor<1x1x345x2xf32>
    %951 = tensor.empty() : tensor<1x1x345x2xf32>
    %952 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%946 : tensor<1x1x345x2xf32>) outs(%951 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb72(%953: f32, %954: f32):
      linalg.yield %953 : f32
    } -> tensor<1x1x345x2xf32>
    %955 = tensor.collapse_shape %952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %956 = tensor.expand_shape %955 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %957 = tensor.empty() : tensor<1x1x2x32xf32>
    %958 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%856 : tensor<1x1x2x32xf32>) outs(%957 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb73(%959: f32, %960: f32):
      linalg.yield %959 : f32
    } -> tensor<1x1x2x32xf32>
    %961 = tensor.collapse_shape %958 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %962 = tensor.expand_shape %961 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %963 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %964 = tensor.splat %963 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%956, %962 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%964 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb74(%966: f32, %967: f32, %968: f32):
      %969 = arith.mulf %966, %967 : f32
      %970 = arith.addf %968, %969 : f32
      linalg.yield %970 : f32
    } -> tensor<1x345x32xf32>
    %971 = tensor.collapse_shape %965 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %972 = tensor.expand_shape %971 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %973 = tensor.empty() : tensor<1x345x1x32xf32>
    %974 = linalg.transpose ins(%972:tensor<1x1x345x32xf32>) outs(%973:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %975 = tensor.collapse_shape %974 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %976 = tensor.expand_shape %975 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %977 = tensor.empty() : tensor<32x32xi8>
    %978 = linalg.transpose ins(%33:tensor<32x32xi8>) outs(%977:tensor<32x32xi8>) permutation = [1, 0]
    %979 = tensor.collapse_shape %976 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %980 = tensor.expand_shape %979 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %981 = tensor.empty() : tensor<32x32xf32>
    %982 = arith.constant 0 : i32
    %983 = tensor.splat %982 : tensor<32xi32>
    %984 = "quant_ext.dequantize_per_channel"(%978, %34, %983) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %985 = tensor.empty() : tensor<345x32xf32>
    %986 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %987 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%986 : f32) outs(%985 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %988 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} ins(%980, %984 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%987 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %989 = tensor.empty() : tensor<345x32xf32>
    %990 = tensor.collapse_shape %988 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %991 = tensor.expand_shape %990 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %992 = tensor.empty() : tensor<1x345x32xf32>
    %993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991, %32 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%992 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb75(%994: f32, %995: f32, %996: f32):
      %997 = arith.addf %994, %995 : f32
      linalg.yield %997 : f32
    } -> tensor<1x345x32xf32>
    %998 = tensor.collapse_shape %993 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %999 = tensor.expand_shape %998 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %1000 = tensor.collapse_shape %999 [[0 : i64, 1 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %1001 = tensor.expand_shape %1000 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1002 = tensor.empty() : tensor<1x345x32xf32>
    %1003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%713, %1001 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1002 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%1004: f32, %1005: f32, %1006: f32):
      %1007 = arith.addf %1004, %1005 : f32
      linalg.yield %1007 : f32
    } -> tensor<1x345x32xf32>
    %1008 = tensor.empty() : tensor<32x256xi8>
    %1009 = linalg.transpose ins(%47:tensor<256x32xi8>) outs(%1008:tensor<32x256xi8>) permutation = [1, 0]
    %1010 = tensor.collapse_shape %1003 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1011 = tensor.expand_shape %1010 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %1012 = tensor.empty() : tensor<32x256xf32>
    %1013 = arith.constant 0 : i32
    %1014 = tensor.splat %1013 : tensor<256xi32>
    %1015 = "quant_ext.dequantize_per_channel"(%1009, %48, %1014) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<32x256xi8>, tensor<256xf32>, tensor<256xi32>) -> tensor<32x256xf32>
    %1016 = tensor.empty() : tensor<345x256xf32>
    %1017 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1018 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1017 : f32) outs(%1016 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %1019 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} ins(%1011, %1015 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%1018 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %1020 = tensor.empty() : tensor<345x256xf32>
    %1021 = tensor.collapse_shape %1019 [[0 : i64, 1 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %1022 = tensor.expand_shape %1021 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %1023 = tensor.empty() : tensor<1x345x256xf32>
    %1024 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1022, %46 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%1023 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb77(%1025: f32, %1026: f32, %1027: f32):
      %1028 = arith.addf %1025, %1026 : f32
      linalg.yield %1028 : f32
    } -> tensor<1x345x256xf32>
    %1029 = tensor.collapse_shape %1024 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %1030 = tensor.expand_shape %1029 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<88320xf32> into tensor<345x256xf32>
    %1031 = tensor.collapse_shape %1030 [[0 : i64, 1 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<345x256xf32> into tensor<88320xf32>
    %1032 = tensor.expand_shape %1031 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %1033 = tensor.empty() : tensor<1x256x345xf32>
    %1034 = linalg.transpose ins(%1032:tensor<1x345x256xf32>) outs(%1033:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %1035 = tensor.collapse_shape %1034 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %1036 = tensor.expand_shape %1035 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1037 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1038 = tensor.splat %1037 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %1039 = "tensor.insert_slice"(%1036, %1038) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %1040 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %1041 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1039 : tensor<1x256x17x25xf32>) outs(%1040 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb78(%1042: f32, %1043: f32):
      linalg.yield %1042 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %1044 = tensor.collapse_shape %1041 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %1045 = tensor.expand_shape %1044 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %1046 = tensor.collapse_shape %50 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %1047 = tensor.expand_shape %1046 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %1048 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1049 = tensor.splat %1048 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1047, %1045 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%1049 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb79(%1051: f32, %1052: f32, %1053: f32):
      %1054 = arith.mulf %1051, %1052 : f32
      %1055 = arith.addf %1053, %1054 : f32
      linalg.yield %1055 : f32
    } -> tensor<32x8x345xf32>
    %1056 = tensor.collapse_shape %1050 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %1057 = tensor.expand_shape %1056 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %1058 = tensor.collapse_shape %1057 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %1059 = tensor.expand_shape %1058 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1060 = tensor.empty() : tensor<1x256x15x23xf32>
    %1061 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1059, %51 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%1060 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb80(%1062: f32, %1063: f32, %1064: f32):
      %1065 = arith.addf %1062, %1063 : f32
      linalg.yield %1065 : f32
    } -> tensor<1x256x15x23xf32>
    %1066 = tensor.collapse_shape %1061 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %1067 = tensor.expand_shape %1066 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %1068 = tensor.empty() : tensor<1x345x256xf32>
    %1069 = linalg.transpose ins(%1067:tensor<1x256x345xf32>) outs(%1068:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %1070 = tensor.empty() : tensor<1x345x256xf32>
    %1071 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1069 : tensor<1x345x256xf32>) outs(%1070 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb81(%1072: f32, %1073: f32):
      %1074 = arith.constant 5.000000e-01 : f32
      %1075 = arith.constant 1.000000e+00 : f32
      %1076 = arith.constant 0.707106769 : f32
      %1077 = arith.mulf %1072, %1076 : f32
      %1078 = math.erf %1077 : f32
      %1079 = arith.addf %1075, %1078 : f32
      %1080 = arith.mulf %1074, %1072 : f32
      %1081 = arith.mulf %1080, %1079 : f32
      linalg.yield %1081 : f32
    } -> tensor<1x345x256xf32>
    %1082 = tensor.empty() : tensor<256x32xi8>
    %1083 = linalg.transpose ins(%53:tensor<32x256xi8>) outs(%1082:tensor<256x32xi8>) permutation = [1, 0]
    %1084 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %1085 = tensor.expand_shape %1084 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %1086 = tensor.empty() : tensor<256x32xf32>
    %1087 = arith.constant 0 : i32
    %1088 = tensor.splat %1087 : tensor<32xi32>
    %1089 = "quant_ext.dequantize_per_channel"(%1083, %54, %1088) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<256x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<256x32xf32>
    %1090 = tensor.empty() : tensor<345x32xf32>
    %1091 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1092 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1091 : f32) outs(%1090 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1093 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} ins(%1085, %1089 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%1092 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1094 = tensor.empty() : tensor<345x32xf32>
    %1095 = tensor.collapse_shape %1093 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %1096 = tensor.expand_shape %1095 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1097 = tensor.empty() : tensor<1x345x32xf32>
    %1098 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1096, %52 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1097 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb82(%1099: f32, %1100: f32, %1101: f32):
      %1102 = arith.addf %1099, %1100 : f32
      linalg.yield %1102 : f32
    } -> tensor<1x345x32xf32>
    %1103 = tensor.collapse_shape %1098 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1104 = tensor.expand_shape %1103 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<345x32xf32>
    %1105 = tensor.collapse_shape %1104 [[0 : i64, 1 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %1106 = tensor.expand_shape %1105 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1107 = tensor.empty() : tensor<1x345x32xf32>
    %1108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1003, %1106 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1107 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb83(%1109: f32, %1110: f32, %1111: f32):
      %1112 = arith.addf %1109, %1110 : f32
      linalg.yield %1112 : f32
    } -> tensor<1x345x32xf32>
    %1113 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1114 = tensor.splat %1113 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1115 = linalg.reduce ins(%1108:tensor<1x345x32xf32>) outs(%1114:tensor<1x345xf32>) dimensions = [2]
    (%1116: f32, %1117: f32) {
      %1118 = arith.addf %1116, %1117 : f32
      linalg.yield %1118 : f32
    }
    %1119 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1120 = tensor.splat %1119 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1121 = tensor.empty() : tensor<1x345xf32>
    %1122 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1115, %1120 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1121 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%1123: f32, %1124: f32, %1125: f32):
      %1126 = arith.divf %1123, %1124 : f32
      linalg.yield %1126 : f32
    } -> tensor<1x345xf32>
    %1127 = tensor.collapse_shape %1122 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1129 = tensor.empty() : tensor<1x345x32xf32>
    %1130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1108, %1128 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1129 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb85(%1131: f32, %1132: f32, %1133: f32):
      %1134 = arith.subf %1131, %1132 : f32
      linalg.yield %1134 : f32
    } -> tensor<1x345x32xf32>
    %1135 = tensor.empty() : tensor<1x345x32xf32>
    %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130, %1130 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1135 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb86(%1137: f32, %1138: f32, %1139: f32):
      %1140 = arith.mulf %1137, %1138 : f32
      linalg.yield %1140 : f32
    } -> tensor<1x345x32xf32>
    %1141 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1142 = tensor.splat %1141 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1143 = linalg.reduce ins(%1136:tensor<1x345x32xf32>) outs(%1142:tensor<1x345xf32>) dimensions = [2]
    (%1144: f32, %1145: f32) {
      %1146 = arith.addf %1144, %1145 : f32
      linalg.yield %1146 : f32
    }
    %1147 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1148 = tensor.splat %1147 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1149 = tensor.empty() : tensor<1x345xf32>
    %1150 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1143, %1148 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1149 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb87(%1151: f32, %1152: f32, %1153: f32):
      %1154 = arith.divf %1151, %1152 : f32
      linalg.yield %1154 : f32
    } -> tensor<1x345xf32>
    %1155 = tensor.collapse_shape %1150 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1156 = tensor.expand_shape %1155 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1157 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1158 = tensor.splat %1157 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1159 = tensor.empty() : tensor<1x345x1xf32>
    %1160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1156, %1158 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1159 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb88(%1161: f32, %1162: f32, %1163: f32):
      %1164 = arith.addf %1161, %1162 : f32
      linalg.yield %1164 : f32
    } -> tensor<1x345x1xf32>
    %1165 = tensor.empty() : tensor<1x345x1xf32>
    %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1160 : tensor<1x345x1xf32>) outs(%1165 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb89(%1167: f32, %1168: f32):
      %1169 = math.rsqrt %1167 : f32
      linalg.yield %1169 : f32
    } -> tensor<1x345x1xf32>
    %1170 = tensor.empty() : tensor<1x345x32xf32>
    %1171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130, %1166 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1170 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb90(%1172: f32, %1173: f32, %1174: f32):
      %1175 = arith.mulf %1172, %1173 : f32
      linalg.yield %1175 : f32
    } -> tensor<1x345x32xf32>
    %1176 = tensor.empty() : tensor<1x345x32xf32>
    %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1171, %58 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1176 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb91(%1178: f32, %1179: f32, %1180: f32):
      %1181 = arith.mulf %1178, %1179 : f32
      linalg.yield %1181 : f32
    } -> tensor<1x345x32xf32>
    %1182 = tensor.empty() : tensor<1x345x32xf32>
    %1183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %59 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1182 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb92(%1184: f32, %1185: f32, %1186: f32):
      %1187 = arith.addf %1184, %1185 : f32
      linalg.yield %1187 : f32
    } -> tensor<1x345x32xf32>
    %1188 = tensor.collapse_shape %1183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1189 = tensor.expand_shape %1188 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1190 = tensor.empty() : tensor<1x32x15x23xf32>
    %1191 = linalg.transpose ins(%1189:tensor<1x15x23x32xf32>) outs(%1190:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1192 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1193 = tensor.splat %1192 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1194 = "tensor.insert_slice"(%1191, %1193) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1195 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1196 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1194 : tensor<1x32x17x25xf32>) outs(%1195 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb93(%1197: f32, %1198: f32):
      linalg.yield %1197 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1199 = tensor.collapse_shape %1196 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1200 = tensor.expand_shape %1199 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1201 = tensor.collapse_shape %60 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1202 = tensor.expand_shape %1201 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1203 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1204 = tensor.splat %1203 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1205 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1202, %1200 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1204 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1206 = tensor.collapse_shape %1205 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1207 = tensor.expand_shape %1206 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1208 = tensor.collapse_shape %1207 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1209 = tensor.expand_shape %1208 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1210 = tensor.empty() : tensor<1x64x8x12xf32>
    %1211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1209, %61 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1210 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb94(%1212: f32, %1213: f32, %1214: f32):
      %1215 = arith.addf %1212, %1213 : f32
      linalg.yield %1215 : f32
    } -> tensor<1x64x8x12xf32>
    %1216 = tensor.collapse_shape %1211 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1217 = tensor.expand_shape %1216 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1218 = tensor.empty() : tensor<1x96x64xf32>
    %1219 = linalg.transpose ins(%1217:tensor<1x64x96xf32>) outs(%1218:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1220 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1221 = tensor.splat %1220 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1222 = linalg.reduce ins(%1219:tensor<1x96x64xf32>) outs(%1221:tensor<1x96xf32>) dimensions = [2]
    (%1223: f32, %1224: f32) {
      %1225 = arith.addf %1223, %1224 : f32
      linalg.yield %1225 : f32
    }
    %1226 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1227 = tensor.splat %1226 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1228 = tensor.empty() : tensor<1x96xf32>
    %1229 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1222, %1227 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1228 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb95(%1230: f32, %1231: f32, %1232: f32):
      %1233 = arith.divf %1230, %1231 : f32
      linalg.yield %1233 : f32
    } -> tensor<1x96xf32>
    %1234 = tensor.collapse_shape %1229 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1235 = tensor.expand_shape %1234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1236 = tensor.empty() : tensor<1x96x64xf32>
    %1237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1219, %1235 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1236 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb96(%1238: f32, %1239: f32, %1240: f32):
      %1241 = arith.subf %1238, %1239 : f32
      linalg.yield %1241 : f32
    } -> tensor<1x96x64xf32>
    %1242 = tensor.empty() : tensor<1x96x64xf32>
    %1243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1237, %1237 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1242 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb97(%1244: f32, %1245: f32, %1246: f32):
      %1247 = arith.mulf %1244, %1245 : f32
      linalg.yield %1247 : f32
    } -> tensor<1x96x64xf32>
    %1248 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1249 = tensor.splat %1248 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1250 = linalg.reduce ins(%1243:tensor<1x96x64xf32>) outs(%1249:tensor<1x96xf32>) dimensions = [2]
    (%1251: f32, %1252: f32) {
      %1253 = arith.addf %1251, %1252 : f32
      linalg.yield %1253 : f32
    }
    %1254 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1255 = tensor.splat %1254 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1256 = tensor.empty() : tensor<1x96xf32>
    %1257 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1250, %1255 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1256 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb98(%1258: f32, %1259: f32, %1260: f32):
      %1261 = arith.divf %1258, %1259 : f32
      linalg.yield %1261 : f32
    } -> tensor<1x96xf32>
    %1262 = tensor.collapse_shape %1257 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1263 = tensor.expand_shape %1262 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1264 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1265 = tensor.splat %1264 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1266 = tensor.empty() : tensor<1x96x1xf32>
    %1267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1263, %1265 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1266 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb99(%1268: f32, %1269: f32, %1270: f32):
      %1271 = arith.addf %1268, %1269 : f32
      linalg.yield %1271 : f32
    } -> tensor<1x96x1xf32>
    %1272 = tensor.empty() : tensor<1x96x1xf32>
    %1273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1267 : tensor<1x96x1xf32>) outs(%1272 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb100(%1274: f32, %1275: f32):
      %1276 = math.rsqrt %1274 : f32
      linalg.yield %1276 : f32
    } -> tensor<1x96x1xf32>
    %1277 = tensor.empty() : tensor<1x96x64xf32>
    %1278 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1237, %1273 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1277 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb101(%1279: f32, %1280: f32, %1281: f32):
      %1282 = arith.mulf %1279, %1280 : f32
      linalg.yield %1282 : f32
    } -> tensor<1x96x64xf32>
    %1283 = tensor.empty() : tensor<1x96x64xf32>
    %1284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1278, %62 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1283 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb102(%1285: f32, %1286: f32, %1287: f32):
      %1288 = arith.mulf %1285, %1286 : f32
      linalg.yield %1288 : f32
    } -> tensor<1x96x64xf32>
    %1289 = tensor.empty() : tensor<1x96x64xf32>
    %1290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1284, %63 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1289 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb103(%1291: f32, %1292: f32, %1293: f32):
      %1294 = arith.addf %1291, %1292 : f32
      linalg.yield %1294 : f32
    } -> tensor<1x96x64xf32>
    %1295 = tensor.empty() : tensor<1x64x96xf32>
    %1296 = linalg.transpose ins(%1290:tensor<1x96x64xf32>) outs(%1295:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1297 = tensor.collapse_shape %1296 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1299 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1298 : tensor<1x64x8x12xf32>) outs(%1299 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb104(%1301: f32, %1302: f32):
      linalg.yield %1301 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1303 = tensor.collapse_shape %1300 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1304 = tensor.expand_shape %1303 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1305 = tensor.collapse_shape %64 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1306 = tensor.expand_shape %1305 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1307 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1308 = tensor.splat %1307 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1309 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1306, %1304 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1308 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1310 = tensor.collapse_shape %1309 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1311 = tensor.expand_shape %1310 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1312 = tensor.collapse_shape %1311 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1313 = tensor.expand_shape %1312 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1314 = tensor.empty() : tensor<1x64x2x3xf32>
    %1315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1313, %65 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1314 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb105(%1316: f32, %1317: f32, %1318: f32):
      %1319 = arith.addf %1316, %1317 : f32
      linalg.yield %1319 : f32
    } -> tensor<1x64x2x3xf32>
    %1320 = tensor.collapse_shape %1315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1321 = tensor.expand_shape %1320 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1322 = tensor.empty() : tensor<1x6x64xf32>
    %1323 = linalg.transpose ins(%1321:tensor<1x64x6xf32>) outs(%1322:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1324 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1325 = tensor.splat %1324 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1326 = linalg.reduce ins(%1323:tensor<1x6x64xf32>) outs(%1325:tensor<1x6xf32>) dimensions = [2]
    (%1327: f32, %1328: f32) {
      %1329 = arith.addf %1327, %1328 : f32
      linalg.yield %1329 : f32
    }
    %1330 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1331 = tensor.splat %1330 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1332 = tensor.empty() : tensor<1x6xf32>
    %1333 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1326, %1331 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1332 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb106(%1334: f32, %1335: f32, %1336: f32):
      %1337 = arith.divf %1334, %1335 : f32
      linalg.yield %1337 : f32
    } -> tensor<1x6xf32>
    %1338 = tensor.collapse_shape %1333 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1339 = tensor.expand_shape %1338 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1340 = tensor.empty() : tensor<1x6x64xf32>
    %1341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1323, %1339 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1340 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb107(%1342: f32, %1343: f32, %1344: f32):
      %1345 = arith.subf %1342, %1343 : f32
      linalg.yield %1345 : f32
    } -> tensor<1x6x64xf32>
    %1346 = tensor.empty() : tensor<1x6x64xf32>
    %1347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1341, %1341 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1346 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb108(%1348: f32, %1349: f32, %1350: f32):
      %1351 = arith.mulf %1348, %1349 : f32
      linalg.yield %1351 : f32
    } -> tensor<1x6x64xf32>
    %1352 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1353 = tensor.splat %1352 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1354 = linalg.reduce ins(%1347:tensor<1x6x64xf32>) outs(%1353:tensor<1x6xf32>) dimensions = [2]
    (%1355: f32, %1356: f32) {
      %1357 = arith.addf %1355, %1356 : f32
      linalg.yield %1357 : f32
    }
    %1358 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1359 = tensor.splat %1358 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1360 = tensor.empty() : tensor<1x6xf32>
    %1361 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1354, %1359 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1360 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb109(%1362: f32, %1363: f32, %1364: f32):
      %1365 = arith.divf %1362, %1363 : f32
      linalg.yield %1365 : f32
    } -> tensor<1x6xf32>
    %1366 = tensor.collapse_shape %1361 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1367 = tensor.expand_shape %1366 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1368 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1369 = tensor.splat %1368 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1370 = tensor.empty() : tensor<1x6x1xf32>
    %1371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1367, %1369 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1370 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb110(%1372: f32, %1373: f32, %1374: f32):
      %1375 = arith.addf %1372, %1373 : f32
      linalg.yield %1375 : f32
    } -> tensor<1x6x1xf32>
    %1376 = tensor.empty() : tensor<1x6x1xf32>
    %1377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1371 : tensor<1x6x1xf32>) outs(%1376 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb111(%1378: f32, %1379: f32):
      %1380 = math.rsqrt %1378 : f32
      linalg.yield %1380 : f32
    } -> tensor<1x6x1xf32>
    %1381 = tensor.empty() : tensor<1x6x64xf32>
    %1382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1341, %1377 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1381 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb112(%1383: f32, %1384: f32, %1385: f32):
      %1386 = arith.mulf %1383, %1384 : f32
      linalg.yield %1386 : f32
    } -> tensor<1x6x64xf32>
    %1387 = tensor.empty() : tensor<1x6x64xf32>
    %1388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1382, %66 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1387 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb113(%1389: f32, %1390: f32, %1391: f32):
      %1392 = arith.mulf %1389, %1390 : f32
      linalg.yield %1392 : f32
    } -> tensor<1x6x64xf32>
    %1393 = tensor.empty() : tensor<1x6x64xf32>
    %1394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1388, %67 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1393 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb114(%1395: f32, %1396: f32, %1397: f32):
      %1398 = arith.addf %1395, %1396 : f32
      linalg.yield %1398 : f32
    } -> tensor<1x6x64xf32>
    %1399 = tensor.empty() : tensor<64x128xi8>
    %1400 = linalg.transpose ins(%69:tensor<128x64xi8>) outs(%1399:tensor<64x128xi8>) permutation = [1, 0]
    %1401 = tensor.collapse_shape %1394 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1402 = tensor.expand_shape %1401 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1403 = tensor.empty() : tensor<64x128xf32>
    %1404 = arith.constant 0 : i32
    %1405 = tensor.splat %1404 : tensor<128xi32>
    %1406 = "quant_ext.dequantize_per_channel"(%1400, %70, %1405) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<64x128xf32>
    %1407 = tensor.empty() : tensor<6x128xf32>
    %1408 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1409 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1408 : f32) outs(%1407 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1410 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} ins(%1402, %1406 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1409 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1411 = tensor.empty() : tensor<6x128xf32>
    %1412 = tensor.collapse_shape %1410 [[0 : i64, 1 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1413 = tensor.expand_shape %1412 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1414 = tensor.empty() : tensor<1x6x128xf32>
    %1415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1413, %68 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1414 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb115(%1416: f32, %1417: f32, %1418: f32):
      %1419 = arith.addf %1416, %1417 : f32
      linalg.yield %1419 : f32
    } -> tensor<1x6x128xf32>
    %1420 = tensor.collapse_shape %1415 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1421 = tensor.expand_shape %1420 [[0 : i64, 1 : i64]] output_shape [6, 128] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<768xf32> into tensor<6x128xf32>
    %1422 = tensor.collapse_shape %1421 [[0 : i64, 1 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6x128xf32> into tensor<768xf32>
    %1423 = tensor.expand_shape %1422 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1424 = tensor.collapse_shape %1423 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1425 = tensor.expand_shape %1424 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1426 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1427 = linalg.transpose ins(%1425:tensor<1x6x2x2x32xf32>) outs(%1426:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1428 = "tensor.extract_slice"(%1427) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1429 = tensor.collapse_shape %1428 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1430 = tensor.expand_shape %1429 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1431 = "tensor.extract_slice"(%1427) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1432 = tensor.collapse_shape %1431 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1433 = tensor.expand_shape %1432 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1434 = tensor.empty() : tensor<64x64xi8>
    %1435 = linalg.transpose ins(%73:tensor<64x64xi8>) outs(%1434:tensor<64x64xi8>) permutation = [1, 0]
    %1436 = tensor.collapse_shape %1290 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1437 = tensor.expand_shape %1436 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1438 = tensor.empty() : tensor<64x64xf32>
    %1439 = arith.constant 0 : i32
    %1440 = tensor.splat %1439 : tensor<64xi32>
    %1441 = "quant_ext.dequantize_per_channel"(%1435, %74, %1440) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1442 = tensor.empty() : tensor<96x64xf32>
    %1443 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1444 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1443 : f32) outs(%1442 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1445 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} ins(%1437, %1441 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1444 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1446 = tensor.empty() : tensor<96x64xf32>
    %1447 = tensor.collapse_shape %1445 [[0 : i64, 1 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1448 = tensor.expand_shape %1447 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1449 = tensor.empty() : tensor<1x96x64xf32>
    %1450 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1448, %72 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1449 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb116(%1451: f32, %1452: f32, %1453: f32):
      %1454 = arith.addf %1451, %1452 : f32
      linalg.yield %1454 : f32
    } -> tensor<1x96x64xf32>
    %1455 = tensor.collapse_shape %1450 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1456 = tensor.expand_shape %1455 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1457 = tensor.collapse_shape %1456 [[0 : i64, 1 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96x64xf32> into tensor<6144xf32>
    %1458 = tensor.expand_shape %1457 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1459 = tensor.collapse_shape %1458 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1460 = tensor.expand_shape %1459 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1461 = tensor.empty() : tensor<1x2x96x32xf32>
    %1462 = linalg.transpose ins(%1460:tensor<1x96x2x32xf32>) outs(%1461:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1463 = tensor.empty() : tensor<1x2x32x6xf32>
    %1464 = linalg.transpose ins(%1430:tensor<1x2x6x32xf32>) outs(%1463:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1465 = tensor.empty() : tensor<1x2x96x32xf32>
    %1466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1462 : tensor<1x2x96x32xf32>) outs(%1465 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb117(%1467: f32, %1468: f32):
      linalg.yield %1467 : f32
    } -> tensor<1x2x96x32xf32>
    %1469 = tensor.collapse_shape %1466 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1470 = tensor.expand_shape %1469 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1471 = tensor.empty() : tensor<1x2x32x6xf32>
    %1472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1464 : tensor<1x2x32x6xf32>) outs(%1471 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb118(%1473: f32, %1474: f32):
      linalg.yield %1473 : f32
    } -> tensor<1x2x32x6xf32>
    %1475 = tensor.collapse_shape %1472 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1476 = tensor.expand_shape %1475 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1477 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1478 = tensor.splat %1477 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1479 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1470, %1476 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1478 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb119(%1480: f32, %1481: f32, %1482: f32):
      %1483 = arith.mulf %1480, %1481 : f32
      %1484 = arith.addf %1482, %1483 : f32
      linalg.yield %1484 : f32
    } -> tensor<2x96x6xf32>
    %1485 = tensor.collapse_shape %1479 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1486 = tensor.expand_shape %1485 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1487 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1488 = tensor.splat %1487 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1489 = tensor.empty() : tensor<1x2x96x6xf32>
    %1490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1486, %1488 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1489 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb120(%1491: f32, %1492: f32, %1493: f32):
      %1494 = arith.divf %1491, %1492 : f32
      linalg.yield %1494 : f32
    } -> tensor<1x2x96x6xf32>
    %1495 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1496 = tensor.splat %1495 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1497 = linalg.reduce ins(%1490:tensor<1x2x96x6xf32>) outs(%1496:tensor<1x2x96xf32>) dimensions = [3]
    (%1498: f32, %1499: f32) {
      %1500 = arith.maximumf %1498, %1499 : f32
      linalg.yield %1500 : f32
    }
    %1501 = tensor.collapse_shape %1497 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1502 = tensor.expand_shape %1501 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1503 = tensor.empty() : tensor<1x2x96x6xf32>
    %1504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1490, %1502 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1503 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb121(%1505: f32, %1506: f32, %1507: f32):
      %1508 = arith.subf %1505, %1506 : f32
      linalg.yield %1508 : f32
    } -> tensor<1x2x96x6xf32>
    %1509 = tensor.empty() : tensor<1x2x96x6xf32>
    %1510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1504 : tensor<1x2x96x6xf32>) outs(%1509 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb122(%1511: f32, %1512: f32):
      %1513 = math.exp %1511 : f32
      linalg.yield %1513 : f32
    } -> tensor<1x2x96x6xf32>
    %1514 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1515 = tensor.splat %1514 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1516 = linalg.reduce ins(%1510:tensor<1x2x96x6xf32>) outs(%1515:tensor<1x2x96xf32>) dimensions = [3]
    (%1517: f32, %1518: f32) {
      %1519 = arith.addf %1517, %1518 : f32
      linalg.yield %1519 : f32
    }
    %1520 = tensor.collapse_shape %1516 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1521 = tensor.expand_shape %1520 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1522 = tensor.empty() : tensor<1x2x96x6xf32>
    %1523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1510, %1521 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1522 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb123(%1524: f32, %1525: f32, %1526: f32):
      %1527 = arith.divf %1524, %1525 : f32
      linalg.yield %1527 : f32
    } -> tensor<1x2x96x6xf32>
    %1528 = tensor.empty() : tensor<1x2x96x6xf32>
    %1529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1523 : tensor<1x2x96x6xf32>) outs(%1528 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb124(%1530: f32, %1531: f32):
      linalg.yield %1530 : f32
    } -> tensor<1x2x96x6xf32>
    %1532 = tensor.collapse_shape %1529 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %1533 = tensor.expand_shape %1532 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %1534 = tensor.empty() : tensor<1x2x6x32xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1433 : tensor<1x2x6x32xf32>) outs(%1534 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb125(%1536: f32, %1537: f32):
      linalg.yield %1536 : f32
    } -> tensor<1x2x6x32xf32>
    %1538 = tensor.collapse_shape %1535 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %1539 = tensor.expand_shape %1538 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %1540 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1541 = tensor.splat %1540 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %1542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1533, %1539 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%1541 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb126(%1543: f32, %1544: f32, %1545: f32):
      %1546 = arith.mulf %1543, %1544 : f32
      %1547 = arith.addf %1545, %1546 : f32
      linalg.yield %1547 : f32
    } -> tensor<2x96x32xf32>
    %1548 = tensor.collapse_shape %1542 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %1549 = tensor.expand_shape %1548 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %1550 = tensor.empty() : tensor<1x96x2x32xf32>
    %1551 = linalg.transpose ins(%1549:tensor<1x2x96x32xf32>) outs(%1550:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1552 = tensor.collapse_shape %1551 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1553 = tensor.expand_shape %1552 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1554 = tensor.empty() : tensor<64x64xi8>
    %1555 = linalg.transpose ins(%77:tensor<64x64xi8>) outs(%1554:tensor<64x64xi8>) permutation = [1, 0]
    %1556 = tensor.collapse_shape %1553 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1557 = tensor.expand_shape %1556 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1558 = tensor.empty() : tensor<64x64xf32>
    %1559 = arith.constant 0 : i32
    %1560 = tensor.splat %1559 : tensor<64xi32>
    %1561 = "quant_ext.dequantize_per_channel"(%1555, %78, %1560) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1562 = tensor.empty() : tensor<96x64xf32>
    %1563 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1564 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1563 : f32) outs(%1562 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1565 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} ins(%1557, %1561 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1564 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1566 = tensor.empty() : tensor<96x64xf32>
    %1567 = tensor.collapse_shape %1565 [[0 : i64, 1 : i64]] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1568 = tensor.expand_shape %1567 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1569 = tensor.empty() : tensor<1x96x64xf32>
    %1570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1568, %76 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1569 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb127(%1571: f32, %1572: f32, %1573: f32):
      %1574 = arith.addf %1571, %1572 : f32
      linalg.yield %1574 : f32
    } -> tensor<1x96x64xf32>
    %1575 = tensor.collapse_shape %1570 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1576 = tensor.expand_shape %1575 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1577 = tensor.collapse_shape %1576 [[0 : i64, 1 : i64]] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1578 = tensor.expand_shape %1577 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1579 = tensor.empty() : tensor<1x96x64xf32>
    %1580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1290, %1578 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1579 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb128(%1581: f32, %1582: f32, %1583: f32):
      %1584 = arith.addf %1581, %1582 : f32
      linalg.yield %1584 : f32
    } -> tensor<1x96x64xf32>
    %1585 = tensor.empty() : tensor<64x512xi8>
    %1586 = linalg.transpose ins(%97:tensor<512x64xi8>) outs(%1585:tensor<64x512xi8>) permutation = [1, 0]
    %1587 = tensor.collapse_shape %1580 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1588 = tensor.expand_shape %1587 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1589 = tensor.empty() : tensor<64x512xf32>
    %1590 = arith.constant 0 : i32
    %1591 = tensor.splat %1590 : tensor<512xi32>
    %1592 = "quant_ext.dequantize_per_channel"(%1586, %98, %1591) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<64x512xf32>
    %1593 = tensor.empty() : tensor<96x512xf32>
    %1594 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1595 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1594 : f32) outs(%1593 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1596 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} ins(%1588, %1592 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1595 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1597 = tensor.empty() : tensor<96x512xf32>
    %1598 = tensor.collapse_shape %1596 [[0 : i64, 1 : i64]] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1599 = tensor.expand_shape %1598 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1600 = tensor.empty() : tensor<1x96x512xf32>
    %1601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1599, %96 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1600 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb129(%1602: f32, %1603: f32, %1604: f32):
      %1605 = arith.addf %1602, %1603 : f32
      linalg.yield %1605 : f32
    } -> tensor<1x96x512xf32>
    %1606 = tensor.collapse_shape %1601 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1607 = tensor.expand_shape %1606 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<49152xf32> into tensor<96x512xf32>
    %1608 = tensor.collapse_shape %1607 [[0 : i64, 1 : i64]] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<96x512xf32> into tensor<49152xf32>
    %1609 = tensor.expand_shape %1608 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1610 = tensor.empty() : tensor<1x512x96xf32>
    %1611 = linalg.transpose ins(%1609:tensor<1x96x512xf32>) outs(%1610:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1612 = tensor.collapse_shape %1611 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1613 = tensor.expand_shape %1612 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1614 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1615 = tensor.splat %1614 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1616 = "tensor.insert_slice"(%1613, %1615) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1617 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1616 : tensor<1x512x10x14xf32>) outs(%1617 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb130(%1619: f32, %1620: f32):
      linalg.yield %1619 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1621 = tensor.collapse_shape %1618 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1622 = tensor.expand_shape %1621 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1623 = tensor.collapse_shape %100 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1624 = tensor.expand_shape %1623 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1625 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1626 = tensor.splat %1625 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1624, %1622 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1626 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb131(%1628: f32, %1629: f32, %1630: f32):
      %1631 = arith.mulf %1628, %1629 : f32
      %1632 = arith.addf %1630, %1631 : f32
      linalg.yield %1632 : f32
    } -> tensor<64x8x96xf32>
    %1633 = tensor.collapse_shape %1627 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1634 = tensor.expand_shape %1633 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1635 = tensor.collapse_shape %1634 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1636 = tensor.expand_shape %1635 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1637 = tensor.empty() : tensor<1x512x8x12xf32>
    %1638 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1636, %101 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1637 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb132(%1639: f32, %1640: f32, %1641: f32):
      %1642 = arith.addf %1639, %1640 : f32
      linalg.yield %1642 : f32
    } -> tensor<1x512x8x12xf32>
    %1643 = tensor.collapse_shape %1638 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1644 = tensor.expand_shape %1643 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1645 = tensor.empty() : tensor<1x96x512xf32>
    %1646 = linalg.transpose ins(%1644:tensor<1x512x96xf32>) outs(%1645:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1647 = tensor.empty() : tensor<1x96x512xf32>
    %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1646 : tensor<1x96x512xf32>) outs(%1647 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb133(%1649: f32, %1650: f32):
      %1651 = arith.constant 5.000000e-01 : f32
      %1652 = arith.constant 1.000000e+00 : f32
      %1653 = arith.constant 0.707106769 : f32
      %1654 = arith.mulf %1649, %1653 : f32
      %1655 = math.erf %1654 : f32
      %1656 = arith.addf %1652, %1655 : f32
      %1657 = arith.mulf %1651, %1649 : f32
      %1658 = arith.mulf %1657, %1656 : f32
      linalg.yield %1658 : f32
    } -> tensor<1x96x512xf32>
    %1659 = tensor.empty() : tensor<512x64xi8>
    %1660 = linalg.transpose ins(%103:tensor<64x512xi8>) outs(%1659:tensor<512x64xi8>) permutation = [1, 0]
    %1661 = tensor.collapse_shape %1648 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1662 = tensor.expand_shape %1661 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1663 = tensor.empty() : tensor<512x64xf32>
    %1664 = arith.constant 0 : i32
    %1665 = tensor.splat %1664 : tensor<64xi32>
    %1666 = "quant_ext.dequantize_per_channel"(%1660, %104, %1665) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<512x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<512x64xf32>
    %1667 = tensor.empty() : tensor<96x64xf32>
    %1668 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1669 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1668 : f32) outs(%1667 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1670 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} ins(%1662, %1666 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1669 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1671 = tensor.empty() : tensor<96x64xf32>
    %1672 = tensor.collapse_shape %1670 [[0 : i64, 1 : i64]] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1673 = tensor.expand_shape %1672 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1674 = tensor.empty() : tensor<1x96x64xf32>
    %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673, %102 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1674 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb134(%1676: f32, %1677: f32, %1678: f32):
      %1679 = arith.addf %1676, %1677 : f32
      linalg.yield %1679 : f32
    } -> tensor<1x96x64xf32>
    %1680 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1681 = tensor.expand_shape %1680 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<96x64xf32>
    %1682 = tensor.collapse_shape %1681 [[0 : i64, 1 : i64]] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1683 = tensor.expand_shape %1682 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1684 = tensor.empty() : tensor<1x96x64xf32>
    %1685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1580, %1683 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1684 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb135(%1686: f32, %1687: f32, %1688: f32):
      %1689 = arith.addf %1686, %1687 : f32
      linalg.yield %1689 : f32
    } -> tensor<1x96x64xf32>
    %1690 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1691 = tensor.splat %1690 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1692 = linalg.reduce ins(%1685:tensor<1x96x64xf32>) outs(%1691:tensor<1x96xf32>) dimensions = [2]
    (%1693: f32, %1694: f32) {
      %1695 = arith.addf %1693, %1694 : f32
      linalg.yield %1695 : f32
    }
    %1696 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1697 = tensor.splat %1696 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1698 = tensor.empty() : tensor<1x96xf32>
    %1699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1692, %1697 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1698 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb136(%1700: f32, %1701: f32, %1702: f32):
      %1703 = arith.divf %1700, %1701 : f32
      linalg.yield %1703 : f32
    } -> tensor<1x96xf32>
    %1704 = tensor.collapse_shape %1699 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1706 = tensor.empty() : tensor<1x96x64xf32>
    %1707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1685, %1705 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1706 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb137(%1708: f32, %1709: f32, %1710: f32):
      %1711 = arith.subf %1708, %1709 : f32
      linalg.yield %1711 : f32
    } -> tensor<1x96x64xf32>
    %1712 = tensor.empty() : tensor<1x96x64xf32>
    %1713 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707, %1707 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1712 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb138(%1714: f32, %1715: f32, %1716: f32):
      %1717 = arith.mulf %1714, %1715 : f32
      linalg.yield %1717 : f32
    } -> tensor<1x96x64xf32>
    %1718 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1719 = tensor.splat %1718 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1720 = linalg.reduce ins(%1713:tensor<1x96x64xf32>) outs(%1719:tensor<1x96xf32>) dimensions = [2]
    (%1721: f32, %1722: f32) {
      %1723 = arith.addf %1721, %1722 : f32
      linalg.yield %1723 : f32
    }
    %1724 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1725 = tensor.splat %1724 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1726 = tensor.empty() : tensor<1x96xf32>
    %1727 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1720, %1725 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1726 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb139(%1728: f32, %1729: f32, %1730: f32):
      %1731 = arith.divf %1728, %1729 : f32
      linalg.yield %1731 : f32
    } -> tensor<1x96xf32>
    %1732 = tensor.collapse_shape %1727 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1733 = tensor.expand_shape %1732 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1734 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1735 = tensor.splat %1734 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1736 = tensor.empty() : tensor<1x96x1xf32>
    %1737 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1733, %1735 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1736 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb140(%1738: f32, %1739: f32, %1740: f32):
      %1741 = arith.addf %1738, %1739 : f32
      linalg.yield %1741 : f32
    } -> tensor<1x96x1xf32>
    %1742 = tensor.empty() : tensor<1x96x1xf32>
    %1743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1737 : tensor<1x96x1xf32>) outs(%1742 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb141(%1744: f32, %1745: f32):
      %1746 = math.rsqrt %1744 : f32
      linalg.yield %1746 : f32
    } -> tensor<1x96x1xf32>
    %1747 = tensor.empty() : tensor<1x96x64xf32>
    %1748 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707, %1743 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1747 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb142(%1749: f32, %1750: f32, %1751: f32):
      %1752 = arith.mulf %1749, %1750 : f32
      linalg.yield %1752 : f32
    } -> tensor<1x96x64xf32>
    %1753 = tensor.empty() : tensor<1x96x64xf32>
    %1754 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1748, %116 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1753 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb143(%1755: f32, %1756: f32, %1757: f32):
      %1758 = arith.mulf %1755, %1756 : f32
      linalg.yield %1758 : f32
    } -> tensor<1x96x64xf32>
    %1759 = tensor.empty() : tensor<1x96x64xf32>
    %1760 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1754, %117 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1759 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb144(%1761: f32, %1762: f32, %1763: f32):
      %1764 = arith.addf %1761, %1762 : f32
      linalg.yield %1764 : f32
    } -> tensor<1x96x64xf32>
    %1765 = tensor.empty() : tensor<1x64x96xf32>
    %1766 = linalg.transpose ins(%1760:tensor<1x96x64xf32>) outs(%1765:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1767 = tensor.collapse_shape %1766 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1768 = tensor.expand_shape %1767 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1769 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1768 : tensor<1x64x8x12xf32>) outs(%1769 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb145(%1771: f32, %1772: f32):
      linalg.yield %1771 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1773 = tensor.collapse_shape %1770 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1774 = tensor.expand_shape %1773 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1775 = tensor.collapse_shape %80 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1776 = tensor.expand_shape %1775 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1777 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1778 = tensor.splat %1777 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1779 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1776, %1774 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1778 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1780 = tensor.collapse_shape %1779 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1781 = tensor.expand_shape %1780 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1782 = tensor.collapse_shape %1781 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1783 = tensor.expand_shape %1782 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1784 = tensor.empty() : tensor<1x64x2x3xf32>
    %1785 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1783, %81 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1784 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb146(%1786: f32, %1787: f32, %1788: f32):
      %1789 = arith.addf %1786, %1787 : f32
      linalg.yield %1789 : f32
    } -> tensor<1x64x2x3xf32>
    %1790 = tensor.collapse_shape %1785 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1791 = tensor.expand_shape %1790 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1792 = tensor.empty() : tensor<1x6x64xf32>
    %1793 = linalg.transpose ins(%1791:tensor<1x64x6xf32>) outs(%1792:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1794 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1795 = tensor.splat %1794 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1796 = linalg.reduce ins(%1793:tensor<1x6x64xf32>) outs(%1795:tensor<1x6xf32>) dimensions = [2]
    (%1797: f32, %1798: f32) {
      %1799 = arith.addf %1797, %1798 : f32
      linalg.yield %1799 : f32
    }
    %1800 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1801 = tensor.splat %1800 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1802 = tensor.empty() : tensor<1x6xf32>
    %1803 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1796, %1801 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1802 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb147(%1804: f32, %1805: f32, %1806: f32):
      %1807 = arith.divf %1804, %1805 : f32
      linalg.yield %1807 : f32
    } -> tensor<1x6xf32>
    %1808 = tensor.collapse_shape %1803 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1809 = tensor.expand_shape %1808 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1810 = tensor.empty() : tensor<1x6x64xf32>
    %1811 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1793, %1809 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1810 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb148(%1812: f32, %1813: f32, %1814: f32):
      %1815 = arith.subf %1812, %1813 : f32
      linalg.yield %1815 : f32
    } -> tensor<1x6x64xf32>
    %1816 = tensor.empty() : tensor<1x6x64xf32>
    %1817 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1811, %1811 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1816 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb149(%1818: f32, %1819: f32, %1820: f32):
      %1821 = arith.mulf %1818, %1819 : f32
      linalg.yield %1821 : f32
    } -> tensor<1x6x64xf32>
    %1822 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1823 = tensor.splat %1822 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1824 = linalg.reduce ins(%1817:tensor<1x6x64xf32>) outs(%1823:tensor<1x6xf32>) dimensions = [2]
    (%1825: f32, %1826: f32) {
      %1827 = arith.addf %1825, %1826 : f32
      linalg.yield %1827 : f32
    }
    %1828 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1829 = tensor.splat %1828 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1830 = tensor.empty() : tensor<1x6xf32>
    %1831 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1824, %1829 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1830 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb150(%1832: f32, %1833: f32, %1834: f32):
      %1835 = arith.divf %1832, %1833 : f32
      linalg.yield %1835 : f32
    } -> tensor<1x6xf32>
    %1836 = tensor.collapse_shape %1831 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1837 = tensor.expand_shape %1836 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1838 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1839 = tensor.splat %1838 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1840 = tensor.empty() : tensor<1x6x1xf32>
    %1841 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1837, %1839 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1840 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb151(%1842: f32, %1843: f32, %1844: f32):
      %1845 = arith.addf %1842, %1843 : f32
      linalg.yield %1845 : f32
    } -> tensor<1x6x1xf32>
    %1846 = tensor.empty() : tensor<1x6x1xf32>
    %1847 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1841 : tensor<1x6x1xf32>) outs(%1846 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb152(%1848: f32, %1849: f32):
      %1850 = math.rsqrt %1848 : f32
      linalg.yield %1850 : f32
    } -> tensor<1x6x1xf32>
    %1851 = tensor.empty() : tensor<1x6x64xf32>
    %1852 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1811, %1847 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1851 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb153(%1853: f32, %1854: f32, %1855: f32):
      %1856 = arith.mulf %1853, %1854 : f32
      linalg.yield %1856 : f32
    } -> tensor<1x6x64xf32>
    %1857 = tensor.empty() : tensor<1x6x64xf32>
    %1858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1852, %82 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1857 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb154(%1859: f32, %1860: f32, %1861: f32):
      %1862 = arith.mulf %1859, %1860 : f32
      linalg.yield %1862 : f32
    } -> tensor<1x6x64xf32>
    %1863 = tensor.empty() : tensor<1x6x64xf32>
    %1864 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1858, %83 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1863 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb155(%1865: f32, %1866: f32, %1867: f32):
      %1868 = arith.addf %1865, %1866 : f32
      linalg.yield %1868 : f32
    } -> tensor<1x6x64xf32>
    %1869 = tensor.empty() : tensor<64x128xi8>
    %1870 = linalg.transpose ins(%85:tensor<128x64xi8>) outs(%1869:tensor<64x128xi8>) permutation = [1, 0]
    %1871 = tensor.collapse_shape %1864 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1872 = tensor.expand_shape %1871 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1873 = tensor.empty() : tensor<64x128xf32>
    %1874 = arith.constant 0 : i32
    %1875 = tensor.splat %1874 : tensor<128xi32>
    %1876 = "quant_ext.dequantize_per_channel"(%1870, %86, %1875) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<64x128xf32>
    %1877 = tensor.empty() : tensor<6x128xf32>
    %1878 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1879 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1878 : f32) outs(%1877 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1880 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} ins(%1872, %1876 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1879 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1881 = tensor.empty() : tensor<6x128xf32>
    %1882 = tensor.collapse_shape %1880 [[0 : i64, 1 : i64]] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1883 = tensor.expand_shape %1882 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1884 = tensor.empty() : tensor<1x6x128xf32>
    %1885 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1883, %84 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1884 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb156(%1886: f32, %1887: f32, %1888: f32):
      %1889 = arith.addf %1886, %1887 : f32
      linalg.yield %1889 : f32
    } -> tensor<1x6x128xf32>
    %1890 = tensor.collapse_shape %1885 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1891 = tensor.expand_shape %1890 [[0 : i64, 1 : i64]] output_shape [6, 128] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<768xf32> into tensor<6x128xf32>
    %1892 = tensor.collapse_shape %1891 [[0 : i64, 1 : i64]] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6x128xf32> into tensor<768xf32>
    %1893 = tensor.expand_shape %1892 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1894 = tensor.collapse_shape %1893 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1895 = tensor.expand_shape %1894 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1896 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1897 = linalg.transpose ins(%1895:tensor<1x6x2x2x32xf32>) outs(%1896:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1898 = "tensor.extract_slice"(%1897) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1899 = tensor.collapse_shape %1898 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1900 = tensor.expand_shape %1899 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1901 = "tensor.extract_slice"(%1897) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1902 = tensor.collapse_shape %1901 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1903 = tensor.expand_shape %1902 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1904 = tensor.empty() : tensor<64x64xi8>
    %1905 = linalg.transpose ins(%89:tensor<64x64xi8>) outs(%1904:tensor<64x64xi8>) permutation = [1, 0]
    %1906 = tensor.collapse_shape %1760 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1907 = tensor.expand_shape %1906 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1908 = tensor.empty() : tensor<64x64xf32>
    %1909 = arith.constant 0 : i32
    %1910 = tensor.splat %1909 : tensor<64xi32>
    %1911 = "quant_ext.dequantize_per_channel"(%1905, %90, %1910) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1912 = tensor.empty() : tensor<96x64xf32>
    %1913 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1914 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1913 : f32) outs(%1912 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1915 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} ins(%1907, %1911 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1914 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1916 = tensor.empty() : tensor<96x64xf32>
    %1917 = tensor.collapse_shape %1915 [[0 : i64, 1 : i64]] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1918 = tensor.expand_shape %1917 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1919 = tensor.empty() : tensor<1x96x64xf32>
    %1920 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1918, %88 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1919 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb157(%1921: f32, %1922: f32, %1923: f32):
      %1924 = arith.addf %1921, %1922 : f32
      linalg.yield %1924 : f32
    } -> tensor<1x96x64xf32>
    %1925 = tensor.collapse_shape %1920 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_111", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1926 = tensor.expand_shape %1925 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_111", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1927 = tensor.collapse_shape %1926 [[0 : i64, 1 : i64]] {prov.region_id = "view_112", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96x64xf32> into tensor<6144xf32>
    %1928 = tensor.expand_shape %1927 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_112", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1929 = tensor.collapse_shape %1928 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_113", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1930 = tensor.expand_shape %1929 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_113", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1931 = tensor.empty() : tensor<1x2x96x32xf32>
    %1932 = linalg.transpose ins(%1930:tensor<1x96x2x32xf32>) outs(%1931:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1933 = tensor.empty() : tensor<1x2x32x6xf32>
    %1934 = linalg.transpose ins(%1900:tensor<1x2x6x32xf32>) outs(%1933:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1935 = tensor.empty() : tensor<1x2x96x32xf32>
    %1936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1932 : tensor<1x2x96x32xf32>) outs(%1935 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb158(%1937: f32, %1938: f32):
      linalg.yield %1937 : f32
    } -> tensor<1x2x96x32xf32>
    %1939 = tensor.collapse_shape %1936 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_114", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1940 = tensor.expand_shape %1939 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_114", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1941 = tensor.empty() : tensor<1x2x32x6xf32>
    %1942 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1934 : tensor<1x2x32x6xf32>) outs(%1941 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb159(%1943: f32, %1944: f32):
      linalg.yield %1943 : f32
    } -> tensor<1x2x32x6xf32>
    %1945 = tensor.collapse_shape %1942 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_115", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1946 = tensor.expand_shape %1945 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_115", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1947 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1948 = tensor.splat %1947 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1949 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1940, %1946 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1948 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%1950: f32, %1951: f32, %1952: f32):
      %1953 = arith.mulf %1950, %1951 : f32
      %1954 = arith.addf %1952, %1953 : f32
      linalg.yield %1954 : f32
    } -> tensor<2x96x6xf32>
    %1955 = tensor.collapse_shape %1949 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_116", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1956 = tensor.expand_shape %1955 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_116", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1957 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1958 = tensor.splat %1957 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1959 = tensor.empty() : tensor<1x2x96x6xf32>
    %1960 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1956, %1958 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1959 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%1961: f32, %1962: f32, %1963: f32):
      %1964 = arith.divf %1961, %1962 : f32
      linalg.yield %1964 : f32
    } -> tensor<1x2x96x6xf32>
    %1965 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1966 = tensor.splat %1965 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1967 = linalg.reduce ins(%1960:tensor<1x2x96x6xf32>) outs(%1966:tensor<1x2x96xf32>) dimensions = [3]
    (%1968: f32, %1969: f32) {
      %1970 = arith.maximumf %1968, %1969 : f32
      linalg.yield %1970 : f32
    }
    %1971 = tensor.collapse_shape %1967 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1972 = tensor.expand_shape %1971 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1973 = tensor.empty() : tensor<1x2x96x6xf32>
    %1974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1960, %1972 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1973 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb162(%1975: f32, %1976: f32, %1977: f32):
      %1978 = arith.subf %1975, %1976 : f32
      linalg.yield %1978 : f32
    } -> tensor<1x2x96x6xf32>
    %1979 = tensor.empty() : tensor<1x2x96x6xf32>
    %1980 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1974 : tensor<1x2x96x6xf32>) outs(%1979 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb163(%1981: f32, %1982: f32):
      %1983 = math.exp %1981 : f32
      linalg.yield %1983 : f32
    } -> tensor<1x2x96x6xf32>
    %1984 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1985 = tensor.splat %1984 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1986 = linalg.reduce ins(%1980:tensor<1x2x96x6xf32>) outs(%1985:tensor<1x2x96xf32>) dimensions = [3]
    (%1987: f32, %1988: f32) {
      %1989 = arith.addf %1987, %1988 : f32
      linalg.yield %1989 : f32
    }
    %1990 = tensor.collapse_shape %1986 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1991 = tensor.expand_shape %1990 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1992 = tensor.empty() : tensor<1x2x96x6xf32>
    %1993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1980, %1991 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1992 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb164(%1994: f32, %1995: f32, %1996: f32):
      %1997 = arith.divf %1994, %1995 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x2x96x6xf32>
    %1998 = tensor.empty() : tensor<1x2x96x6xf32>
    %1999 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1993 : tensor<1x2x96x6xf32>) outs(%1998 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb165(%2000: f32, %2001: f32):
      linalg.yield %2000 : f32
    } -> tensor<1x2x96x6xf32>
    %2002 = tensor.collapse_shape %1999 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_117", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %2003 = tensor.expand_shape %2002 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_117", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %2004 = tensor.empty() : tensor<1x2x6x32xf32>
    %2005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1903 : tensor<1x2x6x32xf32>) outs(%2004 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb166(%2006: f32, %2007: f32):
      linalg.yield %2006 : f32
    } -> tensor<1x2x6x32xf32>
    %2008 = tensor.collapse_shape %2005 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_118", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %2009 = tensor.expand_shape %2008 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_118", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %2010 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2011 = tensor.splat %2010 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %2012 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2003, %2009 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%2011 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb167(%2013: f32, %2014: f32, %2015: f32):
      %2016 = arith.mulf %2013, %2014 : f32
      %2017 = arith.addf %2015, %2016 : f32
      linalg.yield %2017 : f32
    } -> tensor<2x96x32xf32>
    %2018 = tensor.collapse_shape %2012 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_119", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %2019 = tensor.expand_shape %2018 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_119", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %2020 = tensor.empty() : tensor<1x96x2x32xf32>
    %2021 = linalg.transpose ins(%2019:tensor<1x2x96x32xf32>) outs(%2020:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %2022 = tensor.collapse_shape %2021 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_120", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %2023 = tensor.expand_shape %2022 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_120", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2024 = tensor.empty() : tensor<64x64xi8>
    %2025 = linalg.transpose ins(%93:tensor<64x64xi8>) outs(%2024:tensor<64x64xi8>) permutation = [1, 0]
    %2026 = tensor.collapse_shape %2023 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_121", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2027 = tensor.expand_shape %2026 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_121", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %2028 = tensor.empty() : tensor<64x64xf32>
    %2029 = arith.constant 0 : i32
    %2030 = tensor.splat %2029 : tensor<64xi32>
    %2031 = "quant_ext.dequantize_per_channel"(%2025, %94, %2030) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %2032 = tensor.empty() : tensor<96x64xf32>
    %2033 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2034 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2033 : f32) outs(%2032 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2035 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} ins(%2027, %2031 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%2034 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2036 = tensor.empty() : tensor<96x64xf32>
    %2037 = tensor.collapse_shape %2035 [[0 : i64, 1 : i64]] {prov.region_id = "view_122", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %2038 = tensor.expand_shape %2037 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_122", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2039 = tensor.empty() : tensor<1x96x64xf32>
    %2040 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2038, %92 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2039 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb168(%2041: f32, %2042: f32, %2043: f32):
      %2044 = arith.addf %2041, %2042 : f32
      linalg.yield %2044 : f32
    } -> tensor<1x96x64xf32>
    %2045 = tensor.collapse_shape %2040 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_123", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2046 = tensor.expand_shape %2045 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_123", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %2047 = tensor.collapse_shape %2046 [[0 : i64, 1 : i64]] {prov.region_id = "view_124", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %2048 = tensor.expand_shape %2047 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_124", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2049 = tensor.empty() : tensor<1x96x64xf32>
    %2050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1760, %2048 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2049 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb169(%2051: f32, %2052: f32, %2053: f32):
      %2054 = arith.addf %2051, %2052 : f32
      linalg.yield %2054 : f32
    } -> tensor<1x96x64xf32>
    %2055 = tensor.empty() : tensor<64x512xi8>
    %2056 = linalg.transpose ins(%107:tensor<512x64xi8>) outs(%2055:tensor<64x512xi8>) permutation = [1, 0]
    %2057 = tensor.collapse_shape %2050 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_125", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2058 = tensor.expand_shape %2057 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_125", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %2059 = tensor.empty() : tensor<64x512xf32>
    %2060 = arith.constant 0 : i32
    %2061 = tensor.splat %2060 : tensor<512xi32>
    %2062 = "quant_ext.dequantize_per_channel"(%2056, %108, %2061) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<64x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<64x512xf32>
    %2063 = tensor.empty() : tensor<96x512xf32>
    %2064 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2065 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2064 : f32) outs(%2063 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %2066 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} ins(%2058, %2062 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%2065 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %2067 = tensor.empty() : tensor<96x512xf32>
    %2068 = tensor.collapse_shape %2066 [[0 : i64, 1 : i64]] {prov.region_id = "view_126", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %2069 = tensor.expand_shape %2068 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_126", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %2070 = tensor.empty() : tensor<1x96x512xf32>
    %2071 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2069, %106 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%2070 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb170(%2072: f32, %2073: f32, %2074: f32):
      %2075 = arith.addf %2072, %2073 : f32
      linalg.yield %2075 : f32
    } -> tensor<1x96x512xf32>
    %2076 = tensor.collapse_shape %2071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_127", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %2077 = tensor.expand_shape %2076 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_127", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<49152xf32> into tensor<96x512xf32>
    %2078 = tensor.collapse_shape %2077 [[0 : i64, 1 : i64]] {prov.region_id = "view_128", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<96x512xf32> into tensor<49152xf32>
    %2079 = tensor.expand_shape %2078 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_128", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %2080 = tensor.empty() : tensor<1x512x96xf32>
    %2081 = linalg.transpose ins(%2079:tensor<1x96x512xf32>) outs(%2080:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %2082 = tensor.collapse_shape %2081 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_129", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %2083 = tensor.expand_shape %2082 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_129", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2084 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2085 = tensor.splat %2084 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %2086 = "tensor.insert_slice"(%2083, %2085) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %2087 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %2088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2086 : tensor<1x512x10x14xf32>) outs(%2087 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb171(%2089: f32, %2090: f32):
      linalg.yield %2089 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %2091 = tensor.collapse_shape %2088 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %2092 = tensor.expand_shape %2091 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %2093 = tensor.collapse_shape %110 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %2094 = tensor.expand_shape %2093 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %2095 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2096 = tensor.splat %2095 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %2097 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2094, %2092 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%2096 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb172(%2098: f32, %2099: f32, %2100: f32):
      %2101 = arith.mulf %2098, %2099 : f32
      %2102 = arith.addf %2100, %2101 : f32
      linalg.yield %2102 : f32
    } -> tensor<64x8x96xf32>
    %2103 = tensor.collapse_shape %2097 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %2104 = tensor.expand_shape %2103 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %2105 = tensor.collapse_shape %2104 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %2106 = tensor.expand_shape %2105 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2107 = tensor.empty() : tensor<1x512x8x12xf32>
    %2108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2106, %111 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%2107 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb173(%2109: f32, %2110: f32, %2111: f32):
      %2112 = arith.addf %2109, %2110 : f32
      linalg.yield %2112 : f32
    } -> tensor<1x512x8x12xf32>
    %2113 = tensor.collapse_shape %2108 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_130", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %2114 = tensor.expand_shape %2113 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_130", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %2115 = tensor.empty() : tensor<1x96x512xf32>
    %2116 = linalg.transpose ins(%2114:tensor<1x512x96xf32>) outs(%2115:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %2117 = tensor.empty() : tensor<1x96x512xf32>
    %2118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2116 : tensor<1x96x512xf32>) outs(%2117 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb174(%2119: f32, %2120: f32):
      %2121 = arith.constant 5.000000e-01 : f32
      %2122 = arith.constant 1.000000e+00 : f32
      %2123 = arith.constant 0.707106769 : f32
      %2124 = arith.mulf %2119, %2123 : f32
      %2125 = math.erf %2124 : f32
      %2126 = arith.addf %2122, %2125 : f32
      %2127 = arith.mulf %2121, %2119 : f32
      %2128 = arith.mulf %2127, %2126 : f32
      linalg.yield %2128 : f32
    } -> tensor<1x96x512xf32>
    %2129 = tensor.empty() : tensor<512x64xi8>
    %2130 = linalg.transpose ins(%113:tensor<64x512xi8>) outs(%2129:tensor<512x64xi8>) permutation = [1, 0]
    %2131 = tensor.collapse_shape %2118 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_131", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %2132 = tensor.expand_shape %2131 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_131", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %2133 = tensor.empty() : tensor<512x64xf32>
    %2134 = arith.constant 0 : i32
    %2135 = tensor.splat %2134 : tensor<64xi32>
    %2136 = "quant_ext.dequantize_per_channel"(%2130, %114, %2135) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<512x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<512x64xf32>
    %2137 = tensor.empty() : tensor<96x64xf32>
    %2138 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2139 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2138 : f32) outs(%2137 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2140 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} ins(%2132, %2136 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%2139 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2141 = tensor.empty() : tensor<96x64xf32>
    %2142 = tensor.collapse_shape %2140 [[0 : i64, 1 : i64]] {prov.region_id = "view_132", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %2143 = tensor.expand_shape %2142 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_132", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2144 = tensor.empty() : tensor<1x96x64xf32>
    %2145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2143, %112 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2144 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb175(%2146: f32, %2147: f32, %2148: f32):
      %2149 = arith.addf %2146, %2147 : f32
      linalg.yield %2149 : f32
    } -> tensor<1x96x64xf32>
    %2150 = tensor.collapse_shape %2145 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_133", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2151 = tensor.expand_shape %2150 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_133", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<96x64xf32>
    %2152 = tensor.collapse_shape %2151 [[0 : i64, 1 : i64]] {prov.region_id = "view_134", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %2153 = tensor.expand_shape %2152 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_134", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2154 = tensor.empty() : tensor<1x96x64xf32>
    %2155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2050, %2153 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2154 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb176(%2156: f32, %2157: f32, %2158: f32):
      %2159 = arith.addf %2156, %2157 : f32
      linalg.yield %2159 : f32
    } -> tensor<1x96x64xf32>
    %2160 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2161 = tensor.splat %2160 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2162 = linalg.reduce ins(%2155:tensor<1x96x64xf32>) outs(%2161:tensor<1x96xf32>) dimensions = [2]
    (%2163: f32, %2164: f32) {
      %2165 = arith.addf %2163, %2164 : f32
      linalg.yield %2165 : f32
    }
    %2166 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2167 = tensor.splat %2166 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2168 = tensor.empty() : tensor<1x96xf32>
    %2169 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2162, %2167 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2168 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb177(%2170: f32, %2171: f32, %2172: f32):
      %2173 = arith.divf %2170, %2171 : f32
      linalg.yield %2173 : f32
    } -> tensor<1x96xf32>
    %2174 = tensor.collapse_shape %2169 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2175 = tensor.expand_shape %2174 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2176 = tensor.empty() : tensor<1x96x64xf32>
    %2177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2155, %2175 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2176 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb178(%2178: f32, %2179: f32, %2180: f32):
      %2181 = arith.subf %2178, %2179 : f32
      linalg.yield %2181 : f32
    } -> tensor<1x96x64xf32>
    %2182 = tensor.empty() : tensor<1x96x64xf32>
    %2183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2177, %2177 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2182 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb179(%2184: f32, %2185: f32, %2186: f32):
      %2187 = arith.mulf %2184, %2185 : f32
      linalg.yield %2187 : f32
    } -> tensor<1x96x64xf32>
    %2188 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2189 = tensor.splat %2188 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2190 = linalg.reduce ins(%2183:tensor<1x96x64xf32>) outs(%2189:tensor<1x96xf32>) dimensions = [2]
    (%2191: f32, %2192: f32) {
      %2193 = arith.addf %2191, %2192 : f32
      linalg.yield %2193 : f32
    }
    %2194 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2195 = tensor.splat %2194 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2196 = tensor.empty() : tensor<1x96xf32>
    %2197 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2190, %2195 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2196 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb180(%2198: f32, %2199: f32, %2200: f32):
      %2201 = arith.divf %2198, %2199 : f32
      linalg.yield %2201 : f32
    } -> tensor<1x96xf32>
    %2202 = tensor.collapse_shape %2197 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2203 = tensor.expand_shape %2202 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2204 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %2205 = tensor.splat %2204 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %2206 = tensor.empty() : tensor<1x96x1xf32>
    %2207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2203, %2205 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%2206 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb181(%2208: f32, %2209: f32, %2210: f32):
      %2211 = arith.addf %2208, %2209 : f32
      linalg.yield %2211 : f32
    } -> tensor<1x96x1xf32>
    %2212 = tensor.empty() : tensor<1x96x1xf32>
    %2213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2207 : tensor<1x96x1xf32>) outs(%2212 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb182(%2214: f32, %2215: f32):
      %2216 = math.rsqrt %2214 : f32
      linalg.yield %2216 : f32
    } -> tensor<1x96x1xf32>
    %2217 = tensor.empty() : tensor<1x96x64xf32>
    %2218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2177, %2213 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2217 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb183(%2219: f32, %2220: f32, %2221: f32):
      %2222 = arith.mulf %2219, %2220 : f32
      linalg.yield %2222 : f32
    } -> tensor<1x96x64xf32>
    %2223 = tensor.empty() : tensor<1x96x64xf32>
    %2224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2218, %118 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2223 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb184(%2225: f32, %2226: f32, %2227: f32):
      %2228 = arith.mulf %2225, %2226 : f32
      linalg.yield %2228 : f32
    } -> tensor<1x96x64xf32>
    %2229 = tensor.empty() : tensor<1x96x64xf32>
    %2230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2224, %119 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2229 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb185(%2231: f32, %2232: f32, %2233: f32):
      %2234 = arith.addf %2231, %2232 : f32
      linalg.yield %2234 : f32
    } -> tensor<1x96x64xf32>
    %2235 = tensor.collapse_shape %2230 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_135", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2236 = tensor.expand_shape %2235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_135", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2237 = tensor.empty() : tensor<1x64x8x12xf32>
    %2238 = linalg.transpose ins(%2236:tensor<1x8x12x64xf32>) outs(%2237:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2239 = tensor.collapse_shape %2238 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_136", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2240 = tensor.expand_shape %2239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov.region_id = "view_136", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2241 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2242 = linalg.transpose ins(%2240:tensor<1x16x2x2x8x12xf32>) outs(%2241:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2243 = tensor.collapse_shape %2242 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_137", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2244 = tensor.expand_shape %2243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov.region_id = "view_137", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2245 = tensor.empty() : tensor<1x32x23x15xf32>
    %2246 = linalg.transpose ins(%1191:tensor<1x32x15x23xf32>) outs(%2245:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2247 = tensor.collapse_shape %2246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2248 = tensor.expand_shape %2247 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2249 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2250 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2251 = tensor.splat %2250 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2252 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2248, %2249 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2251 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2253 = tensor.collapse_shape %2252 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2254 = tensor.expand_shape %2253 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2255 = tensor.empty() : tensor<1x32x16x23xf32>
    %2256 = linalg.transpose ins(%2254:tensor<1x32x23x16xf32>) outs(%2255:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2257 = tensor.collapse_shape %2256 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2258 = tensor.expand_shape %2257 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2259 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2260 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2261 = tensor.splat %2260 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2262 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2258, %2259 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2261 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2263 = tensor.collapse_shape %2262 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2264 = tensor.expand_shape %2263 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2265 = tensor.concat dim(1) %2244, %2264 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2266 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2267 = tensor.splat %2266 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2268 = "tensor.insert_slice"(%2265, %2267) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2269 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2270 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2268 : tensor<1x48x18x26xf32>) outs(%2269 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb186(%2271: f32, %2272: f32):
      linalg.yield %2271 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2273 = tensor.collapse_shape %2270 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2274 = tensor.expand_shape %2273 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2275 = tensor.collapse_shape %140 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2276 = tensor.expand_shape %2275 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2277 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2278 = tensor.splat %2277 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2279 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2276, %2274 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2278 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2280 = tensor.collapse_shape %2279 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2281 = tensor.expand_shape %2280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2282 = tensor.collapse_shape %2281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2283 = tensor.expand_shape %2282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2284 = tensor.empty() : tensor<1x12x16x24xf32>
    %2285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2283, %141 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2284 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb187(%2286: f32, %2287: f32, %2288: f32):
      %2289 = arith.addf %2286, %2287 : f32
      linalg.yield %2289 : f32
    } -> tensor<1x12x16x24xf32>
    %2290 = tensor.collapse_shape %2285 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_138", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2291 = tensor.expand_shape %2290 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_138", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2292 = tensor.empty() : tensor<4608x512xi8>
    %2293 = linalg.transpose ins(%121:tensor<512x4608xi8>) outs(%2292:tensor<4608x512xi8>) permutation = [1, 0]
    %2294 = tensor.empty() : tensor<4608x512xf32>
    %2295 = arith.constant 0 : i32
    %2296 = tensor.splat %2295 : tensor<512xi32>
    %2297 = "quant_ext.dequantize_per_channel"(%2293, %122, %2296) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<4608x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<4608x512xf32>
    %2298 = tensor.empty() : tensor<1x512xf32>
    %2299 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2300 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2299 : f32) outs(%2298 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2301 = linalg.matmul {prov.region_id = "matmul_28", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%2291, %2297 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2300 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2302 = tensor.empty() : tensor<1x512xf32>
    %2303 = tensor.empty() : tensor<1x512xf32>
    %2304 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2301, %120 : tensor<1x512xf32>, tensor<512xf32>) outs(%2303 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb188(%2305: f32, %2306: f32, %2307: f32):
      %2308 = arith.addf %2305, %2306 : f32
      linalg.yield %2308 : f32
    } -> tensor<1x512xf32>
    %2309 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2310 = tensor.splat %2309 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2311 = tensor.empty() : tensor<1x1xf32>
    %2312 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%143, %2310 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2311 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb189(%2313: f32, %2314: f32, %2315: f32):
      %2316 = arith.divf %2313, %2314 : f32
      linalg.yield %2316 : f32
    } -> tensor<1x1xf32>
    %2317 = tensor.concat dim(1) %2304, %2312, %144 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2318 = tensor.collapse_shape %2317 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2319 = tensor.expand_shape %2318 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2320 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2321 = tensor.splat %2320 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2322 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2323 = tensor.splat %2322 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2324 = "tensor.extract_slice"(%2321) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2325 = "tensor.extract_slice"(%2321) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2326 = "tensor.extract_slice"(%2321) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2327 = tensor.collapse_shape %2324 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2328 = tensor.expand_shape %2327 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2329 = tensor.collapse_shape %2325 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2330 = tensor.expand_shape %2329 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2331 = tensor.collapse_shape %2326 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2332 = tensor.expand_shape %2331 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2333 = "tensor.extract_slice"(%2323) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2334 = "tensor.extract_slice"(%2323) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2335 = "tensor.extract_slice"(%2323) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2336 = tensor.collapse_shape %2333 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2337 = tensor.expand_shape %2336 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2338 = tensor.collapse_shape %2334 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2339 = tensor.expand_shape %2338 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2340 = tensor.collapse_shape %2335 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2341 = tensor.expand_shape %2340 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2342 = tensor.collapse_shape %2328 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2343 = tensor.expand_shape %2342 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2344 = tensor.collapse_shape %2337 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2345 = tensor.expand_shape %2344 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2346 = tensor.collapse_shape %2319 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_143", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2347 = tensor.expand_shape %2346 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov.region_id = "view_143", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2348 = tensor.empty() : tensor<517x512xf32>
    %2349 = linalg.transpose ins(%124:tensor<512x517xf32>) outs(%2348:tensor<517x512xf32>) permutation = [1, 0]
    %2350 = tensor.empty() : tensor<1x512xf32>
    %2351 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2352 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2351 : f32) outs(%2350 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2353 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2347, %2349 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2352 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2354 = tensor.empty() : tensor<1x512xf32>
    %2355 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2353, %126 : tensor<1x512xf32>, tensor<512xf32>) outs(%2354 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb190(%2356: f32, %2357: f32, %2358: f32):
      %2359 = arith.addf %2356, %2357 : f32
      linalg.yield %2359 : f32
    } -> tensor<1x512xf32>
    %2360 = tensor.collapse_shape %2355 [[0 : i64, 1 : i64]] {prov.region_id = "view_144", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2361 = tensor.expand_shape %2360 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_144", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2362 = "tensor.extract_slice"(%2361) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2363 = tensor.collapse_shape %2362 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2364 = tensor.expand_shape %2363 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2365 = tensor.collapse_shape %2343 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_145", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2366 = tensor.expand_shape %2365 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_145", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2367 = tensor.empty() : tensor<128x512xf32>
    %2368 = linalg.transpose ins(%125:tensor<512x128xf32>) outs(%2367:tensor<128x512xf32>) permutation = [1, 0]
    %2369 = tensor.empty() : tensor<1x512xf32>
    %2370 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2371 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2370 : f32) outs(%2369 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2372 = linalg.matmul {prov.region_id = "matmul_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2366, %2368 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2371 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2373 = tensor.empty() : tensor<1x512xf32>
    %2374 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2372, %127 : tensor<1x512xf32>, tensor<512xf32>) outs(%2373 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb191(%2375: f32, %2376: f32, %2377: f32):
      %2378 = arith.addf %2375, %2376 : f32
      linalg.yield %2378 : f32
    } -> tensor<1x512xf32>
    %2379 = tensor.collapse_shape %2374 [[0 : i64, 1 : i64]] {prov.region_id = "view_146", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2380 = tensor.expand_shape %2379 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_146", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2381 = tensor.empty() : tensor<1x1x512xf32>
    %2382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2380, %2364 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2381 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb192(%2383: f32, %2384: f32, %2385: f32):
      %2386 = arith.addf %2383, %2384 : f32
      linalg.yield %2386 : f32
    } -> tensor<1x1x512xf32>
    %2387 = "tensor.extract_slice"(%2382) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2388 = "tensor.extract_slice"(%2382) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2389 = "tensor.extract_slice"(%2382) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2390 = "tensor.extract_slice"(%2382) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2391 = tensor.empty() : tensor<1x1x128xf32>
    %2392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2387 : tensor<1x1x128xf32>) outs(%2391 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb193(%2393: f32, %2394: f32):
      %2395 = arith.constant 1.000000e+00 : f32
      %2396 = arith.negf %2393 : f32
      %2397 = math.exp %2396 : f32
      %2398 = arith.addf %2395, %2397 : f32
      %2399 = arith.divf %2395, %2398 : f32
      linalg.yield %2399 : f32
    } -> tensor<1x1x128xf32>
    %2400 = tensor.empty() : tensor<1x1x128xf32>
    %2401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2388 : tensor<1x1x128xf32>) outs(%2400 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb194(%2402: f32, %2403: f32):
      %2404 = arith.constant 1.000000e+00 : f32
      %2405 = arith.negf %2402 : f32
      %2406 = math.exp %2405 : f32
      %2407 = arith.addf %2404, %2406 : f32
      %2408 = arith.divf %2404, %2407 : f32
      linalg.yield %2408 : f32
    } -> tensor<1x1x128xf32>
    %2409 = tensor.empty() : tensor<1x1x128xf32>
    %2410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2389 : tensor<1x1x128xf32>) outs(%2409 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb195(%2411: f32, %2412: f32):
      %2413 = math.tanh %2411 : f32
      linalg.yield %2413 : f32
    } -> tensor<1x1x128xf32>
    %2414 = tensor.empty() : tensor<1x1x128xf32>
    %2415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2390 : tensor<1x1x128xf32>) outs(%2414 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb196(%2416: f32, %2417: f32):
      %2418 = arith.constant 1.000000e+00 : f32
      %2419 = arith.negf %2416 : f32
      %2420 = math.exp %2419 : f32
      %2421 = arith.addf %2418, %2420 : f32
      %2422 = arith.divf %2418, %2421 : f32
      linalg.yield %2422 : f32
    } -> tensor<1x1x128xf32>
    %2423 = tensor.empty() : tensor<1x1x128xf32>
    %2424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2401, %2345 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2423 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb197(%2425: f32, %2426: f32, %2427: f32):
      %2428 = arith.mulf %2425, %2426 : f32
      linalg.yield %2428 : f32
    } -> tensor<1x1x128xf32>
    %2429 = tensor.empty() : tensor<1x1x128xf32>
    %2430 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2392, %2410 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2429 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb198(%2431: f32, %2432: f32, %2433: f32):
      %2434 = arith.mulf %2431, %2432 : f32
      linalg.yield %2434 : f32
    } -> tensor<1x1x128xf32>
    %2435 = tensor.empty() : tensor<1x1x128xf32>
    %2436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2424, %2430 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2435 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb199(%2437: f32, %2438: f32, %2439: f32):
      %2440 = arith.addf %2437, %2438 : f32
      linalg.yield %2440 : f32
    } -> tensor<1x1x128xf32>
    %2441 = tensor.empty() : tensor<1x1x128xf32>
    %2442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2436 : tensor<1x1x128xf32>) outs(%2441 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_1", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb200(%2443: f32, %2444: f32):
      %2445 = math.tanh %2443 : f32
      linalg.yield %2445 : f32
    } -> tensor<1x1x128xf32>
    %2446 = tensor.empty() : tensor<1x1x128xf32>
    %2447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2415, %2442 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2446 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb201(%2448: f32, %2449: f32, %2450: f32):
      %2451 = arith.mulf %2448, %2449 : f32
      linalg.yield %2451 : f32
    } -> tensor<1x1x128xf32>
    %2452 = tensor.concat dim(0) %2447 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2453 = tensor.collapse_shape %2330 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2454 = tensor.expand_shape %2453 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2455 = tensor.collapse_shape %2339 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2456 = tensor.expand_shape %2455 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2457 = tensor.collapse_shape %2452 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_147", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2458 = tensor.expand_shape %2457 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_147", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2459 = tensor.empty() : tensor<128x512xf32>
    %2460 = linalg.transpose ins(%128:tensor<512x128xf32>) outs(%2459:tensor<128x512xf32>) permutation = [1, 0]
    %2461 = tensor.empty() : tensor<1x512xf32>
    %2462 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2463 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2462 : f32) outs(%2461 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2464 = linalg.matmul {prov.region_id = "matmul_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2458, %2460 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2463 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2465 = tensor.empty() : tensor<1x512xf32>
    %2466 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2464, %130 : tensor<1x512xf32>, tensor<512xf32>) outs(%2465 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb202(%2467: f32, %2468: f32, %2469: f32):
      %2470 = arith.addf %2467, %2468 : f32
      linalg.yield %2470 : f32
    } -> tensor<1x512xf32>
    %2471 = tensor.collapse_shape %2466 [[0 : i64, 1 : i64]] {prov.region_id = "view_148", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2472 = tensor.expand_shape %2471 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_148", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2473 = "tensor.extract_slice"(%2472) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2474 = tensor.collapse_shape %2473 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2475 = tensor.expand_shape %2474 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2476 = tensor.collapse_shape %2454 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_149", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2477 = tensor.expand_shape %2476 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_149", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2478 = tensor.empty() : tensor<128x512xf32>
    %2479 = linalg.transpose ins(%129:tensor<512x128xf32>) outs(%2478:tensor<128x512xf32>) permutation = [1, 0]
    %2480 = tensor.empty() : tensor<1x512xf32>
    %2481 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2482 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2481 : f32) outs(%2480 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2483 = linalg.matmul {prov.region_id = "matmul_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2477, %2479 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2482 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2484 = tensor.empty() : tensor<1x512xf32>
    %2485 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2483, %131 : tensor<1x512xf32>, tensor<512xf32>) outs(%2484 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb203(%2486: f32, %2487: f32, %2488: f32):
      %2489 = arith.addf %2486, %2487 : f32
      linalg.yield %2489 : f32
    } -> tensor<1x512xf32>
    %2490 = tensor.collapse_shape %2485 [[0 : i64, 1 : i64]] {prov.region_id = "view_150", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2491 = tensor.expand_shape %2490 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_150", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2492 = tensor.empty() : tensor<1x1x512xf32>
    %2493 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2491, %2475 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2492 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb204(%2494: f32, %2495: f32, %2496: f32):
      %2497 = arith.addf %2494, %2495 : f32
      linalg.yield %2497 : f32
    } -> tensor<1x1x512xf32>
    %2498 = "tensor.extract_slice"(%2493) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2499 = "tensor.extract_slice"(%2493) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2500 = "tensor.extract_slice"(%2493) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2501 = "tensor.extract_slice"(%2493) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2502 = tensor.empty() : tensor<1x1x128xf32>
    %2503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2498 : tensor<1x1x128xf32>) outs(%2502 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb205(%2504: f32, %2505: f32):
      %2506 = arith.constant 1.000000e+00 : f32
      %2507 = arith.negf %2504 : f32
      %2508 = math.exp %2507 : f32
      %2509 = arith.addf %2506, %2508 : f32
      %2510 = arith.divf %2506, %2509 : f32
      linalg.yield %2510 : f32
    } -> tensor<1x1x128xf32>
    %2511 = tensor.empty() : tensor<1x1x128xf32>
    %2512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2499 : tensor<1x1x128xf32>) outs(%2511 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_4", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb206(%2513: f32, %2514: f32):
      %2515 = arith.constant 1.000000e+00 : f32
      %2516 = arith.negf %2513 : f32
      %2517 = math.exp %2516 : f32
      %2518 = arith.addf %2515, %2517 : f32
      %2519 = arith.divf %2515, %2518 : f32
      linalg.yield %2519 : f32
    } -> tensor<1x1x128xf32>
    %2520 = tensor.empty() : tensor<1x1x128xf32>
    %2521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2500 : tensor<1x1x128xf32>) outs(%2520 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_2", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb207(%2522: f32, %2523: f32):
      %2524 = math.tanh %2522 : f32
      linalg.yield %2524 : f32
    } -> tensor<1x1x128xf32>
    %2525 = tensor.empty() : tensor<1x1x128xf32>
    %2526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2501 : tensor<1x1x128xf32>) outs(%2525 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_5", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb208(%2527: f32, %2528: f32):
      %2529 = arith.constant 1.000000e+00 : f32
      %2530 = arith.negf %2527 : f32
      %2531 = math.exp %2530 : f32
      %2532 = arith.addf %2529, %2531 : f32
      %2533 = arith.divf %2529, %2532 : f32
      linalg.yield %2533 : f32
    } -> tensor<1x1x128xf32>
    %2534 = tensor.empty() : tensor<1x1x128xf32>
    %2535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2512, %2456 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2534 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb209(%2536: f32, %2537: f32, %2538: f32):
      %2539 = arith.mulf %2536, %2537 : f32
      linalg.yield %2539 : f32
    } -> tensor<1x1x128xf32>
    %2540 = tensor.empty() : tensor<1x1x128xf32>
    %2541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2503, %2521 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2540 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb210(%2542: f32, %2543: f32, %2544: f32):
      %2545 = arith.mulf %2542, %2543 : f32
      linalg.yield %2545 : f32
    } -> tensor<1x1x128xf32>
    %2546 = tensor.empty() : tensor<1x1x128xf32>
    %2547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2535, %2541 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2546 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb211(%2548: f32, %2549: f32, %2550: f32):
      %2551 = arith.addf %2548, %2549 : f32
      linalg.yield %2551 : f32
    } -> tensor<1x1x128xf32>
    %2552 = tensor.empty() : tensor<1x1x128xf32>
    %2553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2547 : tensor<1x1x128xf32>) outs(%2552 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_3", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb212(%2554: f32, %2555: f32):
      %2556 = math.tanh %2554 : f32
      linalg.yield %2556 : f32
    } -> tensor<1x1x128xf32>
    %2557 = tensor.empty() : tensor<1x1x128xf32>
    %2558 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2526, %2553 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2557 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb213(%2559: f32, %2560: f32, %2561: f32):
      %2562 = arith.mulf %2559, %2560 : f32
      linalg.yield %2562 : f32
    } -> tensor<1x1x128xf32>
    %2563 = tensor.concat dim(0) %2558 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2564 = tensor.collapse_shape %2332 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2565 = tensor.expand_shape %2564 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2566 = tensor.collapse_shape %2341 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2567 = tensor.expand_shape %2566 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2568 = tensor.collapse_shape %2563 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_151", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2569 = tensor.expand_shape %2568 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_151", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2570 = tensor.empty() : tensor<128x512xf32>
    %2571 = linalg.transpose ins(%132:tensor<512x128xf32>) outs(%2570:tensor<128x512xf32>) permutation = [1, 0]
    %2572 = tensor.empty() : tensor<1x512xf32>
    %2573 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2574 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2573 : f32) outs(%2572 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2575 = linalg.matmul {prov.region_id = "matmul_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2569, %2571 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2574 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2576 = tensor.empty() : tensor<1x512xf32>
    %2577 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2575, %134 : tensor<1x512xf32>, tensor<512xf32>) outs(%2576 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb214(%2578: f32, %2579: f32, %2580: f32):
      %2581 = arith.addf %2578, %2579 : f32
      linalg.yield %2581 : f32
    } -> tensor<1x512xf32>
    %2582 = tensor.collapse_shape %2577 [[0 : i64, 1 : i64]] {prov.region_id = "view_152", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2583 = tensor.expand_shape %2582 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_152", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2584 = "tensor.extract_slice"(%2583) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2585 = tensor.collapse_shape %2584 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2586 = tensor.expand_shape %2585 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2587 = tensor.collapse_shape %2565 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_153", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2588 = tensor.expand_shape %2587 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_153", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2589 = tensor.empty() : tensor<128x512xf32>
    %2590 = linalg.transpose ins(%133:tensor<512x128xf32>) outs(%2589:tensor<128x512xf32>) permutation = [1, 0]
    %2591 = tensor.empty() : tensor<1x512xf32>
    %2592 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2593 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2592 : f32) outs(%2591 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2594 = linalg.matmul {prov.region_id = "matmul_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2588, %2590 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2593 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2595 = tensor.empty() : tensor<1x512xf32>
    %2596 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2594, %135 : tensor<1x512xf32>, tensor<512xf32>) outs(%2595 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb215(%2597: f32, %2598: f32, %2599: f32):
      %2600 = arith.addf %2597, %2598 : f32
      linalg.yield %2600 : f32
    } -> tensor<1x512xf32>
    %2601 = tensor.collapse_shape %2596 [[0 : i64, 1 : i64]] {prov.region_id = "view_154", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2602 = tensor.expand_shape %2601 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_154", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2603 = tensor.empty() : tensor<1x1x512xf32>
    %2604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2602, %2586 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2603 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb216(%2605: f32, %2606: f32, %2607: f32):
      %2608 = arith.addf %2605, %2606 : f32
      linalg.yield %2608 : f32
    } -> tensor<1x1x512xf32>
    %2609 = "tensor.extract_slice"(%2604) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2610 = "tensor.extract_slice"(%2604) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2611 = "tensor.extract_slice"(%2604) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2612 = "tensor.extract_slice"(%2604) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2613 = tensor.empty() : tensor<1x1x128xf32>
    %2614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2609 : tensor<1x1x128xf32>) outs(%2613 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_6", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb217(%2615: f32, %2616: f32):
      %2617 = arith.constant 1.000000e+00 : f32
      %2618 = arith.negf %2615 : f32
      %2619 = math.exp %2618 : f32
      %2620 = arith.addf %2617, %2619 : f32
      %2621 = arith.divf %2617, %2620 : f32
      linalg.yield %2621 : f32
    } -> tensor<1x1x128xf32>
    %2622 = tensor.empty() : tensor<1x1x128xf32>
    %2623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2610 : tensor<1x1x128xf32>) outs(%2622 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_7", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb218(%2624: f32, %2625: f32):
      %2626 = arith.constant 1.000000e+00 : f32
      %2627 = arith.negf %2624 : f32
      %2628 = math.exp %2627 : f32
      %2629 = arith.addf %2626, %2628 : f32
      %2630 = arith.divf %2626, %2629 : f32
      linalg.yield %2630 : f32
    } -> tensor<1x1x128xf32>
    %2631 = tensor.empty() : tensor<1x1x128xf32>
    %2632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2611 : tensor<1x1x128xf32>) outs(%2631 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_4", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb219(%2633: f32, %2634: f32):
      %2635 = math.tanh %2633 : f32
      linalg.yield %2635 : f32
    } -> tensor<1x1x128xf32>
    %2636 = tensor.empty() : tensor<1x1x128xf32>
    %2637 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2612 : tensor<1x1x128xf32>) outs(%2636 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_8", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb220(%2638: f32, %2639: f32):
      %2640 = arith.constant 1.000000e+00 : f32
      %2641 = arith.negf %2638 : f32
      %2642 = math.exp %2641 : f32
      %2643 = arith.addf %2640, %2642 : f32
      %2644 = arith.divf %2640, %2643 : f32
      linalg.yield %2644 : f32
    } -> tensor<1x1x128xf32>
    %2645 = tensor.empty() : tensor<1x1x128xf32>
    %2646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2623, %2567 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2645 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb221(%2647: f32, %2648: f32, %2649: f32):
      %2650 = arith.mulf %2647, %2648 : f32
      linalg.yield %2650 : f32
    } -> tensor<1x1x128xf32>
    %2651 = tensor.empty() : tensor<1x1x128xf32>
    %2652 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2614, %2632 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2651 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb222(%2653: f32, %2654: f32, %2655: f32):
      %2656 = arith.mulf %2653, %2654 : f32
      linalg.yield %2656 : f32
    } -> tensor<1x1x128xf32>
    %2657 = tensor.empty() : tensor<1x1x128xf32>
    %2658 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2646, %2652 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2657 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb223(%2659: f32, %2660: f32, %2661: f32):
      %2662 = arith.addf %2659, %2660 : f32
      linalg.yield %2662 : f32
    } -> tensor<1x1x128xf32>
    %2663 = tensor.empty() : tensor<1x1x128xf32>
    %2664 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2658 : tensor<1x1x128xf32>) outs(%2663 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_5", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb224(%2665: f32, %2666: f32):
      %2667 = math.tanh %2665 : f32
      linalg.yield %2667 : f32
    } -> tensor<1x1x128xf32>
    %2668 = tensor.empty() : tensor<1x1x128xf32>
    %2669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2637, %2664 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2668 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb225(%2670: f32, %2671: f32, %2672: f32):
      %2673 = arith.mulf %2670, %2671 : f32
      linalg.yield %2673 : f32
    } -> tensor<1x1x128xf32>
    %2674 = tensor.concat dim(0) %2669 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2675 = tensor.collapse_shape %2674 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2676 = tensor.expand_shape %2675 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2677 = tensor.empty() : tensor<128x3xi8>
    %2678 = linalg.transpose ins(%137:tensor<3x128xi8>) outs(%2677:tensor<128x3xi8>) permutation = [1, 0]
    %2679 = tensor.empty() : tensor<128x3xf32>
    %2680 = arith.constant 0 : i32
    %2681 = tensor.splat %2680 : tensor<3xi32>
    %2682 = "quant_ext.dequantize_per_channel"(%2678, %138, %2681) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x3xi8>, tensor<3xf32>, tensor<3xi32>) -> tensor<128x3xf32>
    %2683 = tensor.empty() : tensor<1x3xf32>
    %2684 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2685 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2684 : f32) outs(%2683 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2686 = linalg.matmul {prov.region_id = "matmul_35", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2676, %2682 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2685 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2687 = tensor.empty() : tensor<1x3xf32>
    %2688 = tensor.empty() : tensor<1x3xf32>
    %2689 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2686, %136 : tensor<1x3xf32>, tensor<3xf32>) outs(%2688 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb226(%2690: f32, %2691: f32, %2692: f32):
      %2693 = arith.addf %2690, %2691 : f32
      linalg.yield %2693 : f32
    } -> tensor<1x3xf32>
    func.return %2689 : tensor<1x3xf32>
  }
}
