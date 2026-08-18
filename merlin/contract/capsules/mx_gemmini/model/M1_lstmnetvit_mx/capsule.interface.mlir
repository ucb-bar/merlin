builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "float8_weight_only_e4m3"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64xf32>, %9: tensor<64x32xf32>, %10: tensor<64x1xf32>, %11: tensor<32xf32>, %12: tensor<32x32xf32>, %13: tensor<32x1xf32>, %14: tensor<32xf32>, %15: tensor<32x32xf32>, %16: tensor<32x1xf32>, %17: tensor<32x32x8x8xf32>, %18: tensor<32xf32>, %19: tensor<32xf32>, %20: tensor<32xf32>, %21: tensor<64xf32>, %22: tensor<64x32xf32>, %23: tensor<64x1xf32>, %24: tensor<32xf32>, %25: tensor<32x32xf32>, %26: tensor<32x1xf32>, %27: tensor<32xf32>, %28: tensor<32x32xf32>, %29: tensor<32x1xf32>, %30: tensor<256xf32>, %31: tensor<256x32xf32>, %32: tensor<256x1xf32>, %33: tensor<256x8x3x3xf32>, %34: tensor<256xf32>, %35: tensor<32xf32>, %36: tensor<32x256xf32>, %37: tensor<32x1xf32>, %38: tensor<256xf32>, %39: tensor<256x32xf32>, %40: tensor<256x1xf32>, %41: tensor<256x8x3x3xf32>, %42: tensor<256xf32>, %43: tensor<32xf32>, %44: tensor<32x256xf32>, %45: tensor<32x1xf32>, %46: tensor<32xf32>, %47: tensor<32xf32>, %48: tensor<32xf32>, %49: tensor<32xf32>, %50: tensor<64x32x3x3xf32>, %51: tensor<64xf32>, %52: tensor<64xf32>, %53: tensor<64xf32>, %54: tensor<64x64x4x4xf32>, %55: tensor<64xf32>, %56: tensor<64xf32>, %57: tensor<64xf32>, %58: tensor<128xf32>, %59: tensor<128x64xf32>, %60: tensor<128x1xf32>, %61: tensor<64xf32>, %62: tensor<64x64xf32>, %63: tensor<64x1xf32>, %64: tensor<64xf32>, %65: tensor<64x64xf32>, %66: tensor<64x1xf32>, %67: tensor<64x64x4x4xf32>, %68: tensor<64xf32>, %69: tensor<64xf32>, %70: tensor<64xf32>, %71: tensor<128xf32>, %72: tensor<128x64xf32>, %73: tensor<128x1xf32>, %74: tensor<64xf32>, %75: tensor<64x64xf32>, %76: tensor<64x1xf32>, %77: tensor<64xf32>, %78: tensor<64x64xf32>, %79: tensor<64x1xf32>, %80: tensor<512xf32>, %81: tensor<512x64xf32>, %82: tensor<512x1xf32>, %83: tensor<512x8x3x3xf32>, %84: tensor<512xf32>, %85: tensor<64xf32>, %86: tensor<64x512xf32>, %87: tensor<64x1xf32>, %88: tensor<512xf32>, %89: tensor<512x64xf32>, %90: tensor<512x1xf32>, %91: tensor<512x8x3x3xf32>, %92: tensor<512xf32>, %93: tensor<64xf32>, %94: tensor<64x512xf32>, %95: tensor<64x1xf32>, %96: tensor<64xf32>, %97: tensor<64xf32>, %98: tensor<64xf32>, %99: tensor<64xf32>, %100: tensor<512xf32>, %101: tensor<512x4608xf32>, %102: tensor<512x1xf32>, %103: tensor<512x517xf32>, %104: tensor<512x128xf32>, %105: tensor<512xf32>, %106: tensor<512xf32>, %107: tensor<512x128xf32>, %108: tensor<512x128xf32>, %109: tensor<512xf32>, %110: tensor<512xf32>, %111: tensor<512x128xf32>, %112: tensor<512x128xf32>, %113: tensor<512xf32>, %114: tensor<512xf32>, %115: tensor<3xf32>, %116: tensor<3x128xf32>, %117: tensor<3x1xf32>, %118: tensor<12x48x3x3xf32>, %119: tensor<12xf32>, %120: tensor<1x1x60x90xf32>, %121: tensor<1x1xf32>, %122: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %123 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %124 = tensor.splat %123 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %125 = "tensor.insert_slice"(%120, %124) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %126 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%125 : tensor<1x1x66x96xf32>) outs(%126 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%128: f32, %129: f32):
      linalg.yield %128 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %130 = tensor.collapse_shape %127 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %131 = tensor.expand_shape %130 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %132 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %134 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %135 = tensor.splat %134 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %136 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%133, %131 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%135 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %137 = tensor.collapse_shape %136 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %138 = tensor.expand_shape %137 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %139 = tensor.collapse_shape %138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %140 = tensor.expand_shape %139 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %141 = tensor.empty() : tensor<1x32x15x23xf32>
    %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%141 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%143: f32, %144: f32, %145: f32):
      %146 = arith.addf %143, %144 : f32
      linalg.yield %146 : f32
    } -> tensor<1x32x15x23xf32>
    %147 = tensor.collapse_shape %142 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %148 = tensor.expand_shape %147 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %149 = tensor.empty() : tensor<1x345x32xf32>
    %150 = linalg.transpose ins(%148:tensor<1x32x345xf32>) outs(%149:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %151 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %152 = tensor.splat %151 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %153 = linalg.reduce ins(%150:tensor<1x345x32xf32>) outs(%152:tensor<1x345xf32>) dimensions = [2]
    (%154: f32, %155: f32) {
      %156 = arith.addf %154, %155 : f32
      linalg.yield %156 : f32
    }
    %157 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %158 = tensor.splat %157 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %159 = tensor.empty() : tensor<1x345xf32>
    %160 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%153, %158 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%159 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%161: f32, %162: f32, %163: f32):
      %164 = arith.divf %161, %162 : f32
      linalg.yield %164 : f32
    } -> tensor<1x345xf32>
    %165 = tensor.collapse_shape %160 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %166 = tensor.expand_shape %165 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %167 = tensor.empty() : tensor<1x345x32xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150, %166 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%167 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%169: f32, %170: f32, %171: f32):
      %172 = arith.subf %169, %170 : f32
      linalg.yield %172 : f32
    } -> tensor<1x345x32xf32>
    %173 = tensor.empty() : tensor<1x345x32xf32>
    %174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168, %168 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%173 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%175: f32, %176: f32, %177: f32):
      %178 = arith.mulf %175, %176 : f32
      linalg.yield %178 : f32
    } -> tensor<1x345x32xf32>
    %179 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %180 = tensor.splat %179 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %181 = linalg.reduce ins(%174:tensor<1x345x32xf32>) outs(%180:tensor<1x345xf32>) dimensions = [2]
    (%182: f32, %183: f32) {
      %184 = arith.addf %182, %183 : f32
      linalg.yield %184 : f32
    }
    %185 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %186 = tensor.splat %185 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %187 = tensor.empty() : tensor<1x345xf32>
    %188 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%181, %186 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%187 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%189: f32, %190: f32, %191: f32):
      %192 = arith.divf %189, %190 : f32
      linalg.yield %192 : f32
    } -> tensor<1x345xf32>
    %193 = tensor.collapse_shape %188 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %194 = tensor.expand_shape %193 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %195 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %196 = tensor.splat %195 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %197 = tensor.empty() : tensor<1x345x1xf32>
    %198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%194, %196 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%197 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%199: f32, %200: f32, %201: f32):
      %202 = arith.addf %199, %200 : f32
      linalg.yield %202 : f32
    } -> tensor<1x345x1xf32>
    %203 = tensor.empty() : tensor<1x345x1xf32>
    %204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x345x1xf32>) outs(%203 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%205: f32, %206: f32):
      %207 = math.rsqrt %205 : f32
      linalg.yield %207 : f32
    } -> tensor<1x345x1xf32>
    %208 = tensor.empty() : tensor<1x345x32xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168, %204 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%208 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%210: f32, %211: f32, %212: f32):
      %213 = arith.mulf %210, %211 : f32
      linalg.yield %213 : f32
    } -> tensor<1x345x32xf32>
    %214 = tensor.empty() : tensor<1x345x32xf32>
    %215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%209, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%214 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%216: f32, %217: f32, %218: f32):
      %219 = arith.mulf %216, %217 : f32
      linalg.yield %219 : f32
    } -> tensor<1x345x32xf32>
    %220 = tensor.empty() : tensor<1x345x32xf32>
    %221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%215, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%220 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%222: f32, %223: f32, %224: f32):
      %225 = arith.addf %222, %223 : f32
      linalg.yield %225 : f32
    } -> tensor<1x345x32xf32>
    %226 = tensor.empty() : tensor<1x32x345xf32>
    %227 = linalg.transpose ins(%221:tensor<1x345x32xf32>) outs(%226:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %228 = tensor.collapse_shape %227 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %229 = tensor.expand_shape %228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %230 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%229 : tensor<1x32x15x23xf32>) outs(%230 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%232: f32, %233: f32):
      linalg.yield %232 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %234 = tensor.collapse_shape %231 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %235 = tensor.expand_shape %234 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %236 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %237 = tensor.expand_shape %236 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %238 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %239 = tensor.splat %238 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %240 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%237, %235 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%239 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %241 = tensor.collapse_shape %240 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %242 = tensor.expand_shape %241 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %243 = tensor.collapse_shape %242 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %244 = tensor.expand_shape %243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %245 = tensor.empty() : tensor<1x32x1x2xf32>
    %246 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%244, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%245 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%247: f32, %248: f32, %249: f32):
      %250 = arith.addf %247, %248 : f32
      linalg.yield %250 : f32
    } -> tensor<1x32x1x2xf32>
    %251 = tensor.collapse_shape %246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %252 = tensor.expand_shape %251 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %253 = tensor.empty() : tensor<1x2x32xf32>
    %254 = linalg.transpose ins(%252:tensor<1x32x2xf32>) outs(%253:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %255 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %256 = tensor.splat %255 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %257 = linalg.reduce ins(%254:tensor<1x2x32xf32>) outs(%256:tensor<1x2xf32>) dimensions = [2]
    (%258: f32, %259: f32) {
      %260 = arith.addf %258, %259 : f32
      linalg.yield %260 : f32
    }
    %261 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %262 = tensor.splat %261 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %263 = tensor.empty() : tensor<1x2xf32>
    %264 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%257, %262 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%263 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%265: f32, %266: f32, %267: f32):
      %268 = arith.divf %265, %266 : f32
      linalg.yield %268 : f32
    } -> tensor<1x2xf32>
    %269 = tensor.collapse_shape %264 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %270 = tensor.expand_shape %269 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %271 = tensor.empty() : tensor<1x2x32xf32>
    %272 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254, %270 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%271 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%273: f32, %274: f32, %275: f32):
      %276 = arith.subf %273, %274 : f32
      linalg.yield %276 : f32
    } -> tensor<1x2x32xf32>
    %277 = tensor.empty() : tensor<1x2x32xf32>
    %278 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%272, %272 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%277 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%279: f32, %280: f32, %281: f32):
      %282 = arith.mulf %279, %280 : f32
      linalg.yield %282 : f32
    } -> tensor<1x2x32xf32>
    %283 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %284 = tensor.splat %283 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %285 = linalg.reduce ins(%278:tensor<1x2x32xf32>) outs(%284:tensor<1x2xf32>) dimensions = [2]
    (%286: f32, %287: f32) {
      %288 = arith.addf %286, %287 : f32
      linalg.yield %288 : f32
    }
    %289 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %290 = tensor.splat %289 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %291 = tensor.empty() : tensor<1x2xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%285, %290 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%291 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%293: f32, %294: f32, %295: f32):
      %296 = arith.divf %293, %294 : f32
      linalg.yield %296 : f32
    } -> tensor<1x2xf32>
    %297 = tensor.collapse_shape %292 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %298 = tensor.expand_shape %297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %299 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %300 = tensor.splat %299 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %301 = tensor.empty() : tensor<1x2x1xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%298, %300 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%301 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%303: f32, %304: f32, %305: f32):
      %306 = arith.addf %303, %304 : f32
      linalg.yield %306 : f32
    } -> tensor<1x2x1xf32>
    %307 = tensor.empty() : tensor<1x2x1xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%302 : tensor<1x2x1xf32>) outs(%307 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%309: f32, %310: f32):
      %311 = math.rsqrt %309 : f32
      linalg.yield %311 : f32
    } -> tensor<1x2x1xf32>
    %312 = tensor.empty() : tensor<1x2x32xf32>
    %313 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%272, %308 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%312 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%314: f32, %315: f32, %316: f32):
      %317 = arith.mulf %314, %315 : f32
      linalg.yield %317 : f32
    } -> tensor<1x2x32xf32>
    %318 = tensor.empty() : tensor<1x2x32xf32>
    %319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%318 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%320: f32, %321: f32, %322: f32):
      %323 = arith.mulf %320, %321 : f32
      linalg.yield %323 : f32
    } -> tensor<1x2x32xf32>
    %324 = tensor.empty() : tensor<1x2x32xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%319, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%324 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%326: f32, %327: f32, %328: f32):
      %329 = arith.addf %326, %327 : f32
      linalg.yield %329 : f32
    } -> tensor<1x2x32xf32>
    %330 = tensor.empty() : tensor<32x64xf32>
    %331 = linalg.transpose ins(%9:tensor<64x32xf32>) outs(%330:tensor<32x64xf32>) permutation = [1, 0]
    %332 = tensor.empty() : tensor<1x64xf32>
    %333 = linalg.transpose ins(%10:tensor<64x1xf32>) outs(%332:tensor<1x64xf32>) permutation = [1, 0]
    %334 = tensor.empty() : tensor<32x64xf32>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%331, %333 : tensor<32x64xf32>, tensor<1x64xf32>) outs(%334 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%336: f32, %337: f32, %338: f32):
      %339 = arith.mulf %336, %337 : f32
      linalg.yield %339 : f32
    } -> tensor<32x64xf32>
    %340 = tensor.collapse_shape %325 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %341 = tensor.expand_shape %340 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %342 = tensor.empty() : tensor<2x64xf32>
    %343 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %344 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%343 : f32) outs(%342 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %345 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} ins(%341, %335 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%344 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %346 = tensor.collapse_shape %345 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %347 = tensor.expand_shape %346 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %348 = tensor.empty() : tensor<1x2x64xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%347, %8 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%348 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb23(%350: f32, %351: f32, %352: f32):
      %353 = arith.addf %350, %351 : f32
      linalg.yield %353 : f32
    } -> tensor<1x2x64xf32>
    %354 = tensor.collapse_shape %349 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %355 = tensor.expand_shape %354 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %356 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %357 = linalg.transpose ins(%355:tensor<1x2x2x1x32xf32>) outs(%356:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %358 = "tensor.extract_slice"(%357) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %359 = tensor.collapse_shape %358 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %360 = tensor.expand_shape %359 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %361 = "tensor.extract_slice"(%357) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %362 = tensor.collapse_shape %361 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %363 = tensor.expand_shape %362 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %364 = tensor.empty() : tensor<32x32xf32>
    %365 = linalg.transpose ins(%12:tensor<32x32xf32>) outs(%364:tensor<32x32xf32>) permutation = [1, 0]
    %366 = tensor.empty() : tensor<1x32xf32>
    %367 = linalg.transpose ins(%13:tensor<32x1xf32>) outs(%366:tensor<1x32xf32>) permutation = [1, 0]
    %368 = tensor.empty() : tensor<32x32xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%365, %367 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%368 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb24(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<32x32xf32>
    %374 = tensor.collapse_shape %221 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %375 = tensor.expand_shape %374 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %376 = tensor.empty() : tensor<345x32xf32>
    %377 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %378 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%377 : f32) outs(%376 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %379 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} ins(%375, %369 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%378 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %380 = tensor.collapse_shape %379 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %381 = tensor.expand_shape %380 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %382 = tensor.empty() : tensor<1x345x32xf32>
    %383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%381, %11 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%382 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb25(%384: f32, %385: f32, %386: f32):
      %387 = arith.addf %384, %385 : f32
      linalg.yield %387 : f32
    } -> tensor<1x345x32xf32>
    %388 = tensor.collapse_shape %383 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %389 = tensor.expand_shape %388 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %390 = tensor.empty() : tensor<1x1x345x32xf32>
    %391 = linalg.transpose ins(%389:tensor<1x345x1x32xf32>) outs(%390:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %392 = tensor.empty() : tensor<1x1x32x2xf32>
    %393 = linalg.transpose ins(%360:tensor<1x1x2x32xf32>) outs(%392:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %394 = tensor.empty() : tensor<1x1x345x32xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%391 : tensor<1x1x345x32xf32>) outs(%394 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb26(%396: f32, %397: f32):
      linalg.yield %396 : f32
    } -> tensor<1x1x345x32xf32>
    %398 = tensor.collapse_shape %395 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %399 = tensor.expand_shape %398 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %400 = tensor.empty() : tensor<1x1x32x2xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%393 : tensor<1x1x32x2xf32>) outs(%400 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb27(%402: f32, %403: f32):
      linalg.yield %402 : f32
    } -> tensor<1x1x32x2xf32>
    %404 = tensor.collapse_shape %401 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %405 = tensor.expand_shape %404 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %406 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %407 = tensor.splat %406 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%399, %405 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%407 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb28(%409: f32, %410: f32, %411: f32):
      %412 = arith.mulf %409, %410 : f32
      %413 = arith.addf %411, %412 : f32
      linalg.yield %413 : f32
    } -> tensor<1x345x2xf32>
    %414 = tensor.collapse_shape %408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %415 = tensor.expand_shape %414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %416 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %417 = tensor.splat %416 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %418 = tensor.empty() : tensor<1x1x345x2xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %417 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%418 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb29(%420: f32, %421: f32, %422: f32):
      %423 = arith.divf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x1x345x2xf32>
    %424 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %425 = tensor.splat %424 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %426 = linalg.reduce ins(%419:tensor<1x1x345x2xf32>) outs(%425:tensor<1x1x345xf32>) dimensions = [3]
    (%427: f32, %428: f32) {
      %429 = arith.maximumf %427, %428 : f32
      linalg.yield %429 : f32
    }
    %430 = tensor.collapse_shape %426 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %431 = tensor.expand_shape %430 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %432 = tensor.empty() : tensor<1x1x345x2xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %431 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%432 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb30(%434: f32, %435: f32, %436: f32):
      %437 = arith.subf %434, %435 : f32
      linalg.yield %437 : f32
    } -> tensor<1x1x345x2xf32>
    %438 = tensor.empty() : tensor<1x1x345x2xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433 : tensor<1x1x345x2xf32>) outs(%438 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb31(%440: f32, %441: f32):
      %442 = math.exp %440 : f32
      linalg.yield %442 : f32
    } -> tensor<1x1x345x2xf32>
    %443 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %444 = tensor.splat %443 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %445 = linalg.reduce ins(%439:tensor<1x1x345x2xf32>) outs(%444:tensor<1x1x345xf32>) dimensions = [3]
    (%446: f32, %447: f32) {
      %448 = arith.addf %446, %447 : f32
      linalg.yield %448 : f32
    }
    %449 = tensor.collapse_shape %445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %450 = tensor.expand_shape %449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %451 = tensor.empty() : tensor<1x1x345x2xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %450 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%451 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb32(%453: f32, %454: f32, %455: f32):
      %456 = arith.divf %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<1x1x345x2xf32>
    %457 = tensor.empty() : tensor<1x1x345x2xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452 : tensor<1x1x345x2xf32>) outs(%457 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb33(%459: f32, %460: f32):
      linalg.yield %459 : f32
    } -> tensor<1x1x345x2xf32>
    %461 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %462 = tensor.expand_shape %461 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %463 = tensor.empty() : tensor<1x1x2x32xf32>
    %464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363 : tensor<1x1x2x32xf32>) outs(%463 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb34(%465: f32, %466: f32):
      linalg.yield %465 : f32
    } -> tensor<1x1x2x32xf32>
    %467 = tensor.collapse_shape %464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %468 = tensor.expand_shape %467 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %469 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %470 = tensor.splat %469 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%462, %468 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%470 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb35(%472: f32, %473: f32, %474: f32):
      %475 = arith.mulf %472, %473 : f32
      %476 = arith.addf %474, %475 : f32
      linalg.yield %476 : f32
    } -> tensor<1x345x32xf32>
    %477 = tensor.collapse_shape %471 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %478 = tensor.expand_shape %477 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %479 = tensor.empty() : tensor<1x345x1x32xf32>
    %480 = linalg.transpose ins(%478:tensor<1x1x345x32xf32>) outs(%479:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %481 = tensor.collapse_shape %480 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %482 = tensor.expand_shape %481 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %483 = tensor.empty() : tensor<32x32xf32>
    %484 = linalg.transpose ins(%15:tensor<32x32xf32>) outs(%483:tensor<32x32xf32>) permutation = [1, 0]
    %485 = tensor.empty() : tensor<1x32xf32>
    %486 = linalg.transpose ins(%16:tensor<32x1xf32>) outs(%485:tensor<1x32xf32>) permutation = [1, 0]
    %487 = tensor.empty() : tensor<32x32xf32>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%484, %486 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%487 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb36(%489: f32, %490: f32, %491: f32):
      %492 = arith.mulf %489, %490 : f32
      linalg.yield %492 : f32
    } -> tensor<32x32xf32>
    %493 = tensor.collapse_shape %482 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %494 = tensor.expand_shape %493 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %495 = tensor.empty() : tensor<345x32xf32>
    %496 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %497 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%496 : f32) outs(%495 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %498 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} ins(%494, %488 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%497 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %499 = tensor.collapse_shape %498 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %500 = tensor.expand_shape %499 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %501 = tensor.empty() : tensor<1x345x32xf32>
    %502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%500, %14 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%501 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb37(%503: f32, %504: f32, %505: f32):
      %506 = arith.addf %503, %504 : f32
      linalg.yield %506 : f32
    } -> tensor<1x345x32xf32>
    %507 = tensor.empty() : tensor<1x345x32xf32>
    %508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%221, %502 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%507 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb38(%509: f32, %510: f32, %511: f32):
      %512 = arith.addf %509, %510 : f32
      linalg.yield %512 : f32
    } -> tensor<1x345x32xf32>
    %513 = tensor.empty() : tensor<32x256xf32>
    %514 = linalg.transpose ins(%31:tensor<256x32xf32>) outs(%513:tensor<32x256xf32>) permutation = [1, 0]
    %515 = tensor.empty() : tensor<1x256xf32>
    %516 = linalg.transpose ins(%32:tensor<256x1xf32>) outs(%515:tensor<1x256xf32>) permutation = [1, 0]
    %517 = tensor.empty() : tensor<32x256xf32>
    %518 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%514, %516 : tensor<32x256xf32>, tensor<1x256xf32>) outs(%517 : tensor<32x256xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb39(%519: f32, %520: f32, %521: f32):
      %522 = arith.mulf %519, %520 : f32
      linalg.yield %522 : f32
    } -> tensor<32x256xf32>
    %523 = tensor.collapse_shape %508 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %524 = tensor.expand_shape %523 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %525 = tensor.empty() : tensor<345x256xf32>
    %526 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %527 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%526 : f32) outs(%525 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %528 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} ins(%524, %518 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%527 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %529 = tensor.collapse_shape %528 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %530 = tensor.expand_shape %529 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %531 = tensor.empty() : tensor<1x345x256xf32>
    %532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%530, %30 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%531 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb40(%533: f32, %534: f32, %535: f32):
      %536 = arith.addf %533, %534 : f32
      linalg.yield %536 : f32
    } -> tensor<1x345x256xf32>
    %537 = tensor.empty() : tensor<1x256x345xf32>
    %538 = linalg.transpose ins(%532:tensor<1x345x256xf32>) outs(%537:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %539 = tensor.collapse_shape %538 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %540 = tensor.expand_shape %539 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %541 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %542 = tensor.splat %541 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %543 = "tensor.insert_slice"(%540, %542) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %544 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %545 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%543 : tensor<1x256x17x25xf32>) outs(%544 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb41(%546: f32, %547: f32):
      linalg.yield %546 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %548 = tensor.collapse_shape %545 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %549 = tensor.expand_shape %548 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %550 = tensor.collapse_shape %33 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %551 = tensor.expand_shape %550 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %552 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %553 = tensor.splat %552 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%551, %549 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%553 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb42(%555: f32, %556: f32, %557: f32):
      %558 = arith.mulf %555, %556 : f32
      %559 = arith.addf %557, %558 : f32
      linalg.yield %559 : f32
    } -> tensor<32x8x345xf32>
    %560 = tensor.collapse_shape %554 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %561 = tensor.expand_shape %560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %562 = tensor.collapse_shape %561 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %563 = tensor.expand_shape %562 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %564 = tensor.empty() : tensor<1x256x15x23xf32>
    %565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %34 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%564 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb43(%566: f32, %567: f32, %568: f32):
      %569 = arith.addf %566, %567 : f32
      linalg.yield %569 : f32
    } -> tensor<1x256x15x23xf32>
    %570 = tensor.collapse_shape %565 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %571 = tensor.expand_shape %570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %572 = tensor.empty() : tensor<1x345x256xf32>
    %573 = linalg.transpose ins(%571:tensor<1x256x345xf32>) outs(%572:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %574 = tensor.empty() : tensor<1x345x256xf32>
    %575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573 : tensor<1x345x256xf32>) outs(%574 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb44(%576: f32, %577: f32):
      %578 = arith.constant 5.000000e-01 : f32
      %579 = arith.constant 1.000000e+00 : f32
      %580 = arith.constant 0.707106769 : f32
      %581 = arith.mulf %576, %580 : f32
      %582 = math.erf %581 : f32
      %583 = arith.addf %579, %582 : f32
      %584 = arith.mulf %578, %576 : f32
      %585 = arith.mulf %584, %583 : f32
      linalg.yield %585 : f32
    } -> tensor<1x345x256xf32>
    %586 = tensor.empty() : tensor<256x32xf32>
    %587 = linalg.transpose ins(%36:tensor<32x256xf32>) outs(%586:tensor<256x32xf32>) permutation = [1, 0]
    %588 = tensor.empty() : tensor<1x32xf32>
    %589 = linalg.transpose ins(%37:tensor<32x1xf32>) outs(%588:tensor<1x32xf32>) permutation = [1, 0]
    %590 = tensor.empty() : tensor<256x32xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%587, %589 : tensor<256x32xf32>, tensor<1x32xf32>) outs(%590 : tensor<256x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb45(%592: f32, %593: f32, %594: f32):
      %595 = arith.mulf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<256x32xf32>
    %596 = tensor.collapse_shape %575 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %597 = tensor.expand_shape %596 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %598 = tensor.empty() : tensor<345x32xf32>
    %599 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %600 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%599 : f32) outs(%598 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %601 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} ins(%597, %591 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%600 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %602 = tensor.collapse_shape %601 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %603 = tensor.expand_shape %602 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %604 = tensor.empty() : tensor<1x345x32xf32>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%603, %35 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%604 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb46(%606: f32, %607: f32, %608: f32):
      %609 = arith.addf %606, %607 : f32
      linalg.yield %609 : f32
    } -> tensor<1x345x32xf32>
    %610 = tensor.empty() : tensor<1x345x32xf32>
    %611 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%508, %605 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%610 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%612: f32, %613: f32, %614: f32):
      %615 = arith.addf %612, %613 : f32
      linalg.yield %615 : f32
    } -> tensor<1x345x32xf32>
    %616 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %617 = tensor.splat %616 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %618 = linalg.reduce ins(%611:tensor<1x345x32xf32>) outs(%617:tensor<1x345xf32>) dimensions = [2]
    (%619: f32, %620: f32) {
      %621 = arith.addf %619, %620 : f32
      linalg.yield %621 : f32
    }
    %622 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %623 = tensor.splat %622 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %624 = tensor.empty() : tensor<1x345xf32>
    %625 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%618, %623 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%624 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb48(%626: f32, %627: f32, %628: f32):
      %629 = arith.divf %626, %627 : f32
      linalg.yield %629 : f32
    } -> tensor<1x345xf32>
    %630 = tensor.collapse_shape %625 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %631 = tensor.expand_shape %630 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %632 = tensor.empty() : tensor<1x345x32xf32>
    %633 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%611, %631 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%632 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb49(%634: f32, %635: f32, %636: f32):
      %637 = arith.subf %634, %635 : f32
      linalg.yield %637 : f32
    } -> tensor<1x345x32xf32>
    %638 = tensor.empty() : tensor<1x345x32xf32>
    %639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%633, %633 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%638 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb50(%640: f32, %641: f32, %642: f32):
      %643 = arith.mulf %640, %641 : f32
      linalg.yield %643 : f32
    } -> tensor<1x345x32xf32>
    %644 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %645 = tensor.splat %644 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %646 = linalg.reduce ins(%639:tensor<1x345x32xf32>) outs(%645:tensor<1x345xf32>) dimensions = [2]
    (%647: f32, %648: f32) {
      %649 = arith.addf %647, %648 : f32
      linalg.yield %649 : f32
    }
    %650 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %651 = tensor.splat %650 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %652 = tensor.empty() : tensor<1x345xf32>
    %653 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%646, %651 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%652 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb51(%654: f32, %655: f32, %656: f32):
      %657 = arith.divf %654, %655 : f32
      linalg.yield %657 : f32
    } -> tensor<1x345xf32>
    %658 = tensor.collapse_shape %653 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %659 = tensor.expand_shape %658 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %660 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %661 = tensor.splat %660 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %662 = tensor.empty() : tensor<1x345x1xf32>
    %663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%659, %661 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%662 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb52(%664: f32, %665: f32, %666: f32):
      %667 = arith.addf %664, %665 : f32
      linalg.yield %667 : f32
    } -> tensor<1x345x1xf32>
    %668 = tensor.empty() : tensor<1x345x1xf32>
    %669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663 : tensor<1x345x1xf32>) outs(%668 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb53(%670: f32, %671: f32):
      %672 = math.rsqrt %670 : f32
      linalg.yield %672 : f32
    } -> tensor<1x345x1xf32>
    %673 = tensor.empty() : tensor<1x345x32xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%633, %669 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%673 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb54(%675: f32, %676: f32, %677: f32):
      %678 = arith.mulf %675, %676 : f32
      linalg.yield %678 : f32
    } -> tensor<1x345x32xf32>
    %679 = tensor.empty() : tensor<1x345x32xf32>
    %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%674, %46 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%679 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb55(%681: f32, %682: f32, %683: f32):
      %684 = arith.mulf %681, %682 : f32
      linalg.yield %684 : f32
    } -> tensor<1x345x32xf32>
    %685 = tensor.empty() : tensor<1x345x32xf32>
    %686 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%680, %47 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%685 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb56(%687: f32, %688: f32, %689: f32):
      %690 = arith.addf %687, %688 : f32
      linalg.yield %690 : f32
    } -> tensor<1x345x32xf32>
    %691 = tensor.empty() : tensor<1x32x345xf32>
    %692 = linalg.transpose ins(%686:tensor<1x345x32xf32>) outs(%691:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %693 = tensor.collapse_shape %692 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %694 = tensor.expand_shape %693 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %695 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %696 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%694 : tensor<1x32x15x23xf32>) outs(%695 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb57(%697: f32, %698: f32):
      linalg.yield %697 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %699 = tensor.collapse_shape %696 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %700 = tensor.expand_shape %699 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %701 = tensor.collapse_shape %17 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %702 = tensor.expand_shape %701 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %703 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %704 = tensor.splat %703 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %705 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%702, %700 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%704 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %706 = tensor.collapse_shape %705 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %707 = tensor.expand_shape %706 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %708 = tensor.collapse_shape %707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %709 = tensor.expand_shape %708 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %710 = tensor.empty() : tensor<1x32x1x2xf32>
    %711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%709, %18 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%710 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb58(%712: f32, %713: f32, %714: f32):
      %715 = arith.addf %712, %713 : f32
      linalg.yield %715 : f32
    } -> tensor<1x32x1x2xf32>
    %716 = tensor.collapse_shape %711 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %717 = tensor.expand_shape %716 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %718 = tensor.empty() : tensor<1x2x32xf32>
    %719 = linalg.transpose ins(%717:tensor<1x32x2xf32>) outs(%718:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %720 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %721 = tensor.splat %720 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %722 = linalg.reduce ins(%719:tensor<1x2x32xf32>) outs(%721:tensor<1x2xf32>) dimensions = [2]
    (%723: f32, %724: f32) {
      %725 = arith.addf %723, %724 : f32
      linalg.yield %725 : f32
    }
    %726 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %727 = tensor.splat %726 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %728 = tensor.empty() : tensor<1x2xf32>
    %729 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%722, %727 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%728 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb59(%730: f32, %731: f32, %732: f32):
      %733 = arith.divf %730, %731 : f32
      linalg.yield %733 : f32
    } -> tensor<1x2xf32>
    %734 = tensor.collapse_shape %729 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %735 = tensor.expand_shape %734 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %736 = tensor.empty() : tensor<1x2x32xf32>
    %737 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%719, %735 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%736 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb60(%738: f32, %739: f32, %740: f32):
      %741 = arith.subf %738, %739 : f32
      linalg.yield %741 : f32
    } -> tensor<1x2x32xf32>
    %742 = tensor.empty() : tensor<1x2x32xf32>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%737, %737 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%742 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb61(%744: f32, %745: f32, %746: f32):
      %747 = arith.mulf %744, %745 : f32
      linalg.yield %747 : f32
    } -> tensor<1x2x32xf32>
    %748 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %749 = tensor.splat %748 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %750 = linalg.reduce ins(%743:tensor<1x2x32xf32>) outs(%749:tensor<1x2xf32>) dimensions = [2]
    (%751: f32, %752: f32) {
      %753 = arith.addf %751, %752 : f32
      linalg.yield %753 : f32
    }
    %754 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %755 = tensor.splat %754 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %756 = tensor.empty() : tensor<1x2xf32>
    %757 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%750, %755 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%756 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb62(%758: f32, %759: f32, %760: f32):
      %761 = arith.divf %758, %759 : f32
      linalg.yield %761 : f32
    } -> tensor<1x2xf32>
    %762 = tensor.collapse_shape %757 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %763 = tensor.expand_shape %762 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %764 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %765 = tensor.splat %764 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %766 = tensor.empty() : tensor<1x2x1xf32>
    %767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%763, %765 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%766 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb63(%768: f32, %769: f32, %770: f32):
      %771 = arith.addf %768, %769 : f32
      linalg.yield %771 : f32
    } -> tensor<1x2x1xf32>
    %772 = tensor.empty() : tensor<1x2x1xf32>
    %773 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%767 : tensor<1x2x1xf32>) outs(%772 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb64(%774: f32, %775: f32):
      %776 = math.rsqrt %774 : f32
      linalg.yield %776 : f32
    } -> tensor<1x2x1xf32>
    %777 = tensor.empty() : tensor<1x2x32xf32>
    %778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%737, %773 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%777 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb65(%779: f32, %780: f32, %781: f32):
      %782 = arith.mulf %779, %780 : f32
      linalg.yield %782 : f32
    } -> tensor<1x2x32xf32>
    %783 = tensor.empty() : tensor<1x2x32xf32>
    %784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%778, %19 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%783 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb66(%785: f32, %786: f32, %787: f32):
      %788 = arith.mulf %785, %786 : f32
      linalg.yield %788 : f32
    } -> tensor<1x2x32xf32>
    %789 = tensor.empty() : tensor<1x2x32xf32>
    %790 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%784, %20 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%789 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb67(%791: f32, %792: f32, %793: f32):
      %794 = arith.addf %791, %792 : f32
      linalg.yield %794 : f32
    } -> tensor<1x2x32xf32>
    %795 = tensor.empty() : tensor<32x64xf32>
    %796 = linalg.transpose ins(%22:tensor<64x32xf32>) outs(%795:tensor<32x64xf32>) permutation = [1, 0]
    %797 = tensor.empty() : tensor<1x64xf32>
    %798 = linalg.transpose ins(%23:tensor<64x1xf32>) outs(%797:tensor<1x64xf32>) permutation = [1, 0]
    %799 = tensor.empty() : tensor<32x64xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%796, %798 : tensor<32x64xf32>, tensor<1x64xf32>) outs(%799 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb68(%801: f32, %802: f32, %803: f32):
      %804 = arith.mulf %801, %802 : f32
      linalg.yield %804 : f32
    } -> tensor<32x64xf32>
    %805 = tensor.collapse_shape %790 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %806 = tensor.expand_shape %805 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %807 = tensor.empty() : tensor<2x64xf32>
    %808 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %809 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%808 : f32) outs(%807 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %810 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} ins(%806, %800 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%809 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %811 = tensor.collapse_shape %810 [[0 : i64, 1 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %812 = tensor.expand_shape %811 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %813 = tensor.empty() : tensor<1x2x64xf32>
    %814 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%812, %21 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%813 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb69(%815: f32, %816: f32, %817: f32):
      %818 = arith.addf %815, %816 : f32
      linalg.yield %818 : f32
    } -> tensor<1x2x64xf32>
    %819 = tensor.collapse_shape %814 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %820 = tensor.expand_shape %819 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %821 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %822 = linalg.transpose ins(%820:tensor<1x2x2x1x32xf32>) outs(%821:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %823 = "tensor.extract_slice"(%822) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %824 = tensor.collapse_shape %823 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %825 = tensor.expand_shape %824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %826 = "tensor.extract_slice"(%822) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %827 = tensor.collapse_shape %826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %828 = tensor.expand_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %829 = tensor.empty() : tensor<32x32xf32>
    %830 = linalg.transpose ins(%25:tensor<32x32xf32>) outs(%829:tensor<32x32xf32>) permutation = [1, 0]
    %831 = tensor.empty() : tensor<1x32xf32>
    %832 = linalg.transpose ins(%26:tensor<32x1xf32>) outs(%831:tensor<1x32xf32>) permutation = [1, 0]
    %833 = tensor.empty() : tensor<32x32xf32>
    %834 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%830, %832 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%833 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb70(%835: f32, %836: f32, %837: f32):
      %838 = arith.mulf %835, %836 : f32
      linalg.yield %838 : f32
    } -> tensor<32x32xf32>
    %839 = tensor.collapse_shape %686 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %840 = tensor.expand_shape %839 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %841 = tensor.empty() : tensor<345x32xf32>
    %842 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %843 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%842 : f32) outs(%841 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %844 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} ins(%840, %834 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%843 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %845 = tensor.collapse_shape %844 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %846 = tensor.expand_shape %845 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %847 = tensor.empty() : tensor<1x345x32xf32>
    %848 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%846, %24 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%847 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb71(%849: f32, %850: f32, %851: f32):
      %852 = arith.addf %849, %850 : f32
      linalg.yield %852 : f32
    } -> tensor<1x345x32xf32>
    %853 = tensor.collapse_shape %848 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %854 = tensor.expand_shape %853 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %855 = tensor.empty() : tensor<1x1x345x32xf32>
    %856 = linalg.transpose ins(%854:tensor<1x345x1x32xf32>) outs(%855:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %857 = tensor.empty() : tensor<1x1x32x2xf32>
    %858 = linalg.transpose ins(%825:tensor<1x1x2x32xf32>) outs(%857:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %859 = tensor.empty() : tensor<1x1x345x32xf32>
    %860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%856 : tensor<1x1x345x32xf32>) outs(%859 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb72(%861: f32, %862: f32):
      linalg.yield %861 : f32
    } -> tensor<1x1x345x32xf32>
    %863 = tensor.collapse_shape %860 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %864 = tensor.expand_shape %863 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %865 = tensor.empty() : tensor<1x1x32x2xf32>
    %866 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858 : tensor<1x1x32x2xf32>) outs(%865 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb73(%867: f32, %868: f32):
      linalg.yield %867 : f32
    } -> tensor<1x1x32x2xf32>
    %869 = tensor.collapse_shape %866 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %870 = tensor.expand_shape %869 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %871 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %872 = tensor.splat %871 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%864, %870 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%872 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb74(%874: f32, %875: f32, %876: f32):
      %877 = arith.mulf %874, %875 : f32
      %878 = arith.addf %876, %877 : f32
      linalg.yield %878 : f32
    } -> tensor<1x345x2xf32>
    %879 = tensor.collapse_shape %873 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %880 = tensor.expand_shape %879 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %881 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %882 = tensor.splat %881 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %883 = tensor.empty() : tensor<1x1x345x2xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%880, %882 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%883 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb75(%885: f32, %886: f32, %887: f32):
      %888 = arith.divf %885, %886 : f32
      linalg.yield %888 : f32
    } -> tensor<1x1x345x2xf32>
    %889 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %890 = tensor.splat %889 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %891 = linalg.reduce ins(%884:tensor<1x1x345x2xf32>) outs(%890:tensor<1x1x345xf32>) dimensions = [3]
    (%892: f32, %893: f32) {
      %894 = arith.maximumf %892, %893 : f32
      linalg.yield %894 : f32
    }
    %895 = tensor.collapse_shape %891 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %896 = tensor.expand_shape %895 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %897 = tensor.empty() : tensor<1x1x345x2xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%884, %896 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%897 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb76(%899: f32, %900: f32, %901: f32):
      %902 = arith.subf %899, %900 : f32
      linalg.yield %902 : f32
    } -> tensor<1x1x345x2xf32>
    %903 = tensor.empty() : tensor<1x1x345x2xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%898 : tensor<1x1x345x2xf32>) outs(%903 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb77(%905: f32, %906: f32):
      %907 = math.exp %905 : f32
      linalg.yield %907 : f32
    } -> tensor<1x1x345x2xf32>
    %908 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %909 = tensor.splat %908 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %910 = linalg.reduce ins(%904:tensor<1x1x345x2xf32>) outs(%909:tensor<1x1x345xf32>) dimensions = [3]
    (%911: f32, %912: f32) {
      %913 = arith.addf %911, %912 : f32
      linalg.yield %913 : f32
    }
    %914 = tensor.collapse_shape %910 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %915 = tensor.expand_shape %914 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %916 = tensor.empty() : tensor<1x1x345x2xf32>
    %917 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%904, %915 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%916 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb78(%918: f32, %919: f32, %920: f32):
      %921 = arith.divf %918, %919 : f32
      linalg.yield %921 : f32
    } -> tensor<1x1x345x2xf32>
    %922 = tensor.empty() : tensor<1x1x345x2xf32>
    %923 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%917 : tensor<1x1x345x2xf32>) outs(%922 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb79(%924: f32, %925: f32):
      linalg.yield %924 : f32
    } -> tensor<1x1x345x2xf32>
    %926 = tensor.collapse_shape %923 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %927 = tensor.expand_shape %926 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %928 = tensor.empty() : tensor<1x1x2x32xf32>
    %929 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828 : tensor<1x1x2x32xf32>) outs(%928 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb80(%930: f32, %931: f32):
      linalg.yield %930 : f32
    } -> tensor<1x1x2x32xf32>
    %932 = tensor.collapse_shape %929 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %933 = tensor.expand_shape %932 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %934 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %935 = tensor.splat %934 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%927, %933 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%935 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb81(%937: f32, %938: f32, %939: f32):
      %940 = arith.mulf %937, %938 : f32
      %941 = arith.addf %939, %940 : f32
      linalg.yield %941 : f32
    } -> tensor<1x345x32xf32>
    %942 = tensor.collapse_shape %936 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %943 = tensor.expand_shape %942 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %944 = tensor.empty() : tensor<1x345x1x32xf32>
    %945 = linalg.transpose ins(%943:tensor<1x1x345x32xf32>) outs(%944:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %946 = tensor.collapse_shape %945 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %947 = tensor.expand_shape %946 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %948 = tensor.empty() : tensor<32x32xf32>
    %949 = linalg.transpose ins(%28:tensor<32x32xf32>) outs(%948:tensor<32x32xf32>) permutation = [1, 0]
    %950 = tensor.empty() : tensor<1x32xf32>
    %951 = linalg.transpose ins(%29:tensor<32x1xf32>) outs(%950:tensor<1x32xf32>) permutation = [1, 0]
    %952 = tensor.empty() : tensor<32x32xf32>
    %953 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%949, %951 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%952 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb82(%954: f32, %955: f32, %956: f32):
      %957 = arith.mulf %954, %955 : f32
      linalg.yield %957 : f32
    } -> tensor<32x32xf32>
    %958 = tensor.collapse_shape %947 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %959 = tensor.expand_shape %958 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %960 = tensor.empty() : tensor<345x32xf32>
    %961 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %962 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%961 : f32) outs(%960 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %963 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} ins(%959, %953 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%962 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %964 = tensor.collapse_shape %963 [[0 : i64, 1 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %965 = tensor.expand_shape %964 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %966 = tensor.empty() : tensor<1x345x32xf32>
    %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %27 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%966 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb83(%968: f32, %969: f32, %970: f32):
      %971 = arith.addf %968, %969 : f32
      linalg.yield %971 : f32
    } -> tensor<1x345x32xf32>
    %972 = tensor.empty() : tensor<1x345x32xf32>
    %973 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686, %967 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%972 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%974: f32, %975: f32, %976: f32):
      %977 = arith.addf %974, %975 : f32
      linalg.yield %977 : f32
    } -> tensor<1x345x32xf32>
    %978 = tensor.empty() : tensor<32x256xf32>
    %979 = linalg.transpose ins(%39:tensor<256x32xf32>) outs(%978:tensor<32x256xf32>) permutation = [1, 0]
    %980 = tensor.empty() : tensor<1x256xf32>
    %981 = linalg.transpose ins(%40:tensor<256x1xf32>) outs(%980:tensor<1x256xf32>) permutation = [1, 0]
    %982 = tensor.empty() : tensor<32x256xf32>
    %983 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%979, %981 : tensor<32x256xf32>, tensor<1x256xf32>) outs(%982 : tensor<32x256xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb85(%984: f32, %985: f32, %986: f32):
      %987 = arith.mulf %984, %985 : f32
      linalg.yield %987 : f32
    } -> tensor<32x256xf32>
    %988 = tensor.collapse_shape %973 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %989 = tensor.expand_shape %988 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %990 = tensor.empty() : tensor<345x256xf32>
    %991 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %992 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%991 : f32) outs(%990 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %993 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} ins(%989, %983 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%992 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %994 = tensor.collapse_shape %993 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %995 = tensor.expand_shape %994 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %996 = tensor.empty() : tensor<1x345x256xf32>
    %997 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%995, %38 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%996 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb86(%998: f32, %999: f32, %1000: f32):
      %1001 = arith.addf %998, %999 : f32
      linalg.yield %1001 : f32
    } -> tensor<1x345x256xf32>
    %1002 = tensor.empty() : tensor<1x256x345xf32>
    %1003 = linalg.transpose ins(%997:tensor<1x345x256xf32>) outs(%1002:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %1004 = tensor.collapse_shape %1003 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %1005 = tensor.expand_shape %1004 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1006 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1007 = tensor.splat %1006 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %1008 = "tensor.insert_slice"(%1005, %1007) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %1009 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %1010 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1008 : tensor<1x256x17x25xf32>) outs(%1009 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb87(%1011: f32, %1012: f32):
      linalg.yield %1011 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %1013 = tensor.collapse_shape %1010 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %1014 = tensor.expand_shape %1013 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %1015 = tensor.collapse_shape %41 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %1016 = tensor.expand_shape %1015 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %1017 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %1018 = tensor.splat %1017 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %1019 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1016, %1014 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%1018 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb88(%1020: f32, %1021: f32, %1022: f32):
      %1023 = arith.mulf %1020, %1021 : f32
      %1024 = arith.addf %1022, %1023 : f32
      linalg.yield %1024 : f32
    } -> tensor<32x8x345xf32>
    %1025 = tensor.collapse_shape %1019 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %1026 = tensor.expand_shape %1025 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %1027 = tensor.collapse_shape %1026 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %1028 = tensor.expand_shape %1027 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %1029 = tensor.empty() : tensor<1x256x15x23xf32>
    %1030 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1028, %42 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%1029 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb89(%1031: f32, %1032: f32, %1033: f32):
      %1034 = arith.addf %1031, %1032 : f32
      linalg.yield %1034 : f32
    } -> tensor<1x256x15x23xf32>
    %1035 = tensor.collapse_shape %1030 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %1036 = tensor.expand_shape %1035 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %1037 = tensor.empty() : tensor<1x345x256xf32>
    %1038 = linalg.transpose ins(%1036:tensor<1x256x345xf32>) outs(%1037:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %1039 = tensor.empty() : tensor<1x345x256xf32>
    %1040 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1038 : tensor<1x345x256xf32>) outs(%1039 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb90(%1041: f32, %1042: f32):
      %1043 = arith.constant 5.000000e-01 : f32
      %1044 = arith.constant 1.000000e+00 : f32
      %1045 = arith.constant 0.707106769 : f32
      %1046 = arith.mulf %1041, %1045 : f32
      %1047 = math.erf %1046 : f32
      %1048 = arith.addf %1044, %1047 : f32
      %1049 = arith.mulf %1043, %1041 : f32
      %1050 = arith.mulf %1049, %1048 : f32
      linalg.yield %1050 : f32
    } -> tensor<1x345x256xf32>
    %1051 = tensor.empty() : tensor<256x32xf32>
    %1052 = linalg.transpose ins(%44:tensor<32x256xf32>) outs(%1051:tensor<256x32xf32>) permutation = [1, 0]
    %1053 = tensor.empty() : tensor<1x32xf32>
    %1054 = linalg.transpose ins(%45:tensor<32x1xf32>) outs(%1053:tensor<1x32xf32>) permutation = [1, 0]
    %1055 = tensor.empty() : tensor<256x32xf32>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1052, %1054 : tensor<256x32xf32>, tensor<1x32xf32>) outs(%1055 : tensor<256x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb91(%1057: f32, %1058: f32, %1059: f32):
      %1060 = arith.mulf %1057, %1058 : f32
      linalg.yield %1060 : f32
    } -> tensor<256x32xf32>
    %1061 = tensor.collapse_shape %1040 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %1062 = tensor.expand_shape %1061 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %1063 = tensor.empty() : tensor<345x32xf32>
    %1064 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1065 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1064 : f32) outs(%1063 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1066 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} ins(%1062, %1056 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%1065 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1067 = tensor.collapse_shape %1066 [[0 : i64, 1 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %1068 = tensor.expand_shape %1067 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1069 = tensor.empty() : tensor<1x345x32xf32>
    %1070 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1068, %43 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1069 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb92(%1071: f32, %1072: f32, %1073: f32):
      %1074 = arith.addf %1071, %1072 : f32
      linalg.yield %1074 : f32
    } -> tensor<1x345x32xf32>
    %1075 = tensor.empty() : tensor<1x345x32xf32>
    %1076 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%973, %1070 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1075 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb93(%1077: f32, %1078: f32, %1079: f32):
      %1080 = arith.addf %1077, %1078 : f32
      linalg.yield %1080 : f32
    } -> tensor<1x345x32xf32>
    %1081 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1082 = tensor.splat %1081 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1083 = linalg.reduce ins(%1076:tensor<1x345x32xf32>) outs(%1082:tensor<1x345xf32>) dimensions = [2]
    (%1084: f32, %1085: f32) {
      %1086 = arith.addf %1084, %1085 : f32
      linalg.yield %1086 : f32
    }
    %1087 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1088 = tensor.splat %1087 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1089 = tensor.empty() : tensor<1x345xf32>
    %1090 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1083, %1088 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1089 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb94(%1091: f32, %1092: f32, %1093: f32):
      %1094 = arith.divf %1091, %1092 : f32
      linalg.yield %1094 : f32
    } -> tensor<1x345xf32>
    %1095 = tensor.collapse_shape %1090 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1096 = tensor.expand_shape %1095 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1097 = tensor.empty() : tensor<1x345x32xf32>
    %1098 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1076, %1096 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1097 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb95(%1099: f32, %1100: f32, %1101: f32):
      %1102 = arith.subf %1099, %1100 : f32
      linalg.yield %1102 : f32
    } -> tensor<1x345x32xf32>
    %1103 = tensor.empty() : tensor<1x345x32xf32>
    %1104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1098, %1098 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1103 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb96(%1105: f32, %1106: f32, %1107: f32):
      %1108 = arith.mulf %1105, %1106 : f32
      linalg.yield %1108 : f32
    } -> tensor<1x345x32xf32>
    %1109 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1110 = tensor.splat %1109 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1111 = linalg.reduce ins(%1104:tensor<1x345x32xf32>) outs(%1110:tensor<1x345xf32>) dimensions = [2]
    (%1112: f32, %1113: f32) {
      %1114 = arith.addf %1112, %1113 : f32
      linalg.yield %1114 : f32
    }
    %1115 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1116 = tensor.splat %1115 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1117 = tensor.empty() : tensor<1x345xf32>
    %1118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1111, %1116 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1117 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb97(%1119: f32, %1120: f32, %1121: f32):
      %1122 = arith.divf %1119, %1120 : f32
      linalg.yield %1122 : f32
    } -> tensor<1x345xf32>
    %1123 = tensor.collapse_shape %1118 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1124 = tensor.expand_shape %1123 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1125 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1126 = tensor.splat %1125 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1127 = tensor.empty() : tensor<1x345x1xf32>
    %1128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1124, %1126 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1127 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb98(%1129: f32, %1130: f32, %1131: f32):
      %1132 = arith.addf %1129, %1130 : f32
      linalg.yield %1132 : f32
    } -> tensor<1x345x1xf32>
    %1133 = tensor.empty() : tensor<1x345x1xf32>
    %1134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1128 : tensor<1x345x1xf32>) outs(%1133 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb99(%1135: f32, %1136: f32):
      %1137 = math.rsqrt %1135 : f32
      linalg.yield %1137 : f32
    } -> tensor<1x345x1xf32>
    %1138 = tensor.empty() : tensor<1x345x32xf32>
    %1139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1098, %1134 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1138 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb100(%1140: f32, %1141: f32, %1142: f32):
      %1143 = arith.mulf %1140, %1141 : f32
      linalg.yield %1143 : f32
    } -> tensor<1x345x32xf32>
    %1144 = tensor.empty() : tensor<1x345x32xf32>
    %1145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1139, %48 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1144 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb101(%1146: f32, %1147: f32, %1148: f32):
      %1149 = arith.mulf %1146, %1147 : f32
      linalg.yield %1149 : f32
    } -> tensor<1x345x32xf32>
    %1150 = tensor.empty() : tensor<1x345x32xf32>
    %1151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1145, %49 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1150 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb102(%1152: f32, %1153: f32, %1154: f32):
      %1155 = arith.addf %1152, %1153 : f32
      linalg.yield %1155 : f32
    } -> tensor<1x345x32xf32>
    %1156 = tensor.collapse_shape %1151 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1157 = tensor.expand_shape %1156 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1158 = tensor.empty() : tensor<1x32x15x23xf32>
    %1159 = linalg.transpose ins(%1157:tensor<1x15x23x32xf32>) outs(%1158:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1160 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1161 = tensor.splat %1160 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1162 = "tensor.insert_slice"(%1159, %1161) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1163 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1162 : tensor<1x32x17x25xf32>) outs(%1163 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb103(%1165: f32, %1166: f32):
      linalg.yield %1165 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1167 = tensor.collapse_shape %1164 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1168 = tensor.expand_shape %1167 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1169 = tensor.collapse_shape %50 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1170 = tensor.expand_shape %1169 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1171 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1172 = tensor.splat %1171 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1173 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1170, %1168 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1172 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1174 = tensor.collapse_shape %1173 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1175 = tensor.expand_shape %1174 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1176 = tensor.collapse_shape %1175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1177 = tensor.expand_shape %1176 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1178 = tensor.empty() : tensor<1x64x8x12xf32>
    %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1177, %51 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1178 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb104(%1180: f32, %1181: f32, %1182: f32):
      %1183 = arith.addf %1180, %1181 : f32
      linalg.yield %1183 : f32
    } -> tensor<1x64x8x12xf32>
    %1184 = tensor.collapse_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1185 = tensor.expand_shape %1184 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1186 = tensor.empty() : tensor<1x96x64xf32>
    %1187 = linalg.transpose ins(%1185:tensor<1x64x96xf32>) outs(%1186:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1188 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1189 = tensor.splat %1188 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1190 = linalg.reduce ins(%1187:tensor<1x96x64xf32>) outs(%1189:tensor<1x96xf32>) dimensions = [2]
    (%1191: f32, %1192: f32) {
      %1193 = arith.addf %1191, %1192 : f32
      linalg.yield %1193 : f32
    }
    %1194 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1195 = tensor.splat %1194 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1196 = tensor.empty() : tensor<1x96xf32>
    %1197 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1190, %1195 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1196 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb105(%1198: f32, %1199: f32, %1200: f32):
      %1201 = arith.divf %1198, %1199 : f32
      linalg.yield %1201 : f32
    } -> tensor<1x96xf32>
    %1202 = tensor.collapse_shape %1197 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1203 = tensor.expand_shape %1202 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1204 = tensor.empty() : tensor<1x96x64xf32>
    %1205 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1187, %1203 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1204 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb106(%1206: f32, %1207: f32, %1208: f32):
      %1209 = arith.subf %1206, %1207 : f32
      linalg.yield %1209 : f32
    } -> tensor<1x96x64xf32>
    %1210 = tensor.empty() : tensor<1x96x64xf32>
    %1211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1205, %1205 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1210 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb107(%1212: f32, %1213: f32, %1214: f32):
      %1215 = arith.mulf %1212, %1213 : f32
      linalg.yield %1215 : f32
    } -> tensor<1x96x64xf32>
    %1216 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1217 = tensor.splat %1216 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1218 = linalg.reduce ins(%1211:tensor<1x96x64xf32>) outs(%1217:tensor<1x96xf32>) dimensions = [2]
    (%1219: f32, %1220: f32) {
      %1221 = arith.addf %1219, %1220 : f32
      linalg.yield %1221 : f32
    }
    %1222 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1223 = tensor.splat %1222 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1224 = tensor.empty() : tensor<1x96xf32>
    %1225 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1218, %1223 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1224 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb108(%1226: f32, %1227: f32, %1228: f32):
      %1229 = arith.divf %1226, %1227 : f32
      linalg.yield %1229 : f32
    } -> tensor<1x96xf32>
    %1230 = tensor.collapse_shape %1225 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1231 = tensor.expand_shape %1230 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1232 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1233 = tensor.splat %1232 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1234 = tensor.empty() : tensor<1x96x1xf32>
    %1235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1231, %1233 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1234 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb109(%1236: f32, %1237: f32, %1238: f32):
      %1239 = arith.addf %1236, %1237 : f32
      linalg.yield %1239 : f32
    } -> tensor<1x96x1xf32>
    %1240 = tensor.empty() : tensor<1x96x1xf32>
    %1241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1235 : tensor<1x96x1xf32>) outs(%1240 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb110(%1242: f32, %1243: f32):
      %1244 = math.rsqrt %1242 : f32
      linalg.yield %1244 : f32
    } -> tensor<1x96x1xf32>
    %1245 = tensor.empty() : tensor<1x96x64xf32>
    %1246 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1205, %1241 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1245 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb111(%1247: f32, %1248: f32, %1249: f32):
      %1250 = arith.mulf %1247, %1248 : f32
      linalg.yield %1250 : f32
    } -> tensor<1x96x64xf32>
    %1251 = tensor.empty() : tensor<1x96x64xf32>
    %1252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1246, %52 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1251 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb112(%1253: f32, %1254: f32, %1255: f32):
      %1256 = arith.mulf %1253, %1254 : f32
      linalg.yield %1256 : f32
    } -> tensor<1x96x64xf32>
    %1257 = tensor.empty() : tensor<1x96x64xf32>
    %1258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1252, %53 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1257 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb113(%1259: f32, %1260: f32, %1261: f32):
      %1262 = arith.addf %1259, %1260 : f32
      linalg.yield %1262 : f32
    } -> tensor<1x96x64xf32>
    %1263 = tensor.empty() : tensor<1x64x96xf32>
    %1264 = linalg.transpose ins(%1258:tensor<1x96x64xf32>) outs(%1263:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1265 = tensor.collapse_shape %1264 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1266 = tensor.expand_shape %1265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1267 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1266 : tensor<1x64x8x12xf32>) outs(%1267 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb114(%1269: f32, %1270: f32):
      linalg.yield %1269 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1271 = tensor.collapse_shape %1268 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1272 = tensor.expand_shape %1271 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1273 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1274 = tensor.expand_shape %1273 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1275 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1276 = tensor.splat %1275 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1277 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1274, %1272 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1276 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1278 = tensor.collapse_shape %1277 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1279 = tensor.expand_shape %1278 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1280 = tensor.collapse_shape %1279 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1281 = tensor.expand_shape %1280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1282 = tensor.empty() : tensor<1x64x2x3xf32>
    %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1281, %55 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1282 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb115(%1284: f32, %1285: f32, %1286: f32):
      %1287 = arith.addf %1284, %1285 : f32
      linalg.yield %1287 : f32
    } -> tensor<1x64x2x3xf32>
    %1288 = tensor.collapse_shape %1283 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1289 = tensor.expand_shape %1288 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1290 = tensor.empty() : tensor<1x6x64xf32>
    %1291 = linalg.transpose ins(%1289:tensor<1x64x6xf32>) outs(%1290:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1292 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1293 = tensor.splat %1292 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1294 = linalg.reduce ins(%1291:tensor<1x6x64xf32>) outs(%1293:tensor<1x6xf32>) dimensions = [2]
    (%1295: f32, %1296: f32) {
      %1297 = arith.addf %1295, %1296 : f32
      linalg.yield %1297 : f32
    }
    %1298 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1299 = tensor.splat %1298 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1300 = tensor.empty() : tensor<1x6xf32>
    %1301 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1294, %1299 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1300 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb116(%1302: f32, %1303: f32, %1304: f32):
      %1305 = arith.divf %1302, %1303 : f32
      linalg.yield %1305 : f32
    } -> tensor<1x6xf32>
    %1306 = tensor.collapse_shape %1301 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1307 = tensor.expand_shape %1306 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1308 = tensor.empty() : tensor<1x6x64xf32>
    %1309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1291, %1307 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1308 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb117(%1310: f32, %1311: f32, %1312: f32):
      %1313 = arith.subf %1310, %1311 : f32
      linalg.yield %1313 : f32
    } -> tensor<1x6x64xf32>
    %1314 = tensor.empty() : tensor<1x6x64xf32>
    %1315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1309, %1309 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1314 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb118(%1316: f32, %1317: f32, %1318: f32):
      %1319 = arith.mulf %1316, %1317 : f32
      linalg.yield %1319 : f32
    } -> tensor<1x6x64xf32>
    %1320 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1321 = tensor.splat %1320 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1322 = linalg.reduce ins(%1315:tensor<1x6x64xf32>) outs(%1321:tensor<1x6xf32>) dimensions = [2]
    (%1323: f32, %1324: f32) {
      %1325 = arith.addf %1323, %1324 : f32
      linalg.yield %1325 : f32
    }
    %1326 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1327 = tensor.splat %1326 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1328 = tensor.empty() : tensor<1x6xf32>
    %1329 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1322, %1327 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1328 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb119(%1330: f32, %1331: f32, %1332: f32):
      %1333 = arith.divf %1330, %1331 : f32
      linalg.yield %1333 : f32
    } -> tensor<1x6xf32>
    %1334 = tensor.collapse_shape %1329 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1335 = tensor.expand_shape %1334 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1336 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1337 = tensor.splat %1336 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1338 = tensor.empty() : tensor<1x6x1xf32>
    %1339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1335, %1337 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1338 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb120(%1340: f32, %1341: f32, %1342: f32):
      %1343 = arith.addf %1340, %1341 : f32
      linalg.yield %1343 : f32
    } -> tensor<1x6x1xf32>
    %1344 = tensor.empty() : tensor<1x6x1xf32>
    %1345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1339 : tensor<1x6x1xf32>) outs(%1344 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb121(%1346: f32, %1347: f32):
      %1348 = math.rsqrt %1346 : f32
      linalg.yield %1348 : f32
    } -> tensor<1x6x1xf32>
    %1349 = tensor.empty() : tensor<1x6x64xf32>
    %1350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1309, %1345 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1349 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb122(%1351: f32, %1352: f32, %1353: f32):
      %1354 = arith.mulf %1351, %1352 : f32
      linalg.yield %1354 : f32
    } -> tensor<1x6x64xf32>
    %1355 = tensor.empty() : tensor<1x6x64xf32>
    %1356 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1350, %56 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1355 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb123(%1357: f32, %1358: f32, %1359: f32):
      %1360 = arith.mulf %1357, %1358 : f32
      linalg.yield %1360 : f32
    } -> tensor<1x6x64xf32>
    %1361 = tensor.empty() : tensor<1x6x64xf32>
    %1362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1356, %57 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1361 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb124(%1363: f32, %1364: f32, %1365: f32):
      %1366 = arith.addf %1363, %1364 : f32
      linalg.yield %1366 : f32
    } -> tensor<1x6x64xf32>
    %1367 = tensor.empty() : tensor<64x128xf32>
    %1368 = linalg.transpose ins(%59:tensor<128x64xf32>) outs(%1367:tensor<64x128xf32>) permutation = [1, 0]
    %1369 = tensor.empty() : tensor<1x128xf32>
    %1370 = linalg.transpose ins(%60:tensor<128x1xf32>) outs(%1369:tensor<1x128xf32>) permutation = [1, 0]
    %1371 = tensor.empty() : tensor<64x128xf32>
    %1372 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1368, %1370 : tensor<64x128xf32>, tensor<1x128xf32>) outs(%1371 : tensor<64x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb125(%1373: f32, %1374: f32, %1375: f32):
      %1376 = arith.mulf %1373, %1374 : f32
      linalg.yield %1376 : f32
    } -> tensor<64x128xf32>
    %1377 = tensor.collapse_shape %1362 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1378 = tensor.expand_shape %1377 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1379 = tensor.empty() : tensor<6x128xf32>
    %1380 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1381 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1380 : f32) outs(%1379 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1382 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} ins(%1378, %1372 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1381 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1383 = tensor.collapse_shape %1382 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1384 = tensor.expand_shape %1383 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1385 = tensor.empty() : tensor<1x6x128xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1384, %58 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1385 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb126(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.addf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x6x128xf32>
    %1391 = tensor.collapse_shape %1386 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1392 = tensor.expand_shape %1391 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1393 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1394 = linalg.transpose ins(%1392:tensor<1x6x2x2x32xf32>) outs(%1393:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1395 = "tensor.extract_slice"(%1394) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1396 = tensor.collapse_shape %1395 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1397 = tensor.expand_shape %1396 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1398 = "tensor.extract_slice"(%1394) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1399 = tensor.collapse_shape %1398 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1400 = tensor.expand_shape %1399 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1401 = tensor.empty() : tensor<64x64xf32>
    %1402 = linalg.transpose ins(%62:tensor<64x64xf32>) outs(%1401:tensor<64x64xf32>) permutation = [1, 0]
    %1403 = tensor.empty() : tensor<1x64xf32>
    %1404 = linalg.transpose ins(%63:tensor<64x1xf32>) outs(%1403:tensor<1x64xf32>) permutation = [1, 0]
    %1405 = tensor.empty() : tensor<64x64xf32>
    %1406 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1402, %1404 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1405 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb127(%1407: f32, %1408: f32, %1409: f32):
      %1410 = arith.mulf %1407, %1408 : f32
      linalg.yield %1410 : f32
    } -> tensor<64x64xf32>
    %1411 = tensor.collapse_shape %1258 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1412 = tensor.expand_shape %1411 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1413 = tensor.empty() : tensor<96x64xf32>
    %1414 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1415 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1414 : f32) outs(%1413 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1416 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} ins(%1412, %1406 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1415 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1417 = tensor.collapse_shape %1416 [[0 : i64, 1 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1418 = tensor.expand_shape %1417 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1419 = tensor.empty() : tensor<1x96x64xf32>
    %1420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1418, %61 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1419 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb128(%1421: f32, %1422: f32, %1423: f32):
      %1424 = arith.addf %1421, %1422 : f32
      linalg.yield %1424 : f32
    } -> tensor<1x96x64xf32>
    %1425 = tensor.collapse_shape %1420 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1426 = tensor.expand_shape %1425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1427 = tensor.empty() : tensor<1x2x96x32xf32>
    %1428 = linalg.transpose ins(%1426:tensor<1x96x2x32xf32>) outs(%1427:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1429 = tensor.empty() : tensor<1x2x32x6xf32>
    %1430 = linalg.transpose ins(%1397:tensor<1x2x6x32xf32>) outs(%1429:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1431 = tensor.empty() : tensor<1x2x96x32xf32>
    %1432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1428 : tensor<1x2x96x32xf32>) outs(%1431 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb129(%1433: f32, %1434: f32):
      linalg.yield %1433 : f32
    } -> tensor<1x2x96x32xf32>
    %1435 = tensor.collapse_shape %1432 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1436 = tensor.expand_shape %1435 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1437 = tensor.empty() : tensor<1x2x32x6xf32>
    %1438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1430 : tensor<1x2x32x6xf32>) outs(%1437 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb130(%1439: f32, %1440: f32):
      linalg.yield %1439 : f32
    } -> tensor<1x2x32x6xf32>
    %1441 = tensor.collapse_shape %1438 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1442 = tensor.expand_shape %1441 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1443 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1444 = tensor.splat %1443 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1436, %1442 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1444 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb131(%1446: f32, %1447: f32, %1448: f32):
      %1449 = arith.mulf %1446, %1447 : f32
      %1450 = arith.addf %1448, %1449 : f32
      linalg.yield %1450 : f32
    } -> tensor<2x96x6xf32>
    %1451 = tensor.collapse_shape %1445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1452 = tensor.expand_shape %1451 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1453 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1454 = tensor.splat %1453 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1455 = tensor.empty() : tensor<1x2x96x6xf32>
    %1456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1452, %1454 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1455 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb132(%1457: f32, %1458: f32, %1459: f32):
      %1460 = arith.divf %1457, %1458 : f32
      linalg.yield %1460 : f32
    } -> tensor<1x2x96x6xf32>
    %1461 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1462 = tensor.splat %1461 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1463 = linalg.reduce ins(%1456:tensor<1x2x96x6xf32>) outs(%1462:tensor<1x2x96xf32>) dimensions = [3]
    (%1464: f32, %1465: f32) {
      %1466 = arith.maximumf %1464, %1465 : f32
      linalg.yield %1466 : f32
    }
    %1467 = tensor.collapse_shape %1463 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1468 = tensor.expand_shape %1467 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1469 = tensor.empty() : tensor<1x2x96x6xf32>
    %1470 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1456, %1468 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1469 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb133(%1471: f32, %1472: f32, %1473: f32):
      %1474 = arith.subf %1471, %1472 : f32
      linalg.yield %1474 : f32
    } -> tensor<1x2x96x6xf32>
    %1475 = tensor.empty() : tensor<1x2x96x6xf32>
    %1476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1470 : tensor<1x2x96x6xf32>) outs(%1475 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb134(%1477: f32, %1478: f32):
      %1479 = math.exp %1477 : f32
      linalg.yield %1479 : f32
    } -> tensor<1x2x96x6xf32>
    %1480 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1481 = tensor.splat %1480 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1482 = linalg.reduce ins(%1476:tensor<1x2x96x6xf32>) outs(%1481:tensor<1x2x96xf32>) dimensions = [3]
    (%1483: f32, %1484: f32) {
      %1485 = arith.addf %1483, %1484 : f32
      linalg.yield %1485 : f32
    }
    %1486 = tensor.collapse_shape %1482 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1487 = tensor.expand_shape %1486 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1488 = tensor.empty() : tensor<1x2x96x6xf32>
    %1489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1476, %1487 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1488 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb135(%1490: f32, %1491: f32, %1492: f32):
      %1493 = arith.divf %1490, %1491 : f32
      linalg.yield %1493 : f32
    } -> tensor<1x2x96x6xf32>
    %1494 = tensor.empty() : tensor<1x2x96x6xf32>
    %1495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1489 : tensor<1x2x96x6xf32>) outs(%1494 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb136(%1496: f32, %1497: f32):
      linalg.yield %1496 : f32
    } -> tensor<1x2x96x6xf32>
    %1498 = tensor.collapse_shape %1495 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %1499 = tensor.expand_shape %1498 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %1500 = tensor.empty() : tensor<1x2x6x32xf32>
    %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1400 : tensor<1x2x6x32xf32>) outs(%1500 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb137(%1502: f32, %1503: f32):
      linalg.yield %1502 : f32
    } -> tensor<1x2x6x32xf32>
    %1504 = tensor.collapse_shape %1501 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %1505 = tensor.expand_shape %1504 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %1506 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1507 = tensor.splat %1506 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %1508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1499, %1505 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%1507 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb138(%1509: f32, %1510: f32, %1511: f32):
      %1512 = arith.mulf %1509, %1510 : f32
      %1513 = arith.addf %1511, %1512 : f32
      linalg.yield %1513 : f32
    } -> tensor<2x96x32xf32>
    %1514 = tensor.collapse_shape %1508 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %1515 = tensor.expand_shape %1514 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %1516 = tensor.empty() : tensor<1x96x2x32xf32>
    %1517 = linalg.transpose ins(%1515:tensor<1x2x96x32xf32>) outs(%1516:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1518 = tensor.collapse_shape %1517 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1519 = tensor.expand_shape %1518 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1520 = tensor.empty() : tensor<64x64xf32>
    %1521 = linalg.transpose ins(%65:tensor<64x64xf32>) outs(%1520:tensor<64x64xf32>) permutation = [1, 0]
    %1522 = tensor.empty() : tensor<1x64xf32>
    %1523 = linalg.transpose ins(%66:tensor<64x1xf32>) outs(%1522:tensor<1x64xf32>) permutation = [1, 0]
    %1524 = tensor.empty() : tensor<64x64xf32>
    %1525 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1521, %1523 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1524 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb139(%1526: f32, %1527: f32, %1528: f32):
      %1529 = arith.mulf %1526, %1527 : f32
      linalg.yield %1529 : f32
    } -> tensor<64x64xf32>
    %1530 = tensor.collapse_shape %1519 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1531 = tensor.expand_shape %1530 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1532 = tensor.empty() : tensor<96x64xf32>
    %1533 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1534 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1533 : f32) outs(%1532 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1535 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} ins(%1531, %1525 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1534 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1536 = tensor.collapse_shape %1535 [[0 : i64, 1 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1537 = tensor.expand_shape %1536 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1538 = tensor.empty() : tensor<1x96x64xf32>
    %1539 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1537, %64 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1538 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb140(%1540: f32, %1541: f32, %1542: f32):
      %1543 = arith.addf %1540, %1541 : f32
      linalg.yield %1543 : f32
    } -> tensor<1x96x64xf32>
    %1544 = tensor.empty() : tensor<1x96x64xf32>
    %1545 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1258, %1539 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1544 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb141(%1546: f32, %1547: f32, %1548: f32):
      %1549 = arith.addf %1546, %1547 : f32
      linalg.yield %1549 : f32
    } -> tensor<1x96x64xf32>
    %1550 = tensor.empty() : tensor<64x512xf32>
    %1551 = linalg.transpose ins(%81:tensor<512x64xf32>) outs(%1550:tensor<64x512xf32>) permutation = [1, 0]
    %1552 = tensor.empty() : tensor<1x512xf32>
    %1553 = linalg.transpose ins(%82:tensor<512x1xf32>) outs(%1552:tensor<1x512xf32>) permutation = [1, 0]
    %1554 = tensor.empty() : tensor<64x512xf32>
    %1555 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1551, %1553 : tensor<64x512xf32>, tensor<1x512xf32>) outs(%1554 : tensor<64x512xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb142(%1556: f32, %1557: f32, %1558: f32):
      %1559 = arith.mulf %1556, %1557 : f32
      linalg.yield %1559 : f32
    } -> tensor<64x512xf32>
    %1560 = tensor.collapse_shape %1545 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1562 = tensor.empty() : tensor<96x512xf32>
    %1563 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1564 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1563 : f32) outs(%1562 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1565 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} ins(%1561, %1555 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1564 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1566 = tensor.collapse_shape %1565 [[0 : i64, 1 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1567 = tensor.expand_shape %1566 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1568 = tensor.empty() : tensor<1x96x512xf32>
    %1569 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1567, %80 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1568 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb143(%1570: f32, %1571: f32, %1572: f32):
      %1573 = arith.addf %1570, %1571 : f32
      linalg.yield %1573 : f32
    } -> tensor<1x96x512xf32>
    %1574 = tensor.empty() : tensor<1x512x96xf32>
    %1575 = linalg.transpose ins(%1569:tensor<1x96x512xf32>) outs(%1574:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1576 = tensor.collapse_shape %1575 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1577 = tensor.expand_shape %1576 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1578 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1579 = tensor.splat %1578 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1580 = "tensor.insert_slice"(%1577, %1579) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1581 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1580 : tensor<1x512x10x14xf32>) outs(%1581 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb144(%1583: f32, %1584: f32):
      linalg.yield %1583 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1585 = tensor.collapse_shape %1582 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1586 = tensor.expand_shape %1585 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1587 = tensor.collapse_shape %83 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1588 = tensor.expand_shape %1587 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1589 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1590 = tensor.splat %1589 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1588, %1586 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1590 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb145(%1592: f32, %1593: f32, %1594: f32):
      %1595 = arith.mulf %1592, %1593 : f32
      %1596 = arith.addf %1594, %1595 : f32
      linalg.yield %1596 : f32
    } -> tensor<64x8x96xf32>
    %1597 = tensor.collapse_shape %1591 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1598 = tensor.expand_shape %1597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1599 = tensor.collapse_shape %1598 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1600 = tensor.expand_shape %1599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1601 = tensor.empty() : tensor<1x512x8x12xf32>
    %1602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1600, %84 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1601 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb146(%1603: f32, %1604: f32, %1605: f32):
      %1606 = arith.addf %1603, %1604 : f32
      linalg.yield %1606 : f32
    } -> tensor<1x512x8x12xf32>
    %1607 = tensor.collapse_shape %1602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1608 = tensor.expand_shape %1607 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1609 = tensor.empty() : tensor<1x96x512xf32>
    %1610 = linalg.transpose ins(%1608:tensor<1x512x96xf32>) outs(%1609:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1611 = tensor.empty() : tensor<1x96x512xf32>
    %1612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1610 : tensor<1x96x512xf32>) outs(%1611 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb147(%1613: f32, %1614: f32):
      %1615 = arith.constant 5.000000e-01 : f32
      %1616 = arith.constant 1.000000e+00 : f32
      %1617 = arith.constant 0.707106769 : f32
      %1618 = arith.mulf %1613, %1617 : f32
      %1619 = math.erf %1618 : f32
      %1620 = arith.addf %1616, %1619 : f32
      %1621 = arith.mulf %1615, %1613 : f32
      %1622 = arith.mulf %1621, %1620 : f32
      linalg.yield %1622 : f32
    } -> tensor<1x96x512xf32>
    %1623 = tensor.empty() : tensor<512x64xf32>
    %1624 = linalg.transpose ins(%86:tensor<64x512xf32>) outs(%1623:tensor<512x64xf32>) permutation = [1, 0]
    %1625 = tensor.empty() : tensor<1x64xf32>
    %1626 = linalg.transpose ins(%87:tensor<64x1xf32>) outs(%1625:tensor<1x64xf32>) permutation = [1, 0]
    %1627 = tensor.empty() : tensor<512x64xf32>
    %1628 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1624, %1626 : tensor<512x64xf32>, tensor<1x64xf32>) outs(%1627 : tensor<512x64xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb148(%1629: f32, %1630: f32, %1631: f32):
      %1632 = arith.mulf %1629, %1630 : f32
      linalg.yield %1632 : f32
    } -> tensor<512x64xf32>
    %1633 = tensor.collapse_shape %1612 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1634 = tensor.expand_shape %1633 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1635 = tensor.empty() : tensor<96x64xf32>
    %1636 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1637 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1636 : f32) outs(%1635 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1638 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} ins(%1634, %1628 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1637 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1639 = tensor.collapse_shape %1638 [[0 : i64, 1 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1640 = tensor.expand_shape %1639 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1641 = tensor.empty() : tensor<1x96x64xf32>
    %1642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1640, %85 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1641 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb149(%1643: f32, %1644: f32, %1645: f32):
      %1646 = arith.addf %1643, %1644 : f32
      linalg.yield %1646 : f32
    } -> tensor<1x96x64xf32>
    %1647 = tensor.empty() : tensor<1x96x64xf32>
    %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1545, %1642 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1647 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb150(%1649: f32, %1650: f32, %1651: f32):
      %1652 = arith.addf %1649, %1650 : f32
      linalg.yield %1652 : f32
    } -> tensor<1x96x64xf32>
    %1653 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1654 = tensor.splat %1653 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1655 = linalg.reduce ins(%1648:tensor<1x96x64xf32>) outs(%1654:tensor<1x96xf32>) dimensions = [2]
    (%1656: f32, %1657: f32) {
      %1658 = arith.addf %1656, %1657 : f32
      linalg.yield %1658 : f32
    }
    %1659 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1660 = tensor.splat %1659 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1661 = tensor.empty() : tensor<1x96xf32>
    %1662 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1655, %1660 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1661 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb151(%1663: f32, %1664: f32, %1665: f32):
      %1666 = arith.divf %1663, %1664 : f32
      linalg.yield %1666 : f32
    } -> tensor<1x96xf32>
    %1667 = tensor.collapse_shape %1662 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1668 = tensor.expand_shape %1667 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1669 = tensor.empty() : tensor<1x96x64xf32>
    %1670 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1648, %1668 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1669 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb152(%1671: f32, %1672: f32, %1673: f32):
      %1674 = arith.subf %1671, %1672 : f32
      linalg.yield %1674 : f32
    } -> tensor<1x96x64xf32>
    %1675 = tensor.empty() : tensor<1x96x64xf32>
    %1676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1670, %1670 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1675 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb153(%1677: f32, %1678: f32, %1679: f32):
      %1680 = arith.mulf %1677, %1678 : f32
      linalg.yield %1680 : f32
    } -> tensor<1x96x64xf32>
    %1681 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1682 = tensor.splat %1681 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1683 = linalg.reduce ins(%1676:tensor<1x96x64xf32>) outs(%1682:tensor<1x96xf32>) dimensions = [2]
    (%1684: f32, %1685: f32) {
      %1686 = arith.addf %1684, %1685 : f32
      linalg.yield %1686 : f32
    }
    %1687 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1688 = tensor.splat %1687 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1689 = tensor.empty() : tensor<1x96xf32>
    %1690 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1683, %1688 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1689 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb154(%1691: f32, %1692: f32, %1693: f32):
      %1694 = arith.divf %1691, %1692 : f32
      linalg.yield %1694 : f32
    } -> tensor<1x96xf32>
    %1695 = tensor.collapse_shape %1690 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1696 = tensor.expand_shape %1695 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1697 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1698 = tensor.splat %1697 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1699 = tensor.empty() : tensor<1x96x1xf32>
    %1700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1696, %1698 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1699 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb155(%1701: f32, %1702: f32, %1703: f32):
      %1704 = arith.addf %1701, %1702 : f32
      linalg.yield %1704 : f32
    } -> tensor<1x96x1xf32>
    %1705 = tensor.empty() : tensor<1x96x1xf32>
    %1706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1700 : tensor<1x96x1xf32>) outs(%1705 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb156(%1707: f32, %1708: f32):
      %1709 = math.rsqrt %1707 : f32
      linalg.yield %1709 : f32
    } -> tensor<1x96x1xf32>
    %1710 = tensor.empty() : tensor<1x96x64xf32>
    %1711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1670, %1706 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1710 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb157(%1712: f32, %1713: f32, %1714: f32):
      %1715 = arith.mulf %1712, %1713 : f32
      linalg.yield %1715 : f32
    } -> tensor<1x96x64xf32>
    %1716 = tensor.empty() : tensor<1x96x64xf32>
    %1717 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1711, %96 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1716 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb158(%1718: f32, %1719: f32, %1720: f32):
      %1721 = arith.mulf %1718, %1719 : f32
      linalg.yield %1721 : f32
    } -> tensor<1x96x64xf32>
    %1722 = tensor.empty() : tensor<1x96x64xf32>
    %1723 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1717, %97 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1722 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb159(%1724: f32, %1725: f32, %1726: f32):
      %1727 = arith.addf %1724, %1725 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x96x64xf32>
    %1728 = tensor.empty() : tensor<1x64x96xf32>
    %1729 = linalg.transpose ins(%1723:tensor<1x96x64xf32>) outs(%1728:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1730 = tensor.collapse_shape %1729 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1731 = tensor.expand_shape %1730 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1732 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1731 : tensor<1x64x8x12xf32>) outs(%1732 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb160(%1734: f32, %1735: f32):
      linalg.yield %1734 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1736 = tensor.collapse_shape %1733 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1737 = tensor.expand_shape %1736 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1738 = tensor.collapse_shape %67 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1739 = tensor.expand_shape %1738 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1740 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1741 = tensor.splat %1740 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1742 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1739, %1737 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1741 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1743 = tensor.collapse_shape %1742 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1744 = tensor.expand_shape %1743 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1745 = tensor.collapse_shape %1744 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1746 = tensor.expand_shape %1745 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1747 = tensor.empty() : tensor<1x64x2x3xf32>
    %1748 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1746, %68 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1747 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb161(%1749: f32, %1750: f32, %1751: f32):
      %1752 = arith.addf %1749, %1750 : f32
      linalg.yield %1752 : f32
    } -> tensor<1x64x2x3xf32>
    %1753 = tensor.collapse_shape %1748 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1754 = tensor.expand_shape %1753 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1755 = tensor.empty() : tensor<1x6x64xf32>
    %1756 = linalg.transpose ins(%1754:tensor<1x64x6xf32>) outs(%1755:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1757 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1758 = tensor.splat %1757 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1759 = linalg.reduce ins(%1756:tensor<1x6x64xf32>) outs(%1758:tensor<1x6xf32>) dimensions = [2]
    (%1760: f32, %1761: f32) {
      %1762 = arith.addf %1760, %1761 : f32
      linalg.yield %1762 : f32
    }
    %1763 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1764 = tensor.splat %1763 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1765 = tensor.empty() : tensor<1x6xf32>
    %1766 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1759, %1764 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1765 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb162(%1767: f32, %1768: f32, %1769: f32):
      %1770 = arith.divf %1767, %1768 : f32
      linalg.yield %1770 : f32
    } -> tensor<1x6xf32>
    %1771 = tensor.collapse_shape %1766 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1772 = tensor.expand_shape %1771 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1773 = tensor.empty() : tensor<1x6x64xf32>
    %1774 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1756, %1772 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1773 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb163(%1775: f32, %1776: f32, %1777: f32):
      %1778 = arith.subf %1775, %1776 : f32
      linalg.yield %1778 : f32
    } -> tensor<1x6x64xf32>
    %1779 = tensor.empty() : tensor<1x6x64xf32>
    %1780 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1774, %1774 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1779 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb164(%1781: f32, %1782: f32, %1783: f32):
      %1784 = arith.mulf %1781, %1782 : f32
      linalg.yield %1784 : f32
    } -> tensor<1x6x64xf32>
    %1785 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1786 = tensor.splat %1785 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1787 = linalg.reduce ins(%1780:tensor<1x6x64xf32>) outs(%1786:tensor<1x6xf32>) dimensions = [2]
    (%1788: f32, %1789: f32) {
      %1790 = arith.addf %1788, %1789 : f32
      linalg.yield %1790 : f32
    }
    %1791 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1792 = tensor.splat %1791 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1793 = tensor.empty() : tensor<1x6xf32>
    %1794 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1787, %1792 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1793 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb165(%1795: f32, %1796: f32, %1797: f32):
      %1798 = arith.divf %1795, %1796 : f32
      linalg.yield %1798 : f32
    } -> tensor<1x6xf32>
    %1799 = tensor.collapse_shape %1794 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1800 = tensor.expand_shape %1799 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1801 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1802 = tensor.splat %1801 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1803 = tensor.empty() : tensor<1x6x1xf32>
    %1804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1800, %1802 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1803 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb166(%1805: f32, %1806: f32, %1807: f32):
      %1808 = arith.addf %1805, %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<1x6x1xf32>
    %1809 = tensor.empty() : tensor<1x6x1xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804 : tensor<1x6x1xf32>) outs(%1809 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb167(%1811: f32, %1812: f32):
      %1813 = math.rsqrt %1811 : f32
      linalg.yield %1813 : f32
    } -> tensor<1x6x1xf32>
    %1814 = tensor.empty() : tensor<1x6x64xf32>
    %1815 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1774, %1810 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1814 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb168(%1816: f32, %1817: f32, %1818: f32):
      %1819 = arith.mulf %1816, %1817 : f32
      linalg.yield %1819 : f32
    } -> tensor<1x6x64xf32>
    %1820 = tensor.empty() : tensor<1x6x64xf32>
    %1821 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1815, %69 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1820 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb169(%1822: f32, %1823: f32, %1824: f32):
      %1825 = arith.mulf %1822, %1823 : f32
      linalg.yield %1825 : f32
    } -> tensor<1x6x64xf32>
    %1826 = tensor.empty() : tensor<1x6x64xf32>
    %1827 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1821, %70 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1826 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb170(%1828: f32, %1829: f32, %1830: f32):
      %1831 = arith.addf %1828, %1829 : f32
      linalg.yield %1831 : f32
    } -> tensor<1x6x64xf32>
    %1832 = tensor.empty() : tensor<64x128xf32>
    %1833 = linalg.transpose ins(%72:tensor<128x64xf32>) outs(%1832:tensor<64x128xf32>) permutation = [1, 0]
    %1834 = tensor.empty() : tensor<1x128xf32>
    %1835 = linalg.transpose ins(%73:tensor<128x1xf32>) outs(%1834:tensor<1x128xf32>) permutation = [1, 0]
    %1836 = tensor.empty() : tensor<64x128xf32>
    %1837 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1833, %1835 : tensor<64x128xf32>, tensor<1x128xf32>) outs(%1836 : tensor<64x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb171(%1838: f32, %1839: f32, %1840: f32):
      %1841 = arith.mulf %1838, %1839 : f32
      linalg.yield %1841 : f32
    } -> tensor<64x128xf32>
    %1842 = tensor.collapse_shape %1827 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1843 = tensor.expand_shape %1842 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1844 = tensor.empty() : tensor<6x128xf32>
    %1845 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1846 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1845 : f32) outs(%1844 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1847 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} ins(%1843, %1837 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1846 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1848 = tensor.collapse_shape %1847 [[0 : i64, 1 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1849 = tensor.expand_shape %1848 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1850 = tensor.empty() : tensor<1x6x128xf32>
    %1851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1849, %71 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1850 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb172(%1852: f32, %1853: f32, %1854: f32):
      %1855 = arith.addf %1852, %1853 : f32
      linalg.yield %1855 : f32
    } -> tensor<1x6x128xf32>
    %1856 = tensor.collapse_shape %1851 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1857 = tensor.expand_shape %1856 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1858 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1859 = linalg.transpose ins(%1857:tensor<1x6x2x2x32xf32>) outs(%1858:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1860 = "tensor.extract_slice"(%1859) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1861 = tensor.collapse_shape %1860 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1862 = tensor.expand_shape %1861 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1863 = "tensor.extract_slice"(%1859) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1864 = tensor.collapse_shape %1863 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1865 = tensor.expand_shape %1864 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1866 = tensor.empty() : tensor<64x64xf32>
    %1867 = linalg.transpose ins(%75:tensor<64x64xf32>) outs(%1866:tensor<64x64xf32>) permutation = [1, 0]
    %1868 = tensor.empty() : tensor<1x64xf32>
    %1869 = linalg.transpose ins(%76:tensor<64x1xf32>) outs(%1868:tensor<1x64xf32>) permutation = [1, 0]
    %1870 = tensor.empty() : tensor<64x64xf32>
    %1871 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1867, %1869 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1870 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb173(%1872: f32, %1873: f32, %1874: f32):
      %1875 = arith.mulf %1872, %1873 : f32
      linalg.yield %1875 : f32
    } -> tensor<64x64xf32>
    %1876 = tensor.collapse_shape %1723 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1877 = tensor.expand_shape %1876 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1878 = tensor.empty() : tensor<96x64xf32>
    %1879 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1880 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1879 : f32) outs(%1878 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1881 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} ins(%1877, %1871 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1880 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1882 = tensor.collapse_shape %1881 [[0 : i64, 1 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1883 = tensor.expand_shape %1882 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1884 = tensor.empty() : tensor<1x96x64xf32>
    %1885 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1883, %74 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1884 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb174(%1886: f32, %1887: f32, %1888: f32):
      %1889 = arith.addf %1886, %1887 : f32
      linalg.yield %1889 : f32
    } -> tensor<1x96x64xf32>
    %1890 = tensor.collapse_shape %1885 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1891 = tensor.expand_shape %1890 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1892 = tensor.empty() : tensor<1x2x96x32xf32>
    %1893 = linalg.transpose ins(%1891:tensor<1x96x2x32xf32>) outs(%1892:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1894 = tensor.empty() : tensor<1x2x32x6xf32>
    %1895 = linalg.transpose ins(%1862:tensor<1x2x6x32xf32>) outs(%1894:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1896 = tensor.empty() : tensor<1x2x96x32xf32>
    %1897 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1893 : tensor<1x2x96x32xf32>) outs(%1896 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb175(%1898: f32, %1899: f32):
      linalg.yield %1898 : f32
    } -> tensor<1x2x96x32xf32>
    %1900 = tensor.collapse_shape %1897 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1901 = tensor.expand_shape %1900 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1902 = tensor.empty() : tensor<1x2x32x6xf32>
    %1903 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1895 : tensor<1x2x32x6xf32>) outs(%1902 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb176(%1904: f32, %1905: f32):
      linalg.yield %1904 : f32
    } -> tensor<1x2x32x6xf32>
    %1906 = tensor.collapse_shape %1903 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1907 = tensor.expand_shape %1906 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1908 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1909 = tensor.splat %1908 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1901, %1907 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1909 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb177(%1911: f32, %1912: f32, %1913: f32):
      %1914 = arith.mulf %1911, %1912 : f32
      %1915 = arith.addf %1913, %1914 : f32
      linalg.yield %1915 : f32
    } -> tensor<2x96x6xf32>
    %1916 = tensor.collapse_shape %1910 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1917 = tensor.expand_shape %1916 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1918 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1919 = tensor.splat %1918 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1920 = tensor.empty() : tensor<1x2x96x6xf32>
    %1921 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1917, %1919 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1920 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb178(%1922: f32, %1923: f32, %1924: f32):
      %1925 = arith.divf %1922, %1923 : f32
      linalg.yield %1925 : f32
    } -> tensor<1x2x96x6xf32>
    %1926 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1927 = tensor.splat %1926 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1928 = linalg.reduce ins(%1921:tensor<1x2x96x6xf32>) outs(%1927:tensor<1x2x96xf32>) dimensions = [3]
    (%1929: f32, %1930: f32) {
      %1931 = arith.maximumf %1929, %1930 : f32
      linalg.yield %1931 : f32
    }
    %1932 = tensor.collapse_shape %1928 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1933 = tensor.expand_shape %1932 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1934 = tensor.empty() : tensor<1x2x96x6xf32>
    %1935 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1921, %1933 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1934 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb179(%1936: f32, %1937: f32, %1938: f32):
      %1939 = arith.subf %1936, %1937 : f32
      linalg.yield %1939 : f32
    } -> tensor<1x2x96x6xf32>
    %1940 = tensor.empty() : tensor<1x2x96x6xf32>
    %1941 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1935 : tensor<1x2x96x6xf32>) outs(%1940 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb180(%1942: f32, %1943: f32):
      %1944 = math.exp %1942 : f32
      linalg.yield %1944 : f32
    } -> tensor<1x2x96x6xf32>
    %1945 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1946 = tensor.splat %1945 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1947 = linalg.reduce ins(%1941:tensor<1x2x96x6xf32>) outs(%1946:tensor<1x2x96xf32>) dimensions = [3]
    (%1948: f32, %1949: f32) {
      %1950 = arith.addf %1948, %1949 : f32
      linalg.yield %1950 : f32
    }
    %1951 = tensor.collapse_shape %1947 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1952 = tensor.expand_shape %1951 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1953 = tensor.empty() : tensor<1x2x96x6xf32>
    %1954 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1941, %1952 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1953 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb181(%1955: f32, %1956: f32, %1957: f32):
      %1958 = arith.divf %1955, %1956 : f32
      linalg.yield %1958 : f32
    } -> tensor<1x2x96x6xf32>
    %1959 = tensor.empty() : tensor<1x2x96x6xf32>
    %1960 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1954 : tensor<1x2x96x6xf32>) outs(%1959 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb182(%1961: f32, %1962: f32):
      linalg.yield %1961 : f32
    } -> tensor<1x2x96x6xf32>
    %1963 = tensor.collapse_shape %1960 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %1964 = tensor.expand_shape %1963 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %1965 = tensor.empty() : tensor<1x2x6x32xf32>
    %1966 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1865 : tensor<1x2x6x32xf32>) outs(%1965 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb183(%1967: f32, %1968: f32):
      linalg.yield %1967 : f32
    } -> tensor<1x2x6x32xf32>
    %1969 = tensor.collapse_shape %1966 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %1970 = tensor.expand_shape %1969 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %1971 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1972 = tensor.splat %1971 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %1973 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1964, %1970 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%1972 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb184(%1974: f32, %1975: f32, %1976: f32):
      %1977 = arith.mulf %1974, %1975 : f32
      %1978 = arith.addf %1976, %1977 : f32
      linalg.yield %1978 : f32
    } -> tensor<2x96x32xf32>
    %1979 = tensor.collapse_shape %1973 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %1980 = tensor.expand_shape %1979 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %1981 = tensor.empty() : tensor<1x96x2x32xf32>
    %1982 = linalg.transpose ins(%1980:tensor<1x2x96x32xf32>) outs(%1981:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1983 = tensor.collapse_shape %1982 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1984 = tensor.expand_shape %1983 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1985 = tensor.empty() : tensor<64x64xf32>
    %1986 = linalg.transpose ins(%78:tensor<64x64xf32>) outs(%1985:tensor<64x64xf32>) permutation = [1, 0]
    %1987 = tensor.empty() : tensor<1x64xf32>
    %1988 = linalg.transpose ins(%79:tensor<64x1xf32>) outs(%1987:tensor<1x64xf32>) permutation = [1, 0]
    %1989 = tensor.empty() : tensor<64x64xf32>
    %1990 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1986, %1988 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1989 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb185(%1991: f32, %1992: f32, %1993: f32):
      %1994 = arith.mulf %1991, %1992 : f32
      linalg.yield %1994 : f32
    } -> tensor<64x64xf32>
    %1995 = tensor.collapse_shape %1984 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1996 = tensor.expand_shape %1995 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1997 = tensor.empty() : tensor<96x64xf32>
    %1998 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1999 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1998 : f32) outs(%1997 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2000 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} ins(%1996, %1990 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1999 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2001 = tensor.collapse_shape %2000 [[0 : i64, 1 : i64]] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %2002 = tensor.expand_shape %2001 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2003 = tensor.empty() : tensor<1x96x64xf32>
    %2004 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2002, %77 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2003 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb186(%2005: f32, %2006: f32, %2007: f32):
      %2008 = arith.addf %2005, %2006 : f32
      linalg.yield %2008 : f32
    } -> tensor<1x96x64xf32>
    %2009 = tensor.empty() : tensor<1x96x64xf32>
    %2010 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1723, %2004 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2009 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb187(%2011: f32, %2012: f32, %2013: f32):
      %2014 = arith.addf %2011, %2012 : f32
      linalg.yield %2014 : f32
    } -> tensor<1x96x64xf32>
    %2015 = tensor.empty() : tensor<64x512xf32>
    %2016 = linalg.transpose ins(%89:tensor<512x64xf32>) outs(%2015:tensor<64x512xf32>) permutation = [1, 0]
    %2017 = tensor.empty() : tensor<1x512xf32>
    %2018 = linalg.transpose ins(%90:tensor<512x1xf32>) outs(%2017:tensor<1x512xf32>) permutation = [1, 0]
    %2019 = tensor.empty() : tensor<64x512xf32>
    %2020 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2016, %2018 : tensor<64x512xf32>, tensor<1x512xf32>) outs(%2019 : tensor<64x512xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb188(%2021: f32, %2022: f32, %2023: f32):
      %2024 = arith.mulf %2021, %2022 : f32
      linalg.yield %2024 : f32
    } -> tensor<64x512xf32>
    %2025 = tensor.collapse_shape %2010 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2026 = tensor.expand_shape %2025 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %2027 = tensor.empty() : tensor<96x512xf32>
    %2028 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2029 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2028 : f32) outs(%2027 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %2030 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} ins(%2026, %2020 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%2029 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %2031 = tensor.collapse_shape %2030 [[0 : i64, 1 : i64]] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %2032 = tensor.expand_shape %2031 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %2033 = tensor.empty() : tensor<1x96x512xf32>
    %2034 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2032, %88 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%2033 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb189(%2035: f32, %2036: f32, %2037: f32):
      %2038 = arith.addf %2035, %2036 : f32
      linalg.yield %2038 : f32
    } -> tensor<1x96x512xf32>
    %2039 = tensor.empty() : tensor<1x512x96xf32>
    %2040 = linalg.transpose ins(%2034:tensor<1x96x512xf32>) outs(%2039:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %2041 = tensor.collapse_shape %2040 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %2042 = tensor.expand_shape %2041 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2043 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2044 = tensor.splat %2043 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %2045 = "tensor.insert_slice"(%2042, %2044) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %2046 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %2047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2045 : tensor<1x512x10x14xf32>) outs(%2046 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb190(%2048: f32, %2049: f32):
      linalg.yield %2048 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %2050 = tensor.collapse_shape %2047 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %2051 = tensor.expand_shape %2050 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %2052 = tensor.collapse_shape %91 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %2053 = tensor.expand_shape %2052 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %2054 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %2055 = tensor.splat %2054 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %2056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2053, %2051 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%2055 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb191(%2057: f32, %2058: f32, %2059: f32):
      %2060 = arith.mulf %2057, %2058 : f32
      %2061 = arith.addf %2059, %2060 : f32
      linalg.yield %2061 : f32
    } -> tensor<64x8x96xf32>
    %2062 = tensor.collapse_shape %2056 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %2063 = tensor.expand_shape %2062 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %2064 = tensor.collapse_shape %2063 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %2065 = tensor.expand_shape %2064 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %2066 = tensor.empty() : tensor<1x512x8x12xf32>
    %2067 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2065, %92 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%2066 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb192(%2068: f32, %2069: f32, %2070: f32):
      %2071 = arith.addf %2068, %2069 : f32
      linalg.yield %2071 : f32
    } -> tensor<1x512x8x12xf32>
    %2072 = tensor.collapse_shape %2067 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %2073 = tensor.expand_shape %2072 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %2074 = tensor.empty() : tensor<1x96x512xf32>
    %2075 = linalg.transpose ins(%2073:tensor<1x512x96xf32>) outs(%2074:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %2076 = tensor.empty() : tensor<1x96x512xf32>
    %2077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2075 : tensor<1x96x512xf32>) outs(%2076 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb193(%2078: f32, %2079: f32):
      %2080 = arith.constant 5.000000e-01 : f32
      %2081 = arith.constant 1.000000e+00 : f32
      %2082 = arith.constant 0.707106769 : f32
      %2083 = arith.mulf %2078, %2082 : f32
      %2084 = math.erf %2083 : f32
      %2085 = arith.addf %2081, %2084 : f32
      %2086 = arith.mulf %2080, %2078 : f32
      %2087 = arith.mulf %2086, %2085 : f32
      linalg.yield %2087 : f32
    } -> tensor<1x96x512xf32>
    %2088 = tensor.empty() : tensor<512x64xf32>
    %2089 = linalg.transpose ins(%94:tensor<64x512xf32>) outs(%2088:tensor<512x64xf32>) permutation = [1, 0]
    %2090 = tensor.empty() : tensor<1x64xf32>
    %2091 = linalg.transpose ins(%95:tensor<64x1xf32>) outs(%2090:tensor<1x64xf32>) permutation = [1, 0]
    %2092 = tensor.empty() : tensor<512x64xf32>
    %2093 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2089, %2091 : tensor<512x64xf32>, tensor<1x64xf32>) outs(%2092 : tensor<512x64xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb194(%2094: f32, %2095: f32, %2096: f32):
      %2097 = arith.mulf %2094, %2095 : f32
      linalg.yield %2097 : f32
    } -> tensor<512x64xf32>
    %2098 = tensor.collapse_shape %2077 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %2099 = tensor.expand_shape %2098 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %2100 = tensor.empty() : tensor<96x64xf32>
    %2101 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2102 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2101 : f32) outs(%2100 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2103 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} ins(%2099, %2093 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%2102 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %2104 = tensor.collapse_shape %2103 [[0 : i64, 1 : i64]] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %2105 = tensor.expand_shape %2104 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %2106 = tensor.empty() : tensor<1x96x64xf32>
    %2107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2105, %93 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2106 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb195(%2108: f32, %2109: f32, %2110: f32):
      %2111 = arith.addf %2108, %2109 : f32
      linalg.yield %2111 : f32
    } -> tensor<1x96x64xf32>
    %2112 = tensor.empty() : tensor<1x96x64xf32>
    %2113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2010, %2107 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2112 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb196(%2114: f32, %2115: f32, %2116: f32):
      %2117 = arith.addf %2114, %2115 : f32
      linalg.yield %2117 : f32
    } -> tensor<1x96x64xf32>
    %2118 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2119 = tensor.splat %2118 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2120 = linalg.reduce ins(%2113:tensor<1x96x64xf32>) outs(%2119:tensor<1x96xf32>) dimensions = [2]
    (%2121: f32, %2122: f32) {
      %2123 = arith.addf %2121, %2122 : f32
      linalg.yield %2123 : f32
    }
    %2124 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2125 = tensor.splat %2124 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2126 = tensor.empty() : tensor<1x96xf32>
    %2127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2120, %2125 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2126 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb197(%2128: f32, %2129: f32, %2130: f32):
      %2131 = arith.divf %2128, %2129 : f32
      linalg.yield %2131 : f32
    } -> tensor<1x96xf32>
    %2132 = tensor.collapse_shape %2127 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2133 = tensor.expand_shape %2132 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2134 = tensor.empty() : tensor<1x96x64xf32>
    %2135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2113, %2133 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2134 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb198(%2136: f32, %2137: f32, %2138: f32):
      %2139 = arith.subf %2136, %2137 : f32
      linalg.yield %2139 : f32
    } -> tensor<1x96x64xf32>
    %2140 = tensor.empty() : tensor<1x96x64xf32>
    %2141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2135, %2135 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2140 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb199(%2142: f32, %2143: f32, %2144: f32):
      %2145 = arith.mulf %2142, %2143 : f32
      linalg.yield %2145 : f32
    } -> tensor<1x96x64xf32>
    %2146 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2147 = tensor.splat %2146 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2148 = linalg.reduce ins(%2141:tensor<1x96x64xf32>) outs(%2147:tensor<1x96xf32>) dimensions = [2]
    (%2149: f32, %2150: f32) {
      %2151 = arith.addf %2149, %2150 : f32
      linalg.yield %2151 : f32
    }
    %2152 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2153 = tensor.splat %2152 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2154 = tensor.empty() : tensor<1x96xf32>
    %2155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2148, %2153 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2154 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb200(%2156: f32, %2157: f32, %2158: f32):
      %2159 = arith.divf %2156, %2157 : f32
      linalg.yield %2159 : f32
    } -> tensor<1x96xf32>
    %2160 = tensor.collapse_shape %2155 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2161 = tensor.expand_shape %2160 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2162 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %2163 = tensor.splat %2162 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %2164 = tensor.empty() : tensor<1x96x1xf32>
    %2165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2161, %2163 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%2164 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb201(%2166: f32, %2167: f32, %2168: f32):
      %2169 = arith.addf %2166, %2167 : f32
      linalg.yield %2169 : f32
    } -> tensor<1x96x1xf32>
    %2170 = tensor.empty() : tensor<1x96x1xf32>
    %2171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2165 : tensor<1x96x1xf32>) outs(%2170 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb202(%2172: f32, %2173: f32):
      %2174 = math.rsqrt %2172 : f32
      linalg.yield %2174 : f32
    } -> tensor<1x96x1xf32>
    %2175 = tensor.empty() : tensor<1x96x64xf32>
    %2176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2135, %2171 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2175 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb203(%2177: f32, %2178: f32, %2179: f32):
      %2180 = arith.mulf %2177, %2178 : f32
      linalg.yield %2180 : f32
    } -> tensor<1x96x64xf32>
    %2181 = tensor.empty() : tensor<1x96x64xf32>
    %2182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2176, %98 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2181 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb204(%2183: f32, %2184: f32, %2185: f32):
      %2186 = arith.mulf %2183, %2184 : f32
      linalg.yield %2186 : f32
    } -> tensor<1x96x64xf32>
    %2187 = tensor.empty() : tensor<1x96x64xf32>
    %2188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2182, %99 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2187 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb205(%2189: f32, %2190: f32, %2191: f32):
      %2192 = arith.addf %2189, %2190 : f32
      linalg.yield %2192 : f32
    } -> tensor<1x96x64xf32>
    %2193 = tensor.collapse_shape %2188 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2194 = tensor.expand_shape %2193 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2195 = tensor.empty() : tensor<1x64x8x12xf32>
    %2196 = linalg.transpose ins(%2194:tensor<1x8x12x64xf32>) outs(%2195:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2197 = tensor.collapse_shape %2196 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2198 = tensor.expand_shape %2197 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2199 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2200 = linalg.transpose ins(%2198:tensor<1x16x2x2x8x12xf32>) outs(%2199:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2201 = tensor.collapse_shape %2200 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2202 = tensor.expand_shape %2201 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2203 = tensor.empty() : tensor<1x32x23x15xf32>
    %2204 = linalg.transpose ins(%1159:tensor<1x32x15x23xf32>) outs(%2203:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2205 = tensor.collapse_shape %2204 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2206 = tensor.expand_shape %2205 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2207 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2208 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2209 = tensor.splat %2208 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2210 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2206, %2207 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2209 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2211 = tensor.collapse_shape %2210 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2212 = tensor.expand_shape %2211 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2213 = tensor.empty() : tensor<1x32x16x23xf32>
    %2214 = linalg.transpose ins(%2212:tensor<1x32x23x16xf32>) outs(%2213:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2215 = tensor.collapse_shape %2214 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2216 = tensor.expand_shape %2215 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2217 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2218 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2219 = tensor.splat %2218 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2220 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2216, %2217 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2219 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2221 = tensor.collapse_shape %2220 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2222 = tensor.expand_shape %2221 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2223 = tensor.concat dim(1) %2202, %2222 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2224 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2225 = tensor.splat %2224 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2226 = "tensor.insert_slice"(%2223, %2225) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2227 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2226 : tensor<1x48x18x26xf32>) outs(%2227 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb206(%2229: f32, %2230: f32):
      linalg.yield %2229 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2231 = tensor.collapse_shape %2228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2232 = tensor.expand_shape %2231 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2233 = tensor.collapse_shape %118 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2234 = tensor.expand_shape %2233 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2235 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2236 = tensor.splat %2235 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2237 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2234, %2232 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2236 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2238 = tensor.collapse_shape %2237 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2239 = tensor.expand_shape %2238 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2240 = tensor.collapse_shape %2239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2241 = tensor.expand_shape %2240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2242 = tensor.empty() : tensor<1x12x16x24xf32>
    %2243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2241, %119 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2242 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb207(%2244: f32, %2245: f32, %2246: f32):
      %2247 = arith.addf %2244, %2245 : f32
      linalg.yield %2247 : f32
    } -> tensor<1x12x16x24xf32>
    %2248 = tensor.collapse_shape %2243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2249 = tensor.expand_shape %2248 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2250 = tensor.empty() : tensor<4608x512xf32>
    %2251 = linalg.transpose ins(%101:tensor<512x4608xf32>) outs(%2250:tensor<4608x512xf32>) permutation = [1, 0]
    %2252 = tensor.empty() : tensor<1x512xf32>
    %2253 = linalg.transpose ins(%102:tensor<512x1xf32>) outs(%2252:tensor<1x512xf32>) permutation = [1, 0]
    %2254 = tensor.empty() : tensor<4608x512xf32>
    %2255 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2251, %2253 : tensor<4608x512xf32>, tensor<1x512xf32>) outs(%2254 : tensor<4608x512xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb208(%2256: f32, %2257: f32, %2258: f32):
      %2259 = arith.mulf %2256, %2257 : f32
      linalg.yield %2259 : f32
    } -> tensor<4608x512xf32>
    %2260 = tensor.empty() : tensor<1x512xf32>
    %2261 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2262 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2261 : f32) outs(%2260 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2263 = linalg.matmul {prov.region_id = "matmul_28", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%2249, %2255 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2262 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2264 = tensor.empty() : tensor<1x512xf32>
    %2265 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2263, %100 : tensor<1x512xf32>, tensor<512xf32>) outs(%2264 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb209(%2266: f32, %2267: f32, %2268: f32):
      %2269 = arith.addf %2266, %2267 : f32
      linalg.yield %2269 : f32
    } -> tensor<1x512xf32>
    %2270 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2271 = tensor.splat %2270 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2272 = tensor.empty() : tensor<1x1xf32>
    %2273 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%121, %2271 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2272 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb210(%2274: f32, %2275: f32, %2276: f32):
      %2277 = arith.divf %2274, %2275 : f32
      linalg.yield %2277 : f32
    } -> tensor<1x1xf32>
    %2278 = tensor.concat dim(1) %2265, %2273, %122 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2279 = tensor.collapse_shape %2278 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2280 = tensor.expand_shape %2279 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2281 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2282 = tensor.splat %2281 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2283 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2284 = tensor.splat %2283 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2285 = "tensor.extract_slice"(%2282) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2286 = "tensor.extract_slice"(%2282) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2287 = "tensor.extract_slice"(%2282) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2288 = tensor.collapse_shape %2285 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2289 = tensor.expand_shape %2288 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2290 = tensor.collapse_shape %2286 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2291 = tensor.expand_shape %2290 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2292 = tensor.collapse_shape %2287 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2293 = tensor.expand_shape %2292 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2294 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2295 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2296 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2297 = tensor.collapse_shape %2294 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2298 = tensor.expand_shape %2297 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2299 = tensor.collapse_shape %2295 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2300 = tensor.expand_shape %2299 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2301 = tensor.collapse_shape %2296 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2302 = tensor.expand_shape %2301 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2303 = tensor.collapse_shape %2289 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2304 = tensor.expand_shape %2303 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2305 = tensor.collapse_shape %2298 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2306 = tensor.expand_shape %2305 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2307 = tensor.collapse_shape %2280 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2308 = tensor.expand_shape %2307 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2309 = tensor.empty() : tensor<517x512xf32>
    %2310 = linalg.transpose ins(%103:tensor<512x517xf32>) outs(%2309:tensor<517x512xf32>) permutation = [1, 0]
    %2311 = tensor.empty() : tensor<1x512xf32>
    %2312 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2313 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2312 : f32) outs(%2311 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2314 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2308, %2310 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2313 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2315 = tensor.empty() : tensor<1x512xf32>
    %2316 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2314, %105 : tensor<1x512xf32>, tensor<512xf32>) outs(%2315 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb211(%2317: f32, %2318: f32, %2319: f32):
      %2320 = arith.addf %2317, %2318 : f32
      linalg.yield %2320 : f32
    } -> tensor<1x512xf32>
    %2321 = tensor.collapse_shape %2316 [[0 : i64, 1 : i64]] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2322 = tensor.expand_shape %2321 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2323 = "tensor.extract_slice"(%2322) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2324 = tensor.collapse_shape %2323 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2325 = tensor.expand_shape %2324 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2326 = tensor.collapse_shape %2304 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2327 = tensor.expand_shape %2326 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2328 = tensor.empty() : tensor<128x512xf32>
    %2329 = linalg.transpose ins(%104:tensor<512x128xf32>) outs(%2328:tensor<128x512xf32>) permutation = [1, 0]
    %2330 = tensor.empty() : tensor<1x512xf32>
    %2331 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2332 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2331 : f32) outs(%2330 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2333 = linalg.matmul {prov.region_id = "matmul_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2327, %2329 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2332 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2334 = tensor.empty() : tensor<1x512xf32>
    %2335 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2333, %106 : tensor<1x512xf32>, tensor<512xf32>) outs(%2334 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb212(%2336: f32, %2337: f32, %2338: f32):
      %2339 = arith.addf %2336, %2337 : f32
      linalg.yield %2339 : f32
    } -> tensor<1x512xf32>
    %2340 = tensor.collapse_shape %2335 [[0 : i64, 1 : i64]] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2341 = tensor.expand_shape %2340 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2342 = tensor.empty() : tensor<1x1x512xf32>
    %2343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2341, %2325 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2342 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb213(%2344: f32, %2345: f32, %2346: f32):
      %2347 = arith.addf %2344, %2345 : f32
      linalg.yield %2347 : f32
    } -> tensor<1x1x512xf32>
    %2348 = "tensor.extract_slice"(%2343) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2349 = "tensor.extract_slice"(%2343) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2350 = "tensor.extract_slice"(%2343) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2351 = "tensor.extract_slice"(%2343) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2352 = tensor.empty() : tensor<1x1x128xf32>
    %2353 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2348 : tensor<1x1x128xf32>) outs(%2352 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb214(%2354: f32, %2355: f32):
      %2356 = arith.constant 1.000000e+00 : f32
      %2357 = arith.negf %2354 : f32
      %2358 = math.exp %2357 : f32
      %2359 = arith.addf %2356, %2358 : f32
      %2360 = arith.divf %2356, %2359 : f32
      linalg.yield %2360 : f32
    } -> tensor<1x1x128xf32>
    %2361 = tensor.empty() : tensor<1x1x128xf32>
    %2362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2349 : tensor<1x1x128xf32>) outs(%2361 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb215(%2363: f32, %2364: f32):
      %2365 = arith.constant 1.000000e+00 : f32
      %2366 = arith.negf %2363 : f32
      %2367 = math.exp %2366 : f32
      %2368 = arith.addf %2365, %2367 : f32
      %2369 = arith.divf %2365, %2368 : f32
      linalg.yield %2369 : f32
    } -> tensor<1x1x128xf32>
    %2370 = tensor.empty() : tensor<1x1x128xf32>
    %2371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2350 : tensor<1x1x128xf32>) outs(%2370 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb216(%2372: f32, %2373: f32):
      %2374 = math.tanh %2372 : f32
      linalg.yield %2374 : f32
    } -> tensor<1x1x128xf32>
    %2375 = tensor.empty() : tensor<1x1x128xf32>
    %2376 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2351 : tensor<1x1x128xf32>) outs(%2375 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb217(%2377: f32, %2378: f32):
      %2379 = arith.constant 1.000000e+00 : f32
      %2380 = arith.negf %2377 : f32
      %2381 = math.exp %2380 : f32
      %2382 = arith.addf %2379, %2381 : f32
      %2383 = arith.divf %2379, %2382 : f32
      linalg.yield %2383 : f32
    } -> tensor<1x1x128xf32>
    %2384 = tensor.empty() : tensor<1x1x128xf32>
    %2385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2362, %2306 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2384 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb218(%2386: f32, %2387: f32, %2388: f32):
      %2389 = arith.mulf %2386, %2387 : f32
      linalg.yield %2389 : f32
    } -> tensor<1x1x128xf32>
    %2390 = tensor.empty() : tensor<1x1x128xf32>
    %2391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2353, %2371 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2390 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb219(%2392: f32, %2393: f32, %2394: f32):
      %2395 = arith.mulf %2392, %2393 : f32
      linalg.yield %2395 : f32
    } -> tensor<1x1x128xf32>
    %2396 = tensor.empty() : tensor<1x1x128xf32>
    %2397 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2385, %2391 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2396 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb220(%2398: f32, %2399: f32, %2400: f32):
      %2401 = arith.addf %2398, %2399 : f32
      linalg.yield %2401 : f32
    } -> tensor<1x1x128xf32>
    %2402 = tensor.empty() : tensor<1x1x128xf32>
    %2403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2397 : tensor<1x1x128xf32>) outs(%2402 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_1", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb221(%2404: f32, %2405: f32):
      %2406 = math.tanh %2404 : f32
      linalg.yield %2406 : f32
    } -> tensor<1x1x128xf32>
    %2407 = tensor.empty() : tensor<1x1x128xf32>
    %2408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2376, %2403 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2407 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb222(%2409: f32, %2410: f32, %2411: f32):
      %2412 = arith.mulf %2409, %2410 : f32
      linalg.yield %2412 : f32
    } -> tensor<1x1x128xf32>
    %2413 = tensor.concat dim(0) %2408 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2414 = tensor.collapse_shape %2291 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2415 = tensor.expand_shape %2414 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2416 = tensor.collapse_shape %2300 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2417 = tensor.expand_shape %2416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2418 = tensor.collapse_shape %2413 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2419 = tensor.expand_shape %2418 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2420 = tensor.empty() : tensor<128x512xf32>
    %2421 = linalg.transpose ins(%107:tensor<512x128xf32>) outs(%2420:tensor<128x512xf32>) permutation = [1, 0]
    %2422 = tensor.empty() : tensor<1x512xf32>
    %2423 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2424 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2423 : f32) outs(%2422 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2425 = linalg.matmul {prov.region_id = "matmul_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2419, %2421 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2424 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2426 = tensor.empty() : tensor<1x512xf32>
    %2427 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2425, %109 : tensor<1x512xf32>, tensor<512xf32>) outs(%2426 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb223(%2428: f32, %2429: f32, %2430: f32):
      %2431 = arith.addf %2428, %2429 : f32
      linalg.yield %2431 : f32
    } -> tensor<1x512xf32>
    %2432 = tensor.collapse_shape %2427 [[0 : i64, 1 : i64]] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2433 = tensor.expand_shape %2432 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2434 = "tensor.extract_slice"(%2433) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2435 = tensor.collapse_shape %2434 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2436 = tensor.expand_shape %2435 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2437 = tensor.collapse_shape %2415 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2438 = tensor.expand_shape %2437 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2439 = tensor.empty() : tensor<128x512xf32>
    %2440 = linalg.transpose ins(%108:tensor<512x128xf32>) outs(%2439:tensor<128x512xf32>) permutation = [1, 0]
    %2441 = tensor.empty() : tensor<1x512xf32>
    %2442 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2443 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2442 : f32) outs(%2441 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2444 = linalg.matmul {prov.region_id = "matmul_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2438, %2440 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2443 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2445 = tensor.empty() : tensor<1x512xf32>
    %2446 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2444, %110 : tensor<1x512xf32>, tensor<512xf32>) outs(%2445 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb224(%2447: f32, %2448: f32, %2449: f32):
      %2450 = arith.addf %2447, %2448 : f32
      linalg.yield %2450 : f32
    } -> tensor<1x512xf32>
    %2451 = tensor.collapse_shape %2446 [[0 : i64, 1 : i64]] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2452 = tensor.expand_shape %2451 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2453 = tensor.empty() : tensor<1x1x512xf32>
    %2454 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2452, %2436 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2453 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb225(%2455: f32, %2456: f32, %2457: f32):
      %2458 = arith.addf %2455, %2456 : f32
      linalg.yield %2458 : f32
    } -> tensor<1x1x512xf32>
    %2459 = "tensor.extract_slice"(%2454) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2460 = "tensor.extract_slice"(%2454) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2461 = "tensor.extract_slice"(%2454) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2462 = "tensor.extract_slice"(%2454) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2463 = tensor.empty() : tensor<1x1x128xf32>
    %2464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2459 : tensor<1x1x128xf32>) outs(%2463 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb226(%2465: f32, %2466: f32):
      %2467 = arith.constant 1.000000e+00 : f32
      %2468 = arith.negf %2465 : f32
      %2469 = math.exp %2468 : f32
      %2470 = arith.addf %2467, %2469 : f32
      %2471 = arith.divf %2467, %2470 : f32
      linalg.yield %2471 : f32
    } -> tensor<1x1x128xf32>
    %2472 = tensor.empty() : tensor<1x1x128xf32>
    %2473 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2460 : tensor<1x1x128xf32>) outs(%2472 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_4", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb227(%2474: f32, %2475: f32):
      %2476 = arith.constant 1.000000e+00 : f32
      %2477 = arith.negf %2474 : f32
      %2478 = math.exp %2477 : f32
      %2479 = arith.addf %2476, %2478 : f32
      %2480 = arith.divf %2476, %2479 : f32
      linalg.yield %2480 : f32
    } -> tensor<1x1x128xf32>
    %2481 = tensor.empty() : tensor<1x1x128xf32>
    %2482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2461 : tensor<1x1x128xf32>) outs(%2481 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_2", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb228(%2483: f32, %2484: f32):
      %2485 = math.tanh %2483 : f32
      linalg.yield %2485 : f32
    } -> tensor<1x1x128xf32>
    %2486 = tensor.empty() : tensor<1x1x128xf32>
    %2487 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2462 : tensor<1x1x128xf32>) outs(%2486 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_5", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb229(%2488: f32, %2489: f32):
      %2490 = arith.constant 1.000000e+00 : f32
      %2491 = arith.negf %2488 : f32
      %2492 = math.exp %2491 : f32
      %2493 = arith.addf %2490, %2492 : f32
      %2494 = arith.divf %2490, %2493 : f32
      linalg.yield %2494 : f32
    } -> tensor<1x1x128xf32>
    %2495 = tensor.empty() : tensor<1x1x128xf32>
    %2496 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2473, %2417 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2495 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb230(%2497: f32, %2498: f32, %2499: f32):
      %2500 = arith.mulf %2497, %2498 : f32
      linalg.yield %2500 : f32
    } -> tensor<1x1x128xf32>
    %2501 = tensor.empty() : tensor<1x1x128xf32>
    %2502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2464, %2482 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2501 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb231(%2503: f32, %2504: f32, %2505: f32):
      %2506 = arith.mulf %2503, %2504 : f32
      linalg.yield %2506 : f32
    } -> tensor<1x1x128xf32>
    %2507 = tensor.empty() : tensor<1x1x128xf32>
    %2508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2496, %2502 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2507 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb232(%2509: f32, %2510: f32, %2511: f32):
      %2512 = arith.addf %2509, %2510 : f32
      linalg.yield %2512 : f32
    } -> tensor<1x1x128xf32>
    %2513 = tensor.empty() : tensor<1x1x128xf32>
    %2514 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2508 : tensor<1x1x128xf32>) outs(%2513 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_3", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb233(%2515: f32, %2516: f32):
      %2517 = math.tanh %2515 : f32
      linalg.yield %2517 : f32
    } -> tensor<1x1x128xf32>
    %2518 = tensor.empty() : tensor<1x1x128xf32>
    %2519 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2487, %2514 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2518 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb234(%2520: f32, %2521: f32, %2522: f32):
      %2523 = arith.mulf %2520, %2521 : f32
      linalg.yield %2523 : f32
    } -> tensor<1x1x128xf32>
    %2524 = tensor.concat dim(0) %2519 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2525 = tensor.collapse_shape %2293 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2526 = tensor.expand_shape %2525 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2527 = tensor.collapse_shape %2302 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2528 = tensor.expand_shape %2527 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2529 = tensor.collapse_shape %2524 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2530 = tensor.expand_shape %2529 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2531 = tensor.empty() : tensor<128x512xf32>
    %2532 = linalg.transpose ins(%111:tensor<512x128xf32>) outs(%2531:tensor<128x512xf32>) permutation = [1, 0]
    %2533 = tensor.empty() : tensor<1x512xf32>
    %2534 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2535 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2534 : f32) outs(%2533 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2536 = linalg.matmul {prov.region_id = "matmul_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2530, %2532 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2535 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2537 = tensor.empty() : tensor<1x512xf32>
    %2538 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2536, %113 : tensor<1x512xf32>, tensor<512xf32>) outs(%2537 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb235(%2539: f32, %2540: f32, %2541: f32):
      %2542 = arith.addf %2539, %2540 : f32
      linalg.yield %2542 : f32
    } -> tensor<1x512xf32>
    %2543 = tensor.collapse_shape %2538 [[0 : i64, 1 : i64]] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2544 = tensor.expand_shape %2543 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2545 = "tensor.extract_slice"(%2544) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2546 = tensor.collapse_shape %2545 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2547 = tensor.expand_shape %2546 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2548 = tensor.collapse_shape %2526 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2549 = tensor.expand_shape %2548 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2550 = tensor.empty() : tensor<128x512xf32>
    %2551 = linalg.transpose ins(%112:tensor<512x128xf32>) outs(%2550:tensor<128x512xf32>) permutation = [1, 0]
    %2552 = tensor.empty() : tensor<1x512xf32>
    %2553 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2554 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2553 : f32) outs(%2552 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2555 = linalg.matmul {prov.region_id = "matmul_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2549, %2551 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2554 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2556 = tensor.empty() : tensor<1x512xf32>
    %2557 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2555, %114 : tensor<1x512xf32>, tensor<512xf32>) outs(%2556 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb236(%2558: f32, %2559: f32, %2560: f32):
      %2561 = arith.addf %2558, %2559 : f32
      linalg.yield %2561 : f32
    } -> tensor<1x512xf32>
    %2562 = tensor.collapse_shape %2557 [[0 : i64, 1 : i64]] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2563 = tensor.expand_shape %2562 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2564 = tensor.empty() : tensor<1x1x512xf32>
    %2565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2563, %2547 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2564 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb237(%2566: f32, %2567: f32, %2568: f32):
      %2569 = arith.addf %2566, %2567 : f32
      linalg.yield %2569 : f32
    } -> tensor<1x1x512xf32>
    %2570 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2571 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2572 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2573 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2574 = tensor.empty() : tensor<1x1x128xf32>
    %2575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2570 : tensor<1x1x128xf32>) outs(%2574 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_6", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb238(%2576: f32, %2577: f32):
      %2578 = arith.constant 1.000000e+00 : f32
      %2579 = arith.negf %2576 : f32
      %2580 = math.exp %2579 : f32
      %2581 = arith.addf %2578, %2580 : f32
      %2582 = arith.divf %2578, %2581 : f32
      linalg.yield %2582 : f32
    } -> tensor<1x1x128xf32>
    %2583 = tensor.empty() : tensor<1x1x128xf32>
    %2584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2571 : tensor<1x1x128xf32>) outs(%2583 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_7", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb239(%2585: f32, %2586: f32):
      %2587 = arith.constant 1.000000e+00 : f32
      %2588 = arith.negf %2585 : f32
      %2589 = math.exp %2588 : f32
      %2590 = arith.addf %2587, %2589 : f32
      %2591 = arith.divf %2587, %2590 : f32
      linalg.yield %2591 : f32
    } -> tensor<1x1x128xf32>
    %2592 = tensor.empty() : tensor<1x1x128xf32>
    %2593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2572 : tensor<1x1x128xf32>) outs(%2592 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_4", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb240(%2594: f32, %2595: f32):
      %2596 = math.tanh %2594 : f32
      linalg.yield %2596 : f32
    } -> tensor<1x1x128xf32>
    %2597 = tensor.empty() : tensor<1x1x128xf32>
    %2598 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2573 : tensor<1x1x128xf32>) outs(%2597 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_8", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb241(%2599: f32, %2600: f32):
      %2601 = arith.constant 1.000000e+00 : f32
      %2602 = arith.negf %2599 : f32
      %2603 = math.exp %2602 : f32
      %2604 = arith.addf %2601, %2603 : f32
      %2605 = arith.divf %2601, %2604 : f32
      linalg.yield %2605 : f32
    } -> tensor<1x1x128xf32>
    %2606 = tensor.empty() : tensor<1x1x128xf32>
    %2607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2584, %2528 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2606 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb242(%2608: f32, %2609: f32, %2610: f32):
      %2611 = arith.mulf %2608, %2609 : f32
      linalg.yield %2611 : f32
    } -> tensor<1x1x128xf32>
    %2612 = tensor.empty() : tensor<1x1x128xf32>
    %2613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2575, %2593 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2612 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb243(%2614: f32, %2615: f32, %2616: f32):
      %2617 = arith.mulf %2614, %2615 : f32
      linalg.yield %2617 : f32
    } -> tensor<1x1x128xf32>
    %2618 = tensor.empty() : tensor<1x1x128xf32>
    %2619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2607, %2613 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2618 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb244(%2620: f32, %2621: f32, %2622: f32):
      %2623 = arith.addf %2620, %2621 : f32
      linalg.yield %2623 : f32
    } -> tensor<1x1x128xf32>
    %2624 = tensor.empty() : tensor<1x1x128xf32>
    %2625 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2619 : tensor<1x1x128xf32>) outs(%2624 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_5", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb245(%2626: f32, %2627: f32):
      %2628 = math.tanh %2626 : f32
      linalg.yield %2628 : f32
    } -> tensor<1x1x128xf32>
    %2629 = tensor.empty() : tensor<1x1x128xf32>
    %2630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2598, %2625 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2629 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb246(%2631: f32, %2632: f32, %2633: f32):
      %2634 = arith.mulf %2631, %2632 : f32
      linalg.yield %2634 : f32
    } -> tensor<1x1x128xf32>
    %2635 = tensor.concat dim(0) %2630 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2636 = tensor.collapse_shape %2635 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2637 = tensor.expand_shape %2636 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2638 = tensor.empty() : tensor<128x3xf32>
    %2639 = linalg.transpose ins(%116:tensor<3x128xf32>) outs(%2638:tensor<128x3xf32>) permutation = [1, 0]
    %2640 = tensor.empty() : tensor<1x3xf32>
    %2641 = linalg.transpose ins(%117:tensor<3x1xf32>) outs(%2640:tensor<1x3xf32>) permutation = [1, 0]
    %2642 = tensor.empty() : tensor<128x3xf32>
    %2643 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2639, %2641 : tensor<128x3xf32>, tensor<1x3xf32>) outs(%2642 : tensor<128x3xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb247(%2644: f32, %2645: f32, %2646: f32):
      %2647 = arith.mulf %2644, %2645 : f32
      linalg.yield %2647 : f32
    } -> tensor<128x3xf32>
    %2648 = tensor.empty() : tensor<1x3xf32>
    %2649 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2650 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2649 : f32) outs(%2648 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2651 = linalg.matmul {prov.region_id = "matmul_35", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2637, %2643 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2650 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2652 = tensor.empty() : tensor<1x3xf32>
    %2653 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2651, %115 : tensor<1x3xf32>, tensor<3xf32>) outs(%2652 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb248(%2654: f32, %2655: f32, %2656: f32):
      %2657 = arith.addf %2654, %2655 : f32
      linalg.yield %2657 : f32
    } -> tensor<1x3xf32>
    func.return %2653 : tensor<1x3xf32>
  }
}
