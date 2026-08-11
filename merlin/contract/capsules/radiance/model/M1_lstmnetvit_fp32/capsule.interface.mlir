builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64x32xf32>, %9: tensor<64xf32>, %10: tensor<32x32xf32>, %11: tensor<32xf32>, %12: tensor<32x32xf32>, %13: tensor<32xf32>, %14: tensor<32x32x8x8xf32>, %15: tensor<32xf32>, %16: tensor<32xf32>, %17: tensor<32xf32>, %18: tensor<64x32xf32>, %19: tensor<64xf32>, %20: tensor<32x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32xf32>, %24: tensor<256x32xf32>, %25: tensor<256xf32>, %26: tensor<256x8x3x3xf32>, %27: tensor<256xf32>, %28: tensor<32x256xf32>, %29: tensor<32xf32>, %30: tensor<256x32xf32>, %31: tensor<256xf32>, %32: tensor<256x8x3x3xf32>, %33: tensor<256xf32>, %34: tensor<32x256xf32>, %35: tensor<32xf32>, %36: tensor<32xf32>, %37: tensor<32xf32>, %38: tensor<32xf32>, %39: tensor<32xf32>, %40: tensor<64x32x3x3xf32>, %41: tensor<64xf32>, %42: tensor<64xf32>, %43: tensor<64xf32>, %44: tensor<64x64x4x4xf32>, %45: tensor<64xf32>, %46: tensor<64xf32>, %47: tensor<64xf32>, %48: tensor<128x64xf32>, %49: tensor<128xf32>, %50: tensor<64x64xf32>, %51: tensor<64xf32>, %52: tensor<64x64xf32>, %53: tensor<64xf32>, %54: tensor<64x64x4x4xf32>, %55: tensor<64xf32>, %56: tensor<64xf32>, %57: tensor<64xf32>, %58: tensor<128x64xf32>, %59: tensor<128xf32>, %60: tensor<64x64xf32>, %61: tensor<64xf32>, %62: tensor<64x64xf32>, %63: tensor<64xf32>, %64: tensor<512x64xf32>, %65: tensor<512xf32>, %66: tensor<512x8x3x3xf32>, %67: tensor<512xf32>, %68: tensor<64x512xf32>, %69: tensor<64xf32>, %70: tensor<512x64xf32>, %71: tensor<512xf32>, %72: tensor<512x8x3x3xf32>, %73: tensor<512xf32>, %74: tensor<64x512xf32>, %75: tensor<64xf32>, %76: tensor<64xf32>, %77: tensor<64xf32>, %78: tensor<64xf32>, %79: tensor<64xf32>, %80: tensor<512xf32>, %81: tensor<512x4608xf32>, %82: tensor<512x517xf32>, %83: tensor<512x128xf32>, %84: tensor<512xf32>, %85: tensor<512xf32>, %86: tensor<512x128xf32>, %87: tensor<512x128xf32>, %88: tensor<512xf32>, %89: tensor<512xf32>, %90: tensor<512x128xf32>, %91: tensor<512x128xf32>, %92: tensor<512xf32>, %93: tensor<512xf32>, %94: tensor<3xf32>, %95: tensor<3x128xf32>, %96: tensor<12x48x3x3xf32>, %97: tensor<12xf32>, %98: tensor<1x1x60x90xf32>, %99: tensor<1x1xf32>, %100: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %101 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %102 = tensor.splat %101 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %103 = "tensor.insert_slice"(%98, %102) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %104 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%103 : tensor<1x1x66x96xf32>) outs(%104 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%106: f32, %107: f32):
      linalg.yield %106 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %108 = tensor.collapse_shape %105 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %109 = tensor.expand_shape %108 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %110 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %111 = tensor.expand_shape %110 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %112 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %113 = tensor.splat %112 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %114 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%111, %109 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%113 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %115 = tensor.collapse_shape %114 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %116 = tensor.expand_shape %115 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %117 = tensor.collapse_shape %116 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %118 = tensor.expand_shape %117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %119 = tensor.empty() : tensor<1x32x15x23xf32>
    %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%119 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%121: f32, %122: f32, %123: f32):
      %124 = arith.addf %121, %122 : f32
      linalg.yield %124 : f32
    } -> tensor<1x32x15x23xf32>
    %125 = tensor.collapse_shape %120 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %126 = tensor.expand_shape %125 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %127 = tensor.empty() : tensor<1x345x32xf32>
    %128 = linalg.transpose ins(%126:tensor<1x32x345xf32>) outs(%127:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %129 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %130 = tensor.splat %129 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %131 = linalg.reduce ins(%128:tensor<1x345x32xf32>) outs(%130:tensor<1x345xf32>) dimensions = [2]
    (%132: f32, %133: f32) {
      %134 = arith.addf %132, %133 : f32
      linalg.yield %134 : f32
    }
    %135 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %136 = tensor.splat %135 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %137 = tensor.empty() : tensor<1x345xf32>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%131, %136 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%137 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%139: f32, %140: f32, %141: f32):
      %142 = arith.divf %139, %140 : f32
      linalg.yield %142 : f32
    } -> tensor<1x345xf32>
    %143 = tensor.collapse_shape %138 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %144 = tensor.expand_shape %143 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %145 = tensor.empty() : tensor<1x345x32xf32>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %144 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%145 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%147: f32, %148: f32, %149: f32):
      %150 = arith.subf %147, %148 : f32
      linalg.yield %150 : f32
    } -> tensor<1x345x32xf32>
    %151 = tensor.empty() : tensor<1x345x32xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%146, %146 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%151 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%153: f32, %154: f32, %155: f32):
      %156 = arith.mulf %153, %154 : f32
      linalg.yield %156 : f32
    } -> tensor<1x345x32xf32>
    %157 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %158 = tensor.splat %157 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %159 = linalg.reduce ins(%152:tensor<1x345x32xf32>) outs(%158:tensor<1x345xf32>) dimensions = [2]
    (%160: f32, %161: f32) {
      %162 = arith.addf %160, %161 : f32
      linalg.yield %162 : f32
    }
    %163 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %164 = tensor.splat %163 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %165 = tensor.empty() : tensor<1x345xf32>
    %166 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%159, %164 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%165 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%167: f32, %168: f32, %169: f32):
      %170 = arith.divf %167, %168 : f32
      linalg.yield %170 : f32
    } -> tensor<1x345xf32>
    %171 = tensor.collapse_shape %166 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %172 = tensor.expand_shape %171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %173 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %174 = tensor.splat %173 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %175 = tensor.empty() : tensor<1x345x1xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %174 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%175 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%177: f32, %178: f32, %179: f32):
      %180 = arith.addf %177, %178 : f32
      linalg.yield %180 : f32
    } -> tensor<1x345x1xf32>
    %181 = tensor.empty() : tensor<1x345x1xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176 : tensor<1x345x1xf32>) outs(%181 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%183: f32, %184: f32):
      %185 = math.rsqrt %183 : f32
      linalg.yield %185 : f32
    } -> tensor<1x345x1xf32>
    %186 = tensor.empty() : tensor<1x345x32xf32>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%146, %182 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%186 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%188: f32, %189: f32, %190: f32):
      %191 = arith.mulf %188, %189 : f32
      linalg.yield %191 : f32
    } -> tensor<1x345x32xf32>
    %192 = tensor.empty() : tensor<1x345x32xf32>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%192 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%194: f32, %195: f32, %196: f32):
      %197 = arith.mulf %194, %195 : f32
      linalg.yield %197 : f32
    } -> tensor<1x345x32xf32>
    %198 = tensor.empty() : tensor<1x345x32xf32>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%193, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%198 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%200: f32, %201: f32, %202: f32):
      %203 = arith.addf %200, %201 : f32
      linalg.yield %203 : f32
    } -> tensor<1x345x32xf32>
    %204 = tensor.empty() : tensor<1x32x345xf32>
    %205 = linalg.transpose ins(%199:tensor<1x345x32xf32>) outs(%204:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %206 = tensor.collapse_shape %205 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %207 = tensor.expand_shape %206 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %208 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%207 : tensor<1x32x15x23xf32>) outs(%208 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%210: f32, %211: f32):
      linalg.yield %210 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %212 = tensor.collapse_shape %209 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %213 = tensor.expand_shape %212 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %214 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %215 = tensor.expand_shape %214 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %216 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %217 = tensor.splat %216 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %218 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%215, %213 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%217 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %219 = tensor.collapse_shape %218 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %220 = tensor.expand_shape %219 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %221 = tensor.collapse_shape %220 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %222 = tensor.expand_shape %221 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %223 = tensor.empty() : tensor<1x32x1x2xf32>
    %224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%223 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%225: f32, %226: f32, %227: f32):
      %228 = arith.addf %225, %226 : f32
      linalg.yield %228 : f32
    } -> tensor<1x32x1x2xf32>
    %229 = tensor.collapse_shape %224 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %230 = tensor.expand_shape %229 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %231 = tensor.empty() : tensor<1x2x32xf32>
    %232 = linalg.transpose ins(%230:tensor<1x32x2xf32>) outs(%231:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %233 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %234 = tensor.splat %233 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %235 = linalg.reduce ins(%232:tensor<1x2x32xf32>) outs(%234:tensor<1x2xf32>) dimensions = [2]
    (%236: f32, %237: f32) {
      %238 = arith.addf %236, %237 : f32
      linalg.yield %238 : f32
    }
    %239 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %240 = tensor.splat %239 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %241 = tensor.empty() : tensor<1x2xf32>
    %242 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%235, %240 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%241 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%243: f32, %244: f32, %245: f32):
      %246 = arith.divf %243, %244 : f32
      linalg.yield %246 : f32
    } -> tensor<1x2xf32>
    %247 = tensor.collapse_shape %242 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %248 = tensor.expand_shape %247 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %249 = tensor.empty() : tensor<1x2x32xf32>
    %250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%232, %248 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%249 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%251: f32, %252: f32, %253: f32):
      %254 = arith.subf %251, %252 : f32
      linalg.yield %254 : f32
    } -> tensor<1x2x32xf32>
    %255 = tensor.empty() : tensor<1x2x32xf32>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %250 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%255 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%257: f32, %258: f32, %259: f32):
      %260 = arith.mulf %257, %258 : f32
      linalg.yield %260 : f32
    } -> tensor<1x2x32xf32>
    %261 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %262 = tensor.splat %261 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %263 = linalg.reduce ins(%256:tensor<1x2x32xf32>) outs(%262:tensor<1x2xf32>) dimensions = [2]
    (%264: f32, %265: f32) {
      %266 = arith.addf %264, %265 : f32
      linalg.yield %266 : f32
    }
    %267 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %268 = tensor.splat %267 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %269 = tensor.empty() : tensor<1x2xf32>
    %270 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%263, %268 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%269 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%271: f32, %272: f32, %273: f32):
      %274 = arith.divf %271, %272 : f32
      linalg.yield %274 : f32
    } -> tensor<1x2xf32>
    %275 = tensor.collapse_shape %270 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %276 = tensor.expand_shape %275 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %277 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %278 = tensor.splat %277 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %279 = tensor.empty() : tensor<1x2x1xf32>
    %280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%276, %278 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%279 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%281: f32, %282: f32, %283: f32):
      %284 = arith.addf %281, %282 : f32
      linalg.yield %284 : f32
    } -> tensor<1x2x1xf32>
    %285 = tensor.empty() : tensor<1x2x1xf32>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%280 : tensor<1x2x1xf32>) outs(%285 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%287: f32, %288: f32):
      %289 = math.rsqrt %287 : f32
      linalg.yield %289 : f32
    } -> tensor<1x2x1xf32>
    %290 = tensor.empty() : tensor<1x2x32xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %286 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%290 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%292: f32, %293: f32, %294: f32):
      %295 = arith.mulf %292, %293 : f32
      linalg.yield %295 : f32
    } -> tensor<1x2x32xf32>
    %296 = tensor.empty() : tensor<1x2x32xf32>
    %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%291, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%296 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%298: f32, %299: f32, %300: f32):
      %301 = arith.mulf %298, %299 : f32
      linalg.yield %301 : f32
    } -> tensor<1x2x32xf32>
    %302 = tensor.empty() : tensor<1x2x32xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%297, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%302 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%304: f32, %305: f32, %306: f32):
      %307 = arith.addf %304, %305 : f32
      linalg.yield %307 : f32
    } -> tensor<1x2x32xf32>
    %308 = tensor.collapse_shape %303 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %309 = tensor.expand_shape %308 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %310 = tensor.empty() : tensor<32x64xf32>
    %311 = linalg.transpose ins(%8:tensor<64x32xf32>) outs(%310:tensor<32x64xf32>) permutation = [1, 0]
    %312 = tensor.empty() : tensor<2x64xf32>
    %313 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %314 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%313 : f32) outs(%312 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %315 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor", prov.transposed_b = "true"} ins(%309, %311 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%314 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %316 = tensor.empty() : tensor<2x64xf32>
    %317 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%315, %9 : tensor<2x64xf32>, tensor<64xf32>) outs(%316 : tensor<2x64xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%318: f32, %319: f32, %320: f32):
      %321 = arith.addf %318, %319 : f32
      linalg.yield %321 : f32
    } -> tensor<2x64xf32>
    %322 = tensor.collapse_shape %317 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %323 = tensor.expand_shape %322 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %324 = tensor.collapse_shape %323 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %325 = tensor.expand_shape %324 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %326 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %327 = linalg.transpose ins(%325:tensor<1x2x2x1x32xf32>) outs(%326:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %328 = "tensor.extract_slice"(%327) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %329 = tensor.collapse_shape %328 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %330 = tensor.expand_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %331 = "tensor.extract_slice"(%327) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %332 = tensor.collapse_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %333 = tensor.expand_shape %332 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %334 = tensor.collapse_shape %199 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %335 = tensor.expand_shape %334 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %336 = tensor.empty() : tensor<32x32xf32>
    %337 = linalg.transpose ins(%10:tensor<32x32xf32>) outs(%336:tensor<32x32xf32>) permutation = [1, 0]
    %338 = tensor.empty() : tensor<345x32xf32>
    %339 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %340 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%339 : f32) outs(%338 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %341 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query", prov.transposed_b = "true"} ins(%335, %337 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%340 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %342 = tensor.empty() : tensor<345x32xf32>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%341, %11 : tensor<345x32xf32>, tensor<32xf32>) outs(%342 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb23(%344: f32, %345: f32, %346: f32):
      %347 = arith.addf %344, %345 : f32
      linalg.yield %347 : f32
    } -> tensor<345x32xf32>
    %348 = tensor.collapse_shape %343 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %349 = tensor.expand_shape %348 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %350 = tensor.collapse_shape %349 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %351 = tensor.expand_shape %350 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %352 = tensor.empty() : tensor<1x1x345x32xf32>
    %353 = linalg.transpose ins(%351:tensor<1x345x1x32xf32>) outs(%352:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %354 = tensor.empty() : tensor<1x1x32x2xf32>
    %355 = linalg.transpose ins(%330:tensor<1x1x2x32xf32>) outs(%354:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %356 = tensor.empty() : tensor<1x1x345x32xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353 : tensor<1x1x345x32xf32>) outs(%356 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb24(%358: f32, %359: f32):
      linalg.yield %358 : f32
    } -> tensor<1x1x345x32xf32>
    %360 = tensor.collapse_shape %357 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %361 = tensor.expand_shape %360 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %362 = tensor.empty() : tensor<1x1x32x2xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355 : tensor<1x1x32x2xf32>) outs(%362 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb25(%364: f32, %365: f32):
      linalg.yield %364 : f32
    } -> tensor<1x1x32x2xf32>
    %366 = tensor.collapse_shape %363 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %367 = tensor.expand_shape %366 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %368 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %369 = tensor.splat %368 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %370 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%361, %367 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%369 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb26(%371: f32, %372: f32, %373: f32):
      %374 = arith.mulf %371, %372 : f32
      %375 = arith.addf %373, %374 : f32
      linalg.yield %375 : f32
    } -> tensor<1x345x2xf32>
    %376 = tensor.collapse_shape %370 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %377 = tensor.expand_shape %376 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %378 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %379 = tensor.splat %378 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %380 = tensor.empty() : tensor<1x1x345x2xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%377, %379 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%380 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb27(%382: f32, %383: f32, %384: f32):
      %385 = arith.divf %382, %383 : f32
      linalg.yield %385 : f32
    } -> tensor<1x1x345x2xf32>
    %386 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %387 = tensor.splat %386 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %388 = linalg.reduce ins(%381:tensor<1x1x345x2xf32>) outs(%387:tensor<1x1x345xf32>) dimensions = [3]
    (%389: f32, %390: f32) {
      %391 = arith.maximumf %389, %390 : f32
      linalg.yield %391 : f32
    }
    %392 = tensor.collapse_shape %388 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %393 = tensor.expand_shape %392 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %394 = tensor.empty() : tensor<1x1x345x2xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %393 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%394 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb28(%396: f32, %397: f32, %398: f32):
      %399 = arith.subf %396, %397 : f32
      linalg.yield %399 : f32
    } -> tensor<1x1x345x2xf32>
    %400 = tensor.empty() : tensor<1x1x345x2xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395 : tensor<1x1x345x2xf32>) outs(%400 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb29(%402: f32, %403: f32):
      %404 = math.exp %402 : f32
      linalg.yield %404 : f32
    } -> tensor<1x1x345x2xf32>
    %405 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %406 = tensor.splat %405 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %407 = linalg.reduce ins(%401:tensor<1x1x345x2xf32>) outs(%406:tensor<1x1x345xf32>) dimensions = [3]
    (%408: f32, %409: f32) {
      %410 = arith.addf %408, %409 : f32
      linalg.yield %410 : f32
    }
    %411 = tensor.collapse_shape %407 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %412 = tensor.expand_shape %411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %413 = tensor.empty() : tensor<1x1x345x2xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401, %412 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%413 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb30(%415: f32, %416: f32, %417: f32):
      %418 = arith.divf %415, %416 : f32
      linalg.yield %418 : f32
    } -> tensor<1x1x345x2xf32>
    %419 = tensor.empty() : tensor<1x1x345x2xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414 : tensor<1x1x345x2xf32>) outs(%419 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb31(%421: f32, %422: f32):
      linalg.yield %421 : f32
    } -> tensor<1x1x345x2xf32>
    %423 = tensor.collapse_shape %420 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %424 = tensor.expand_shape %423 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %425 = tensor.empty() : tensor<1x1x2x32xf32>
    %426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333 : tensor<1x1x2x32xf32>) outs(%425 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb32(%427: f32, %428: f32):
      linalg.yield %427 : f32
    } -> tensor<1x1x2x32xf32>
    %429 = tensor.collapse_shape %426 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %430 = tensor.expand_shape %429 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %431 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %432 = tensor.splat %431 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%424, %430 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%432 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb33(%434: f32, %435: f32, %436: f32):
      %437 = arith.mulf %434, %435 : f32
      %438 = arith.addf %436, %437 : f32
      linalg.yield %438 : f32
    } -> tensor<1x345x32xf32>
    %439 = tensor.collapse_shape %433 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %440 = tensor.expand_shape %439 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %441 = tensor.empty() : tensor<1x345x1x32xf32>
    %442 = linalg.transpose ins(%440:tensor<1x1x345x32xf32>) outs(%441:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %443 = tensor.collapse_shape %442 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %444 = tensor.expand_shape %443 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %445 = tensor.collapse_shape %444 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %446 = tensor.expand_shape %445 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %447 = tensor.empty() : tensor<32x32xf32>
    %448 = linalg.transpose ins(%12:tensor<32x32xf32>) outs(%447:tensor<32x32xf32>) permutation = [1, 0]
    %449 = tensor.empty() : tensor<345x32xf32>
    %450 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %451 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%450 : f32) outs(%449 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %452 = linalg.matmul {prov.region_id = "matmul_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer", prov.transposed_b = "true"} ins(%446, %448 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%451 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %453 = tensor.empty() : tensor<345x32xf32>
    %454 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%452, %13 : tensor<345x32xf32>, tensor<32xf32>) outs(%453 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb34(%455: f32, %456: f32, %457: f32):
      %458 = arith.addf %455, %456 : f32
      linalg.yield %458 : f32
    } -> tensor<345x32xf32>
    %459 = tensor.collapse_shape %454 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %460 = tensor.expand_shape %459 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %461 = tensor.empty() : tensor<1x345x32xf32>
    %462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%199, %460 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%461 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb35(%463: f32, %464: f32, %465: f32):
      %466 = arith.addf %463, %464 : f32
      linalg.yield %466 : f32
    } -> tensor<1x345x32xf32>
    %467 = tensor.collapse_shape %462 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %468 = tensor.expand_shape %467 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %469 = tensor.empty() : tensor<32x256xf32>
    %470 = linalg.transpose ins(%24:tensor<256x32xf32>) outs(%469:tensor<32x256xf32>) permutation = [1, 0]
    %471 = tensor.empty() : tensor<345x256xf32>
    %472 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %473 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%472 : f32) outs(%471 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %474 = linalg.matmul {prov.region_id = "matmul_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1", prov.transposed_b = "true"} ins(%468, %470 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%473 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %475 = tensor.empty() : tensor<345x256xf32>
    %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%474, %25 : tensor<345x256xf32>, tensor<256xf32>) outs(%475 : tensor<345x256xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb36(%477: f32, %478: f32, %479: f32):
      %480 = arith.addf %477, %478 : f32
      linalg.yield %480 : f32
    } -> tensor<345x256xf32>
    %481 = tensor.collapse_shape %476 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %482 = tensor.expand_shape %481 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %483 = tensor.empty() : tensor<1x256x345xf32>
    %484 = linalg.transpose ins(%482:tensor<1x345x256xf32>) outs(%483:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %485 = tensor.collapse_shape %484 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %486 = tensor.expand_shape %485 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %487 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %488 = tensor.splat %487 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %489 = "tensor.insert_slice"(%486, %488) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %490 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%489 : tensor<1x256x17x25xf32>) outs(%490 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb37(%492: f32, %493: f32):
      linalg.yield %492 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %494 = tensor.collapse_shape %491 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %495 = tensor.expand_shape %494 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %496 = tensor.collapse_shape %26 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %497 = tensor.expand_shape %496 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %498 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %499 = tensor.splat %498 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %500 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%497, %495 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%499 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb38(%501: f32, %502: f32, %503: f32):
      %504 = arith.mulf %501, %502 : f32
      %505 = arith.addf %503, %504 : f32
      linalg.yield %505 : f32
    } -> tensor<32x8x345xf32>
    %506 = tensor.collapse_shape %500 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %507 = tensor.expand_shape %506 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %508 = tensor.collapse_shape %507 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %509 = tensor.expand_shape %508 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %510 = tensor.empty() : tensor<1x256x15x23xf32>
    %511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%509, %27 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%510 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb39(%512: f32, %513: f32, %514: f32):
      %515 = arith.addf %512, %513 : f32
      linalg.yield %515 : f32
    } -> tensor<1x256x15x23xf32>
    %516 = tensor.collapse_shape %511 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %517 = tensor.expand_shape %516 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %518 = tensor.empty() : tensor<1x345x256xf32>
    %519 = linalg.transpose ins(%517:tensor<1x256x345xf32>) outs(%518:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %520 = tensor.empty() : tensor<1x345x256xf32>
    %521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%519 : tensor<1x345x256xf32>) outs(%520 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb40(%522: f32, %523: f32):
      %524 = arith.constant 5.000000e-01 : f32
      %525 = arith.constant 1.000000e+00 : f32
      %526 = arith.constant 0.707106769 : f32
      %527 = arith.mulf %522, %526 : f32
      %528 = math.erf %527 : f32
      %529 = arith.addf %525, %528 : f32
      %530 = arith.mulf %524, %522 : f32
      %531 = arith.mulf %530, %529 : f32
      linalg.yield %531 : f32
    } -> tensor<1x345x256xf32>
    %532 = tensor.collapse_shape %521 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %533 = tensor.expand_shape %532 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %534 = tensor.empty() : tensor<256x32xf32>
    %535 = linalg.transpose ins(%28:tensor<32x256xf32>) outs(%534:tensor<256x32xf32>) permutation = [1, 0]
    %536 = tensor.empty() : tensor<345x32xf32>
    %537 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %538 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%537 : f32) outs(%536 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %539 = linalg.matmul {prov.region_id = "matmul_6", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2", prov.transposed_b = "true"} ins(%533, %535 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%538 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %540 = tensor.empty() : tensor<345x32xf32>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%539, %29 : tensor<345x32xf32>, tensor<32xf32>) outs(%540 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb41(%542: f32, %543: f32, %544: f32):
      %545 = arith.addf %542, %543 : f32
      linalg.yield %545 : f32
    } -> tensor<345x32xf32>
    %546 = tensor.collapse_shape %541 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %547 = tensor.expand_shape %546 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %548 = tensor.empty() : tensor<1x345x32xf32>
    %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%462, %547 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%548 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb42(%550: f32, %551: f32, %552: f32):
      %553 = arith.addf %550, %551 : f32
      linalg.yield %553 : f32
    } -> tensor<1x345x32xf32>
    %554 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %555 = tensor.splat %554 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %556 = linalg.reduce ins(%549:tensor<1x345x32xf32>) outs(%555:tensor<1x345xf32>) dimensions = [2]
    (%557: f32, %558: f32) {
      %559 = arith.addf %557, %558 : f32
      linalg.yield %559 : f32
    }
    %560 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %561 = tensor.splat %560 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %562 = tensor.empty() : tensor<1x345xf32>
    %563 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%556, %561 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%562 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb43(%564: f32, %565: f32, %566: f32):
      %567 = arith.divf %564, %565 : f32
      linalg.yield %567 : f32
    } -> tensor<1x345xf32>
    %568 = tensor.collapse_shape %563 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %569 = tensor.expand_shape %568 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %570 = tensor.empty() : tensor<1x345x32xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%549, %569 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%570 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb44(%572: f32, %573: f32, %574: f32):
      %575 = arith.subf %572, %573 : f32
      linalg.yield %575 : f32
    } -> tensor<1x345x32xf32>
    %576 = tensor.empty() : tensor<1x345x32xf32>
    %577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%571, %571 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%576 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb45(%578: f32, %579: f32, %580: f32):
      %581 = arith.mulf %578, %579 : f32
      linalg.yield %581 : f32
    } -> tensor<1x345x32xf32>
    %582 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %583 = tensor.splat %582 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %584 = linalg.reduce ins(%577:tensor<1x345x32xf32>) outs(%583:tensor<1x345xf32>) dimensions = [2]
    (%585: f32, %586: f32) {
      %587 = arith.addf %585, %586 : f32
      linalg.yield %587 : f32
    }
    %588 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %589 = tensor.splat %588 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %590 = tensor.empty() : tensor<1x345xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%584, %589 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%590 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb46(%592: f32, %593: f32, %594: f32):
      %595 = arith.divf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<1x345xf32>
    %596 = tensor.collapse_shape %591 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %597 = tensor.expand_shape %596 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %598 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %599 = tensor.splat %598 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %600 = tensor.empty() : tensor<1x345x1xf32>
    %601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %599 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%600 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%602: f32, %603: f32, %604: f32):
      %605 = arith.addf %602, %603 : f32
      linalg.yield %605 : f32
    } -> tensor<1x345x1xf32>
    %606 = tensor.empty() : tensor<1x345x1xf32>
    %607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601 : tensor<1x345x1xf32>) outs(%606 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb48(%608: f32, %609: f32):
      %610 = math.rsqrt %608 : f32
      linalg.yield %610 : f32
    } -> tensor<1x345x1xf32>
    %611 = tensor.empty() : tensor<1x345x32xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%571, %607 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%611 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb49(%613: f32, %614: f32, %615: f32):
      %616 = arith.mulf %613, %614 : f32
      linalg.yield %616 : f32
    } -> tensor<1x345x32xf32>
    %617 = tensor.empty() : tensor<1x345x32xf32>
    %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%612, %36 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%617 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb50(%619: f32, %620: f32, %621: f32):
      %622 = arith.mulf %619, %620 : f32
      linalg.yield %622 : f32
    } -> tensor<1x345x32xf32>
    %623 = tensor.empty() : tensor<1x345x32xf32>
    %624 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%618, %37 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%623 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb51(%625: f32, %626: f32, %627: f32):
      %628 = arith.addf %625, %626 : f32
      linalg.yield %628 : f32
    } -> tensor<1x345x32xf32>
    %629 = tensor.empty() : tensor<1x32x345xf32>
    %630 = linalg.transpose ins(%624:tensor<1x345x32xf32>) outs(%629:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %631 = tensor.collapse_shape %630 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %632 = tensor.expand_shape %631 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %633 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%632 : tensor<1x32x15x23xf32>) outs(%633 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb52(%635: f32, %636: f32):
      linalg.yield %635 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %637 = tensor.collapse_shape %634 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %638 = tensor.expand_shape %637 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %639 = tensor.collapse_shape %14 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %640 = tensor.expand_shape %639 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %641 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %642 = tensor.splat %641 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %643 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%640, %638 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%642 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %644 = tensor.collapse_shape %643 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %645 = tensor.expand_shape %644 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %646 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %647 = tensor.expand_shape %646 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %648 = tensor.empty() : tensor<1x32x1x2xf32>
    %649 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%647, %15 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%648 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb53(%650: f32, %651: f32, %652: f32):
      %653 = arith.addf %650, %651 : f32
      linalg.yield %653 : f32
    } -> tensor<1x32x1x2xf32>
    %654 = tensor.collapse_shape %649 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %655 = tensor.expand_shape %654 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %656 = tensor.empty() : tensor<1x2x32xf32>
    %657 = linalg.transpose ins(%655:tensor<1x32x2xf32>) outs(%656:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %658 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %659 = tensor.splat %658 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %660 = linalg.reduce ins(%657:tensor<1x2x32xf32>) outs(%659:tensor<1x2xf32>) dimensions = [2]
    (%661: f32, %662: f32) {
      %663 = arith.addf %661, %662 : f32
      linalg.yield %663 : f32
    }
    %664 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %665 = tensor.splat %664 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %666 = tensor.empty() : tensor<1x2xf32>
    %667 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%660, %665 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%666 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb54(%668: f32, %669: f32, %670: f32):
      %671 = arith.divf %668, %669 : f32
      linalg.yield %671 : f32
    } -> tensor<1x2xf32>
    %672 = tensor.collapse_shape %667 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %673 = tensor.expand_shape %672 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %674 = tensor.empty() : tensor<1x2x32xf32>
    %675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%657, %673 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%674 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb55(%676: f32, %677: f32, %678: f32):
      %679 = arith.subf %676, %677 : f32
      linalg.yield %679 : f32
    } -> tensor<1x2x32xf32>
    %680 = tensor.empty() : tensor<1x2x32xf32>
    %681 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675, %675 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%680 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb56(%682: f32, %683: f32, %684: f32):
      %685 = arith.mulf %682, %683 : f32
      linalg.yield %685 : f32
    } -> tensor<1x2x32xf32>
    %686 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %687 = tensor.splat %686 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %688 = linalg.reduce ins(%681:tensor<1x2x32xf32>) outs(%687:tensor<1x2xf32>) dimensions = [2]
    (%689: f32, %690: f32) {
      %691 = arith.addf %689, %690 : f32
      linalg.yield %691 : f32
    }
    %692 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %693 = tensor.splat %692 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %694 = tensor.empty() : tensor<1x2xf32>
    %695 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%688, %693 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%694 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb57(%696: f32, %697: f32, %698: f32):
      %699 = arith.divf %696, %697 : f32
      linalg.yield %699 : f32
    } -> tensor<1x2xf32>
    %700 = tensor.collapse_shape %695 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %701 = tensor.expand_shape %700 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %702 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %703 = tensor.splat %702 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %704 = tensor.empty() : tensor<1x2x1xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%701, %703 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%704 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb58(%706: f32, %707: f32, %708: f32):
      %709 = arith.addf %706, %707 : f32
      linalg.yield %709 : f32
    } -> tensor<1x2x1xf32>
    %710 = tensor.empty() : tensor<1x2x1xf32>
    %711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%705 : tensor<1x2x1xf32>) outs(%710 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb59(%712: f32, %713: f32):
      %714 = math.rsqrt %712 : f32
      linalg.yield %714 : f32
    } -> tensor<1x2x1xf32>
    %715 = tensor.empty() : tensor<1x2x32xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675, %711 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%715 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb60(%717: f32, %718: f32, %719: f32):
      %720 = arith.mulf %717, %718 : f32
      linalg.yield %720 : f32
    } -> tensor<1x2x32xf32>
    %721 = tensor.empty() : tensor<1x2x32xf32>
    %722 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716, %16 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%721 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb61(%723: f32, %724: f32, %725: f32):
      %726 = arith.mulf %723, %724 : f32
      linalg.yield %726 : f32
    } -> tensor<1x2x32xf32>
    %727 = tensor.empty() : tensor<1x2x32xf32>
    %728 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722, %17 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%727 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb62(%729: f32, %730: f32, %731: f32):
      %732 = arith.addf %729, %730 : f32
      linalg.yield %732 : f32
    } -> tensor<1x2x32xf32>
    %733 = tensor.collapse_shape %728 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %734 = tensor.expand_shape %733 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %735 = tensor.empty() : tensor<32x64xf32>
    %736 = linalg.transpose ins(%18:tensor<64x32xf32>) outs(%735:tensor<32x64xf32>) permutation = [1, 0]
    %737 = tensor.empty() : tensor<2x64xf32>
    %738 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %739 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%738 : f32) outs(%737 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %740 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor", prov.transposed_b = "true"} ins(%734, %736 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%739 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %741 = tensor.empty() : tensor<2x64xf32>
    %742 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%740, %19 : tensor<2x64xf32>, tensor<64xf32>) outs(%741 : tensor<2x64xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb63(%743: f32, %744: f32, %745: f32):
      %746 = arith.addf %743, %744 : f32
      linalg.yield %746 : f32
    } -> tensor<2x64xf32>
    %747 = tensor.collapse_shape %742 [[0 : i64, 1 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %748 = tensor.expand_shape %747 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %749 = tensor.collapse_shape %748 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %750 = tensor.expand_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %751 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %752 = linalg.transpose ins(%750:tensor<1x2x2x1x32xf32>) outs(%751:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %753 = "tensor.extract_slice"(%752) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %754 = tensor.collapse_shape %753 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %756 = "tensor.extract_slice"(%752) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %757 = tensor.collapse_shape %756 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %758 = tensor.expand_shape %757 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %759 = tensor.collapse_shape %624 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %760 = tensor.expand_shape %759 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %761 = tensor.empty() : tensor<32x32xf32>
    %762 = linalg.transpose ins(%20:tensor<32x32xf32>) outs(%761:tensor<32x32xf32>) permutation = [1, 0]
    %763 = tensor.empty() : tensor<345x32xf32>
    %764 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %765 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%764 : f32) outs(%763 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %766 = linalg.matmul {prov.region_id = "matmul_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query", prov.transposed_b = "true"} ins(%760, %762 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%765 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %767 = tensor.empty() : tensor<345x32xf32>
    %768 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%766, %21 : tensor<345x32xf32>, tensor<32xf32>) outs(%767 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb64(%769: f32, %770: f32, %771: f32):
      %772 = arith.addf %769, %770 : f32
      linalg.yield %772 : f32
    } -> tensor<345x32xf32>
    %773 = tensor.collapse_shape %768 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %774 = tensor.expand_shape %773 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %775 = tensor.collapse_shape %774 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %776 = tensor.expand_shape %775 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %777 = tensor.empty() : tensor<1x1x345x32xf32>
    %778 = linalg.transpose ins(%776:tensor<1x345x1x32xf32>) outs(%777:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %779 = tensor.empty() : tensor<1x1x32x2xf32>
    %780 = linalg.transpose ins(%755:tensor<1x1x2x32xf32>) outs(%779:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %781 = tensor.empty() : tensor<1x1x345x32xf32>
    %782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%778 : tensor<1x1x345x32xf32>) outs(%781 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb65(%783: f32, %784: f32):
      linalg.yield %783 : f32
    } -> tensor<1x1x345x32xf32>
    %785 = tensor.collapse_shape %782 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32> into tensor<11040xf32>
    %786 = tensor.expand_shape %785 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %787 = tensor.empty() : tensor<1x1x32x2xf32>
    %788 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%780 : tensor<1x1x32x2xf32>) outs(%787 : tensor<1x1x32x2xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb66(%789: f32, %790: f32):
      linalg.yield %789 : f32
    } -> tensor<1x1x32x2xf32>
    %791 = tensor.collapse_shape %788 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x32x2xf32> into tensor<64xf32>
    %792 = tensor.expand_shape %791 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %793 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %794 = tensor.splat %793 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32>
    %795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%786, %792 : tensor<1x345x32xf32>, tensor<1x32x2xf32>) outs(%794 : tensor<1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb67(%796: f32, %797: f32, %798: f32):
      %799 = arith.mulf %796, %797 : f32
      %800 = arith.addf %798, %799 : f32
      linalg.yield %800 : f32
    } -> tensor<1x345x2xf32>
    %801 = tensor.collapse_shape %795 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x2xf32> into tensor<690xf32>
    %802 = tensor.expand_shape %801 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 2] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x1x345x2xf32>
    %803 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %804 = tensor.splat %803 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %805 = tensor.empty() : tensor<1x1x345x2xf32>
    %806 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%802, %804 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%805 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb68(%807: f32, %808: f32, %809: f32):
      %810 = arith.divf %807, %808 : f32
      linalg.yield %810 : f32
    } -> tensor<1x1x345x2xf32>
    %811 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %812 = tensor.splat %811 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %813 = linalg.reduce ins(%806:tensor<1x1x345x2xf32>) outs(%812:tensor<1x1x345xf32>) dimensions = [3]
    (%814: f32, %815: f32) {
      %816 = arith.maximumf %814, %815 : f32
      linalg.yield %816 : f32
    }
    %817 = tensor.collapse_shape %813 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %818 = tensor.expand_shape %817 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %819 = tensor.empty() : tensor<1x1x345x2xf32>
    %820 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%806, %818 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%819 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb69(%821: f32, %822: f32, %823: f32):
      %824 = arith.subf %821, %822 : f32
      linalg.yield %824 : f32
    } -> tensor<1x1x345x2xf32>
    %825 = tensor.empty() : tensor<1x1x345x2xf32>
    %826 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820 : tensor<1x1x345x2xf32>) outs(%825 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb70(%827: f32, %828: f32):
      %829 = math.exp %827 : f32
      linalg.yield %829 : f32
    } -> tensor<1x1x345x2xf32>
    %830 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %831 = tensor.splat %830 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %832 = linalg.reduce ins(%826:tensor<1x1x345x2xf32>) outs(%831:tensor<1x1x345xf32>) dimensions = [3]
    (%833: f32, %834: f32) {
      %835 = arith.addf %833, %834 : f32
      linalg.yield %835 : f32
    }
    %836 = tensor.collapse_shape %832 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %837 = tensor.expand_shape %836 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %838 = tensor.empty() : tensor<1x1x345x2xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%826, %837 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%838 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb71(%840: f32, %841: f32, %842: f32):
      %843 = arith.divf %840, %841 : f32
      linalg.yield %843 : f32
    } -> tensor<1x1x345x2xf32>
    %844 = tensor.empty() : tensor<1x1x345x2xf32>
    %845 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%839 : tensor<1x1x345x2xf32>) outs(%844 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb72(%846: f32, %847: f32):
      linalg.yield %846 : f32
    } -> tensor<1x1x345x2xf32>
    %848 = tensor.collapse_shape %845 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32> into tensor<690xf32>
    %849 = tensor.expand_shape %848 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 2] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<690xf32> into tensor<1x345x2xf32>
    %850 = tensor.empty() : tensor<1x1x2x32xf32>
    %851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%758 : tensor<1x1x2x32xf32>) outs(%850 : tensor<1x1x2x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb73(%852: f32, %853: f32):
      linalg.yield %852 : f32
    } -> tensor<1x1x2x32xf32>
    %854 = tensor.collapse_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x2x32xf32> into tensor<64xf32>
    %855 = tensor.expand_shape %854 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 32] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x2x32xf32>
    %856 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %857 = tensor.splat %856 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32>
    %858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%849, %855 : tensor<1x345x2xf32>, tensor<1x2x32xf32>) outs(%857 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb74(%859: f32, %860: f32, %861: f32):
      %862 = arith.mulf %859, %860 : f32
      %863 = arith.addf %861, %862 : f32
      linalg.yield %863 : f32
    } -> tensor<1x345x32xf32>
    %864 = tensor.collapse_shape %858 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %865 = tensor.expand_shape %864 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x1x345x32xf32>
    %866 = tensor.empty() : tensor<1x345x1x32xf32>
    %867 = linalg.transpose ins(%865:tensor<1x1x345x32xf32>) outs(%866:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %868 = tensor.collapse_shape %867 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %869 = tensor.expand_shape %868 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %870 = tensor.collapse_shape %869 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %871 = tensor.expand_shape %870 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %872 = tensor.empty() : tensor<32x32xf32>
    %873 = linalg.transpose ins(%22:tensor<32x32xf32>) outs(%872:tensor<32x32xf32>) permutation = [1, 0]
    %874 = tensor.empty() : tensor<345x32xf32>
    %875 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %876 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%875 : f32) outs(%874 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %877 = linalg.matmul {prov.region_id = "matmul_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer", prov.transposed_b = "true"} ins(%871, %873 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%876 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %878 = tensor.empty() : tensor<345x32xf32>
    %879 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%877, %23 : tensor<345x32xf32>, tensor<32xf32>) outs(%878 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb75(%880: f32, %881: f32, %882: f32):
      %883 = arith.addf %880, %881 : f32
      linalg.yield %883 : f32
    } -> tensor<345x32xf32>
    %884 = tensor.collapse_shape %879 [[0 : i64, 1 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %885 = tensor.expand_shape %884 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %886 = tensor.empty() : tensor<1x345x32xf32>
    %887 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%624, %885 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%886 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%888: f32, %889: f32, %890: f32):
      %891 = arith.addf %888, %889 : f32
      linalg.yield %891 : f32
    } -> tensor<1x345x32xf32>
    %892 = tensor.collapse_shape %887 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %893 = tensor.expand_shape %892 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %894 = tensor.empty() : tensor<32x256xf32>
    %895 = linalg.transpose ins(%30:tensor<256x32xf32>) outs(%894:tensor<32x256xf32>) permutation = [1, 0]
    %896 = tensor.empty() : tensor<345x256xf32>
    %897 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %898 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%897 : f32) outs(%896 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %899 = linalg.matmul {prov.region_id = "matmul_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1", prov.transposed_b = "true"} ins(%893, %895 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%898 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %900 = tensor.empty() : tensor<345x256xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%899, %31 : tensor<345x256xf32>, tensor<256xf32>) outs(%900 : tensor<345x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb77(%902: f32, %903: f32, %904: f32):
      %905 = arith.addf %902, %903 : f32
      linalg.yield %905 : f32
    } -> tensor<345x256xf32>
    %906 = tensor.collapse_shape %901 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %907 = tensor.expand_shape %906 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %908 = tensor.empty() : tensor<1x256x345xf32>
    %909 = linalg.transpose ins(%907:tensor<1x345x256xf32>) outs(%908:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %910 = tensor.collapse_shape %909 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %911 = tensor.expand_shape %910 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %912 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %913 = tensor.splat %912 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %914 = "tensor.insert_slice"(%911, %913) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %915 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%914 : tensor<1x256x17x25xf32>) outs(%915 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb78(%917: f32, %918: f32):
      linalg.yield %917 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %919 = tensor.collapse_shape %916 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %920 = tensor.expand_shape %919 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %921 = tensor.collapse_shape %32 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %922 = tensor.expand_shape %921 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %923 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %924 = tensor.splat %923 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %925 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%922, %920 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%924 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb79(%926: f32, %927: f32, %928: f32):
      %929 = arith.mulf %926, %927 : f32
      %930 = arith.addf %928, %929 : f32
      linalg.yield %930 : f32
    } -> tensor<32x8x345xf32>
    %931 = tensor.collapse_shape %925 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %932 = tensor.expand_shape %931 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %933 = tensor.collapse_shape %932 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %934 = tensor.expand_shape %933 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %935 = tensor.empty() : tensor<1x256x15x23xf32>
    %936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%934, %33 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%935 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb80(%937: f32, %938: f32, %939: f32):
      %940 = arith.addf %937, %938 : f32
      linalg.yield %940 : f32
    } -> tensor<1x256x15x23xf32>
    %941 = tensor.collapse_shape %936 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %942 = tensor.expand_shape %941 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %943 = tensor.empty() : tensor<1x345x256xf32>
    %944 = linalg.transpose ins(%942:tensor<1x256x345xf32>) outs(%943:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %945 = tensor.empty() : tensor<1x345x256xf32>
    %946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944 : tensor<1x345x256xf32>) outs(%945 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb81(%947: f32, %948: f32):
      %949 = arith.constant 5.000000e-01 : f32
      %950 = arith.constant 1.000000e+00 : f32
      %951 = arith.constant 0.707106769 : f32
      %952 = arith.mulf %947, %951 : f32
      %953 = math.erf %952 : f32
      %954 = arith.addf %950, %953 : f32
      %955 = arith.mulf %949, %947 : f32
      %956 = arith.mulf %955, %954 : f32
      linalg.yield %956 : f32
    } -> tensor<1x345x256xf32>
    %957 = tensor.collapse_shape %946 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %958 = tensor.expand_shape %957 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %959 = tensor.empty() : tensor<256x32xf32>
    %960 = linalg.transpose ins(%34:tensor<32x256xf32>) outs(%959:tensor<256x32xf32>) permutation = [1, 0]
    %961 = tensor.empty() : tensor<345x32xf32>
    %962 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %963 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%962 : f32) outs(%961 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %964 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2", prov.transposed_b = "true"} ins(%958, %960 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%963 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %965 = tensor.empty() : tensor<345x32xf32>
    %966 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%964, %35 : tensor<345x32xf32>, tensor<32xf32>) outs(%965 : tensor<345x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb82(%967: f32, %968: f32, %969: f32):
      %970 = arith.addf %967, %968 : f32
      linalg.yield %970 : f32
    } -> tensor<345x32xf32>
    %971 = tensor.collapse_shape %966 [[0 : i64, 1 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %972 = tensor.expand_shape %971 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %973 = tensor.empty() : tensor<1x345x32xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%887, %972 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%973 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb83(%975: f32, %976: f32, %977: f32):
      %978 = arith.addf %975, %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x345x32xf32>
    %979 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %980 = tensor.splat %979 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %981 = linalg.reduce ins(%974:tensor<1x345x32xf32>) outs(%980:tensor<1x345xf32>) dimensions = [2]
    (%982: f32, %983: f32) {
      %984 = arith.addf %982, %983 : f32
      linalg.yield %984 : f32
    }
    %985 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %986 = tensor.splat %985 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %987 = tensor.empty() : tensor<1x345xf32>
    %988 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%981, %986 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%987 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%989: f32, %990: f32, %991: f32):
      %992 = arith.divf %989, %990 : f32
      linalg.yield %992 : f32
    } -> tensor<1x345xf32>
    %993 = tensor.collapse_shape %988 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %994 = tensor.expand_shape %993 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %995 = tensor.empty() : tensor<1x345x32xf32>
    %996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %994 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%995 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb85(%997: f32, %998: f32, %999: f32):
      %1000 = arith.subf %997, %998 : f32
      linalg.yield %1000 : f32
    } -> tensor<1x345x32xf32>
    %1001 = tensor.empty() : tensor<1x345x32xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%996, %996 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1001 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb86(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.mulf %1003, %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x345x32xf32>
    %1007 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1008 = tensor.splat %1007 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1009 = linalg.reduce ins(%1002:tensor<1x345x32xf32>) outs(%1008:tensor<1x345xf32>) dimensions = [2]
    (%1010: f32, %1011: f32) {
      %1012 = arith.addf %1010, %1011 : f32
      linalg.yield %1012 : f32
    }
    %1013 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1014 = tensor.splat %1013 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1015 = tensor.empty() : tensor<1x345xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1009, %1014 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1015 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb87(%1017: f32, %1018: f32, %1019: f32):
      %1020 = arith.divf %1017, %1018 : f32
      linalg.yield %1020 : f32
    } -> tensor<1x345xf32>
    %1021 = tensor.collapse_shape %1016 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1022 = tensor.expand_shape %1021 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1023 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1024 = tensor.splat %1023 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1025 = tensor.empty() : tensor<1x345x1xf32>
    %1026 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1022, %1024 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1025 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb88(%1027: f32, %1028: f32, %1029: f32):
      %1030 = arith.addf %1027, %1028 : f32
      linalg.yield %1030 : f32
    } -> tensor<1x345x1xf32>
    %1031 = tensor.empty() : tensor<1x345x1xf32>
    %1032 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1026 : tensor<1x345x1xf32>) outs(%1031 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb89(%1033: f32, %1034: f32):
      %1035 = math.rsqrt %1033 : f32
      linalg.yield %1035 : f32
    } -> tensor<1x345x1xf32>
    %1036 = tensor.empty() : tensor<1x345x32xf32>
    %1037 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%996, %1032 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1036 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb90(%1038: f32, %1039: f32, %1040: f32):
      %1041 = arith.mulf %1038, %1039 : f32
      linalg.yield %1041 : f32
    } -> tensor<1x345x32xf32>
    %1042 = tensor.empty() : tensor<1x345x32xf32>
    %1043 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %38 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1042 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb91(%1044: f32, %1045: f32, %1046: f32):
      %1047 = arith.mulf %1044, %1045 : f32
      linalg.yield %1047 : f32
    } -> tensor<1x345x32xf32>
    %1048 = tensor.empty() : tensor<1x345x32xf32>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1043, %39 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1048 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb92(%1050: f32, %1051: f32, %1052: f32):
      %1053 = arith.addf %1050, %1051 : f32
      linalg.yield %1053 : f32
    } -> tensor<1x345x32xf32>
    %1054 = tensor.collapse_shape %1049 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1055 = tensor.expand_shape %1054 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1056 = tensor.empty() : tensor<1x32x15x23xf32>
    %1057 = linalg.transpose ins(%1055:tensor<1x15x23x32xf32>) outs(%1056:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1058 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1059 = tensor.splat %1058 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1060 = "tensor.insert_slice"(%1057, %1059) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1061 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1060 : tensor<1x32x17x25xf32>) outs(%1061 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb93(%1063: f32, %1064: f32):
      linalg.yield %1063 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1065 = tensor.collapse_shape %1062 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1066 = tensor.expand_shape %1065 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1067 = tensor.collapse_shape %40 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1068 = tensor.expand_shape %1067 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1069 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1071 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1068, %1066 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1070 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1072 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1073 = tensor.expand_shape %1072 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1074 = tensor.collapse_shape %1073 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1075 = tensor.expand_shape %1074 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1076 = tensor.empty() : tensor<1x64x8x12xf32>
    %1077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1075, %41 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1076 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb94(%1078: f32, %1079: f32, %1080: f32):
      %1081 = arith.addf %1078, %1079 : f32
      linalg.yield %1081 : f32
    } -> tensor<1x64x8x12xf32>
    %1082 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1083 = tensor.expand_shape %1082 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1084 = tensor.empty() : tensor<1x96x64xf32>
    %1085 = linalg.transpose ins(%1083:tensor<1x64x96xf32>) outs(%1084:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1086 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1087 = tensor.splat %1086 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1088 = linalg.reduce ins(%1085:tensor<1x96x64xf32>) outs(%1087:tensor<1x96xf32>) dimensions = [2]
    (%1089: f32, %1090: f32) {
      %1091 = arith.addf %1089, %1090 : f32
      linalg.yield %1091 : f32
    }
    %1092 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1093 = tensor.splat %1092 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1094 = tensor.empty() : tensor<1x96xf32>
    %1095 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1088, %1093 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1094 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb95(%1096: f32, %1097: f32, %1098: f32):
      %1099 = arith.divf %1096, %1097 : f32
      linalg.yield %1099 : f32
    } -> tensor<1x96xf32>
    %1100 = tensor.collapse_shape %1095 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1101 = tensor.expand_shape %1100 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1102 = tensor.empty() : tensor<1x96x64xf32>
    %1103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1085, %1101 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1102 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb96(%1104: f32, %1105: f32, %1106: f32):
      %1107 = arith.subf %1104, %1105 : f32
      linalg.yield %1107 : f32
    } -> tensor<1x96x64xf32>
    %1108 = tensor.empty() : tensor<1x96x64xf32>
    %1109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1103, %1103 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1108 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb97(%1110: f32, %1111: f32, %1112: f32):
      %1113 = arith.mulf %1110, %1111 : f32
      linalg.yield %1113 : f32
    } -> tensor<1x96x64xf32>
    %1114 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1115 = tensor.splat %1114 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1116 = linalg.reduce ins(%1109:tensor<1x96x64xf32>) outs(%1115:tensor<1x96xf32>) dimensions = [2]
    (%1117: f32, %1118: f32) {
      %1119 = arith.addf %1117, %1118 : f32
      linalg.yield %1119 : f32
    }
    %1120 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1121 = tensor.splat %1120 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1122 = tensor.empty() : tensor<1x96xf32>
    %1123 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1116, %1121 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1122 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb98(%1124: f32, %1125: f32, %1126: f32):
      %1127 = arith.divf %1124, %1125 : f32
      linalg.yield %1127 : f32
    } -> tensor<1x96xf32>
    %1128 = tensor.collapse_shape %1123 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1129 = tensor.expand_shape %1128 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1130 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1131 = tensor.splat %1130 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1132 = tensor.empty() : tensor<1x96x1xf32>
    %1133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1129, %1131 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1132 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb99(%1134: f32, %1135: f32, %1136: f32):
      %1137 = arith.addf %1134, %1135 : f32
      linalg.yield %1137 : f32
    } -> tensor<1x96x1xf32>
    %1138 = tensor.empty() : tensor<1x96x1xf32>
    %1139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1133 : tensor<1x96x1xf32>) outs(%1138 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb100(%1140: f32, %1141: f32):
      %1142 = math.rsqrt %1140 : f32
      linalg.yield %1142 : f32
    } -> tensor<1x96x1xf32>
    %1143 = tensor.empty() : tensor<1x96x64xf32>
    %1144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1103, %1139 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1143 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb101(%1145: f32, %1146: f32, %1147: f32):
      %1148 = arith.mulf %1145, %1146 : f32
      linalg.yield %1148 : f32
    } -> tensor<1x96x64xf32>
    %1149 = tensor.empty() : tensor<1x96x64xf32>
    %1150 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1144, %42 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1149 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb102(%1151: f32, %1152: f32, %1153: f32):
      %1154 = arith.mulf %1151, %1152 : f32
      linalg.yield %1154 : f32
    } -> tensor<1x96x64xf32>
    %1155 = tensor.empty() : tensor<1x96x64xf32>
    %1156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1150, %43 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1155 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb103(%1157: f32, %1158: f32, %1159: f32):
      %1160 = arith.addf %1157, %1158 : f32
      linalg.yield %1160 : f32
    } -> tensor<1x96x64xf32>
    %1161 = tensor.empty() : tensor<1x64x96xf32>
    %1162 = linalg.transpose ins(%1156:tensor<1x96x64xf32>) outs(%1161:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1163 = tensor.collapse_shape %1162 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1164 = tensor.expand_shape %1163 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1165 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1164 : tensor<1x64x8x12xf32>) outs(%1165 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb104(%1167: f32, %1168: f32):
      linalg.yield %1167 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1169 = tensor.collapse_shape %1166 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1170 = tensor.expand_shape %1169 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1171 = tensor.collapse_shape %44 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1172 = tensor.expand_shape %1171 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1173 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1174 = tensor.splat %1173 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1175 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1172, %1170 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1174 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1176 = tensor.collapse_shape %1175 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1177 = tensor.expand_shape %1176 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1178 = tensor.collapse_shape %1177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1179 = tensor.expand_shape %1178 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1180 = tensor.empty() : tensor<1x64x2x3xf32>
    %1181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1179, %45 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1180 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb105(%1182: f32, %1183: f32, %1184: f32):
      %1185 = arith.addf %1182, %1183 : f32
      linalg.yield %1185 : f32
    } -> tensor<1x64x2x3xf32>
    %1186 = tensor.collapse_shape %1181 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1187 = tensor.expand_shape %1186 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1188 = tensor.empty() : tensor<1x6x64xf32>
    %1189 = linalg.transpose ins(%1187:tensor<1x64x6xf32>) outs(%1188:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1190 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1191 = tensor.splat %1190 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1192 = linalg.reduce ins(%1189:tensor<1x6x64xf32>) outs(%1191:tensor<1x6xf32>) dimensions = [2]
    (%1193: f32, %1194: f32) {
      %1195 = arith.addf %1193, %1194 : f32
      linalg.yield %1195 : f32
    }
    %1196 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1197 = tensor.splat %1196 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1198 = tensor.empty() : tensor<1x6xf32>
    %1199 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1192, %1197 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1198 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb106(%1200: f32, %1201: f32, %1202: f32):
      %1203 = arith.divf %1200, %1201 : f32
      linalg.yield %1203 : f32
    } -> tensor<1x6xf32>
    %1204 = tensor.collapse_shape %1199 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1205 = tensor.expand_shape %1204 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1206 = tensor.empty() : tensor<1x6x64xf32>
    %1207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1189, %1205 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1206 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb107(%1208: f32, %1209: f32, %1210: f32):
      %1211 = arith.subf %1208, %1209 : f32
      linalg.yield %1211 : f32
    } -> tensor<1x6x64xf32>
    %1212 = tensor.empty() : tensor<1x6x64xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207, %1207 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1212 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb108(%1214: f32, %1215: f32, %1216: f32):
      %1217 = arith.mulf %1214, %1215 : f32
      linalg.yield %1217 : f32
    } -> tensor<1x6x64xf32>
    %1218 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1219 = tensor.splat %1218 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1220 = linalg.reduce ins(%1213:tensor<1x6x64xf32>) outs(%1219:tensor<1x6xf32>) dimensions = [2]
    (%1221: f32, %1222: f32) {
      %1223 = arith.addf %1221, %1222 : f32
      linalg.yield %1223 : f32
    }
    %1224 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1225 = tensor.splat %1224 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1226 = tensor.empty() : tensor<1x6xf32>
    %1227 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1220, %1225 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1226 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb109(%1228: f32, %1229: f32, %1230: f32):
      %1231 = arith.divf %1228, %1229 : f32
      linalg.yield %1231 : f32
    } -> tensor<1x6xf32>
    %1232 = tensor.collapse_shape %1227 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1233 = tensor.expand_shape %1232 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1234 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1235 = tensor.splat %1234 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1236 = tensor.empty() : tensor<1x6x1xf32>
    %1237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1233, %1235 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1236 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb110(%1238: f32, %1239: f32, %1240: f32):
      %1241 = arith.addf %1238, %1239 : f32
      linalg.yield %1241 : f32
    } -> tensor<1x6x1xf32>
    %1242 = tensor.empty() : tensor<1x6x1xf32>
    %1243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1237 : tensor<1x6x1xf32>) outs(%1242 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb111(%1244: f32, %1245: f32):
      %1246 = math.rsqrt %1244 : f32
      linalg.yield %1246 : f32
    } -> tensor<1x6x1xf32>
    %1247 = tensor.empty() : tensor<1x6x64xf32>
    %1248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207, %1243 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1247 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb112(%1249: f32, %1250: f32, %1251: f32):
      %1252 = arith.mulf %1249, %1250 : f32
      linalg.yield %1252 : f32
    } -> tensor<1x6x64xf32>
    %1253 = tensor.empty() : tensor<1x6x64xf32>
    %1254 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1248, %46 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1253 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb113(%1255: f32, %1256: f32, %1257: f32):
      %1258 = arith.mulf %1255, %1256 : f32
      linalg.yield %1258 : f32
    } -> tensor<1x6x64xf32>
    %1259 = tensor.empty() : tensor<1x6x64xf32>
    %1260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1254, %47 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1259 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb114(%1261: f32, %1262: f32, %1263: f32):
      %1264 = arith.addf %1261, %1262 : f32
      linalg.yield %1264 : f32
    } -> tensor<1x6x64xf32>
    %1265 = tensor.collapse_shape %1260 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1266 = tensor.expand_shape %1265 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1267 = tensor.empty() : tensor<64x128xf32>
    %1268 = linalg.transpose ins(%48:tensor<128x64xf32>) outs(%1267:tensor<64x128xf32>) permutation = [1, 0]
    %1269 = tensor.empty() : tensor<6x128xf32>
    %1270 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1271 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1270 : f32) outs(%1269 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1272 = linalg.matmul {prov.region_id = "matmul_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor", prov.transposed_b = "true"} ins(%1266, %1268 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1271 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1273 = tensor.empty() : tensor<6x128xf32>
    %1274 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1272, %49 : tensor<6x128xf32>, tensor<128xf32>) outs(%1273 : tensor<6x128xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb115(%1275: f32, %1276: f32, %1277: f32):
      %1278 = arith.addf %1275, %1276 : f32
      linalg.yield %1278 : f32
    } -> tensor<6x128xf32>
    %1279 = tensor.collapse_shape %1274 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1280 = tensor.expand_shape %1279 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1281 = tensor.collapse_shape %1280 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1282 = tensor.expand_shape %1281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1283 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1284 = linalg.transpose ins(%1282:tensor<1x6x2x2x32xf32>) outs(%1283:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1285 = "tensor.extract_slice"(%1284) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1286 = tensor.collapse_shape %1285 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1287 = tensor.expand_shape %1286 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1288 = "tensor.extract_slice"(%1284) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1289 = tensor.collapse_shape %1288 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1290 = tensor.expand_shape %1289 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1291 = tensor.collapse_shape %1156 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1292 = tensor.expand_shape %1291 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1293 = tensor.empty() : tensor<64x64xf32>
    %1294 = linalg.transpose ins(%50:tensor<64x64xf32>) outs(%1293:tensor<64x64xf32>) permutation = [1, 0]
    %1295 = tensor.empty() : tensor<96x64xf32>
    %1296 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1297 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1296 : f32) outs(%1295 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1298 = linalg.matmul {prov.region_id = "matmul_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query", prov.transposed_b = "true"} ins(%1292, %1294 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1297 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1299 = tensor.empty() : tensor<96x64xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1298, %51 : tensor<96x64xf32>, tensor<64xf32>) outs(%1299 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb116(%1301: f32, %1302: f32, %1303: f32):
      %1304 = arith.addf %1301, %1302 : f32
      linalg.yield %1304 : f32
    } -> tensor<96x64xf32>
    %1305 = tensor.collapse_shape %1300 [[0 : i64, 1 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1306 = tensor.expand_shape %1305 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1307 = tensor.collapse_shape %1306 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1308 = tensor.expand_shape %1307 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1309 = tensor.empty() : tensor<1x2x96x32xf32>
    %1310 = linalg.transpose ins(%1308:tensor<1x96x2x32xf32>) outs(%1309:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1311 = tensor.empty() : tensor<1x2x32x6xf32>
    %1312 = linalg.transpose ins(%1287:tensor<1x2x6x32xf32>) outs(%1311:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1313 = tensor.empty() : tensor<1x2x96x32xf32>
    %1314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1310 : tensor<1x2x96x32xf32>) outs(%1313 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb117(%1315: f32, %1316: f32):
      linalg.yield %1315 : f32
    } -> tensor<1x2x96x32xf32>
    %1317 = tensor.collapse_shape %1314 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1318 = tensor.expand_shape %1317 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1319 = tensor.empty() : tensor<1x2x32x6xf32>
    %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1312 : tensor<1x2x32x6xf32>) outs(%1319 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb118(%1321: f32, %1322: f32):
      linalg.yield %1321 : f32
    } -> tensor<1x2x32x6xf32>
    %1323 = tensor.collapse_shape %1320 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1324 = tensor.expand_shape %1323 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1325 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1326 = tensor.splat %1325 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1327 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1318, %1324 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1326 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb119(%1328: f32, %1329: f32, %1330: f32):
      %1331 = arith.mulf %1328, %1329 : f32
      %1332 = arith.addf %1330, %1331 : f32
      linalg.yield %1332 : f32
    } -> tensor<2x96x6xf32>
    %1333 = tensor.collapse_shape %1327 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1334 = tensor.expand_shape %1333 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1335 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1336 = tensor.splat %1335 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1337 = tensor.empty() : tensor<1x2x96x6xf32>
    %1338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1334, %1336 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1337 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb120(%1339: f32, %1340: f32, %1341: f32):
      %1342 = arith.divf %1339, %1340 : f32
      linalg.yield %1342 : f32
    } -> tensor<1x2x96x6xf32>
    %1343 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1344 = tensor.splat %1343 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1345 = linalg.reduce ins(%1338:tensor<1x2x96x6xf32>) outs(%1344:tensor<1x2x96xf32>) dimensions = [3]
    (%1346: f32, %1347: f32) {
      %1348 = arith.maximumf %1346, %1347 : f32
      linalg.yield %1348 : f32
    }
    %1349 = tensor.collapse_shape %1345 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1350 = tensor.expand_shape %1349 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1351 = tensor.empty() : tensor<1x2x96x6xf32>
    %1352 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1338, %1350 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1351 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb121(%1353: f32, %1354: f32, %1355: f32):
      %1356 = arith.subf %1353, %1354 : f32
      linalg.yield %1356 : f32
    } -> tensor<1x2x96x6xf32>
    %1357 = tensor.empty() : tensor<1x2x96x6xf32>
    %1358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1352 : tensor<1x2x96x6xf32>) outs(%1357 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb122(%1359: f32, %1360: f32):
      %1361 = math.exp %1359 : f32
      linalg.yield %1361 : f32
    } -> tensor<1x2x96x6xf32>
    %1362 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1363 = tensor.splat %1362 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1364 = linalg.reduce ins(%1358:tensor<1x2x96x6xf32>) outs(%1363:tensor<1x2x96xf32>) dimensions = [3]
    (%1365: f32, %1366: f32) {
      %1367 = arith.addf %1365, %1366 : f32
      linalg.yield %1367 : f32
    }
    %1368 = tensor.collapse_shape %1364 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1369 = tensor.expand_shape %1368 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1370 = tensor.empty() : tensor<1x2x96x6xf32>
    %1371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1358, %1369 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1370 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb123(%1372: f32, %1373: f32, %1374: f32):
      %1375 = arith.divf %1372, %1373 : f32
      linalg.yield %1375 : f32
    } -> tensor<1x2x96x6xf32>
    %1376 = tensor.empty() : tensor<1x2x96x6xf32>
    %1377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1371 : tensor<1x2x96x6xf32>) outs(%1376 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb124(%1378: f32, %1379: f32):
      linalg.yield %1378 : f32
    } -> tensor<1x2x96x6xf32>
    %1380 = tensor.collapse_shape %1377 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %1381 = tensor.expand_shape %1380 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %1382 = tensor.empty() : tensor<1x2x6x32xf32>
    %1383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1290 : tensor<1x2x6x32xf32>) outs(%1382 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb125(%1384: f32, %1385: f32):
      linalg.yield %1384 : f32
    } -> tensor<1x2x6x32xf32>
    %1386 = tensor.collapse_shape %1383 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %1387 = tensor.expand_shape %1386 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %1388 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1389 = tensor.splat %1388 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %1390 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1381, %1387 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%1389 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb126(%1391: f32, %1392: f32, %1393: f32):
      %1394 = arith.mulf %1391, %1392 : f32
      %1395 = arith.addf %1393, %1394 : f32
      linalg.yield %1395 : f32
    } -> tensor<2x96x32xf32>
    %1396 = tensor.collapse_shape %1390 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %1397 = tensor.expand_shape %1396 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %1398 = tensor.empty() : tensor<1x96x2x32xf32>
    %1399 = linalg.transpose ins(%1397:tensor<1x2x96x32xf32>) outs(%1398:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1400 = tensor.collapse_shape %1399 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1401 = tensor.expand_shape %1400 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1402 = tensor.collapse_shape %1401 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1403 = tensor.expand_shape %1402 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1404 = tensor.empty() : tensor<64x64xf32>
    %1405 = linalg.transpose ins(%52:tensor<64x64xf32>) outs(%1404:tensor<64x64xf32>) permutation = [1, 0]
    %1406 = tensor.empty() : tensor<96x64xf32>
    %1407 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1408 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1407 : f32) outs(%1406 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1409 = linalg.matmul {prov.region_id = "matmul_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer", prov.transposed_b = "true"} ins(%1403, %1405 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1408 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1410 = tensor.empty() : tensor<96x64xf32>
    %1411 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1409, %53 : tensor<96x64xf32>, tensor<64xf32>) outs(%1410 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb127(%1412: f32, %1413: f32, %1414: f32):
      %1415 = arith.addf %1412, %1413 : f32
      linalg.yield %1415 : f32
    } -> tensor<96x64xf32>
    %1416 = tensor.collapse_shape %1411 [[0 : i64, 1 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1417 = tensor.expand_shape %1416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1418 = tensor.empty() : tensor<1x96x64xf32>
    %1419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1156, %1417 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1418 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb128(%1420: f32, %1421: f32, %1422: f32):
      %1423 = arith.addf %1420, %1421 : f32
      linalg.yield %1423 : f32
    } -> tensor<1x96x64xf32>
    %1424 = tensor.collapse_shape %1419 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1425 = tensor.expand_shape %1424 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1426 = tensor.empty() : tensor<64x512xf32>
    %1427 = linalg.transpose ins(%64:tensor<512x64xf32>) outs(%1426:tensor<64x512xf32>) permutation = [1, 0]
    %1428 = tensor.empty() : tensor<96x512xf32>
    %1429 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1430 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1429 : f32) outs(%1428 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1431 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1", prov.transposed_b = "true"} ins(%1425, %1427 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1430 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1432 = tensor.empty() : tensor<96x512xf32>
    %1433 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1431, %65 : tensor<96x512xf32>, tensor<512xf32>) outs(%1432 : tensor<96x512xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb129(%1434: f32, %1435: f32, %1436: f32):
      %1437 = arith.addf %1434, %1435 : f32
      linalg.yield %1437 : f32
    } -> tensor<96x512xf32>
    %1438 = tensor.collapse_shape %1433 [[0 : i64, 1 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1439 = tensor.expand_shape %1438 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1440 = tensor.empty() : tensor<1x512x96xf32>
    %1441 = linalg.transpose ins(%1439:tensor<1x96x512xf32>) outs(%1440:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1442 = tensor.collapse_shape %1441 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1443 = tensor.expand_shape %1442 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1444 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1445 = tensor.splat %1444 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1446 = "tensor.insert_slice"(%1443, %1445) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1447 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1446 : tensor<1x512x10x14xf32>) outs(%1447 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb130(%1449: f32, %1450: f32):
      linalg.yield %1449 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1451 = tensor.collapse_shape %1448 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1452 = tensor.expand_shape %1451 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1453 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1454 = tensor.expand_shape %1453 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1455 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1456 = tensor.splat %1455 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1454, %1452 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1456 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb131(%1458: f32, %1459: f32, %1460: f32):
      %1461 = arith.mulf %1458, %1459 : f32
      %1462 = arith.addf %1460, %1461 : f32
      linalg.yield %1462 : f32
    } -> tensor<64x8x96xf32>
    %1463 = tensor.collapse_shape %1457 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1464 = tensor.expand_shape %1463 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1465 = tensor.collapse_shape %1464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1466 = tensor.expand_shape %1465 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1467 = tensor.empty() : tensor<1x512x8x12xf32>
    %1468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1466, %67 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1467 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb132(%1469: f32, %1470: f32, %1471: f32):
      %1472 = arith.addf %1469, %1470 : f32
      linalg.yield %1472 : f32
    } -> tensor<1x512x8x12xf32>
    %1473 = tensor.collapse_shape %1468 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1474 = tensor.expand_shape %1473 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1475 = tensor.empty() : tensor<1x96x512xf32>
    %1476 = linalg.transpose ins(%1474:tensor<1x512x96xf32>) outs(%1475:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1477 = tensor.empty() : tensor<1x96x512xf32>
    %1478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x96x512xf32>) outs(%1477 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb133(%1479: f32, %1480: f32):
      %1481 = arith.constant 5.000000e-01 : f32
      %1482 = arith.constant 1.000000e+00 : f32
      %1483 = arith.constant 0.707106769 : f32
      %1484 = arith.mulf %1479, %1483 : f32
      %1485 = math.erf %1484 : f32
      %1486 = arith.addf %1482, %1485 : f32
      %1487 = arith.mulf %1481, %1479 : f32
      %1488 = arith.mulf %1487, %1486 : f32
      linalg.yield %1488 : f32
    } -> tensor<1x96x512xf32>
    %1489 = tensor.collapse_shape %1478 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1490 = tensor.expand_shape %1489 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1491 = tensor.empty() : tensor<512x64xf32>
    %1492 = linalg.transpose ins(%68:tensor<64x512xf32>) outs(%1491:tensor<512x64xf32>) permutation = [1, 0]
    %1493 = tensor.empty() : tensor<96x64xf32>
    %1494 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1495 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1494 : f32) outs(%1493 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1496 = linalg.matmul {prov.region_id = "matmul_20", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2", prov.transposed_b = "true"} ins(%1490, %1492 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1495 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1497 = tensor.empty() : tensor<96x64xf32>
    %1498 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1496, %69 : tensor<96x64xf32>, tensor<64xf32>) outs(%1497 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb134(%1499: f32, %1500: f32, %1501: f32):
      %1502 = arith.addf %1499, %1500 : f32
      linalg.yield %1502 : f32
    } -> tensor<96x64xf32>
    %1503 = tensor.collapse_shape %1498 [[0 : i64, 1 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1504 = tensor.expand_shape %1503 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1505 = tensor.empty() : tensor<1x96x64xf32>
    %1506 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1419, %1504 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1505 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb135(%1507: f32, %1508: f32, %1509: f32):
      %1510 = arith.addf %1507, %1508 : f32
      linalg.yield %1510 : f32
    } -> tensor<1x96x64xf32>
    %1511 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1512 = tensor.splat %1511 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1513 = linalg.reduce ins(%1506:tensor<1x96x64xf32>) outs(%1512:tensor<1x96xf32>) dimensions = [2]
    (%1514: f32, %1515: f32) {
      %1516 = arith.addf %1514, %1515 : f32
      linalg.yield %1516 : f32
    }
    %1517 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1518 = tensor.splat %1517 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1519 = tensor.empty() : tensor<1x96xf32>
    %1520 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1513, %1518 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1519 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb136(%1521: f32, %1522: f32, %1523: f32):
      %1524 = arith.divf %1521, %1522 : f32
      linalg.yield %1524 : f32
    } -> tensor<1x96xf32>
    %1525 = tensor.collapse_shape %1520 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1526 = tensor.expand_shape %1525 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1527 = tensor.empty() : tensor<1x96x64xf32>
    %1528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1506, %1526 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1527 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb137(%1529: f32, %1530: f32, %1531: f32):
      %1532 = arith.subf %1529, %1530 : f32
      linalg.yield %1532 : f32
    } -> tensor<1x96x64xf32>
    %1533 = tensor.empty() : tensor<1x96x64xf32>
    %1534 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1528, %1528 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1533 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb138(%1535: f32, %1536: f32, %1537: f32):
      %1538 = arith.mulf %1535, %1536 : f32
      linalg.yield %1538 : f32
    } -> tensor<1x96x64xf32>
    %1539 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1540 = tensor.splat %1539 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1541 = linalg.reduce ins(%1534:tensor<1x96x64xf32>) outs(%1540:tensor<1x96xf32>) dimensions = [2]
    (%1542: f32, %1543: f32) {
      %1544 = arith.addf %1542, %1543 : f32
      linalg.yield %1544 : f32
    }
    %1545 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1546 = tensor.splat %1545 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1547 = tensor.empty() : tensor<1x96xf32>
    %1548 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1541, %1546 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1547 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb139(%1549: f32, %1550: f32, %1551: f32):
      %1552 = arith.divf %1549, %1550 : f32
      linalg.yield %1552 : f32
    } -> tensor<1x96xf32>
    %1553 = tensor.collapse_shape %1548 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1554 = tensor.expand_shape %1553 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1555 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1556 = tensor.splat %1555 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1557 = tensor.empty() : tensor<1x96x1xf32>
    %1558 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1554, %1556 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1557 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb140(%1559: f32, %1560: f32, %1561: f32):
      %1562 = arith.addf %1559, %1560 : f32
      linalg.yield %1562 : f32
    } -> tensor<1x96x1xf32>
    %1563 = tensor.empty() : tensor<1x96x1xf32>
    %1564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1558 : tensor<1x96x1xf32>) outs(%1563 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb141(%1565: f32, %1566: f32):
      %1567 = math.rsqrt %1565 : f32
      linalg.yield %1567 : f32
    } -> tensor<1x96x1xf32>
    %1568 = tensor.empty() : tensor<1x96x64xf32>
    %1569 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1528, %1564 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1568 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb142(%1570: f32, %1571: f32, %1572: f32):
      %1573 = arith.mulf %1570, %1571 : f32
      linalg.yield %1573 : f32
    } -> tensor<1x96x64xf32>
    %1574 = tensor.empty() : tensor<1x96x64xf32>
    %1575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1569, %76 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1574 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb143(%1576: f32, %1577: f32, %1578: f32):
      %1579 = arith.mulf %1576, %1577 : f32
      linalg.yield %1579 : f32
    } -> tensor<1x96x64xf32>
    %1580 = tensor.empty() : tensor<1x96x64xf32>
    %1581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1575, %77 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1580 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb144(%1582: f32, %1583: f32, %1584: f32):
      %1585 = arith.addf %1582, %1583 : f32
      linalg.yield %1585 : f32
    } -> tensor<1x96x64xf32>
    %1586 = tensor.empty() : tensor<1x64x96xf32>
    %1587 = linalg.transpose ins(%1581:tensor<1x96x64xf32>) outs(%1586:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1588 = tensor.collapse_shape %1587 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1589 = tensor.expand_shape %1588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1590 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1589 : tensor<1x64x8x12xf32>) outs(%1590 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb145(%1592: f32, %1593: f32):
      linalg.yield %1592 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1594 = tensor.collapse_shape %1591 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1595 = tensor.expand_shape %1594 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1596 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1597 = tensor.expand_shape %1596 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1598 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1599 = tensor.splat %1598 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1600 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1597, %1595 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1599 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1601 = tensor.collapse_shape %1600 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1602 = tensor.expand_shape %1601 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1603 = tensor.collapse_shape %1602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1604 = tensor.expand_shape %1603 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1605 = tensor.empty() : tensor<1x64x2x3xf32>
    %1606 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1604, %55 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1605 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb146(%1607: f32, %1608: f32, %1609: f32):
      %1610 = arith.addf %1607, %1608 : f32
      linalg.yield %1610 : f32
    } -> tensor<1x64x2x3xf32>
    %1611 = tensor.collapse_shape %1606 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1612 = tensor.expand_shape %1611 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1613 = tensor.empty() : tensor<1x6x64xf32>
    %1614 = linalg.transpose ins(%1612:tensor<1x64x6xf32>) outs(%1613:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1615 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1616 = tensor.splat %1615 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1617 = linalg.reduce ins(%1614:tensor<1x6x64xf32>) outs(%1616:tensor<1x6xf32>) dimensions = [2]
    (%1618: f32, %1619: f32) {
      %1620 = arith.addf %1618, %1619 : f32
      linalg.yield %1620 : f32
    }
    %1621 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1622 = tensor.splat %1621 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1623 = tensor.empty() : tensor<1x6xf32>
    %1624 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1617, %1622 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1623 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb147(%1625: f32, %1626: f32, %1627: f32):
      %1628 = arith.divf %1625, %1626 : f32
      linalg.yield %1628 : f32
    } -> tensor<1x6xf32>
    %1629 = tensor.collapse_shape %1624 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1630 = tensor.expand_shape %1629 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1631 = tensor.empty() : tensor<1x6x64xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1614, %1630 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1631 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb148(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.subf %1633, %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<1x6x64xf32>
    %1637 = tensor.empty() : tensor<1x6x64xf32>
    %1638 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1632, %1632 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1637 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb149(%1639: f32, %1640: f32, %1641: f32):
      %1642 = arith.mulf %1639, %1640 : f32
      linalg.yield %1642 : f32
    } -> tensor<1x6x64xf32>
    %1643 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1644 = tensor.splat %1643 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1645 = linalg.reduce ins(%1638:tensor<1x6x64xf32>) outs(%1644:tensor<1x6xf32>) dimensions = [2]
    (%1646: f32, %1647: f32) {
      %1648 = arith.addf %1646, %1647 : f32
      linalg.yield %1648 : f32
    }
    %1649 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1650 = tensor.splat %1649 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1651 = tensor.empty() : tensor<1x6xf32>
    %1652 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1645, %1650 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1651 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb150(%1653: f32, %1654: f32, %1655: f32):
      %1656 = arith.divf %1653, %1654 : f32
      linalg.yield %1656 : f32
    } -> tensor<1x6xf32>
    %1657 = tensor.collapse_shape %1652 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1658 = tensor.expand_shape %1657 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1659 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1660 = tensor.splat %1659 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1661 = tensor.empty() : tensor<1x6x1xf32>
    %1662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1658, %1660 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1661 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb151(%1663: f32, %1664: f32, %1665: f32):
      %1666 = arith.addf %1663, %1664 : f32
      linalg.yield %1666 : f32
    } -> tensor<1x6x1xf32>
    %1667 = tensor.empty() : tensor<1x6x1xf32>
    %1668 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1662 : tensor<1x6x1xf32>) outs(%1667 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb152(%1669: f32, %1670: f32):
      %1671 = math.rsqrt %1669 : f32
      linalg.yield %1671 : f32
    } -> tensor<1x6x1xf32>
    %1672 = tensor.empty() : tensor<1x6x64xf32>
    %1673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1632, %1668 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1672 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb153(%1674: f32, %1675: f32, %1676: f32):
      %1677 = arith.mulf %1674, %1675 : f32
      linalg.yield %1677 : f32
    } -> tensor<1x6x64xf32>
    %1678 = tensor.empty() : tensor<1x6x64xf32>
    %1679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673, %56 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1678 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb154(%1680: f32, %1681: f32, %1682: f32):
      %1683 = arith.mulf %1680, %1681 : f32
      linalg.yield %1683 : f32
    } -> tensor<1x6x64xf32>
    %1684 = tensor.empty() : tensor<1x6x64xf32>
    %1685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1679, %57 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1684 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb155(%1686: f32, %1687: f32, %1688: f32):
      %1689 = arith.addf %1686, %1687 : f32
      linalg.yield %1689 : f32
    } -> tensor<1x6x64xf32>
    %1690 = tensor.collapse_shape %1685 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1691 = tensor.expand_shape %1690 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1692 = tensor.empty() : tensor<64x128xf32>
    %1693 = linalg.transpose ins(%58:tensor<128x64xf32>) outs(%1692:tensor<64x128xf32>) permutation = [1, 0]
    %1694 = tensor.empty() : tensor<6x128xf32>
    %1695 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1696 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1695 : f32) outs(%1694 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1697 = linalg.matmul {prov.region_id = "matmul_21", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor", prov.transposed_b = "true"} ins(%1691, %1693 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1696 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1698 = tensor.empty() : tensor<6x128xf32>
    %1699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1697, %59 : tensor<6x128xf32>, tensor<128xf32>) outs(%1698 : tensor<6x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb156(%1700: f32, %1701: f32, %1702: f32):
      %1703 = arith.addf %1700, %1701 : f32
      linalg.yield %1703 : f32
    } -> tensor<6x128xf32>
    %1704 = tensor.collapse_shape %1699 [[0 : i64, 1 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1706 = tensor.collapse_shape %1705 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1707 = tensor.expand_shape %1706 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1708 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1709 = linalg.transpose ins(%1707:tensor<1x6x2x2x32xf32>) outs(%1708:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1710 = "tensor.extract_slice"(%1709) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1711 = tensor.collapse_shape %1710 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1712 = tensor.expand_shape %1711 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1713 = "tensor.extract_slice"(%1709) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1714 = tensor.collapse_shape %1713 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1715 = tensor.expand_shape %1714 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1716 = tensor.collapse_shape %1581 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1717 = tensor.expand_shape %1716 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1718 = tensor.empty() : tensor<64x64xf32>
    %1719 = linalg.transpose ins(%60:tensor<64x64xf32>) outs(%1718:tensor<64x64xf32>) permutation = [1, 0]
    %1720 = tensor.empty() : tensor<96x64xf32>
    %1721 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1722 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1721 : f32) outs(%1720 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1723 = linalg.matmul {prov.region_id = "matmul_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query", prov.transposed_b = "true"} ins(%1717, %1719 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1722 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1724 = tensor.empty() : tensor<96x64xf32>
    %1725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1723, %61 : tensor<96x64xf32>, tensor<64xf32>) outs(%1724 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb157(%1726: f32, %1727: f32, %1728: f32):
      %1729 = arith.addf %1726, %1727 : f32
      linalg.yield %1729 : f32
    } -> tensor<96x64xf32>
    %1730 = tensor.collapse_shape %1725 [[0 : i64, 1 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1731 = tensor.expand_shape %1730 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1732 = tensor.collapse_shape %1731 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1733 = tensor.expand_shape %1732 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1734 = tensor.empty() : tensor<1x2x96x32xf32>
    %1735 = linalg.transpose ins(%1733:tensor<1x96x2x32xf32>) outs(%1734:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1736 = tensor.empty() : tensor<1x2x32x6xf32>
    %1737 = linalg.transpose ins(%1712:tensor<1x2x6x32xf32>) outs(%1736:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1738 = tensor.empty() : tensor<1x2x96x32xf32>
    %1739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1735 : tensor<1x2x96x32xf32>) outs(%1738 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb158(%1740: f32, %1741: f32):
      linalg.yield %1740 : f32
    } -> tensor<1x2x96x32xf32>
    %1742 = tensor.collapse_shape %1739 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32> into tensor<6144xf32>
    %1743 = tensor.expand_shape %1742 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 32] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<2x96x32xf32>
    %1744 = tensor.empty() : tensor<1x2x32x6xf32>
    %1745 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1737 : tensor<1x2x32x6xf32>) outs(%1744 : tensor<1x2x32x6xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb159(%1746: f32, %1747: f32):
      linalg.yield %1746 : f32
    } -> tensor<1x2x32x6xf32>
    %1748 = tensor.collapse_shape %1745 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x32x6xf32> into tensor<384xf32>
    %1749 = tensor.expand_shape %1748 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 6] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x32x6xf32>
    %1750 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1751 = tensor.splat %1750 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32>
    %1752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1743, %1749 : tensor<2x96x32xf32>, tensor<2x32x6xf32>) outs(%1751 : tensor<2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%1753: f32, %1754: f32, %1755: f32):
      %1756 = arith.mulf %1753, %1754 : f32
      %1757 = arith.addf %1755, %1756 : f32
      linalg.yield %1757 : f32
    } -> tensor<2x96x6xf32>
    %1758 = tensor.collapse_shape %1752 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x6xf32> into tensor<1152xf32>
    %1759 = tensor.expand_shape %1758 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 6] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<1x2x96x6xf32>
    %1760 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1761 = tensor.splat %1760 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1762 = tensor.empty() : tensor<1x2x96x6xf32>
    %1763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1759, %1761 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1762 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%1764: f32, %1765: f32, %1766: f32):
      %1767 = arith.divf %1764, %1765 : f32
      linalg.yield %1767 : f32
    } -> tensor<1x2x96x6xf32>
    %1768 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1769 = tensor.splat %1768 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1770 = linalg.reduce ins(%1763:tensor<1x2x96x6xf32>) outs(%1769:tensor<1x2x96xf32>) dimensions = [3]
    (%1771: f32, %1772: f32) {
      %1773 = arith.maximumf %1771, %1772 : f32
      linalg.yield %1773 : f32
    }
    %1774 = tensor.collapse_shape %1770 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1775 = tensor.expand_shape %1774 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1776 = tensor.empty() : tensor<1x2x96x6xf32>
    %1777 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1763, %1775 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1776 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb162(%1778: f32, %1779: f32, %1780: f32):
      %1781 = arith.subf %1778, %1779 : f32
      linalg.yield %1781 : f32
    } -> tensor<1x2x96x6xf32>
    %1782 = tensor.empty() : tensor<1x2x96x6xf32>
    %1783 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1777 : tensor<1x2x96x6xf32>) outs(%1782 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb163(%1784: f32, %1785: f32):
      %1786 = math.exp %1784 : f32
      linalg.yield %1786 : f32
    } -> tensor<1x2x96x6xf32>
    %1787 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1788 = tensor.splat %1787 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1789 = linalg.reduce ins(%1783:tensor<1x2x96x6xf32>) outs(%1788:tensor<1x2x96xf32>) dimensions = [3]
    (%1790: f32, %1791: f32) {
      %1792 = arith.addf %1790, %1791 : f32
      linalg.yield %1792 : f32
    }
    %1793 = tensor.collapse_shape %1789 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1794 = tensor.expand_shape %1793 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1795 = tensor.empty() : tensor<1x2x96x6xf32>
    %1796 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1783, %1794 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1795 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb164(%1797: f32, %1798: f32, %1799: f32):
      %1800 = arith.divf %1797, %1798 : f32
      linalg.yield %1800 : f32
    } -> tensor<1x2x96x6xf32>
    %1801 = tensor.empty() : tensor<1x2x96x6xf32>
    %1802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1796 : tensor<1x2x96x6xf32>) outs(%1801 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb165(%1803: f32, %1804: f32):
      linalg.yield %1803 : f32
    } -> tensor<1x2x96x6xf32>
    %1805 = tensor.collapse_shape %1802 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32> into tensor<1152xf32>
    %1806 = tensor.expand_shape %1805 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 96, 6] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1152xf32> into tensor<2x96x6xf32>
    %1807 = tensor.empty() : tensor<1x2x6x32xf32>
    %1808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1715 : tensor<1x2x6x32xf32>) outs(%1807 : tensor<1x2x6x32xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb166(%1809: f32, %1810: f32):
      linalg.yield %1809 : f32
    } -> tensor<1x2x6x32xf32>
    %1811 = tensor.collapse_shape %1808 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x6x32xf32> into tensor<384xf32>
    %1812 = tensor.expand_shape %1811 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 6, 32] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<2x6x32xf32>
    %1813 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1814 = tensor.splat %1813 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32>
    %1815 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1806, %1812 : tensor<2x96x6xf32>, tensor<2x6x32xf32>) outs(%1814 : tensor<2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb167(%1816: f32, %1817: f32, %1818: f32):
      %1819 = arith.mulf %1816, %1817 : f32
      %1820 = arith.addf %1818, %1819 : f32
      linalg.yield %1820 : f32
    } -> tensor<2x96x32xf32>
    %1821 = tensor.collapse_shape %1815 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<2x96x32xf32> into tensor<6144xf32>
    %1822 = tensor.expand_shape %1821 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 32] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x2x96x32xf32>
    %1823 = tensor.empty() : tensor<1x96x2x32xf32>
    %1824 = linalg.transpose ins(%1822:tensor<1x2x96x32xf32>) outs(%1823:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1825 = tensor.collapse_shape %1824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1826 = tensor.expand_shape %1825 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1827 = tensor.collapse_shape %1826 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1828 = tensor.expand_shape %1827 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1829 = tensor.empty() : tensor<64x64xf32>
    %1830 = linalg.transpose ins(%62:tensor<64x64xf32>) outs(%1829:tensor<64x64xf32>) permutation = [1, 0]
    %1831 = tensor.empty() : tensor<96x64xf32>
    %1832 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1833 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1832 : f32) outs(%1831 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1834 = linalg.matmul {prov.region_id = "matmul_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer", prov.transposed_b = "true"} ins(%1828, %1830 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1833 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1835 = tensor.empty() : tensor<96x64xf32>
    %1836 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1834, %63 : tensor<96x64xf32>, tensor<64xf32>) outs(%1835 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb168(%1837: f32, %1838: f32, %1839: f32):
      %1840 = arith.addf %1837, %1838 : f32
      linalg.yield %1840 : f32
    } -> tensor<96x64xf32>
    %1841 = tensor.collapse_shape %1836 [[0 : i64, 1 : i64]] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1842 = tensor.expand_shape %1841 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1843 = tensor.empty() : tensor<1x96x64xf32>
    %1844 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1581, %1842 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1843 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb169(%1845: f32, %1846: f32, %1847: f32):
      %1848 = arith.addf %1845, %1846 : f32
      linalg.yield %1848 : f32
    } -> tensor<1x96x64xf32>
    %1849 = tensor.collapse_shape %1844 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1850 = tensor.expand_shape %1849 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1851 = tensor.empty() : tensor<64x512xf32>
    %1852 = linalg.transpose ins(%70:tensor<512x64xf32>) outs(%1851:tensor<64x512xf32>) permutation = [1, 0]
    %1853 = tensor.empty() : tensor<96x512xf32>
    %1854 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1855 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1854 : f32) outs(%1853 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1856 = linalg.matmul {prov.region_id = "matmul_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1", prov.transposed_b = "true"} ins(%1850, %1852 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1855 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1857 = tensor.empty() : tensor<96x512xf32>
    %1858 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1856, %71 : tensor<96x512xf32>, tensor<512xf32>) outs(%1857 : tensor<96x512xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb170(%1859: f32, %1860: f32, %1861: f32):
      %1862 = arith.addf %1859, %1860 : f32
      linalg.yield %1862 : f32
    } -> tensor<96x512xf32>
    %1863 = tensor.collapse_shape %1858 [[0 : i64, 1 : i64]] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1864 = tensor.expand_shape %1863 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1865 = tensor.empty() : tensor<1x512x96xf32>
    %1866 = linalg.transpose ins(%1864:tensor<1x96x512xf32>) outs(%1865:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1867 = tensor.collapse_shape %1866 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1868 = tensor.expand_shape %1867 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1869 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1870 = tensor.splat %1869 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %1871 = "tensor.insert_slice"(%1868, %1870) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1872 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1871 : tensor<1x512x10x14xf32>) outs(%1872 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb171(%1874: f32, %1875: f32):
      linalg.yield %1874 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1876 = tensor.collapse_shape %1873 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1877 = tensor.expand_shape %1876 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1878 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1879 = tensor.expand_shape %1878 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1880 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1881 = tensor.splat %1880 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %1882 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1879, %1877 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1881 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb172(%1883: f32, %1884: f32, %1885: f32):
      %1886 = arith.mulf %1883, %1884 : f32
      %1887 = arith.addf %1885, %1886 : f32
      linalg.yield %1887 : f32
    } -> tensor<64x8x96xf32>
    %1888 = tensor.collapse_shape %1882 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1889 = tensor.expand_shape %1888 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1890 = tensor.collapse_shape %1889 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1891 = tensor.expand_shape %1890 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1892 = tensor.empty() : tensor<1x512x8x12xf32>
    %1893 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1891, %73 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1892 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb173(%1894: f32, %1895: f32, %1896: f32):
      %1897 = arith.addf %1894, %1895 : f32
      linalg.yield %1897 : f32
    } -> tensor<1x512x8x12xf32>
    %1898 = tensor.collapse_shape %1893 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1899 = tensor.expand_shape %1898 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1900 = tensor.empty() : tensor<1x96x512xf32>
    %1901 = linalg.transpose ins(%1899:tensor<1x512x96xf32>) outs(%1900:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1902 = tensor.empty() : tensor<1x96x512xf32>
    %1903 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1901 : tensor<1x96x512xf32>) outs(%1902 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb174(%1904: f32, %1905: f32):
      %1906 = arith.constant 5.000000e-01 : f32
      %1907 = arith.constant 1.000000e+00 : f32
      %1908 = arith.constant 0.707106769 : f32
      %1909 = arith.mulf %1904, %1908 : f32
      %1910 = math.erf %1909 : f32
      %1911 = arith.addf %1907, %1910 : f32
      %1912 = arith.mulf %1906, %1904 : f32
      %1913 = arith.mulf %1912, %1911 : f32
      linalg.yield %1913 : f32
    } -> tensor<1x96x512xf32>
    %1914 = tensor.collapse_shape %1903 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1915 = tensor.expand_shape %1914 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1916 = tensor.empty() : tensor<512x64xf32>
    %1917 = linalg.transpose ins(%74:tensor<64x512xf32>) outs(%1916:tensor<512x64xf32>) permutation = [1, 0]
    %1918 = tensor.empty() : tensor<96x64xf32>
    %1919 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1920 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1919 : f32) outs(%1918 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1921 = linalg.matmul {prov.region_id = "matmul_27", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2", prov.transposed_b = "true"} ins(%1915, %1917 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1920 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1922 = tensor.empty() : tensor<96x64xf32>
    %1923 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1921, %75 : tensor<96x64xf32>, tensor<64xf32>) outs(%1922 : tensor<96x64xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb175(%1924: f32, %1925: f32, %1926: f32):
      %1927 = arith.addf %1924, %1925 : f32
      linalg.yield %1927 : f32
    } -> tensor<96x64xf32>
    %1928 = tensor.collapse_shape %1923 [[0 : i64, 1 : i64]] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1929 = tensor.expand_shape %1928 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1930 = tensor.empty() : tensor<1x96x64xf32>
    %1931 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1844, %1929 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1930 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb176(%1932: f32, %1933: f32, %1934: f32):
      %1935 = arith.addf %1932, %1933 : f32
      linalg.yield %1935 : f32
    } -> tensor<1x96x64xf32>
    %1936 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1937 = tensor.splat %1936 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1938 = linalg.reduce ins(%1931:tensor<1x96x64xf32>) outs(%1937:tensor<1x96xf32>) dimensions = [2]
    (%1939: f32, %1940: f32) {
      %1941 = arith.addf %1939, %1940 : f32
      linalg.yield %1941 : f32
    }
    %1942 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1943 = tensor.splat %1942 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1944 = tensor.empty() : tensor<1x96xf32>
    %1945 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1938, %1943 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1944 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb177(%1946: f32, %1947: f32, %1948: f32):
      %1949 = arith.divf %1946, %1947 : f32
      linalg.yield %1949 : f32
    } -> tensor<1x96xf32>
    %1950 = tensor.collapse_shape %1945 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1951 = tensor.expand_shape %1950 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1952 = tensor.empty() : tensor<1x96x64xf32>
    %1953 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1931, %1951 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1952 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb178(%1954: f32, %1955: f32, %1956: f32):
      %1957 = arith.subf %1954, %1955 : f32
      linalg.yield %1957 : f32
    } -> tensor<1x96x64xf32>
    %1958 = tensor.empty() : tensor<1x96x64xf32>
    %1959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1953, %1953 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1958 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb179(%1960: f32, %1961: f32, %1962: f32):
      %1963 = arith.mulf %1960, %1961 : f32
      linalg.yield %1963 : f32
    } -> tensor<1x96x64xf32>
    %1964 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1965 = tensor.splat %1964 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1966 = linalg.reduce ins(%1959:tensor<1x96x64xf32>) outs(%1965:tensor<1x96xf32>) dimensions = [2]
    (%1967: f32, %1968: f32) {
      %1969 = arith.addf %1967, %1968 : f32
      linalg.yield %1969 : f32
    }
    %1970 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1971 = tensor.splat %1970 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1972 = tensor.empty() : tensor<1x96xf32>
    %1973 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1966, %1971 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1972 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb180(%1974: f32, %1975: f32, %1976: f32):
      %1977 = arith.divf %1974, %1975 : f32
      linalg.yield %1977 : f32
    } -> tensor<1x96xf32>
    %1978 = tensor.collapse_shape %1973 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1979 = tensor.expand_shape %1978 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1980 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1981 = tensor.splat %1980 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1982 = tensor.empty() : tensor<1x96x1xf32>
    %1983 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1979, %1981 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1982 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb181(%1984: f32, %1985: f32, %1986: f32):
      %1987 = arith.addf %1984, %1985 : f32
      linalg.yield %1987 : f32
    } -> tensor<1x96x1xf32>
    %1988 = tensor.empty() : tensor<1x96x1xf32>
    %1989 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1983 : tensor<1x96x1xf32>) outs(%1988 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb182(%1990: f32, %1991: f32):
      %1992 = math.rsqrt %1990 : f32
      linalg.yield %1992 : f32
    } -> tensor<1x96x1xf32>
    %1993 = tensor.empty() : tensor<1x96x64xf32>
    %1994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1953, %1989 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1993 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb183(%1995: f32, %1996: f32, %1997: f32):
      %1998 = arith.mulf %1995, %1996 : f32
      linalg.yield %1998 : f32
    } -> tensor<1x96x64xf32>
    %1999 = tensor.empty() : tensor<1x96x64xf32>
    %2000 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1994, %78 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1999 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb184(%2001: f32, %2002: f32, %2003: f32):
      %2004 = arith.mulf %2001, %2002 : f32
      linalg.yield %2004 : f32
    } -> tensor<1x96x64xf32>
    %2005 = tensor.empty() : tensor<1x96x64xf32>
    %2006 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2000, %79 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2005 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb185(%2007: f32, %2008: f32, %2009: f32):
      %2010 = arith.addf %2007, %2008 : f32
      linalg.yield %2010 : f32
    } -> tensor<1x96x64xf32>
    %2011 = tensor.collapse_shape %2006 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2012 = tensor.expand_shape %2011 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2013 = tensor.empty() : tensor<1x64x8x12xf32>
    %2014 = linalg.transpose ins(%2012:tensor<1x8x12x64xf32>) outs(%2013:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2015 = tensor.collapse_shape %2014 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2016 = tensor.expand_shape %2015 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2017 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2018 = linalg.transpose ins(%2016:tensor<1x16x2x2x8x12xf32>) outs(%2017:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2019 = tensor.collapse_shape %2018 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2020 = tensor.expand_shape %2019 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2021 = tensor.empty() : tensor<1x32x23x15xf32>
    %2022 = linalg.transpose ins(%1057:tensor<1x32x15x23xf32>) outs(%2021:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2023 = tensor.collapse_shape %2022 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2024 = tensor.expand_shape %2023 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2025 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2026 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2027 = tensor.splat %2026 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2028 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2024, %2025 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2027 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2029 = tensor.collapse_shape %2028 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2030 = tensor.expand_shape %2029 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2031 = tensor.empty() : tensor<1x32x16x23xf32>
    %2032 = linalg.transpose ins(%2030:tensor<1x32x23x16xf32>) outs(%2031:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2033 = tensor.collapse_shape %2032 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2034 = tensor.expand_shape %2033 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2035 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2036 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2037 = tensor.splat %2036 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2038 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2034, %2035 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2037 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2039 = tensor.collapse_shape %2038 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2040 = tensor.expand_shape %2039 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2041 = tensor.concat dim(1) %2020, %2040 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2042 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2043 = tensor.splat %2042 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2044 = "tensor.insert_slice"(%2041, %2043) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2045 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2046 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2044 : tensor<1x48x18x26xf32>) outs(%2045 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb186(%2047: f32, %2048: f32):
      linalg.yield %2047 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2049 = tensor.collapse_shape %2046 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2050 = tensor.expand_shape %2049 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2051 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2052 = tensor.expand_shape %2051 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2053 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2054 = tensor.splat %2053 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2055 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2052, %2050 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2054 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2056 = tensor.collapse_shape %2055 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2057 = tensor.expand_shape %2056 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2058 = tensor.collapse_shape %2057 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2059 = tensor.expand_shape %2058 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2060 = tensor.empty() : tensor<1x12x16x24xf32>
    %2061 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2059, %97 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2060 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb187(%2062: f32, %2063: f32, %2064: f32):
      %2065 = arith.addf %2062, %2063 : f32
      linalg.yield %2065 : f32
    } -> tensor<1x12x16x24xf32>
    %2066 = tensor.collapse_shape %2061 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2067 = tensor.expand_shape %2066 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2068 = tensor.empty() : tensor<4608x512xf32>
    %2069 = linalg.transpose ins(%81:tensor<512x4608xf32>) outs(%2068:tensor<4608x512xf32>) permutation = [1, 0]
    %2070 = tensor.empty() : tensor<1x512xf32>
    %2071 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2072 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2071 : f32) outs(%2070 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2073 = linalg.matmul {prov.region_id = "matmul_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder", prov.transposed_b = "true"} ins(%2067, %2069 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2072 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2074 = tensor.empty() : tensor<1x512xf32>
    %2075 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2073, %80 : tensor<1x512xf32>, tensor<512xf32>) outs(%2074 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb188(%2076: f32, %2077: f32, %2078: f32):
      %2079 = arith.addf %2076, %2077 : f32
      linalg.yield %2079 : f32
    } -> tensor<1x512xf32>
    %2080 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2081 = tensor.splat %2080 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2082 = tensor.empty() : tensor<1x1xf32>
    %2083 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %2081 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2082 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb189(%2084: f32, %2085: f32, %2086: f32):
      %2087 = arith.divf %2084, %2085 : f32
      linalg.yield %2087 : f32
    } -> tensor<1x1xf32>
    %2088 = tensor.concat dim(1) %2075, %2083, %100 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2089 = tensor.collapse_shape %2088 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2090 = tensor.expand_shape %2089 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2091 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2092 = tensor.splat %2091 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2093 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} 0.000000e+00 : f32
    %2094 = tensor.splat %2093 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32>
    %2095 = "tensor.extract_slice"(%2092) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2096 = "tensor.extract_slice"(%2092) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2097 = "tensor.extract_slice"(%2092) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2098 = tensor.collapse_shape %2095 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2099 = tensor.expand_shape %2098 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2100 = tensor.collapse_shape %2096 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2101 = tensor.expand_shape %2100 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2102 = tensor.collapse_shape %2097 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2103 = tensor.expand_shape %2102 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2104 = "tensor.extract_slice"(%2094) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2105 = "tensor.extract_slice"(%2094) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2106 = "tensor.extract_slice"(%2094) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2107 = tensor.collapse_shape %2104 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2108 = tensor.expand_shape %2107 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2109 = tensor.collapse_shape %2105 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2110 = tensor.expand_shape %2109 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2111 = tensor.collapse_shape %2106 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2112 = tensor.expand_shape %2111 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2113 = tensor.collapse_shape %2099 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2114 = tensor.expand_shape %2113 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2115 = tensor.collapse_shape %2108 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2116 = tensor.expand_shape %2115 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2117 = tensor.collapse_shape %2090 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2118 = tensor.expand_shape %2117 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2119 = tensor.empty() : tensor<517x512xf32>
    %2120 = linalg.transpose ins(%82:tensor<512x517xf32>) outs(%2119:tensor<517x512xf32>) permutation = [1, 0]
    %2121 = tensor.empty() : tensor<1x512xf32>
    %2122 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2123 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2122 : f32) outs(%2121 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2124 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2118, %2120 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2123 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2125 = tensor.empty() : tensor<1x512xf32>
    %2126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2124, %84 : tensor<1x512xf32>, tensor<512xf32>) outs(%2125 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb190(%2127: f32, %2128: f32, %2129: f32):
      %2130 = arith.addf %2127, %2128 : f32
      linalg.yield %2130 : f32
    } -> tensor<1x512xf32>
    %2131 = tensor.collapse_shape %2126 [[0 : i64, 1 : i64]] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2132 = tensor.expand_shape %2131 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2133 = "tensor.extract_slice"(%2132) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2134 = tensor.collapse_shape %2133 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2135 = tensor.expand_shape %2134 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2136 = tensor.collapse_shape %2114 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2137 = tensor.expand_shape %2136 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2138 = tensor.empty() : tensor<128x512xf32>
    %2139 = linalg.transpose ins(%83:tensor<512x128xf32>) outs(%2138:tensor<128x512xf32>) permutation = [1, 0]
    %2140 = tensor.empty() : tensor<1x512xf32>
    %2141 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2142 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2141 : f32) outs(%2140 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2143 = linalg.matmul {prov.region_id = "matmul_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2137, %2139 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2142 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2144 = tensor.empty() : tensor<1x512xf32>
    %2145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2143, %85 : tensor<1x512xf32>, tensor<512xf32>) outs(%2144 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb191(%2146: f32, %2147: f32, %2148: f32):
      %2149 = arith.addf %2146, %2147 : f32
      linalg.yield %2149 : f32
    } -> tensor<1x512xf32>
    %2150 = tensor.collapse_shape %2145 [[0 : i64, 1 : i64]] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2151 = tensor.expand_shape %2150 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2152 = tensor.empty() : tensor<1x1x512xf32>
    %2153 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2151, %2135 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2152 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb192(%2154: f32, %2155: f32, %2156: f32):
      %2157 = arith.addf %2154, %2155 : f32
      linalg.yield %2157 : f32
    } -> tensor<1x1x512xf32>
    %2158 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2159 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2160 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2161 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2162 = tensor.empty() : tensor<1x1x128xf32>
    %2163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2158 : tensor<1x1x128xf32>) outs(%2162 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb193(%2164: f32, %2165: f32):
      %2166 = arith.constant 1.000000e+00 : f32
      %2167 = arith.negf %2164 : f32
      %2168 = math.exp %2167 : f32
      %2169 = arith.addf %2166, %2168 : f32
      %2170 = arith.divf %2166, %2169 : f32
      linalg.yield %2170 : f32
    } -> tensor<1x1x128xf32>
    %2171 = tensor.empty() : tensor<1x1x128xf32>
    %2172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2159 : tensor<1x1x128xf32>) outs(%2171 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb194(%2173: f32, %2174: f32):
      %2175 = arith.constant 1.000000e+00 : f32
      %2176 = arith.negf %2173 : f32
      %2177 = math.exp %2176 : f32
      %2178 = arith.addf %2175, %2177 : f32
      %2179 = arith.divf %2175, %2178 : f32
      linalg.yield %2179 : f32
    } -> tensor<1x1x128xf32>
    %2180 = tensor.empty() : tensor<1x1x128xf32>
    %2181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2160 : tensor<1x1x128xf32>) outs(%2180 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb195(%2182: f32, %2183: f32):
      %2184 = math.tanh %2182 : f32
      linalg.yield %2184 : f32
    } -> tensor<1x1x128xf32>
    %2185 = tensor.empty() : tensor<1x1x128xf32>
    %2186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2161 : tensor<1x1x128xf32>) outs(%2185 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb196(%2187: f32, %2188: f32):
      %2189 = arith.constant 1.000000e+00 : f32
      %2190 = arith.negf %2187 : f32
      %2191 = math.exp %2190 : f32
      %2192 = arith.addf %2189, %2191 : f32
      %2193 = arith.divf %2189, %2192 : f32
      linalg.yield %2193 : f32
    } -> tensor<1x1x128xf32>
    %2194 = tensor.empty() : tensor<1x1x128xf32>
    %2195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2172, %2116 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2194 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb197(%2196: f32, %2197: f32, %2198: f32):
      %2199 = arith.mulf %2196, %2197 : f32
      linalg.yield %2199 : f32
    } -> tensor<1x1x128xf32>
    %2200 = tensor.empty() : tensor<1x1x128xf32>
    %2201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2163, %2181 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2200 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb198(%2202: f32, %2203: f32, %2204: f32):
      %2205 = arith.mulf %2202, %2203 : f32
      linalg.yield %2205 : f32
    } -> tensor<1x1x128xf32>
    %2206 = tensor.empty() : tensor<1x1x128xf32>
    %2207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2195, %2201 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2206 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb199(%2208: f32, %2209: f32, %2210: f32):
      %2211 = arith.addf %2208, %2209 : f32
      linalg.yield %2211 : f32
    } -> tensor<1x1x128xf32>
    %2212 = tensor.empty() : tensor<1x1x128xf32>
    %2213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2207 : tensor<1x1x128xf32>) outs(%2212 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_1", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb200(%2214: f32, %2215: f32):
      %2216 = math.tanh %2214 : f32
      linalg.yield %2216 : f32
    } -> tensor<1x1x128xf32>
    %2217 = tensor.empty() : tensor<1x1x128xf32>
    %2218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2186, %2213 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2217 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb201(%2219: f32, %2220: f32, %2221: f32):
      %2222 = arith.mulf %2219, %2220 : f32
      linalg.yield %2222 : f32
    } -> tensor<1x1x128xf32>
    %2223 = tensor.concat dim(0) %2218 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2224 = tensor.collapse_shape %2101 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2225 = tensor.expand_shape %2224 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2226 = tensor.collapse_shape %2110 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2227 = tensor.expand_shape %2226 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2228 = tensor.collapse_shape %2223 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2229 = tensor.expand_shape %2228 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2230 = tensor.empty() : tensor<128x512xf32>
    %2231 = linalg.transpose ins(%86:tensor<512x128xf32>) outs(%2230:tensor<128x512xf32>) permutation = [1, 0]
    %2232 = tensor.empty() : tensor<1x512xf32>
    %2233 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2234 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2233 : f32) outs(%2232 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2235 = linalg.matmul {prov.region_id = "matmul_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2229, %2231 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2234 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2236 = tensor.empty() : tensor<1x512xf32>
    %2237 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2235, %88 : tensor<1x512xf32>, tensor<512xf32>) outs(%2236 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb202(%2238: f32, %2239: f32, %2240: f32):
      %2241 = arith.addf %2238, %2239 : f32
      linalg.yield %2241 : f32
    } -> tensor<1x512xf32>
    %2242 = tensor.collapse_shape %2237 [[0 : i64, 1 : i64]] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2243 = tensor.expand_shape %2242 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2244 = "tensor.extract_slice"(%2243) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2245 = tensor.collapse_shape %2244 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2246 = tensor.expand_shape %2245 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2247 = tensor.collapse_shape %2225 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2248 = tensor.expand_shape %2247 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2249 = tensor.empty() : tensor<128x512xf32>
    %2250 = linalg.transpose ins(%87:tensor<512x128xf32>) outs(%2249:tensor<128x512xf32>) permutation = [1, 0]
    %2251 = tensor.empty() : tensor<1x512xf32>
    %2252 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2253 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2252 : f32) outs(%2251 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2254 = linalg.matmul {prov.region_id = "matmul_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2248, %2250 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2253 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2255 = tensor.empty() : tensor<1x512xf32>
    %2256 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2254, %89 : tensor<1x512xf32>, tensor<512xf32>) outs(%2255 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb203(%2257: f32, %2258: f32, %2259: f32):
      %2260 = arith.addf %2257, %2258 : f32
      linalg.yield %2260 : f32
    } -> tensor<1x512xf32>
    %2261 = tensor.collapse_shape %2256 [[0 : i64, 1 : i64]] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2262 = tensor.expand_shape %2261 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2263 = tensor.empty() : tensor<1x1x512xf32>
    %2264 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2262, %2246 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2263 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb204(%2265: f32, %2266: f32, %2267: f32):
      %2268 = arith.addf %2265, %2266 : f32
      linalg.yield %2268 : f32
    } -> tensor<1x1x512xf32>
    %2269 = "tensor.extract_slice"(%2264) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2270 = "tensor.extract_slice"(%2264) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2271 = "tensor.extract_slice"(%2264) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2272 = "tensor.extract_slice"(%2264) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2273 = tensor.empty() : tensor<1x1x128xf32>
    %2274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2269 : tensor<1x1x128xf32>) outs(%2273 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb205(%2275: f32, %2276: f32):
      %2277 = arith.constant 1.000000e+00 : f32
      %2278 = arith.negf %2275 : f32
      %2279 = math.exp %2278 : f32
      %2280 = arith.addf %2277, %2279 : f32
      %2281 = arith.divf %2277, %2280 : f32
      linalg.yield %2281 : f32
    } -> tensor<1x1x128xf32>
    %2282 = tensor.empty() : tensor<1x1x128xf32>
    %2283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2270 : tensor<1x1x128xf32>) outs(%2282 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_4", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb206(%2284: f32, %2285: f32):
      %2286 = arith.constant 1.000000e+00 : f32
      %2287 = arith.negf %2284 : f32
      %2288 = math.exp %2287 : f32
      %2289 = arith.addf %2286, %2288 : f32
      %2290 = arith.divf %2286, %2289 : f32
      linalg.yield %2290 : f32
    } -> tensor<1x1x128xf32>
    %2291 = tensor.empty() : tensor<1x1x128xf32>
    %2292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2271 : tensor<1x1x128xf32>) outs(%2291 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_2", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb207(%2293: f32, %2294: f32):
      %2295 = math.tanh %2293 : f32
      linalg.yield %2295 : f32
    } -> tensor<1x1x128xf32>
    %2296 = tensor.empty() : tensor<1x1x128xf32>
    %2297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2272 : tensor<1x1x128xf32>) outs(%2296 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_5", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb208(%2298: f32, %2299: f32):
      %2300 = arith.constant 1.000000e+00 : f32
      %2301 = arith.negf %2298 : f32
      %2302 = math.exp %2301 : f32
      %2303 = arith.addf %2300, %2302 : f32
      %2304 = arith.divf %2300, %2303 : f32
      linalg.yield %2304 : f32
    } -> tensor<1x1x128xf32>
    %2305 = tensor.empty() : tensor<1x1x128xf32>
    %2306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2283, %2227 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2305 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb209(%2307: f32, %2308: f32, %2309: f32):
      %2310 = arith.mulf %2307, %2308 : f32
      linalg.yield %2310 : f32
    } -> tensor<1x1x128xf32>
    %2311 = tensor.empty() : tensor<1x1x128xf32>
    %2312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2274, %2292 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2311 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb210(%2313: f32, %2314: f32, %2315: f32):
      %2316 = arith.mulf %2313, %2314 : f32
      linalg.yield %2316 : f32
    } -> tensor<1x1x128xf32>
    %2317 = tensor.empty() : tensor<1x1x128xf32>
    %2318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2306, %2312 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2317 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb211(%2319: f32, %2320: f32, %2321: f32):
      %2322 = arith.addf %2319, %2320 : f32
      linalg.yield %2322 : f32
    } -> tensor<1x1x128xf32>
    %2323 = tensor.empty() : tensor<1x1x128xf32>
    %2324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2318 : tensor<1x1x128xf32>) outs(%2323 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_3", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb212(%2325: f32, %2326: f32):
      %2327 = math.tanh %2325 : f32
      linalg.yield %2327 : f32
    } -> tensor<1x1x128xf32>
    %2328 = tensor.empty() : tensor<1x1x128xf32>
    %2329 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2297, %2324 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2328 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb213(%2330: f32, %2331: f32, %2332: f32):
      %2333 = arith.mulf %2330, %2331 : f32
      linalg.yield %2333 : f32
    } -> tensor<1x1x128xf32>
    %2334 = tensor.concat dim(0) %2329 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2335 = tensor.collapse_shape %2103 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2336 = tensor.expand_shape %2335 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2337 = tensor.collapse_shape %2112 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2338 = tensor.expand_shape %2337 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2339 = tensor.collapse_shape %2334 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2340 = tensor.expand_shape %2339 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2341 = tensor.empty() : tensor<128x512xf32>
    %2342 = linalg.transpose ins(%90:tensor<512x128xf32>) outs(%2341:tensor<128x512xf32>) permutation = [1, 0]
    %2343 = tensor.empty() : tensor<1x512xf32>
    %2344 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2345 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2344 : f32) outs(%2343 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2346 = linalg.matmul {prov.region_id = "matmul_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2340, %2342 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2345 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2347 = tensor.empty() : tensor<1x512xf32>
    %2348 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2346, %92 : tensor<1x512xf32>, tensor<512xf32>) outs(%2347 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb214(%2349: f32, %2350: f32, %2351: f32):
      %2352 = arith.addf %2349, %2350 : f32
      linalg.yield %2352 : f32
    } -> tensor<1x512xf32>
    %2353 = tensor.collapse_shape %2348 [[0 : i64, 1 : i64]] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2354 = tensor.expand_shape %2353 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2355 = "tensor.extract_slice"(%2354) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %2356 = tensor.collapse_shape %2355 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x512xf32> into tensor<512xf32>
    %2357 = tensor.expand_shape %2356 [[0 : i64, 1 : i64]] output_shape [1, 512] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x512xf32>
    %2358 = tensor.collapse_shape %2336 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2359 = tensor.expand_shape %2358 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2360 = tensor.empty() : tensor<128x512xf32>
    %2361 = linalg.transpose ins(%91:tensor<512x128xf32>) outs(%2360:tensor<128x512xf32>) permutation = [1, 0]
    %2362 = tensor.empty() : tensor<1x512xf32>
    %2363 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2364 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2363 : f32) outs(%2362 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2365 = linalg.matmul {prov.region_id = "matmul_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2359, %2361 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2364 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2366 = tensor.empty() : tensor<1x512xf32>
    %2367 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2365, %93 : tensor<1x512xf32>, tensor<512xf32>) outs(%2366 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb215(%2368: f32, %2369: f32, %2370: f32):
      %2371 = arith.addf %2368, %2369 : f32
      linalg.yield %2371 : f32
    } -> tensor<1x512xf32>
    %2372 = tensor.collapse_shape %2367 [[0 : i64, 1 : i64]] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x512xf32> into tensor<512xf32>
    %2373 = tensor.expand_shape %2372 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<512xf32> into tensor<1x1x512xf32>
    %2374 = tensor.empty() : tensor<1x1x512xf32>
    %2375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2373, %2357 : tensor<1x1x512xf32>, tensor<1x512xf32>) outs(%2374 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb216(%2376: f32, %2377: f32, %2378: f32):
      %2379 = arith.addf %2376, %2377 : f32
      linalg.yield %2379 : f32
    } -> tensor<1x1x512xf32>
    %2380 = "tensor.extract_slice"(%2375) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2381 = "tensor.extract_slice"(%2375) <{static_offsets = array<i64: 0, 0, 128>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2382 = "tensor.extract_slice"(%2375) <{static_offsets = array<i64: 0, 0, 256>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2383 = "tensor.extract_slice"(%2375) <{static_offsets = array<i64: 0, 0, 384>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x512xf32>) -> tensor<1x1x128xf32>
    %2384 = tensor.empty() : tensor<1x1x128xf32>
    %2385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2380 : tensor<1x1x128xf32>) outs(%2384 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_6", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb217(%2386: f32, %2387: f32):
      %2388 = arith.constant 1.000000e+00 : f32
      %2389 = arith.negf %2386 : f32
      %2390 = math.exp %2389 : f32
      %2391 = arith.addf %2388, %2390 : f32
      %2392 = arith.divf %2388, %2391 : f32
      linalg.yield %2392 : f32
    } -> tensor<1x1x128xf32>
    %2393 = tensor.empty() : tensor<1x1x128xf32>
    %2394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2381 : tensor<1x1x128xf32>) outs(%2393 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_7", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb218(%2395: f32, %2396: f32):
      %2397 = arith.constant 1.000000e+00 : f32
      %2398 = arith.negf %2395 : f32
      %2399 = math.exp %2398 : f32
      %2400 = arith.addf %2397, %2399 : f32
      %2401 = arith.divf %2397, %2400 : f32
      linalg.yield %2401 : f32
    } -> tensor<1x1x128xf32>
    %2402 = tensor.empty() : tensor<1x1x128xf32>
    %2403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2382 : tensor<1x1x128xf32>) outs(%2402 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_4", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb219(%2404: f32, %2405: f32):
      %2406 = math.tanh %2404 : f32
      linalg.yield %2406 : f32
    } -> tensor<1x1x128xf32>
    %2407 = tensor.empty() : tensor<1x1x128xf32>
    %2408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2383 : tensor<1x1x128xf32>) outs(%2407 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sigmoid_8", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb220(%2409: f32, %2410: f32):
      %2411 = arith.constant 1.000000e+00 : f32
      %2412 = arith.negf %2409 : f32
      %2413 = math.exp %2412 : f32
      %2414 = arith.addf %2411, %2413 : f32
      %2415 = arith.divf %2411, %2414 : f32
      linalg.yield %2415 : f32
    } -> tensor<1x1x128xf32>
    %2416 = tensor.empty() : tensor<1x1x128xf32>
    %2417 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2394, %2338 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2416 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb221(%2418: f32, %2419: f32, %2420: f32):
      %2421 = arith.mulf %2418, %2419 : f32
      linalg.yield %2421 : f32
    } -> tensor<1x1x128xf32>
    %2422 = tensor.empty() : tensor<1x1x128xf32>
    %2423 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2385, %2403 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2422 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb222(%2424: f32, %2425: f32, %2426: f32):
      %2427 = arith.mulf %2424, %2425 : f32
      linalg.yield %2427 : f32
    } -> tensor<1x1x128xf32>
    %2428 = tensor.empty() : tensor<1x1x128xf32>
    %2429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2417, %2423 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2428 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb223(%2430: f32, %2431: f32, %2432: f32):
      %2433 = arith.addf %2430, %2431 : f32
      linalg.yield %2433 : f32
    } -> tensor<1x1x128xf32>
    %2434 = tensor.empty() : tensor<1x1x128xf32>
    %2435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2429 : tensor<1x1x128xf32>) outs(%2434 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "tanh_5", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb224(%2436: f32, %2437: f32):
      %2438 = math.tanh %2436 : f32
      linalg.yield %2438 : f32
    } -> tensor<1x1x128xf32>
    %2439 = tensor.empty() : tensor<1x1x128xf32>
    %2440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2408, %2435 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%2439 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb225(%2441: f32, %2442: f32, %2443: f32):
      %2444 = arith.mulf %2441, %2442 : f32
      linalg.yield %2444 : f32
    } -> tensor<1x1x128xf32>
    %2445 = tensor.concat dim(0) %2440 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2446 = tensor.collapse_shape %2445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2447 = tensor.expand_shape %2446 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2448 = tensor.empty() : tensor<128x3xf32>
    %2449 = linalg.transpose ins(%95:tensor<3x128xf32>) outs(%2448:tensor<128x3xf32>) permutation = [1, 0]
    %2450 = tensor.empty() : tensor<1x3xf32>
    %2451 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2452 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2451 : f32) outs(%2450 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2453 = linalg.matmul {prov.region_id = "matmul_35", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2", prov.transposed_b = "true"} ins(%2447, %2449 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2452 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2454 = tensor.empty() : tensor<1x3xf32>
    %2455 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2453, %94 : tensor<1x3xf32>, tensor<3xf32>) outs(%2454 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb226(%2456: f32, %2457: f32, %2458: f32):
      %2459 = arith.addf %2456, %2457 : f32
      linalg.yield %2459 : f32
    } -> tensor<1x3xf32>
    func.return %2455 : tensor<1x3xf32>
  }
}
