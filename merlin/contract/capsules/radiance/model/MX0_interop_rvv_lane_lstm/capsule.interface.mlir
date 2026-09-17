builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64x32xf32>, %9: tensor<64xf32>, %10: tensor<32x32xf32>, %11: tensor<32xf32>, %12: tensor<32x32xf32>, %13: tensor<32xf32>, %14: tensor<32x32x8x8xf32>, %15: tensor<32xf32>, %16: tensor<32xf32>, %17: tensor<32xf32>, %18: tensor<64x32xf32>, %19: tensor<64xf32>, %20: tensor<32x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32xf32>, %24: tensor<256x32xf32>, %25: tensor<256xf32>, %26: tensor<256x8x3x3xf32>, %27: tensor<256xf32>, %28: tensor<32x256xf32>, %29: tensor<32xf32>, %30: tensor<256x32xf32>, %31: tensor<256xf32>, %32: tensor<256x8x3x3xf32>, %33: tensor<256xf32>, %34: tensor<32x256xf32>, %35: tensor<32xf32>, %36: tensor<32xf32>, %37: tensor<32xf32>, %38: tensor<32xf32>, %39: tensor<32xf32>, %40: tensor<64x32x3x3xf32>, %41: tensor<64xf32>, %42: tensor<64xf32>, %43: tensor<64xf32>, %44: tensor<64x64x4x4xf32>, %45: tensor<64xf32>, %46: tensor<64xf32>, %47: tensor<64xf32>, %48: tensor<128x64xf32>, %49: tensor<128xf32>, %50: tensor<64x64xf32>, %51: tensor<64xf32>, %52: tensor<64x64xf32>, %53: tensor<64xf32>, %54: tensor<64x64x4x4xf32>, %55: tensor<64xf32>, %56: tensor<64xf32>, %57: tensor<64xf32>, %58: tensor<128x64xf32>, %59: tensor<128xf32>, %60: tensor<64x64xf32>, %61: tensor<64xf32>, %62: tensor<64x64xf32>, %63: tensor<64xf32>, %64: tensor<512x64xf32>, %65: tensor<512xf32>, %66: tensor<512x8x3x3xf32>, %67: tensor<512xf32>, %68: tensor<64x512xf32>, %69: tensor<64xf32>, %70: tensor<512x64xf32>, %71: tensor<512xf32>, %72: tensor<512x8x3x3xf32>, %73: tensor<512xf32>, %74: tensor<64x512xf32>, %75: tensor<64xf32>, %76: tensor<64xf32>, %77: tensor<64xf32>, %78: tensor<64xf32>, %79: tensor<64xf32>, %80: tensor<512xf32>, %81: tensor<512x4608xf32>, %82: tensor<512x517xf32>, %83: tensor<512x128xf32>, %84: tensor<512xf32>, %85: tensor<512xf32>, %86: tensor<512x128xf32>, %87: tensor<512x128xf32>, %88: tensor<512xf32>, %89: tensor<512xf32>, %90: tensor<512x128xf32>, %91: tensor<512x128xf32>, %92: tensor<512xf32>, %93: tensor<512xf32>, %94: tensor<3xf32>, %95: tensor<3x128xf32>, %96: tensor<12x48x3x3xf32>, %97: tensor<12xf32>, %98: tensor<1x1x60x90xf32>, %99: tensor<1x1xf32>, %100: tensor<1x4xf32>, %101: tensor<3x128xf32>, %102: tensor<3x128xf32>) -> (tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>) {
    %103 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %104 = tensor.splat %103 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %105 = "tensor.insert_slice"(%98, %104) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %106 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%105 : tensor<1x1x66x96xf32>) outs(%106 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%108: f32, %109: f32):
      linalg.yield %108 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %110 = tensor.collapse_shape %107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %111 = tensor.expand_shape %110 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %112 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %113 = tensor.expand_shape %112 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %114 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %115 = tensor.splat %114 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %116 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%113, %111 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%115 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %117 = tensor.collapse_shape %116 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %118 = tensor.expand_shape %117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %119 = tensor.collapse_shape %118 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %120 = tensor.expand_shape %119 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %121 = tensor.empty() : tensor<1x32x15x23xf32>
    %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%121 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%123: f32, %124: f32, %125: f32):
      %126 = arith.addf %123, %124 : f32
      linalg.yield %126 : f32
    } -> tensor<1x32x15x23xf32>
    %127 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %128 = tensor.expand_shape %127 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %129 = tensor.empty() : tensor<1x345x32xf32>
    %130 = linalg.transpose ins(%128:tensor<1x32x345xf32>) outs(%129:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %131 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %132 = tensor.splat %131 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %133 = linalg.reduce ins(%130:tensor<1x345x32xf32>) outs(%132:tensor<1x345xf32>) dimensions = [2]
    (%134: f32, %135: f32) {
      %136 = arith.addf %134, %135 : f32
      linalg.yield %136 : f32
    }
    %137 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %138 = tensor.splat %137 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %139 = tensor.empty() : tensor<1x345xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%133, %138 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%139 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%141: f32, %142: f32, %143: f32):
      %144 = arith.divf %141, %142 : f32
      linalg.yield %144 : f32
    } -> tensor<1x345xf32>
    %145 = tensor.collapse_shape %140 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %146 = tensor.expand_shape %145 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %147 = tensor.empty() : tensor<1x345x32xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %146 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%147 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%149: f32, %150: f32, %151: f32):
      %152 = arith.subf %149, %150 : f32
      linalg.yield %152 : f32
    } -> tensor<1x345x32xf32>
    %153 = tensor.empty() : tensor<1x345x32xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%148, %148 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%153 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%155: f32, %156: f32, %157: f32):
      %158 = arith.mulf %155, %156 : f32
      linalg.yield %158 : f32
    } -> tensor<1x345x32xf32>
    %159 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %160 = tensor.splat %159 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %161 = linalg.reduce ins(%154:tensor<1x345x32xf32>) outs(%160:tensor<1x345xf32>) dimensions = [2]
    (%162: f32, %163: f32) {
      %164 = arith.addf %162, %163 : f32
      linalg.yield %164 : f32
    }
    %165 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %166 = tensor.splat %165 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %167 = tensor.empty() : tensor<1x345xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%161, %166 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%167 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%169: f32, %170: f32, %171: f32):
      %172 = arith.divf %169, %170 : f32
      linalg.yield %172 : f32
    } -> tensor<1x345xf32>
    %173 = tensor.collapse_shape %168 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %174 = tensor.expand_shape %173 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %175 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %176 = tensor.splat %175 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %177 = tensor.empty() : tensor<1x345x1xf32>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174, %176 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%177 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%179: f32, %180: f32, %181: f32):
      %182 = arith.addf %179, %180 : f32
      linalg.yield %182 : f32
    } -> tensor<1x345x1xf32>
    %183 = tensor.empty() : tensor<1x345x1xf32>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178 : tensor<1x345x1xf32>) outs(%183 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%185: f32, %186: f32):
      %187 = math.rsqrt %185 : f32
      linalg.yield %187 : f32
    } -> tensor<1x345x1xf32>
    %188 = tensor.empty() : tensor<1x345x32xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%148, %184 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%188 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%190: f32, %191: f32, %192: f32):
      %193 = arith.mulf %190, %191 : f32
      linalg.yield %193 : f32
    } -> tensor<1x345x32xf32>
    %194 = tensor.empty() : tensor<1x345x32xf32>
    %195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%189, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%194 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%196: f32, %197: f32, %198: f32):
      %199 = arith.mulf %196, %197 : f32
      linalg.yield %199 : f32
    } -> tensor<1x345x32xf32>
    %200 = tensor.empty() : tensor<1x345x32xf32>
    %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%195, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%200 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%202: f32, %203: f32, %204: f32):
      %205 = arith.addf %202, %203 : f32
      linalg.yield %205 : f32
    } -> tensor<1x345x32xf32>
    %206 = tensor.empty() : tensor<1x32x345xf32>
    %207 = linalg.transpose ins(%201:tensor<1x345x32xf32>) outs(%206:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %208 = tensor.collapse_shape %207 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %209 = tensor.expand_shape %208 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %210 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%209 : tensor<1x32x15x23xf32>) outs(%210 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%212: f32, %213: f32):
      linalg.yield %212 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %214 = tensor.collapse_shape %211 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %215 = tensor.expand_shape %214 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %216 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %217 = tensor.expand_shape %216 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %218 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %219 = tensor.splat %218 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %220 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%217, %215 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%219 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %221 = tensor.collapse_shape %220 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %222 = tensor.expand_shape %221 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %223 = tensor.collapse_shape %222 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %224 = tensor.expand_shape %223 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %225 = tensor.empty() : tensor<1x32x1x2xf32>
    %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%224, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%225 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%227: f32, %228: f32, %229: f32):
      %230 = arith.addf %227, %228 : f32
      linalg.yield %230 : f32
    } -> tensor<1x32x1x2xf32>
    %231 = tensor.collapse_shape %226 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %232 = tensor.expand_shape %231 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %233 = tensor.empty() : tensor<1x2x32xf32>
    %234 = linalg.transpose ins(%232:tensor<1x32x2xf32>) outs(%233:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %235 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %236 = tensor.splat %235 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %237 = linalg.reduce ins(%234:tensor<1x2x32xf32>) outs(%236:tensor<1x2xf32>) dimensions = [2]
    (%238: f32, %239: f32) {
      %240 = arith.addf %238, %239 : f32
      linalg.yield %240 : f32
    }
    %241 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %242 = tensor.splat %241 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %243 = tensor.empty() : tensor<1x2xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%237, %242 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%243 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%245: f32, %246: f32, %247: f32):
      %248 = arith.divf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x2xf32>
    %249 = tensor.collapse_shape %244 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %250 = tensor.expand_shape %249 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %251 = tensor.empty() : tensor<1x2x32xf32>
    %252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %250 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%251 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%253: f32, %254: f32, %255: f32):
      %256 = arith.subf %253, %254 : f32
      linalg.yield %256 : f32
    } -> tensor<1x2x32xf32>
    %257 = tensor.empty() : tensor<1x2x32xf32>
    %258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252, %252 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%257 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%259: f32, %260: f32, %261: f32):
      %262 = arith.mulf %259, %260 : f32
      linalg.yield %262 : f32
    } -> tensor<1x2x32xf32>
    %263 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %264 = tensor.splat %263 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %265 = linalg.reduce ins(%258:tensor<1x2x32xf32>) outs(%264:tensor<1x2xf32>) dimensions = [2]
    (%266: f32, %267: f32) {
      %268 = arith.addf %266, %267 : f32
      linalg.yield %268 : f32
    }
    %269 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %270 = tensor.splat %269 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %271 = tensor.empty() : tensor<1x2xf32>
    %272 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%265, %270 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%271 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%273: f32, %274: f32, %275: f32):
      %276 = arith.divf %273, %274 : f32
      linalg.yield %276 : f32
    } -> tensor<1x2xf32>
    %277 = tensor.collapse_shape %272 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %278 = tensor.expand_shape %277 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %279 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %280 = tensor.splat %279 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %281 = tensor.empty() : tensor<1x2x1xf32>
    %282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278, %280 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%281 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%283: f32, %284: f32, %285: f32):
      %286 = arith.addf %283, %284 : f32
      linalg.yield %286 : f32
    } -> tensor<1x2x1xf32>
    %287 = tensor.empty() : tensor<1x2x1xf32>
    %288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%282 : tensor<1x2x1xf32>) outs(%287 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%289: f32, %290: f32):
      %291 = math.rsqrt %289 : f32
      linalg.yield %291 : f32
    } -> tensor<1x2x1xf32>
    %292 = tensor.empty() : tensor<1x2x32xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252, %288 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%292 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%294: f32, %295: f32, %296: f32):
      %297 = arith.mulf %294, %295 : f32
      linalg.yield %297 : f32
    } -> tensor<1x2x32xf32>
    %298 = tensor.empty() : tensor<1x2x32xf32>
    %299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%293, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%298 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%300: f32, %301: f32, %302: f32):
      %303 = arith.mulf %300, %301 : f32
      linalg.yield %303 : f32
    } -> tensor<1x2x32xf32>
    %304 = tensor.empty() : tensor<1x2x32xf32>
    %305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%299, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%304 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%306: f32, %307: f32, %308: f32):
      %309 = arith.addf %306, %307 : f32
      linalg.yield %309 : f32
    } -> tensor<1x2x32xf32>
    %310 = tensor.empty() : tensor<32x64xf32>
    %311 = linalg.transpose ins(%8:tensor<64x32xf32>) outs(%310:tensor<32x64xf32>) permutation = [1, 0]
    %312 = tensor.empty() : tensor<1x2x64xf32>
    %313 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %314 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%313 : f32) outs(%312 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %315 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} ins(%305, %311 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%314 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %316 = tensor.empty() : tensor<1x2x64xf32>
    %317 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%315, %9 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%316 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%318: f32, %319: f32, %320: f32):
      %321 = arith.addf %318, %319 : f32
      linalg.yield %321 : f32
    } -> tensor<1x2x64xf32>
    %322 = tensor.collapse_shape %317 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %323 = tensor.expand_shape %322 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %324 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %325 = linalg.transpose ins(%323:tensor<1x2x2x1x32xf32>) outs(%324:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %326 = "tensor.extract_slice"(%325) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %327 = tensor.collapse_shape %326 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %328 = tensor.expand_shape %327 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %329 = "tensor.extract_slice"(%325) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %330 = tensor.collapse_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %331 = tensor.expand_shape %330 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %332 = tensor.empty() : tensor<32x32xf32>
    %333 = linalg.transpose ins(%10:tensor<32x32xf32>) outs(%332:tensor<32x32xf32>) permutation = [1, 0]
    %334 = tensor.empty() : tensor<1x345x32xf32>
    %335 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %336 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%335 : f32) outs(%334 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %337 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} ins(%201, %333 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%336 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %338 = tensor.empty() : tensor<1x345x32xf32>
    %339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337, %11 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%338 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb23(%340: f32, %341: f32, %342: f32):
      %343 = arith.addf %340, %341 : f32
      linalg.yield %343 : f32
    } -> tensor<1x345x32xf32>
    %344 = tensor.collapse_shape %339 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %345 = tensor.expand_shape %344 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %346 = tensor.empty() : tensor<1x1x345x32xf32>
    %347 = linalg.transpose ins(%345:tensor<1x345x1x32xf32>) outs(%346:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %348 = tensor.empty() : tensor<1x1x32x2xf32>
    %349 = linalg.transpose ins(%328:tensor<1x1x2x32xf32>) outs(%348:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %350 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %351 = tensor.splat %350 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %352 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%347, %349 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%351 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb24(%353: f32, %354: f32, %355: f32):
      %356 = arith.mulf %353, %354 : f32
      %357 = arith.addf %355, %356 : f32
      linalg.yield %357 : f32
    } -> tensor<1x1x345x2xf32>
    %358 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %359 = tensor.splat %358 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %360 = tensor.empty() : tensor<1x1x345x2xf32>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%352, %359 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%360 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb25(%362: f32, %363: f32, %364: f32):
      %365 = arith.divf %362, %363 : f32
      linalg.yield %365 : f32
    } -> tensor<1x1x345x2xf32>
    %366 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %367 = tensor.splat %366 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %368 = linalg.reduce ins(%361:tensor<1x1x345x2xf32>) outs(%367:tensor<1x1x345xf32>) dimensions = [3]
    (%369: f32, %370: f32) {
      %371 = arith.maximumf %369, %370 : f32
      linalg.yield %371 : f32
    }
    %372 = tensor.collapse_shape %368 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %373 = tensor.expand_shape %372 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %374 = tensor.empty() : tensor<1x1x345x2xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %373 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%374 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb26(%376: f32, %377: f32, %378: f32):
      %379 = arith.subf %376, %377 : f32
      linalg.yield %379 : f32
    } -> tensor<1x1x345x2xf32>
    %380 = tensor.empty() : tensor<1x1x345x2xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%375 : tensor<1x1x345x2xf32>) outs(%380 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb27(%382: f32, %383: f32):
      %384 = math.exp %382 : f32
      linalg.yield %384 : f32
    } -> tensor<1x1x345x2xf32>
    %385 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %386 = tensor.splat %385 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %387 = linalg.reduce ins(%381:tensor<1x1x345x2xf32>) outs(%386:tensor<1x1x345xf32>) dimensions = [3]
    (%388: f32, %389: f32) {
      %390 = arith.addf %388, %389 : f32
      linalg.yield %390 : f32
    }
    %391 = tensor.collapse_shape %387 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %392 = tensor.expand_shape %391 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %393 = tensor.empty() : tensor<1x1x345x2xf32>
    %394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %392 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%393 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb28(%395: f32, %396: f32, %397: f32):
      %398 = arith.divf %395, %396 : f32
      linalg.yield %398 : f32
    } -> tensor<1x1x345x2xf32>
    %399 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %400 = tensor.splat %399 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%394, %331 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%400 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb29(%402: f32, %403: f32, %404: f32):
      %405 = arith.mulf %402, %403 : f32
      %406 = arith.addf %404, %405 : f32
      linalg.yield %406 : f32
    } -> tensor<1x1x345x32xf32>
    %407 = tensor.empty() : tensor<1x345x1x32xf32>
    %408 = linalg.transpose ins(%401:tensor<1x1x345x32xf32>) outs(%407:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %409 = tensor.collapse_shape %408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %410 = tensor.expand_shape %409 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %411 = tensor.empty() : tensor<32x32xf32>
    %412 = linalg.transpose ins(%12:tensor<32x32xf32>) outs(%411:tensor<32x32xf32>) permutation = [1, 0]
    %413 = tensor.empty() : tensor<1x345x32xf32>
    %414 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %415 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%414 : f32) outs(%413 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %416 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} ins(%410, %412 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%415 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %417 = tensor.empty() : tensor<1x345x32xf32>
    %418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%416, %13 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%417 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb30(%419: f32, %420: f32, %421: f32):
      %422 = arith.addf %419, %420 : f32
      linalg.yield %422 : f32
    } -> tensor<1x345x32xf32>
    %423 = tensor.empty() : tensor<1x345x32xf32>
    %424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201, %418 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%423 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb31(%425: f32, %426: f32, %427: f32):
      %428 = arith.addf %425, %426 : f32
      linalg.yield %428 : f32
    } -> tensor<1x345x32xf32>
    %429 = tensor.empty() : tensor<32x256xf32>
    %430 = linalg.transpose ins(%24:tensor<256x32xf32>) outs(%429:tensor<32x256xf32>) permutation = [1, 0]
    %431 = tensor.empty() : tensor<1x345x256xf32>
    %432 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %433 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%432 : f32) outs(%431 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %434 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} ins(%424, %430 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%433 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %435 = tensor.empty() : tensor<1x345x256xf32>
    %436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%434, %25 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%435 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb32(%437: f32, %438: f32, %439: f32):
      %440 = arith.addf %437, %438 : f32
      linalg.yield %440 : f32
    } -> tensor<1x345x256xf32>
    %441 = tensor.empty() : tensor<1x256x345xf32>
    %442 = linalg.transpose ins(%436:tensor<1x345x256xf32>) outs(%441:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %443 = tensor.collapse_shape %442 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %444 = tensor.expand_shape %443 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %445 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %446 = tensor.splat %445 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %447 = "tensor.insert_slice"(%444, %446) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %448 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%447 : tensor<1x256x17x25xf32>) outs(%448 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb33(%450: f32, %451: f32):
      linalg.yield %450 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %452 = tensor.collapse_shape %449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %453 = tensor.expand_shape %452 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %454 = tensor.collapse_shape %26 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %455 = tensor.expand_shape %454 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %456 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %457 = tensor.splat %456 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%455, %453 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%457 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb34(%459: f32, %460: f32, %461: f32):
      %462 = arith.mulf %459, %460 : f32
      %463 = arith.addf %461, %462 : f32
      linalg.yield %463 : f32
    } -> tensor<32x8x345xf32>
    %464 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %465 = tensor.expand_shape %464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %466 = tensor.collapse_shape %465 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %467 = tensor.expand_shape %466 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %468 = tensor.empty() : tensor<1x256x15x23xf32>
    %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467, %27 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%468 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb35(%470: f32, %471: f32, %472: f32):
      %473 = arith.addf %470, %471 : f32
      linalg.yield %473 : f32
    } -> tensor<1x256x15x23xf32>
    %474 = tensor.collapse_shape %469 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %475 = tensor.expand_shape %474 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %476 = tensor.empty() : tensor<1x345x256xf32>
    %477 = linalg.transpose ins(%475:tensor<1x256x345xf32>) outs(%476:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %478 = tensor.empty() : tensor<1x345x256xf32>
    %479 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%477 : tensor<1x345x256xf32>) outs(%478 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb36(%480: f32, %481: f32):
      %482 = arith.constant 5.000000e-01 : f32
      %483 = arith.constant 1.000000e+00 : f32
      %484 = arith.constant 0.707106769 : f32
      %485 = arith.mulf %480, %484 : f32
      %486 = math.erf %485 : f32
      %487 = arith.addf %483, %486 : f32
      %488 = arith.mulf %482, %480 : f32
      %489 = arith.mulf %488, %487 : f32
      linalg.yield %489 : f32
    } -> tensor<1x345x256xf32>
    %490 = tensor.empty() : tensor<256x32xf32>
    %491 = linalg.transpose ins(%28:tensor<32x256xf32>) outs(%490:tensor<256x32xf32>) permutation = [1, 0]
    %492 = tensor.empty() : tensor<1x345x32xf32>
    %493 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %494 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%493 : f32) outs(%492 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %495 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} ins(%479, %491 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%494 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %496 = tensor.empty() : tensor<1x345x32xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%495, %29 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%496 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb37(%498: f32, %499: f32, %500: f32):
      %501 = arith.addf %498, %499 : f32
      linalg.yield %501 : f32
    } -> tensor<1x345x32xf32>
    %502 = tensor.empty() : tensor<1x345x32xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%424, %497 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%502 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb38(%504: f32, %505: f32, %506: f32):
      %507 = arith.addf %504, %505 : f32
      linalg.yield %507 : f32
    } -> tensor<1x345x32xf32>
    %508 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %509 = tensor.splat %508 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %510 = linalg.reduce ins(%503:tensor<1x345x32xf32>) outs(%509:tensor<1x345xf32>) dimensions = [2]
    (%511: f32, %512: f32) {
      %513 = arith.addf %511, %512 : f32
      linalg.yield %513 : f32
    }
    %514 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %515 = tensor.splat %514 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %516 = tensor.empty() : tensor<1x345xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510, %515 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%516 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb39(%518: f32, %519: f32, %520: f32):
      %521 = arith.divf %518, %519 : f32
      linalg.yield %521 : f32
    } -> tensor<1x345xf32>
    %522 = tensor.collapse_shape %517 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %524 = tensor.empty() : tensor<1x345x32xf32>
    %525 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%503, %523 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%524 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb40(%526: f32, %527: f32, %528: f32):
      %529 = arith.subf %526, %527 : f32
      linalg.yield %529 : f32
    } -> tensor<1x345x32xf32>
    %530 = tensor.empty() : tensor<1x345x32xf32>
    %531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525, %525 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%530 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb41(%532: f32, %533: f32, %534: f32):
      %535 = arith.mulf %532, %533 : f32
      linalg.yield %535 : f32
    } -> tensor<1x345x32xf32>
    %536 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %537 = tensor.splat %536 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %538 = linalg.reduce ins(%531:tensor<1x345x32xf32>) outs(%537:tensor<1x345xf32>) dimensions = [2]
    (%539: f32, %540: f32) {
      %541 = arith.addf %539, %540 : f32
      linalg.yield %541 : f32
    }
    %542 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %543 = tensor.splat %542 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %544 = tensor.empty() : tensor<1x345xf32>
    %545 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%538, %543 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%544 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb42(%546: f32, %547: f32, %548: f32):
      %549 = arith.divf %546, %547 : f32
      linalg.yield %549 : f32
    } -> tensor<1x345xf32>
    %550 = tensor.collapse_shape %545 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %551 = tensor.expand_shape %550 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %552 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %553 = tensor.splat %552 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %554 = tensor.empty() : tensor<1x345x1xf32>
    %555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%551, %553 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%554 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb43(%556: f32, %557: f32, %558: f32):
      %559 = arith.addf %556, %557 : f32
      linalg.yield %559 : f32
    } -> tensor<1x345x1xf32>
    %560 = tensor.empty() : tensor<1x345x1xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555 : tensor<1x345x1xf32>) outs(%560 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb44(%562: f32, %563: f32):
      %564 = math.rsqrt %562 : f32
      linalg.yield %564 : f32
    } -> tensor<1x345x1xf32>
    %565 = tensor.empty() : tensor<1x345x32xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525, %561 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%565 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb45(%567: f32, %568: f32, %569: f32):
      %570 = arith.mulf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x345x32xf32>
    %571 = tensor.empty() : tensor<1x345x32xf32>
    %572 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %36 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%571 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb46(%573: f32, %574: f32, %575: f32):
      %576 = arith.mulf %573, %574 : f32
      linalg.yield %576 : f32
    } -> tensor<1x345x32xf32>
    %577 = tensor.empty() : tensor<1x345x32xf32>
    %578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%572, %37 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%577 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%579: f32, %580: f32, %581: f32):
      %582 = arith.addf %579, %580 : f32
      linalg.yield %582 : f32
    } -> tensor<1x345x32xf32>
    %583 = tensor.empty() : tensor<1x32x345xf32>
    %584 = linalg.transpose ins(%578:tensor<1x345x32xf32>) outs(%583:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %585 = tensor.collapse_shape %584 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %586 = tensor.expand_shape %585 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %587 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%586 : tensor<1x32x15x23xf32>) outs(%587 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb48(%589: f32, %590: f32):
      linalg.yield %589 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %591 = tensor.collapse_shape %588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %592 = tensor.expand_shape %591 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %593 = tensor.collapse_shape %14 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %594 = tensor.expand_shape %593 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %595 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %596 = tensor.splat %595 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %597 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%594, %592 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%596 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %598 = tensor.collapse_shape %597 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %599 = tensor.expand_shape %598 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %600 = tensor.collapse_shape %599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %601 = tensor.expand_shape %600 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %602 = tensor.empty() : tensor<1x32x1x2xf32>
    %603 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%601, %15 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%602 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb49(%604: f32, %605: f32, %606: f32):
      %607 = arith.addf %604, %605 : f32
      linalg.yield %607 : f32
    } -> tensor<1x32x1x2xf32>
    %608 = tensor.collapse_shape %603 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %609 = tensor.expand_shape %608 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %610 = tensor.empty() : tensor<1x2x32xf32>
    %611 = linalg.transpose ins(%609:tensor<1x32x2xf32>) outs(%610:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %612 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %613 = tensor.splat %612 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %614 = linalg.reduce ins(%611:tensor<1x2x32xf32>) outs(%613:tensor<1x2xf32>) dimensions = [2]
    (%615: f32, %616: f32) {
      %617 = arith.addf %615, %616 : f32
      linalg.yield %617 : f32
    }
    %618 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %619 = tensor.splat %618 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %620 = tensor.empty() : tensor<1x2xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%614, %619 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%620 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb50(%622: f32, %623: f32, %624: f32):
      %625 = arith.divf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<1x2xf32>
    %626 = tensor.collapse_shape %621 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %627 = tensor.expand_shape %626 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %628 = tensor.empty() : tensor<1x2x32xf32>
    %629 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%611, %627 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%628 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb51(%630: f32, %631: f32, %632: f32):
      %633 = arith.subf %630, %631 : f32
      linalg.yield %633 : f32
    } -> tensor<1x2x32xf32>
    %634 = tensor.empty() : tensor<1x2x32xf32>
    %635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629, %629 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%634 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb52(%636: f32, %637: f32, %638: f32):
      %639 = arith.mulf %636, %637 : f32
      linalg.yield %639 : f32
    } -> tensor<1x2x32xf32>
    %640 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %641 = tensor.splat %640 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %642 = linalg.reduce ins(%635:tensor<1x2x32xf32>) outs(%641:tensor<1x2xf32>) dimensions = [2]
    (%643: f32, %644: f32) {
      %645 = arith.addf %643, %644 : f32
      linalg.yield %645 : f32
    }
    %646 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %647 = tensor.splat %646 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %648 = tensor.empty() : tensor<1x2xf32>
    %649 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%642, %647 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%648 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb53(%650: f32, %651: f32, %652: f32):
      %653 = arith.divf %650, %651 : f32
      linalg.yield %653 : f32
    } -> tensor<1x2xf32>
    %654 = tensor.collapse_shape %649 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %655 = tensor.expand_shape %654 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %656 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %657 = tensor.splat %656 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %658 = tensor.empty() : tensor<1x2x1xf32>
    %659 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%655, %657 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%658 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb54(%660: f32, %661: f32, %662: f32):
      %663 = arith.addf %660, %661 : f32
      linalg.yield %663 : f32
    } -> tensor<1x2x1xf32>
    %664 = tensor.empty() : tensor<1x2x1xf32>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%659 : tensor<1x2x1xf32>) outs(%664 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb55(%666: f32, %667: f32):
      %668 = math.rsqrt %666 : f32
      linalg.yield %668 : f32
    } -> tensor<1x2x1xf32>
    %669 = tensor.empty() : tensor<1x2x32xf32>
    %670 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629, %665 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%669 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb56(%671: f32, %672: f32, %673: f32):
      %674 = arith.mulf %671, %672 : f32
      linalg.yield %674 : f32
    } -> tensor<1x2x32xf32>
    %675 = tensor.empty() : tensor<1x2x32xf32>
    %676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670, %16 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%675 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb57(%677: f32, %678: f32, %679: f32):
      %680 = arith.mulf %677, %678 : f32
      linalg.yield %680 : f32
    } -> tensor<1x2x32xf32>
    %681 = tensor.empty() : tensor<1x2x32xf32>
    %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%676, %17 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%681 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb58(%683: f32, %684: f32, %685: f32):
      %686 = arith.addf %683, %684 : f32
      linalg.yield %686 : f32
    } -> tensor<1x2x32xf32>
    %687 = tensor.empty() : tensor<32x64xf32>
    %688 = linalg.transpose ins(%18:tensor<64x32xf32>) outs(%687:tensor<32x64xf32>) permutation = [1, 0]
    %689 = tensor.empty() : tensor<1x2x64xf32>
    %690 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %691 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%690 : f32) outs(%689 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %692 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} ins(%682, %688 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%691 : tensor<1x2x64xf32>) -> tensor<1x2x64xf32>
    %693 = tensor.empty() : tensor<1x2x64xf32>
    %694 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%692, %19 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%693 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb59(%695: f32, %696: f32, %697: f32):
      %698 = arith.addf %695, %696 : f32
      linalg.yield %698 : f32
    } -> tensor<1x2x64xf32>
    %699 = tensor.collapse_shape %694 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %700 = tensor.expand_shape %699 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %701 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %702 = linalg.transpose ins(%700:tensor<1x2x2x1x32xf32>) outs(%701:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %703 = "tensor.extract_slice"(%702) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %704 = tensor.collapse_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %706 = "tensor.extract_slice"(%702) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %707 = tensor.collapse_shape %706 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %708 = tensor.expand_shape %707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %709 = tensor.empty() : tensor<32x32xf32>
    %710 = linalg.transpose ins(%20:tensor<32x32xf32>) outs(%709:tensor<32x32xf32>) permutation = [1, 0]
    %711 = tensor.empty() : tensor<1x345x32xf32>
    %712 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %713 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%712 : f32) outs(%711 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %714 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} ins(%578, %710 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%713 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %715 = tensor.empty() : tensor<1x345x32xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%714, %21 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%715 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb60(%717: f32, %718: f32, %719: f32):
      %720 = arith.addf %717, %718 : f32
      linalg.yield %720 : f32
    } -> tensor<1x345x32xf32>
    %721 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %722 = tensor.expand_shape %721 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %723 = tensor.empty() : tensor<1x1x345x32xf32>
    %724 = linalg.transpose ins(%722:tensor<1x345x1x32xf32>) outs(%723:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %725 = tensor.empty() : tensor<1x1x32x2xf32>
    %726 = linalg.transpose ins(%705:tensor<1x1x2x32xf32>) outs(%725:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %727 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %728 = tensor.splat %727 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%724, %726 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%728 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb61(%730: f32, %731: f32, %732: f32):
      %733 = arith.mulf %730, %731 : f32
      %734 = arith.addf %732, %733 : f32
      linalg.yield %734 : f32
    } -> tensor<1x1x345x2xf32>
    %735 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %736 = tensor.splat %735 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %737 = tensor.empty() : tensor<1x1x345x2xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%729, %736 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%737 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb62(%739: f32, %740: f32, %741: f32):
      %742 = arith.divf %739, %740 : f32
      linalg.yield %742 : f32
    } -> tensor<1x1x345x2xf32>
    %743 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %744 = tensor.splat %743 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %745 = linalg.reduce ins(%738:tensor<1x1x345x2xf32>) outs(%744:tensor<1x1x345xf32>) dimensions = [3]
    (%746: f32, %747: f32) {
      %748 = arith.maximumf %746, %747 : f32
      linalg.yield %748 : f32
    }
    %749 = tensor.collapse_shape %745 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %750 = tensor.expand_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %751 = tensor.empty() : tensor<1x1x345x2xf32>
    %752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%738, %750 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%751 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb63(%753: f32, %754: f32, %755: f32):
      %756 = arith.subf %753, %754 : f32
      linalg.yield %756 : f32
    } -> tensor<1x1x345x2xf32>
    %757 = tensor.empty() : tensor<1x1x345x2xf32>
    %758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752 : tensor<1x1x345x2xf32>) outs(%757 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb64(%759: f32, %760: f32):
      %761 = math.exp %759 : f32
      linalg.yield %761 : f32
    } -> tensor<1x1x345x2xf32>
    %762 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %763 = tensor.splat %762 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %764 = linalg.reduce ins(%758:tensor<1x1x345x2xf32>) outs(%763:tensor<1x1x345xf32>) dimensions = [3]
    (%765: f32, %766: f32) {
      %767 = arith.addf %765, %766 : f32
      linalg.yield %767 : f32
    }
    %768 = tensor.collapse_shape %764 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %769 = tensor.expand_shape %768 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %770 = tensor.empty() : tensor<1x1x345x2xf32>
    %771 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%758, %769 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%770 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb65(%772: f32, %773: f32, %774: f32):
      %775 = arith.divf %772, %773 : f32
      linalg.yield %775 : f32
    } -> tensor<1x1x345x2xf32>
    %776 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %777 = tensor.splat %776 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%771, %708 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%777 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb66(%779: f32, %780: f32, %781: f32):
      %782 = arith.mulf %779, %780 : f32
      %783 = arith.addf %781, %782 : f32
      linalg.yield %783 : f32
    } -> tensor<1x1x345x32xf32>
    %784 = tensor.empty() : tensor<1x345x1x32xf32>
    %785 = linalg.transpose ins(%778:tensor<1x1x345x32xf32>) outs(%784:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %786 = tensor.collapse_shape %785 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %787 = tensor.expand_shape %786 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %788 = tensor.empty() : tensor<32x32xf32>
    %789 = linalg.transpose ins(%22:tensor<32x32xf32>) outs(%788:tensor<32x32xf32>) permutation = [1, 0]
    %790 = tensor.empty() : tensor<1x345x32xf32>
    %791 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %792 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%791 : f32) outs(%790 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %793 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} ins(%787, %789 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%792 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %794 = tensor.empty() : tensor<1x345x32xf32>
    %795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793, %23 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%794 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb67(%796: f32, %797: f32, %798: f32):
      %799 = arith.addf %796, %797 : f32
      linalg.yield %799 : f32
    } -> tensor<1x345x32xf32>
    %800 = tensor.empty() : tensor<1x345x32xf32>
    %801 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%578, %795 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%800 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb68(%802: f32, %803: f32, %804: f32):
      %805 = arith.addf %802, %803 : f32
      linalg.yield %805 : f32
    } -> tensor<1x345x32xf32>
    %806 = tensor.empty() : tensor<32x256xf32>
    %807 = linalg.transpose ins(%30:tensor<256x32xf32>) outs(%806:tensor<32x256xf32>) permutation = [1, 0]
    %808 = tensor.empty() : tensor<1x345x256xf32>
    %809 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %810 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%809 : f32) outs(%808 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %811 = linalg.matmul {prov.region_id = "matmul_12", prov.dispatch_id = "matmul_12", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} ins(%801, %807 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%810 : tensor<1x345x256xf32>) -> tensor<1x345x256xf32>
    %812 = tensor.empty() : tensor<1x345x256xf32>
    %813 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%811, %31 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%812 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb69(%814: f32, %815: f32, %816: f32):
      %817 = arith.addf %814, %815 : f32
      linalg.yield %817 : f32
    } -> tensor<1x345x256xf32>
    %818 = tensor.empty() : tensor<1x256x345xf32>
    %819 = linalg.transpose ins(%813:tensor<1x345x256xf32>) outs(%818:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %820 = tensor.collapse_shape %819 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %821 = tensor.expand_shape %820 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %822 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %823 = tensor.splat %822 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %824 = "tensor.insert_slice"(%821, %823) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %825 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %826 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%824 : tensor<1x256x17x25xf32>) outs(%825 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb70(%827: f32, %828: f32):
      linalg.yield %827 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %829 = tensor.collapse_shape %826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %830 = tensor.expand_shape %829 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %831 = tensor.collapse_shape %32 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %832 = tensor.expand_shape %831 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %833 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %834 = tensor.splat %833 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %835 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%832, %830 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%834 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb71(%836: f32, %837: f32, %838: f32):
      %839 = arith.mulf %836, %837 : f32
      %840 = arith.addf %838, %839 : f32
      linalg.yield %840 : f32
    } -> tensor<32x8x345xf32>
    %841 = tensor.collapse_shape %835 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %842 = tensor.expand_shape %841 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %843 = tensor.collapse_shape %842 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %844 = tensor.expand_shape %843 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %845 = tensor.empty() : tensor<1x256x15x23xf32>
    %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%844, %33 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%845 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb72(%847: f32, %848: f32, %849: f32):
      %850 = arith.addf %847, %848 : f32
      linalg.yield %850 : f32
    } -> tensor<1x256x15x23xf32>
    %851 = tensor.collapse_shape %846 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %852 = tensor.expand_shape %851 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %853 = tensor.empty() : tensor<1x345x256xf32>
    %854 = linalg.transpose ins(%852:tensor<1x256x345xf32>) outs(%853:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %855 = tensor.empty() : tensor<1x345x256xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%854 : tensor<1x345x256xf32>) outs(%855 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb73(%857: f32, %858: f32):
      %859 = arith.constant 5.000000e-01 : f32
      %860 = arith.constant 1.000000e+00 : f32
      %861 = arith.constant 0.707106769 : f32
      %862 = arith.mulf %857, %861 : f32
      %863 = math.erf %862 : f32
      %864 = arith.addf %860, %863 : f32
      %865 = arith.mulf %859, %857 : f32
      %866 = arith.mulf %865, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x345x256xf32>
    %867 = tensor.empty() : tensor<256x32xf32>
    %868 = linalg.transpose ins(%34:tensor<32x256xf32>) outs(%867:tensor<256x32xf32>) permutation = [1, 0]
    %869 = tensor.empty() : tensor<1x345x32xf32>
    %870 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %871 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%870 : f32) outs(%869 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %872 = linalg.matmul {prov.region_id = "matmul_13", prov.dispatch_id = "matmul_13", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} ins(%856, %868 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%871 : tensor<1x345x32xf32>) -> tensor<1x345x32xf32>
    %873 = tensor.empty() : tensor<1x345x32xf32>
    %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%872, %35 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%873 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb74(%875: f32, %876: f32, %877: f32):
      %878 = arith.addf %875, %876 : f32
      linalg.yield %878 : f32
    } -> tensor<1x345x32xf32>
    %879 = tensor.empty() : tensor<1x345x32xf32>
    %880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%801, %874 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%879 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb75(%881: f32, %882: f32, %883: f32):
      %884 = arith.addf %881, %882 : f32
      linalg.yield %884 : f32
    } -> tensor<1x345x32xf32>
    %885 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %886 = tensor.splat %885 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %887 = linalg.reduce ins(%880:tensor<1x345x32xf32>) outs(%886:tensor<1x345xf32>) dimensions = [2]
    (%888: f32, %889: f32) {
      %890 = arith.addf %888, %889 : f32
      linalg.yield %890 : f32
    }
    %891 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %892 = tensor.splat %891 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %893 = tensor.empty() : tensor<1x345xf32>
    %894 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%887, %892 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%893 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%895: f32, %896: f32, %897: f32):
      %898 = arith.divf %895, %896 : f32
      linalg.yield %898 : f32
    } -> tensor<1x345xf32>
    %899 = tensor.collapse_shape %894 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %900 = tensor.expand_shape %899 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %901 = tensor.empty() : tensor<1x345x32xf32>
    %902 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%880, %900 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%901 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb77(%903: f32, %904: f32, %905: f32):
      %906 = arith.subf %903, %904 : f32
      linalg.yield %906 : f32
    } -> tensor<1x345x32xf32>
    %907 = tensor.empty() : tensor<1x345x32xf32>
    %908 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%902, %902 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%907 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb78(%909: f32, %910: f32, %911: f32):
      %912 = arith.mulf %909, %910 : f32
      linalg.yield %912 : f32
    } -> tensor<1x345x32xf32>
    %913 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %914 = tensor.splat %913 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %915 = linalg.reduce ins(%908:tensor<1x345x32xf32>) outs(%914:tensor<1x345xf32>) dimensions = [2]
    (%916: f32, %917: f32) {
      %918 = arith.addf %916, %917 : f32
      linalg.yield %918 : f32
    }
    %919 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %920 = tensor.splat %919 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %921 = tensor.empty() : tensor<1x345xf32>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%915, %920 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%921 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb79(%923: f32, %924: f32, %925: f32):
      %926 = arith.divf %923, %924 : f32
      linalg.yield %926 : f32
    } -> tensor<1x345xf32>
    %927 = tensor.collapse_shape %922 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %928 = tensor.expand_shape %927 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %929 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %930 = tensor.splat %929 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %931 = tensor.empty() : tensor<1x345x1xf32>
    %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%928, %930 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%931 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb80(%933: f32, %934: f32, %935: f32):
      %936 = arith.addf %933, %934 : f32
      linalg.yield %936 : f32
    } -> tensor<1x345x1xf32>
    %937 = tensor.empty() : tensor<1x345x1xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%932 : tensor<1x345x1xf32>) outs(%937 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb81(%939: f32, %940: f32):
      %941 = math.rsqrt %939 : f32
      linalg.yield %941 : f32
    } -> tensor<1x345x1xf32>
    %942 = tensor.empty() : tensor<1x345x32xf32>
    %943 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%902, %938 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%942 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb82(%944: f32, %945: f32, %946: f32):
      %947 = arith.mulf %944, %945 : f32
      linalg.yield %947 : f32
    } -> tensor<1x345x32xf32>
    %948 = tensor.empty() : tensor<1x345x32xf32>
    %949 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%943, %38 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%948 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb83(%950: f32, %951: f32, %952: f32):
      %953 = arith.mulf %950, %951 : f32
      linalg.yield %953 : f32
    } -> tensor<1x345x32xf32>
    %954 = tensor.empty() : tensor<1x345x32xf32>
    %955 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%949, %39 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%954 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%956: f32, %957: f32, %958: f32):
      %959 = arith.addf %956, %957 : f32
      linalg.yield %959 : f32
    } -> tensor<1x345x32xf32>
    %960 = tensor.collapse_shape %955 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %961 = tensor.expand_shape %960 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %962 = tensor.empty() : tensor<1x32x15x23xf32>
    %963 = linalg.transpose ins(%961:tensor<1x15x23x32xf32>) outs(%962:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %964 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %965 = tensor.splat %964 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %966 = "tensor.insert_slice"(%963, %965) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %967 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%966 : tensor<1x32x17x25xf32>) outs(%967 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb85(%969: f32, %970: f32):
      linalg.yield %969 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %971 = tensor.collapse_shape %968 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %972 = tensor.expand_shape %971 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %973 = tensor.collapse_shape %40 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %974 = tensor.expand_shape %973 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %975 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %976 = tensor.splat %975 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %977 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%974, %972 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%976 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %978 = tensor.collapse_shape %977 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %979 = tensor.expand_shape %978 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %980 = tensor.collapse_shape %979 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %981 = tensor.expand_shape %980 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %982 = tensor.empty() : tensor<1x64x8x12xf32>
    %983 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%981, %41 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%982 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb86(%984: f32, %985: f32, %986: f32):
      %987 = arith.addf %984, %985 : f32
      linalg.yield %987 : f32
    } -> tensor<1x64x8x12xf32>
    %988 = tensor.collapse_shape %983 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %989 = tensor.expand_shape %988 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %990 = tensor.empty() : tensor<1x96x64xf32>
    %991 = linalg.transpose ins(%989:tensor<1x64x96xf32>) outs(%990:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %992 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %993 = tensor.splat %992 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %994 = linalg.reduce ins(%991:tensor<1x96x64xf32>) outs(%993:tensor<1x96xf32>) dimensions = [2]
    (%995: f32, %996: f32) {
      %997 = arith.addf %995, %996 : f32
      linalg.yield %997 : f32
    }
    %998 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %999 = tensor.splat %998 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1000 = tensor.empty() : tensor<1x96xf32>
    %1001 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%994, %999 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1000 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb87(%1002: f32, %1003: f32, %1004: f32):
      %1005 = arith.divf %1002, %1003 : f32
      linalg.yield %1005 : f32
    } -> tensor<1x96xf32>
    %1006 = tensor.collapse_shape %1001 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1007 = tensor.expand_shape %1006 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1008 = tensor.empty() : tensor<1x96x64xf32>
    %1009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991, %1007 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1008 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb88(%1010: f32, %1011: f32, %1012: f32):
      %1013 = arith.subf %1010, %1011 : f32
      linalg.yield %1013 : f32
    } -> tensor<1x96x64xf32>
    %1014 = tensor.empty() : tensor<1x96x64xf32>
    %1015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1009, %1009 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1014 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb89(%1016: f32, %1017: f32, %1018: f32):
      %1019 = arith.mulf %1016, %1017 : f32
      linalg.yield %1019 : f32
    } -> tensor<1x96x64xf32>
    %1020 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1021 = tensor.splat %1020 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1022 = linalg.reduce ins(%1015:tensor<1x96x64xf32>) outs(%1021:tensor<1x96xf32>) dimensions = [2]
    (%1023: f32, %1024: f32) {
      %1025 = arith.addf %1023, %1024 : f32
      linalg.yield %1025 : f32
    }
    %1026 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1027 = tensor.splat %1026 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1028 = tensor.empty() : tensor<1x96xf32>
    %1029 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1022, %1027 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1028 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb90(%1030: f32, %1031: f32, %1032: f32):
      %1033 = arith.divf %1030, %1031 : f32
      linalg.yield %1033 : f32
    } -> tensor<1x96xf32>
    %1034 = tensor.collapse_shape %1029 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1035 = tensor.expand_shape %1034 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1036 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1037 = tensor.splat %1036 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1038 = tensor.empty() : tensor<1x96x1xf32>
    %1039 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1035, %1037 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1038 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb91(%1040: f32, %1041: f32, %1042: f32):
      %1043 = arith.addf %1040, %1041 : f32
      linalg.yield %1043 : f32
    } -> tensor<1x96x1xf32>
    %1044 = tensor.empty() : tensor<1x96x1xf32>
    %1045 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1039 : tensor<1x96x1xf32>) outs(%1044 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb92(%1046: f32, %1047: f32):
      %1048 = math.rsqrt %1046 : f32
      linalg.yield %1048 : f32
    } -> tensor<1x96x1xf32>
    %1049 = tensor.empty() : tensor<1x96x64xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1009, %1045 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1049 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb93(%1051: f32, %1052: f32, %1053: f32):
      %1054 = arith.mulf %1051, %1052 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x96x64xf32>
    %1055 = tensor.empty() : tensor<1x96x64xf32>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1050, %42 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1055 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb94(%1057: f32, %1058: f32, %1059: f32):
      %1060 = arith.mulf %1057, %1058 : f32
      linalg.yield %1060 : f32
    } -> tensor<1x96x64xf32>
    %1061 = tensor.empty() : tensor<1x96x64xf32>
    %1062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1056, %43 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1061 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb95(%1063: f32, %1064: f32, %1065: f32):
      %1066 = arith.addf %1063, %1064 : f32
      linalg.yield %1066 : f32
    } -> tensor<1x96x64xf32>
    %1067 = tensor.empty() : tensor<1x64x96xf32>
    %1068 = linalg.transpose ins(%1062:tensor<1x96x64xf32>) outs(%1067:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1069 = tensor.collapse_shape %1068 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1070 = tensor.expand_shape %1069 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1071 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1070 : tensor<1x64x8x12xf32>) outs(%1071 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb96(%1073: f32, %1074: f32):
      linalg.yield %1073 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1075 = tensor.collapse_shape %1072 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1076 = tensor.expand_shape %1075 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1077 = tensor.collapse_shape %44 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1078 = tensor.expand_shape %1077 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1079 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1080 = tensor.splat %1079 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1081 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1078, %1076 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1080 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1082 = tensor.collapse_shape %1081 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1083 = tensor.expand_shape %1082 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1084 = tensor.collapse_shape %1083 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1085 = tensor.expand_shape %1084 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1086 = tensor.empty() : tensor<1x64x2x3xf32>
    %1087 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1085, %45 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1086 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb97(%1088: f32, %1089: f32, %1090: f32):
      %1091 = arith.addf %1088, %1089 : f32
      linalg.yield %1091 : f32
    } -> tensor<1x64x2x3xf32>
    %1092 = tensor.collapse_shape %1087 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1093 = tensor.expand_shape %1092 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1094 = tensor.empty() : tensor<1x6x64xf32>
    %1095 = linalg.transpose ins(%1093:tensor<1x64x6xf32>) outs(%1094:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1096 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1097 = tensor.splat %1096 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1098 = linalg.reduce ins(%1095:tensor<1x6x64xf32>) outs(%1097:tensor<1x6xf32>) dimensions = [2]
    (%1099: f32, %1100: f32) {
      %1101 = arith.addf %1099, %1100 : f32
      linalg.yield %1101 : f32
    }
    %1102 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1103 = tensor.splat %1102 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1104 = tensor.empty() : tensor<1x6xf32>
    %1105 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1098, %1103 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1104 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb98(%1106: f32, %1107: f32, %1108: f32):
      %1109 = arith.divf %1106, %1107 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x6xf32>
    %1110 = tensor.collapse_shape %1105 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1111 = tensor.expand_shape %1110 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1112 = tensor.empty() : tensor<1x6x64xf32>
    %1113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1095, %1111 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1112 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb99(%1114: f32, %1115: f32, %1116: f32):
      %1117 = arith.subf %1114, %1115 : f32
      linalg.yield %1117 : f32
    } -> tensor<1x6x64xf32>
    %1118 = tensor.empty() : tensor<1x6x64xf32>
    %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113, %1113 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1118 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb100(%1120: f32, %1121: f32, %1122: f32):
      %1123 = arith.mulf %1120, %1121 : f32
      linalg.yield %1123 : f32
    } -> tensor<1x6x64xf32>
    %1124 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1125 = tensor.splat %1124 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1126 = linalg.reduce ins(%1119:tensor<1x6x64xf32>) outs(%1125:tensor<1x6xf32>) dimensions = [2]
    (%1127: f32, %1128: f32) {
      %1129 = arith.addf %1127, %1128 : f32
      linalg.yield %1129 : f32
    }
    %1130 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1131 = tensor.splat %1130 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1132 = tensor.empty() : tensor<1x6xf32>
    %1133 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1126, %1131 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1132 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb101(%1134: f32, %1135: f32, %1136: f32):
      %1137 = arith.divf %1134, %1135 : f32
      linalg.yield %1137 : f32
    } -> tensor<1x6xf32>
    %1138 = tensor.collapse_shape %1133 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1139 = tensor.expand_shape %1138 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1140 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1141 = tensor.splat %1140 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1142 = tensor.empty() : tensor<1x6x1xf32>
    %1143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1139, %1141 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1142 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb102(%1144: f32, %1145: f32, %1146: f32):
      %1147 = arith.addf %1144, %1145 : f32
      linalg.yield %1147 : f32
    } -> tensor<1x6x1xf32>
    %1148 = tensor.empty() : tensor<1x6x1xf32>
    %1149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1143 : tensor<1x6x1xf32>) outs(%1148 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb103(%1150: f32, %1151: f32):
      %1152 = math.rsqrt %1150 : f32
      linalg.yield %1152 : f32
    } -> tensor<1x6x1xf32>
    %1153 = tensor.empty() : tensor<1x6x64xf32>
    %1154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113, %1149 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1153 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb104(%1155: f32, %1156: f32, %1157: f32):
      %1158 = arith.mulf %1155, %1156 : f32
      linalg.yield %1158 : f32
    } -> tensor<1x6x64xf32>
    %1159 = tensor.empty() : tensor<1x6x64xf32>
    %1160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1154, %46 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1159 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb105(%1161: f32, %1162: f32, %1163: f32):
      %1164 = arith.mulf %1161, %1162 : f32
      linalg.yield %1164 : f32
    } -> tensor<1x6x64xf32>
    %1165 = tensor.empty() : tensor<1x6x64xf32>
    %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1160, %47 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1165 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb106(%1167: f32, %1168: f32, %1169: f32):
      %1170 = arith.addf %1167, %1168 : f32
      linalg.yield %1170 : f32
    } -> tensor<1x6x64xf32>
    %1171 = tensor.empty() : tensor<64x128xf32>
    %1172 = linalg.transpose ins(%48:tensor<128x64xf32>) outs(%1171:tensor<64x128xf32>) permutation = [1, 0]
    %1173 = tensor.empty() : tensor<1x6x128xf32>
    %1174 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1175 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1174 : f32) outs(%1173 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1176 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} ins(%1166, %1172 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1175 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1177 = tensor.empty() : tensor<1x6x128xf32>
    %1178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1176, %49 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1177 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb107(%1179: f32, %1180: f32, %1181: f32):
      %1182 = arith.addf %1179, %1180 : f32
      linalg.yield %1182 : f32
    } -> tensor<1x6x128xf32>
    %1183 = tensor.collapse_shape %1178 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1184 = tensor.expand_shape %1183 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1185 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1186 = linalg.transpose ins(%1184:tensor<1x6x2x2x32xf32>) outs(%1185:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1187 = "tensor.extract_slice"(%1186) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1188 = tensor.collapse_shape %1187 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1189 = tensor.expand_shape %1188 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1190 = "tensor.extract_slice"(%1186) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1191 = tensor.collapse_shape %1190 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1192 = tensor.expand_shape %1191 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1193 = tensor.empty() : tensor<64x64xf32>
    %1194 = linalg.transpose ins(%50:tensor<64x64xf32>) outs(%1193:tensor<64x64xf32>) permutation = [1, 0]
    %1195 = tensor.empty() : tensor<1x96x64xf32>
    %1196 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1197 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1196 : f32) outs(%1195 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1198 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} ins(%1062, %1194 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1197 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1199 = tensor.empty() : tensor<1x96x64xf32>
    %1200 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1198, %51 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1199 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb108(%1201: f32, %1202: f32, %1203: f32):
      %1204 = arith.addf %1201, %1202 : f32
      linalg.yield %1204 : f32
    } -> tensor<1x96x64xf32>
    %1205 = tensor.collapse_shape %1200 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1206 = tensor.expand_shape %1205 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1207 = tensor.empty() : tensor<1x2x96x32xf32>
    %1208 = linalg.transpose ins(%1206:tensor<1x96x2x32xf32>) outs(%1207:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1209 = tensor.empty() : tensor<1x2x32x6xf32>
    %1210 = linalg.transpose ins(%1189:tensor<1x2x6x32xf32>) outs(%1209:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1211 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1212 = tensor.splat %1211 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1208, %1210 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1212 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb109(%1214: f32, %1215: f32, %1216: f32):
      %1217 = arith.mulf %1214, %1215 : f32
      %1218 = arith.addf %1216, %1217 : f32
      linalg.yield %1218 : f32
    } -> tensor<1x2x96x6xf32>
    %1219 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1220 = tensor.splat %1219 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1221 = tensor.empty() : tensor<1x2x96x6xf32>
    %1222 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1213, %1220 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1221 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb110(%1223: f32, %1224: f32, %1225: f32):
      %1226 = arith.divf %1223, %1224 : f32
      linalg.yield %1226 : f32
    } -> tensor<1x2x96x6xf32>
    %1227 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1228 = tensor.splat %1227 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1229 = linalg.reduce ins(%1222:tensor<1x2x96x6xf32>) outs(%1228:tensor<1x2x96xf32>) dimensions = [3]
    (%1230: f32, %1231: f32) {
      %1232 = arith.maximumf %1230, %1231 : f32
      linalg.yield %1232 : f32
    }
    %1233 = tensor.collapse_shape %1229 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1235 = tensor.empty() : tensor<1x2x96x6xf32>
    %1236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1222, %1234 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1235 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb111(%1237: f32, %1238: f32, %1239: f32):
      %1240 = arith.subf %1237, %1238 : f32
      linalg.yield %1240 : f32
    } -> tensor<1x2x96x6xf32>
    %1241 = tensor.empty() : tensor<1x2x96x6xf32>
    %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1236 : tensor<1x2x96x6xf32>) outs(%1241 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb112(%1243: f32, %1244: f32):
      %1245 = math.exp %1243 : f32
      linalg.yield %1245 : f32
    } -> tensor<1x2x96x6xf32>
    %1246 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1247 = tensor.splat %1246 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1248 = linalg.reduce ins(%1242:tensor<1x2x96x6xf32>) outs(%1247:tensor<1x2x96xf32>) dimensions = [3]
    (%1249: f32, %1250: f32) {
      %1251 = arith.addf %1249, %1250 : f32
      linalg.yield %1251 : f32
    }
    %1252 = tensor.collapse_shape %1248 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1253 = tensor.expand_shape %1252 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1254 = tensor.empty() : tensor<1x2x96x6xf32>
    %1255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1242, %1253 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1254 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb113(%1256: f32, %1257: f32, %1258: f32):
      %1259 = arith.divf %1256, %1257 : f32
      linalg.yield %1259 : f32
    } -> tensor<1x2x96x6xf32>
    %1260 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1261 = tensor.splat %1260 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1255, %1192 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1261 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb114(%1263: f32, %1264: f32, %1265: f32):
      %1266 = arith.mulf %1263, %1264 : f32
      %1267 = arith.addf %1265, %1266 : f32
      linalg.yield %1267 : f32
    } -> tensor<1x2x96x32xf32>
    %1268 = tensor.empty() : tensor<1x96x2x32xf32>
    %1269 = linalg.transpose ins(%1262:tensor<1x2x96x32xf32>) outs(%1268:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1270 = tensor.collapse_shape %1269 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1271 = tensor.expand_shape %1270 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1272 = tensor.empty() : tensor<64x64xf32>
    %1273 = linalg.transpose ins(%52:tensor<64x64xf32>) outs(%1272:tensor<64x64xf32>) permutation = [1, 0]
    %1274 = tensor.empty() : tensor<1x96x64xf32>
    %1275 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1276 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1275 : f32) outs(%1274 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1277 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} ins(%1271, %1273 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1276 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1278 = tensor.empty() : tensor<1x96x64xf32>
    %1279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277, %53 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1278 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb115(%1280: f32, %1281: f32, %1282: f32):
      %1283 = arith.addf %1280, %1281 : f32
      linalg.yield %1283 : f32
    } -> tensor<1x96x64xf32>
    %1284 = tensor.empty() : tensor<1x96x64xf32>
    %1285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1062, %1279 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1284 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb116(%1286: f32, %1287: f32, %1288: f32):
      %1289 = arith.addf %1286, %1287 : f32
      linalg.yield %1289 : f32
    } -> tensor<1x96x64xf32>
    %1290 = tensor.empty() : tensor<64x512xf32>
    %1291 = linalg.transpose ins(%64:tensor<512x64xf32>) outs(%1290:tensor<64x512xf32>) permutation = [1, 0]
    %1292 = tensor.empty() : tensor<1x96x512xf32>
    %1293 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1294 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1293 : f32) outs(%1292 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1295 = linalg.matmul {prov.region_id = "matmul_19", prov.dispatch_id = "matmul_19", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} ins(%1285, %1291 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%1294 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1296 = tensor.empty() : tensor<1x96x512xf32>
    %1297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295, %65 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1296 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb117(%1298: f32, %1299: f32, %1300: f32):
      %1301 = arith.addf %1298, %1299 : f32
      linalg.yield %1301 : f32
    } -> tensor<1x96x512xf32>
    %1302 = tensor.empty() : tensor<1x512x96xf32>
    %1303 = linalg.transpose ins(%1297:tensor<1x96x512xf32>) outs(%1302:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1304 = tensor.collapse_shape %1303 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1305 = tensor.expand_shape %1304 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1306 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1307 = tensor.splat %1306 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1308 = "tensor.insert_slice"(%1305, %1307) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1309 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1308 : tensor<1x512x10x14xf32>) outs(%1309 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb118(%1311: f32, %1312: f32):
      linalg.yield %1311 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1313 = tensor.collapse_shape %1310 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1314 = tensor.expand_shape %1313 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1315 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1316 = tensor.expand_shape %1315 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1317 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1318 = tensor.splat %1317 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1316, %1314 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1318 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb119(%1320: f32, %1321: f32, %1322: f32):
      %1323 = arith.mulf %1320, %1321 : f32
      %1324 = arith.addf %1322, %1323 : f32
      linalg.yield %1324 : f32
    } -> tensor<64x8x96xf32>
    %1325 = tensor.collapse_shape %1319 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1326 = tensor.expand_shape %1325 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1327 = tensor.collapse_shape %1326 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1328 = tensor.expand_shape %1327 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1329 = tensor.empty() : tensor<1x512x8x12xf32>
    %1330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1328, %67 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1329 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb120(%1331: f32, %1332: f32, %1333: f32):
      %1334 = arith.addf %1331, %1332 : f32
      linalg.yield %1334 : f32
    } -> tensor<1x512x8x12xf32>
    %1335 = tensor.collapse_shape %1330 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1336 = tensor.expand_shape %1335 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1337 = tensor.empty() : tensor<1x96x512xf32>
    %1338 = linalg.transpose ins(%1336:tensor<1x512x96xf32>) outs(%1337:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1339 = tensor.empty() : tensor<1x96x512xf32>
    %1340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1338 : tensor<1x96x512xf32>) outs(%1339 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb121(%1341: f32, %1342: f32):
      %1343 = arith.constant 5.000000e-01 : f32
      %1344 = arith.constant 1.000000e+00 : f32
      %1345 = arith.constant 0.707106769 : f32
      %1346 = arith.mulf %1341, %1345 : f32
      %1347 = math.erf %1346 : f32
      %1348 = arith.addf %1344, %1347 : f32
      %1349 = arith.mulf %1343, %1341 : f32
      %1350 = arith.mulf %1349, %1348 : f32
      linalg.yield %1350 : f32
    } -> tensor<1x96x512xf32>
    %1351 = tensor.empty() : tensor<512x64xf32>
    %1352 = linalg.transpose ins(%68:tensor<64x512xf32>) outs(%1351:tensor<512x64xf32>) permutation = [1, 0]
    %1353 = tensor.empty() : tensor<1x96x64xf32>
    %1354 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1355 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1354 : f32) outs(%1353 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1356 = linalg.matmul {prov.region_id = "matmul_20", prov.dispatch_id = "matmul_20", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} ins(%1340, %1352 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%1355 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1357 = tensor.empty() : tensor<1x96x64xf32>
    %1358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1356, %69 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1357 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb122(%1359: f32, %1360: f32, %1361: f32):
      %1362 = arith.addf %1359, %1360 : f32
      linalg.yield %1362 : f32
    } -> tensor<1x96x64xf32>
    %1363 = tensor.empty() : tensor<1x96x64xf32>
    %1364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1285, %1358 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1363 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb123(%1365: f32, %1366: f32, %1367: f32):
      %1368 = arith.addf %1365, %1366 : f32
      linalg.yield %1368 : f32
    } -> tensor<1x96x64xf32>
    %1369 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1370 = tensor.splat %1369 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1371 = linalg.reduce ins(%1364:tensor<1x96x64xf32>) outs(%1370:tensor<1x96xf32>) dimensions = [2]
    (%1372: f32, %1373: f32) {
      %1374 = arith.addf %1372, %1373 : f32
      linalg.yield %1374 : f32
    }
    %1375 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1376 = tensor.splat %1375 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1377 = tensor.empty() : tensor<1x96xf32>
    %1378 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1371, %1376 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1377 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb124(%1379: f32, %1380: f32, %1381: f32):
      %1382 = arith.divf %1379, %1380 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x96xf32>
    %1383 = tensor.collapse_shape %1378 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1384 = tensor.expand_shape %1383 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1385 = tensor.empty() : tensor<1x96x64xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1364, %1384 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1385 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb125(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.subf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x96x64xf32>
    %1391 = tensor.empty() : tensor<1x96x64xf32>
    %1392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1386, %1386 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1391 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb126(%1393: f32, %1394: f32, %1395: f32):
      %1396 = arith.mulf %1393, %1394 : f32
      linalg.yield %1396 : f32
    } -> tensor<1x96x64xf32>
    %1397 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1398 = tensor.splat %1397 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1399 = linalg.reduce ins(%1392:tensor<1x96x64xf32>) outs(%1398:tensor<1x96xf32>) dimensions = [2]
    (%1400: f32, %1401: f32) {
      %1402 = arith.addf %1400, %1401 : f32
      linalg.yield %1402 : f32
    }
    %1403 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1404 = tensor.splat %1403 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1405 = tensor.empty() : tensor<1x96xf32>
    %1406 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1399, %1404 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1405 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb127(%1407: f32, %1408: f32, %1409: f32):
      %1410 = arith.divf %1407, %1408 : f32
      linalg.yield %1410 : f32
    } -> tensor<1x96xf32>
    %1411 = tensor.collapse_shape %1406 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1412 = tensor.expand_shape %1411 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1413 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1414 = tensor.splat %1413 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1415 = tensor.empty() : tensor<1x96x1xf32>
    %1416 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1412, %1414 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1415 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb128(%1417: f32, %1418: f32, %1419: f32):
      %1420 = arith.addf %1417, %1418 : f32
      linalg.yield %1420 : f32
    } -> tensor<1x96x1xf32>
    %1421 = tensor.empty() : tensor<1x96x1xf32>
    %1422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1416 : tensor<1x96x1xf32>) outs(%1421 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb129(%1423: f32, %1424: f32):
      %1425 = math.rsqrt %1423 : f32
      linalg.yield %1425 : f32
    } -> tensor<1x96x1xf32>
    %1426 = tensor.empty() : tensor<1x96x64xf32>
    %1427 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1386, %1422 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1426 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb130(%1428: f32, %1429: f32, %1430: f32):
      %1431 = arith.mulf %1428, %1429 : f32
      linalg.yield %1431 : f32
    } -> tensor<1x96x64xf32>
    %1432 = tensor.empty() : tensor<1x96x64xf32>
    %1433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1427, %76 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1432 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb131(%1434: f32, %1435: f32, %1436: f32):
      %1437 = arith.mulf %1434, %1435 : f32
      linalg.yield %1437 : f32
    } -> tensor<1x96x64xf32>
    %1438 = tensor.empty() : tensor<1x96x64xf32>
    %1439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1433, %77 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1438 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb132(%1440: f32, %1441: f32, %1442: f32):
      %1443 = arith.addf %1440, %1441 : f32
      linalg.yield %1443 : f32
    } -> tensor<1x96x64xf32>
    %1444 = tensor.empty() : tensor<1x64x96xf32>
    %1445 = linalg.transpose ins(%1439:tensor<1x96x64xf32>) outs(%1444:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1446 = tensor.collapse_shape %1445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1447 = tensor.expand_shape %1446 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1448 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1447 : tensor<1x64x8x12xf32>) outs(%1448 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb133(%1450: f32, %1451: f32):
      linalg.yield %1450 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1452 = tensor.collapse_shape %1449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1453 = tensor.expand_shape %1452 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1454 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1455 = tensor.expand_shape %1454 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1456 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1457 = tensor.splat %1456 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1458 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1455, %1453 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1457 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1459 = tensor.collapse_shape %1458 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1460 = tensor.expand_shape %1459 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1461 = tensor.collapse_shape %1460 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1462 = tensor.expand_shape %1461 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1463 = tensor.empty() : tensor<1x64x2x3xf32>
    %1464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1462, %55 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1463 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb134(%1465: f32, %1466: f32, %1467: f32):
      %1468 = arith.addf %1465, %1466 : f32
      linalg.yield %1468 : f32
    } -> tensor<1x64x2x3xf32>
    %1469 = tensor.collapse_shape %1464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1470 = tensor.expand_shape %1469 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1471 = tensor.empty() : tensor<1x6x64xf32>
    %1472 = linalg.transpose ins(%1470:tensor<1x64x6xf32>) outs(%1471:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1473 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1474 = tensor.splat %1473 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1475 = linalg.reduce ins(%1472:tensor<1x6x64xf32>) outs(%1474:tensor<1x6xf32>) dimensions = [2]
    (%1476: f32, %1477: f32) {
      %1478 = arith.addf %1476, %1477 : f32
      linalg.yield %1478 : f32
    }
    %1479 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1480 = tensor.splat %1479 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1481 = tensor.empty() : tensor<1x6xf32>
    %1482 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1475, %1480 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1481 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb135(%1483: f32, %1484: f32, %1485: f32):
      %1486 = arith.divf %1483, %1484 : f32
      linalg.yield %1486 : f32
    } -> tensor<1x6xf32>
    %1487 = tensor.collapse_shape %1482 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1488 = tensor.expand_shape %1487 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1489 = tensor.empty() : tensor<1x6x64xf32>
    %1490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1472, %1488 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1489 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb136(%1491: f32, %1492: f32, %1493: f32):
      %1494 = arith.subf %1491, %1492 : f32
      linalg.yield %1494 : f32
    } -> tensor<1x6x64xf32>
    %1495 = tensor.empty() : tensor<1x6x64xf32>
    %1496 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1490, %1490 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1495 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb137(%1497: f32, %1498: f32, %1499: f32):
      %1500 = arith.mulf %1497, %1498 : f32
      linalg.yield %1500 : f32
    } -> tensor<1x6x64xf32>
    %1501 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1502 = tensor.splat %1501 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1503 = linalg.reduce ins(%1496:tensor<1x6x64xf32>) outs(%1502:tensor<1x6xf32>) dimensions = [2]
    (%1504: f32, %1505: f32) {
      %1506 = arith.addf %1504, %1505 : f32
      linalg.yield %1506 : f32
    }
    %1507 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1508 = tensor.splat %1507 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1509 = tensor.empty() : tensor<1x6xf32>
    %1510 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1503, %1508 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1509 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb138(%1511: f32, %1512: f32, %1513: f32):
      %1514 = arith.divf %1511, %1512 : f32
      linalg.yield %1514 : f32
    } -> tensor<1x6xf32>
    %1515 = tensor.collapse_shape %1510 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1516 = tensor.expand_shape %1515 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1517 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1518 = tensor.splat %1517 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1519 = tensor.empty() : tensor<1x6x1xf32>
    %1520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1516, %1518 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1519 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb139(%1521: f32, %1522: f32, %1523: f32):
      %1524 = arith.addf %1521, %1522 : f32
      linalg.yield %1524 : f32
    } -> tensor<1x6x1xf32>
    %1525 = tensor.empty() : tensor<1x6x1xf32>
    %1526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1520 : tensor<1x6x1xf32>) outs(%1525 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb140(%1527: f32, %1528: f32):
      %1529 = math.rsqrt %1527 : f32
      linalg.yield %1529 : f32
    } -> tensor<1x6x1xf32>
    %1530 = tensor.empty() : tensor<1x6x64xf32>
    %1531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1490, %1526 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1530 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb141(%1532: f32, %1533: f32, %1534: f32):
      %1535 = arith.mulf %1532, %1533 : f32
      linalg.yield %1535 : f32
    } -> tensor<1x6x64xf32>
    %1536 = tensor.empty() : tensor<1x6x64xf32>
    %1537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531, %56 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1536 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb142(%1538: f32, %1539: f32, %1540: f32):
      %1541 = arith.mulf %1538, %1539 : f32
      linalg.yield %1541 : f32
    } -> tensor<1x6x64xf32>
    %1542 = tensor.empty() : tensor<1x6x64xf32>
    %1543 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1537, %57 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1542 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb143(%1544: f32, %1545: f32, %1546: f32):
      %1547 = arith.addf %1544, %1545 : f32
      linalg.yield %1547 : f32
    } -> tensor<1x6x64xf32>
    %1548 = tensor.empty() : tensor<64x128xf32>
    %1549 = linalg.transpose ins(%58:tensor<128x64xf32>) outs(%1548:tensor<64x128xf32>) permutation = [1, 0]
    %1550 = tensor.empty() : tensor<1x6x128xf32>
    %1551 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1552 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1551 : f32) outs(%1550 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1553 = linalg.matmul {prov.region_id = "matmul_21", prov.dispatch_id = "matmul_21", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} ins(%1543, %1549 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1552 : tensor<1x6x128xf32>) -> tensor<1x6x128xf32>
    %1554 = tensor.empty() : tensor<1x6x128xf32>
    %1555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1553, %59 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1554 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb144(%1556: f32, %1557: f32, %1558: f32):
      %1559 = arith.addf %1556, %1557 : f32
      linalg.yield %1559 : f32
    } -> tensor<1x6x128xf32>
    %1560 = tensor.collapse_shape %1555 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1562 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1563 = linalg.transpose ins(%1561:tensor<1x6x2x2x32xf32>) outs(%1562:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1564 = "tensor.extract_slice"(%1563) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1565 = tensor.collapse_shape %1564 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1566 = tensor.expand_shape %1565 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1567 = "tensor.extract_slice"(%1563) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1568 = tensor.collapse_shape %1567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1569 = tensor.expand_shape %1568 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1570 = tensor.empty() : tensor<64x64xf32>
    %1571 = linalg.transpose ins(%60:tensor<64x64xf32>) outs(%1570:tensor<64x64xf32>) permutation = [1, 0]
    %1572 = tensor.empty() : tensor<1x96x64xf32>
    %1573 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1574 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1573 : f32) outs(%1572 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1575 = linalg.matmul {prov.region_id = "matmul_22", prov.dispatch_id = "matmul_22", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} ins(%1439, %1571 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1574 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1576 = tensor.empty() : tensor<1x96x64xf32>
    %1577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1575, %61 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1576 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb145(%1578: f32, %1579: f32, %1580: f32):
      %1581 = arith.addf %1578, %1579 : f32
      linalg.yield %1581 : f32
    } -> tensor<1x96x64xf32>
    %1582 = tensor.collapse_shape %1577 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1583 = tensor.expand_shape %1582 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1584 = tensor.empty() : tensor<1x2x96x32xf32>
    %1585 = linalg.transpose ins(%1583:tensor<1x96x2x32xf32>) outs(%1584:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1586 = tensor.empty() : tensor<1x2x32x6xf32>
    %1587 = linalg.transpose ins(%1566:tensor<1x2x6x32xf32>) outs(%1586:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1588 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1589 = tensor.splat %1588 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1590 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1585, %1587 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1589 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb146(%1591: f32, %1592: f32, %1593: f32):
      %1594 = arith.mulf %1591, %1592 : f32
      %1595 = arith.addf %1593, %1594 : f32
      linalg.yield %1595 : f32
    } -> tensor<1x2x96x6xf32>
    %1596 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1597 = tensor.splat %1596 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1598 = tensor.empty() : tensor<1x2x96x6xf32>
    %1599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1590, %1597 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1598 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb147(%1600: f32, %1601: f32, %1602: f32):
      %1603 = arith.divf %1600, %1601 : f32
      linalg.yield %1603 : f32
    } -> tensor<1x2x96x6xf32>
    %1604 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1605 = tensor.splat %1604 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1606 = linalg.reduce ins(%1599:tensor<1x2x96x6xf32>) outs(%1605:tensor<1x2x96xf32>) dimensions = [3]
    (%1607: f32, %1608: f32) {
      %1609 = arith.maximumf %1607, %1608 : f32
      linalg.yield %1609 : f32
    }
    %1610 = tensor.collapse_shape %1606 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1611 = tensor.expand_shape %1610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1612 = tensor.empty() : tensor<1x2x96x6xf32>
    %1613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1599, %1611 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1612 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb148(%1614: f32, %1615: f32, %1616: f32):
      %1617 = arith.subf %1614, %1615 : f32
      linalg.yield %1617 : f32
    } -> tensor<1x2x96x6xf32>
    %1618 = tensor.empty() : tensor<1x2x96x6xf32>
    %1619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1613 : tensor<1x2x96x6xf32>) outs(%1618 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb149(%1620: f32, %1621: f32):
      %1622 = math.exp %1620 : f32
      linalg.yield %1622 : f32
    } -> tensor<1x2x96x6xf32>
    %1623 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1624 = tensor.splat %1623 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1625 = linalg.reduce ins(%1619:tensor<1x2x96x6xf32>) outs(%1624:tensor<1x2x96xf32>) dimensions = [3]
    (%1626: f32, %1627: f32) {
      %1628 = arith.addf %1626, %1627 : f32
      linalg.yield %1628 : f32
    }
    %1629 = tensor.collapse_shape %1625 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1630 = tensor.expand_shape %1629 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1631 = tensor.empty() : tensor<1x2x96x6xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1619, %1630 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1631 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb150(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.divf %1633, %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<1x2x96x6xf32>
    %1637 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1638 = tensor.splat %1637 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1632, %1569 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1638 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb151(%1640: f32, %1641: f32, %1642: f32):
      %1643 = arith.mulf %1640, %1641 : f32
      %1644 = arith.addf %1642, %1643 : f32
      linalg.yield %1644 : f32
    } -> tensor<1x2x96x32xf32>
    %1645 = tensor.empty() : tensor<1x96x2x32xf32>
    %1646 = linalg.transpose ins(%1639:tensor<1x2x96x32xf32>) outs(%1645:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1647 = tensor.collapse_shape %1646 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1648 = tensor.expand_shape %1647 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1649 = tensor.empty() : tensor<64x64xf32>
    %1650 = linalg.transpose ins(%62:tensor<64x64xf32>) outs(%1649:tensor<64x64xf32>) permutation = [1, 0]
    %1651 = tensor.empty() : tensor<1x96x64xf32>
    %1652 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1653 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1652 : f32) outs(%1651 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1654 = linalg.matmul {prov.region_id = "matmul_25", prov.dispatch_id = "matmul_25", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} ins(%1648, %1650 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1653 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1655 = tensor.empty() : tensor<1x96x64xf32>
    %1656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1654, %63 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1655 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb152(%1657: f32, %1658: f32, %1659: f32):
      %1660 = arith.addf %1657, %1658 : f32
      linalg.yield %1660 : f32
    } -> tensor<1x96x64xf32>
    %1661 = tensor.empty() : tensor<1x96x64xf32>
    %1662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1439, %1656 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1661 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb153(%1663: f32, %1664: f32, %1665: f32):
      %1666 = arith.addf %1663, %1664 : f32
      linalg.yield %1666 : f32
    } -> tensor<1x96x64xf32>
    %1667 = tensor.empty() : tensor<64x512xf32>
    %1668 = linalg.transpose ins(%70:tensor<512x64xf32>) outs(%1667:tensor<64x512xf32>) permutation = [1, 0]
    %1669 = tensor.empty() : tensor<1x96x512xf32>
    %1670 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1671 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1670 : f32) outs(%1669 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1672 = linalg.matmul {prov.region_id = "matmul_26", prov.dispatch_id = "matmul_26", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} ins(%1662, %1668 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%1671 : tensor<1x96x512xf32>) -> tensor<1x96x512xf32>
    %1673 = tensor.empty() : tensor<1x96x512xf32>
    %1674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1672, %71 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1673 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb154(%1675: f32, %1676: f32, %1677: f32):
      %1678 = arith.addf %1675, %1676 : f32
      linalg.yield %1678 : f32
    } -> tensor<1x96x512xf32>
    %1679 = tensor.empty() : tensor<1x512x96xf32>
    %1680 = linalg.transpose ins(%1674:tensor<1x96x512xf32>) outs(%1679:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1681 = tensor.collapse_shape %1680 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1682 = tensor.expand_shape %1681 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1683 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1684 = tensor.splat %1683 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %1685 = "tensor.insert_slice"(%1682, %1684) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1686 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1687 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1685 : tensor<1x512x10x14xf32>) outs(%1686 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb155(%1688: f32, %1689: f32):
      linalg.yield %1688 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1690 = tensor.collapse_shape %1687 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1691 = tensor.expand_shape %1690 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1692 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1693 = tensor.expand_shape %1692 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1694 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1695 = tensor.splat %1694 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %1696 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1693, %1691 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1695 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb156(%1697: f32, %1698: f32, %1699: f32):
      %1700 = arith.mulf %1697, %1698 : f32
      %1701 = arith.addf %1699, %1700 : f32
      linalg.yield %1701 : f32
    } -> tensor<64x8x96xf32>
    %1702 = tensor.collapse_shape %1696 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1703 = tensor.expand_shape %1702 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1704 = tensor.collapse_shape %1703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1706 = tensor.empty() : tensor<1x512x8x12xf32>
    %1707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1705, %73 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1706 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb157(%1708: f32, %1709: f32, %1710: f32):
      %1711 = arith.addf %1708, %1709 : f32
      linalg.yield %1711 : f32
    } -> tensor<1x512x8x12xf32>
    %1712 = tensor.collapse_shape %1707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1713 = tensor.expand_shape %1712 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1714 = tensor.empty() : tensor<1x96x512xf32>
    %1715 = linalg.transpose ins(%1713:tensor<1x512x96xf32>) outs(%1714:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1716 = tensor.empty() : tensor<1x96x512xf32>
    %1717 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1715 : tensor<1x96x512xf32>) outs(%1716 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb158(%1718: f32, %1719: f32):
      %1720 = arith.constant 5.000000e-01 : f32
      %1721 = arith.constant 1.000000e+00 : f32
      %1722 = arith.constant 0.707106769 : f32
      %1723 = arith.mulf %1718, %1722 : f32
      %1724 = math.erf %1723 : f32
      %1725 = arith.addf %1721, %1724 : f32
      %1726 = arith.mulf %1720, %1718 : f32
      %1727 = arith.mulf %1726, %1725 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x96x512xf32>
    %1728 = tensor.empty() : tensor<512x64xf32>
    %1729 = linalg.transpose ins(%74:tensor<64x512xf32>) outs(%1728:tensor<512x64xf32>) permutation = [1, 0]
    %1730 = tensor.empty() : tensor<1x96x64xf32>
    %1731 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1732 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1731 : f32) outs(%1730 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1733 = linalg.matmul {prov.region_id = "matmul_27", prov.dispatch_id = "matmul_27", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} ins(%1717, %1729 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%1732 : tensor<1x96x64xf32>) -> tensor<1x96x64xf32>
    %1734 = tensor.empty() : tensor<1x96x64xf32>
    %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1733, %75 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1734 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb159(%1736: f32, %1737: f32, %1738: f32):
      %1739 = arith.addf %1736, %1737 : f32
      linalg.yield %1739 : f32
    } -> tensor<1x96x64xf32>
    %1740 = tensor.empty() : tensor<1x96x64xf32>
    %1741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1662, %1735 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1740 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%1742: f32, %1743: f32, %1744: f32):
      %1745 = arith.addf %1742, %1743 : f32
      linalg.yield %1745 : f32
    } -> tensor<1x96x64xf32>
    %1746 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1747 = tensor.splat %1746 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1748 = linalg.reduce ins(%1741:tensor<1x96x64xf32>) outs(%1747:tensor<1x96xf32>) dimensions = [2]
    (%1749: f32, %1750: f32) {
      %1751 = arith.addf %1749, %1750 : f32
      linalg.yield %1751 : f32
    }
    %1752 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1753 = tensor.splat %1752 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1754 = tensor.empty() : tensor<1x96xf32>
    %1755 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1748, %1753 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1754 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%1756: f32, %1757: f32, %1758: f32):
      %1759 = arith.divf %1756, %1757 : f32
      linalg.yield %1759 : f32
    } -> tensor<1x96xf32>
    %1760 = tensor.collapse_shape %1755 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1761 = tensor.expand_shape %1760 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1762 = tensor.empty() : tensor<1x96x64xf32>
    %1763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1741, %1761 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1762 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb162(%1764: f32, %1765: f32, %1766: f32):
      %1767 = arith.subf %1764, %1765 : f32
      linalg.yield %1767 : f32
    } -> tensor<1x96x64xf32>
    %1768 = tensor.empty() : tensor<1x96x64xf32>
    %1769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763, %1763 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1768 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb163(%1770: f32, %1771: f32, %1772: f32):
      %1773 = arith.mulf %1770, %1771 : f32
      linalg.yield %1773 : f32
    } -> tensor<1x96x64xf32>
    %1774 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1775 = tensor.splat %1774 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1776 = linalg.reduce ins(%1769:tensor<1x96x64xf32>) outs(%1775:tensor<1x96xf32>) dimensions = [2]
    (%1777: f32, %1778: f32) {
      %1779 = arith.addf %1777, %1778 : f32
      linalg.yield %1779 : f32
    }
    %1780 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1781 = tensor.splat %1780 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1782 = tensor.empty() : tensor<1x96xf32>
    %1783 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1776, %1781 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1782 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb164(%1784: f32, %1785: f32, %1786: f32):
      %1787 = arith.divf %1784, %1785 : f32
      linalg.yield %1787 : f32
    } -> tensor<1x96xf32>
    %1788 = tensor.collapse_shape %1783 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1789 = tensor.expand_shape %1788 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1790 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1791 = tensor.splat %1790 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1792 = tensor.empty() : tensor<1x96x1xf32>
    %1793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1789, %1791 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1792 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb165(%1794: f32, %1795: f32, %1796: f32):
      %1797 = arith.addf %1794, %1795 : f32
      linalg.yield %1797 : f32
    } -> tensor<1x96x1xf32>
    %1798 = tensor.empty() : tensor<1x96x1xf32>
    %1799 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1793 : tensor<1x96x1xf32>) outs(%1798 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb166(%1800: f32, %1801: f32):
      %1802 = math.rsqrt %1800 : f32
      linalg.yield %1802 : f32
    } -> tensor<1x96x1xf32>
    %1803 = tensor.empty() : tensor<1x96x64xf32>
    %1804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763, %1799 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1803 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb167(%1805: f32, %1806: f32, %1807: f32):
      %1808 = arith.mulf %1805, %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<1x96x64xf32>
    %1809 = tensor.empty() : tensor<1x96x64xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804, %78 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1809 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb168(%1811: f32, %1812: f32, %1813: f32):
      %1814 = arith.mulf %1811, %1812 : f32
      linalg.yield %1814 : f32
    } -> tensor<1x96x64xf32>
    %1815 = tensor.empty() : tensor<1x96x64xf32>
    %1816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1810, %79 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1815 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb169(%1817: f32, %1818: f32, %1819: f32):
      %1820 = arith.addf %1817, %1818 : f32
      linalg.yield %1820 : f32
    } -> tensor<1x96x64xf32>
    %1821 = tensor.collapse_shape %1816 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1822 = tensor.expand_shape %1821 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %1823 = tensor.empty() : tensor<1x64x8x12xf32>
    %1824 = linalg.transpose ins(%1822:tensor<1x8x12x64xf32>) outs(%1823:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %1825 = tensor.collapse_shape %1824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1826 = tensor.expand_shape %1825 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %1827 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %1828 = linalg.transpose ins(%1826:tensor<1x16x2x2x8x12xf32>) outs(%1827:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %1829 = tensor.collapse_shape %1828 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %1830 = tensor.expand_shape %1829 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %1831 = tensor.empty() : tensor<1x32x23x15xf32>
    %1832 = linalg.transpose ins(%963:tensor<1x32x15x23xf32>) outs(%1831:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %1833 = tensor.collapse_shape %1832 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %1834 = tensor.expand_shape %1833 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %1835 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %1836 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %1837 = tensor.splat %1836 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %1838 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%1834, %1835 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%1837 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %1839 = tensor.collapse_shape %1838 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %1840 = tensor.expand_shape %1839 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %1841 = tensor.empty() : tensor<1x32x16x23xf32>
    %1842 = linalg.transpose ins(%1840:tensor<1x32x23x16xf32>) outs(%1841:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %1843 = tensor.collapse_shape %1842 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %1844 = tensor.expand_shape %1843 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %1845 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %1846 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %1847 = tensor.splat %1846 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %1848 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%1844, %1845 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%1847 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %1849 = tensor.collapse_shape %1848 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %1850 = tensor.expand_shape %1849 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %1851 = tensor.concat dim(1) %1830, %1850 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %1852 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %1853 = tensor.splat %1852 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %1854 = "tensor.insert_slice"(%1851, %1853) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %1855 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %1856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1854 : tensor<1x48x18x26xf32>) outs(%1855 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb170(%1857: f32, %1858: f32):
      linalg.yield %1857 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %1859 = tensor.collapse_shape %1856 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %1860 = tensor.expand_shape %1859 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %1861 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %1862 = tensor.expand_shape %1861 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %1863 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %1864 = tensor.splat %1863 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %1865 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%1862, %1860 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%1864 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %1866 = tensor.collapse_shape %1865 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %1867 = tensor.expand_shape %1866 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %1868 = tensor.collapse_shape %1867 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %1869 = tensor.expand_shape %1868 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %1870 = tensor.empty() : tensor<1x12x16x24xf32>
    %1871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1869, %97 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%1870 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb171(%1872: f32, %1873: f32, %1874: f32):
      %1875 = arith.addf %1872, %1873 : f32
      linalg.yield %1875 : f32
    } -> tensor<1x12x16x24xf32>
    %1876 = tensor.collapse_shape %1871 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %1877 = tensor.expand_shape %1876 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %1878 = tensor.empty() : tensor<4608x512xf32>
    %1879 = linalg.transpose ins(%81:tensor<512x4608xf32>) outs(%1878:tensor<4608x512xf32>) permutation = [1, 0]
    %1880 = tensor.empty() : tensor<1x512xf32>
    %1881 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1882 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1881 : f32) outs(%1880 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1883 = linalg.matmul {prov.region_id = "matmul_28", prov.dispatch_id = "matmul_28", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%1877, %1879 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%1882 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1884 = tensor.empty() : tensor<1x512xf32>
    %1885 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1883, %80 : tensor<1x512xf32>, tensor<512xf32>) outs(%1884 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb172(%1886: f32, %1887: f32, %1888: f32):
      %1889 = arith.addf %1886, %1887 : f32
      linalg.yield %1889 : f32
    } -> tensor<1x512xf32>
    %1890 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %1891 = tensor.splat %1890 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %1892 = tensor.empty() : tensor<1x1xf32>
    %1893 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %1891 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1892 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb173(%1894: f32, %1895: f32, %1896: f32):
      %1897 = arith.divf %1894, %1895 : f32
      linalg.yield %1897 : f32
    } -> tensor<1x1xf32>
    %1898 = tensor.concat dim(1) %1885, %1893, %100 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %1899 = tensor.collapse_shape %1898 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %1900 = tensor.expand_shape %1899 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %1901 = tensor.collapse_shape %101 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %1902 = tensor.expand_shape %1901 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %1903 = tensor.collapse_shape %102 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %1904 = tensor.expand_shape %1903 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %1905 = "tensor.extract_slice"(%1900) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 517>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x517xf32>) -> tensor<1x1x517xf32>
    %1906 = tensor.collapse_shape %1905 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %1907 = tensor.expand_shape %1906 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %1908 = "tensor.extract_slice"(%1902) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %1909 = tensor.collapse_shape %1908 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %1910 = tensor.expand_shape %1909 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %1911 = "tensor.extract_slice"(%1904) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %1912 = tensor.collapse_shape %1911 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %1913 = tensor.expand_shape %1912 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %1914 = tensor.empty() : tensor<517x512xf32>
    %1915 = linalg.transpose ins(%82:tensor<512x517xf32>) outs(%1914:tensor<517x512xf32>) permutation = [1, 0]
    %1916 = tensor.empty() : tensor<1x512xf32>
    %1917 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1918 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1917 : f32) outs(%1916 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1919 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%1907, %1915 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%1918 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1920 = tensor.empty() : tensor<128x512xf32>
    %1921 = linalg.transpose ins(%83:tensor<512x128xf32>) outs(%1920:tensor<128x512xf32>) permutation = [1, 0]
    %1922 = tensor.empty() : tensor<1x512xf32>
    %1923 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1924 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1923 : f32) outs(%1922 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1925 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%1910, %1921 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%1924 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1926 = tensor.empty() : tensor<1x512xf32>
    %1927 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1919, %1925, %84, %85 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%1926 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb174(%1928: f32, %1929: f32, %1930: f32, %1931: f32, %1932: f32):
      %1933 = arith.addf %1928, %1929 : f32
      %1934 = arith.addf %1933, %1930 : f32
      %1935 = arith.addf %1934, %1931 : f32
      linalg.yield %1935 : f32
    } -> tensor<1x512xf32>
    %1936 = "tensor.extract_slice"(%1927) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %1937 = "tensor.extract_slice"(%1927) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %1938 = "tensor.extract_slice"(%1927) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %1939 = "tensor.extract_slice"(%1927) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %1940 = tensor.empty() : tensor<1x128xf32>
    %1941 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1936, %1937, %1938, %1913 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%1940 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb175(%1942: f32, %1943: f32, %1944: f32, %1945: f32, %1946: f32):
      %1947 = arith.constant 1.000000e+00 : f32
      %1948 = arith.negf %1943 : f32
      %1949 = math.exp %1948 : f32
      %1950 = arith.addf %1947, %1949 : f32
      %1951 = arith.divf %1947, %1950 : f32
      %1952 = arith.constant 1.000000e+00 : f32
      %1953 = arith.negf %1942 : f32
      %1954 = math.exp %1953 : f32
      %1955 = arith.addf %1952, %1954 : f32
      %1956 = arith.divf %1952, %1955 : f32
      %1957 = math.tanh %1944 : f32
      %1958 = arith.mulf %1951, %1945 : f32
      %1959 = arith.mulf %1956, %1957 : f32
      %1960 = arith.addf %1958, %1959 : f32
      linalg.yield %1960 : f32
    } -> tensor<1x128xf32>
    %1961 = tensor.empty() : tensor<1x128xf32>
    %1962 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1939, %1941 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%1961 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb176(%1963: f32, %1964: f32, %1965: f32):
      %1966 = arith.constant 1.000000e+00 : f32
      %1967 = arith.negf %1963 : f32
      %1968 = math.exp %1967 : f32
      %1969 = arith.addf %1966, %1968 : f32
      %1970 = arith.divf %1966, %1969 : f32
      %1971 = math.tanh %1964 : f32
      %1972 = arith.mulf %1970, %1971 : f32
      linalg.yield %1972 : f32
    } -> tensor<1x128xf32>
    %1973 = "tensor.extract_slice"(%1902) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %1974 = tensor.collapse_shape %1973 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %1975 = tensor.expand_shape %1974 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %1976 = "tensor.extract_slice"(%1904) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %1977 = tensor.collapse_shape %1976 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %1978 = tensor.expand_shape %1977 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %1979 = tensor.empty() : tensor<128x512xf32>
    %1980 = linalg.transpose ins(%86:tensor<512x128xf32>) outs(%1979:tensor<128x512xf32>) permutation = [1, 0]
    %1981 = tensor.empty() : tensor<1x512xf32>
    %1982 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1983 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1982 : f32) outs(%1981 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1984 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%1962, %1980 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%1983 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1985 = tensor.empty() : tensor<128x512xf32>
    %1986 = linalg.transpose ins(%87:tensor<512x128xf32>) outs(%1985:tensor<128x512xf32>) permutation = [1, 0]
    %1987 = tensor.empty() : tensor<1x512xf32>
    %1988 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1989 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1988 : f32) outs(%1987 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1990 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%1975, %1986 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%1989 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %1991 = tensor.empty() : tensor<1x512xf32>
    %1992 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1984, %1990, %88, %89 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%1991 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb177(%1993: f32, %1994: f32, %1995: f32, %1996: f32, %1997: f32):
      %1998 = arith.addf %1993, %1994 : f32
      %1999 = arith.addf %1998, %1995 : f32
      %2000 = arith.addf %1999, %1996 : f32
      linalg.yield %2000 : f32
    } -> tensor<1x512xf32>
    %2001 = "tensor.extract_slice"(%1992) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2002 = "tensor.extract_slice"(%1992) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2003 = "tensor.extract_slice"(%1992) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2004 = "tensor.extract_slice"(%1992) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2005 = tensor.empty() : tensor<1x128xf32>
    %2006 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2001, %2002, %2003, %1978 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2005 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb178(%2007: f32, %2008: f32, %2009: f32, %2010: f32, %2011: f32):
      %2012 = arith.constant 1.000000e+00 : f32
      %2013 = arith.negf %2008 : f32
      %2014 = math.exp %2013 : f32
      %2015 = arith.addf %2012, %2014 : f32
      %2016 = arith.divf %2012, %2015 : f32
      %2017 = arith.constant 1.000000e+00 : f32
      %2018 = arith.negf %2007 : f32
      %2019 = math.exp %2018 : f32
      %2020 = arith.addf %2017, %2019 : f32
      %2021 = arith.divf %2017, %2020 : f32
      %2022 = math.tanh %2009 : f32
      %2023 = arith.mulf %2016, %2010 : f32
      %2024 = arith.mulf %2021, %2022 : f32
      %2025 = arith.addf %2023, %2024 : f32
      linalg.yield %2025 : f32
    } -> tensor<1x128xf32>
    %2026 = tensor.empty() : tensor<1x128xf32>
    %2027 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2004, %2006 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2026 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb179(%2028: f32, %2029: f32, %2030: f32):
      %2031 = arith.constant 1.000000e+00 : f32
      %2032 = arith.negf %2028 : f32
      %2033 = math.exp %2032 : f32
      %2034 = arith.addf %2031, %2033 : f32
      %2035 = arith.divf %2031, %2034 : f32
      %2036 = math.tanh %2029 : f32
      %2037 = arith.mulf %2035, %2036 : f32
      linalg.yield %2037 : f32
    } -> tensor<1x128xf32>
    %2038 = "tensor.extract_slice"(%1902) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2039 = tensor.collapse_shape %2038 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2040 = tensor.expand_shape %2039 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2041 = "tensor.extract_slice"(%1904) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2042 = tensor.collapse_shape %2041 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2043 = tensor.expand_shape %2042 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2044 = tensor.empty() : tensor<128x512xf32>
    %2045 = linalg.transpose ins(%90:tensor<512x128xf32>) outs(%2044:tensor<128x512xf32>) permutation = [1, 0]
    %2046 = tensor.empty() : tensor<1x512xf32>
    %2047 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2048 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2047 : f32) outs(%2046 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2049 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2027, %2045 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2048 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2050 = tensor.empty() : tensor<128x512xf32>
    %2051 = linalg.transpose ins(%91:tensor<512x128xf32>) outs(%2050:tensor<128x512xf32>) permutation = [1, 0]
    %2052 = tensor.empty() : tensor<1x512xf32>
    %2053 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2054 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2053 : f32) outs(%2052 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2055 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2040, %2051 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2054 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2056 = tensor.empty() : tensor<1x512xf32>
    %2057 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2049, %2055, %92, %93 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2056 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb180(%2058: f32, %2059: f32, %2060: f32, %2061: f32, %2062: f32):
      %2063 = arith.addf %2058, %2059 : f32
      %2064 = arith.addf %2063, %2060 : f32
      %2065 = arith.addf %2064, %2061 : f32
      linalg.yield %2065 : f32
    } -> tensor<1x512xf32>
    %2066 = "tensor.extract_slice"(%2057) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2067 = "tensor.extract_slice"(%2057) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2068 = "tensor.extract_slice"(%2057) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2069 = "tensor.extract_slice"(%2057) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2070 = tensor.empty() : tensor<1x128xf32>
    %2071 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2066, %2067, %2068, %2043 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2070 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb181(%2072: f32, %2073: f32, %2074: f32, %2075: f32, %2076: f32):
      %2077 = arith.constant 1.000000e+00 : f32
      %2078 = arith.negf %2073 : f32
      %2079 = math.exp %2078 : f32
      %2080 = arith.addf %2077, %2079 : f32
      %2081 = arith.divf %2077, %2080 : f32
      %2082 = arith.constant 1.000000e+00 : f32
      %2083 = arith.negf %2072 : f32
      %2084 = math.exp %2083 : f32
      %2085 = arith.addf %2082, %2084 : f32
      %2086 = arith.divf %2082, %2085 : f32
      %2087 = math.tanh %2074 : f32
      %2088 = arith.mulf %2081, %2075 : f32
      %2089 = arith.mulf %2086, %2087 : f32
      %2090 = arith.addf %2088, %2089 : f32
      linalg.yield %2090 : f32
    } -> tensor<1x128xf32>
    %2091 = tensor.empty() : tensor<1x128xf32>
    %2092 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2069, %2071 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2091 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb182(%2093: f32, %2094: f32, %2095: f32):
      %2096 = arith.constant 1.000000e+00 : f32
      %2097 = arith.negf %2093 : f32
      %2098 = math.exp %2097 : f32
      %2099 = arith.addf %2096, %2098 : f32
      %2100 = arith.divf %2096, %2099 : f32
      %2101 = math.tanh %2094 : f32
      %2102 = arith.mulf %2100, %2101 : f32
      linalg.yield %2102 : f32
    } -> tensor<1x128xf32>
    %2103 = tensor.empty() : tensor<1x1x128xf32>
    %2104 = tensor.collapse_shape %2092 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2105 = tensor.expand_shape %2104 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2106 = "tensor.insert_slice"(%2105, %2103) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2107 = tensor.empty() : tensor<3x1x128xf32>
    %2108 = tensor.collapse_shape %1962 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2109 = tensor.expand_shape %2108 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2110 = "tensor.insert_slice"(%2109, %2107) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2111 = tensor.collapse_shape %2027 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2112 = tensor.expand_shape %2111 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2113 = "tensor.insert_slice"(%2112, %2110) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2114 = tensor.collapse_shape %2092 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2115 = tensor.expand_shape %2114 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2116 = "tensor.insert_slice"(%2115, %2113) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2117 = tensor.empty() : tensor<3x1x128xf32>
    %2118 = tensor.collapse_shape %1941 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2119 = tensor.expand_shape %2118 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2120 = "tensor.insert_slice"(%2119, %2117) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2121 = tensor.collapse_shape %2006 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2122 = tensor.expand_shape %2121 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2123 = "tensor.insert_slice"(%2122, %2120) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2124 = tensor.collapse_shape %2071 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2125 = tensor.expand_shape %2124 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2126 = "tensor.insert_slice"(%2125, %2123) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2127 = tensor.collapse_shape %2106 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2128 = tensor.expand_shape %2127 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2129 = tensor.collapse_shape %2116 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2130 = tensor.expand_shape %2129 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2131 = tensor.collapse_shape %2126 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2132 = tensor.expand_shape %2131 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2133 = tensor.empty() : tensor<128x3xf32>
    %2134 = linalg.transpose ins(%95:tensor<3x128xf32>) outs(%2133:tensor<128x3xf32>) permutation = [1, 0]
    %2135 = tensor.empty() : tensor<1x3xf32>
    %2136 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2137 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2136 : f32) outs(%2135 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2138 = linalg.matmul {prov.region_id = "matmul_29", prov.dispatch_id = "matmul_29", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2128, %2134 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2137 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2139 = tensor.empty() : tensor<1x3xf32>
    %2140 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2138, %94 : tensor<1x3xf32>, tensor<3xf32>) outs(%2139 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_29", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb183(%2141: f32, %2142: f32, %2143: f32):
      %2144 = arith.addf %2141, %2142 : f32
      linalg.yield %2144 : f32
    } -> tensor<1x3xf32>
    func.return %2140, %2130, %2132 : tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>
  }
}
