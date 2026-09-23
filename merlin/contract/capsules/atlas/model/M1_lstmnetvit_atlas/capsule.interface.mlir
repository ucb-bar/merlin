builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "float8_weight_only_e4m3"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64x32xf32>, %9: tensor<64xf32>, %10: tensor<32x32xf32>, %11: tensor<32xf32>, %12: tensor<32x32xf32>, %13: tensor<32xf32>, %14: tensor<32x32x8x8xf32>, %15: tensor<32xf32>, %16: tensor<32xf32>, %17: tensor<32xf32>, %18: tensor<64x32xf32>, %19: tensor<64xf32>, %20: tensor<32x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32xf32>, %24: tensor<256x32xf32>, %25: tensor<256xf32>, %26: tensor<256x8x3x3xf32>, %27: tensor<256xf32>, %28: tensor<32x256xf32>, %29: tensor<32xf32>, %30: tensor<256x32xf32>, %31: tensor<256xf32>, %32: tensor<256x8x3x3xf32>, %33: tensor<256xf32>, %34: tensor<32x256xf32>, %35: tensor<32xf32>, %36: tensor<32xf32>, %37: tensor<32xf32>, %38: tensor<32xf32>, %39: tensor<32xf32>, %40: tensor<64x32x3x3xf32>, %41: tensor<64xf32>, %42: tensor<64xf32>, %43: tensor<64xf32>, %44: tensor<64x64x4x4xf32>, %45: tensor<64xf32>, %46: tensor<64xf32>, %47: tensor<64xf32>, %48: tensor<128x64xf32>, %49: tensor<128xf32>, %50: tensor<64x64xf32>, %51: tensor<64xf32>, %52: tensor<64x64xf32>, %53: tensor<64xf32>, %54: tensor<64x64x4x4xf32>, %55: tensor<64xf32>, %56: tensor<64xf32>, %57: tensor<64xf32>, %58: tensor<128x64xf32>, %59: tensor<128xf32>, %60: tensor<64x64xf32>, %61: tensor<64xf32>, %62: tensor<64x64xf32>, %63: tensor<64xf32>, %64: tensor<512x64xf32>, %65: tensor<512xf32>, %66: tensor<512x8x3x3xf32>, %67: tensor<512xf32>, %68: tensor<64x512xf32>, %69: tensor<64xf32>, %70: tensor<512x64xf32>, %71: tensor<512xf32>, %72: tensor<512x8x3x3xf32>, %73: tensor<512xf32>, %74: tensor<64x512xf32>, %75: tensor<64xf32>, %76: tensor<64xf32>, %77: tensor<64xf32>, %78: tensor<64xf32>, %79: tensor<64xf32>, %80: tensor<512xf32>, %81: tensor<512x4608xf32>, %82: tensor<512x517xf32>, %83: tensor<512x128xf32>, %84: tensor<512xf32>, %85: tensor<512xf32>, %86: tensor<512x128xf32>, %87: tensor<512x128xf32>, %88: tensor<512xf32>, %89: tensor<512xf32>, %90: tensor<512x128xf32>, %91: tensor<512x128xf32>, %92: tensor<512xf32>, %93: tensor<512xf32>, %94: tensor<3xf32>, %95: tensor<3x128xf32>, %96: tensor<12x48x3x3xf32>, %97: tensor<12xf32>, %98: tensor<1x1x60x90xf32>, %99: tensor<1x1xf32>, %100: tensor<1x4xf32>, %101: tensor<3x128xf32>, %102: tensor<3x128xf32>) -> (tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>) {
    %103 = tensor.empty() : tensor<64x32xf32>
    %104 = tensor.empty() : tensor<64x1xf32>
    %105 = tensor.empty() : tensor<32x32xf32>
    %106 = tensor.empty() : tensor<32x1xf32>
    %107 = tensor.empty() : tensor<32x32xf32>
    %108 = tensor.empty() : tensor<32x1xf32>
    %109 = tensor.empty() : tensor<64x32xf32>
    %110 = tensor.empty() : tensor<64x1xf32>
    %111 = tensor.empty() : tensor<32x32xf32>
    %112 = tensor.empty() : tensor<32x1xf32>
    %113 = tensor.empty() : tensor<32x32xf32>
    %114 = tensor.empty() : tensor<32x1xf32>
    %115 = tensor.empty() : tensor<256x32xf32>
    %116 = tensor.empty() : tensor<256x1xf32>
    %117 = tensor.empty() : tensor<32x256xf32>
    %118 = tensor.empty() : tensor<32x1xf32>
    %119 = tensor.empty() : tensor<256x32xf32>
    %120 = tensor.empty() : tensor<256x1xf32>
    %121 = tensor.empty() : tensor<32x256xf32>
    %122 = tensor.empty() : tensor<32x1xf32>
    %123 = tensor.empty() : tensor<128x64xf32>
    %124 = tensor.empty() : tensor<128x1xf32>
    %125 = tensor.empty() : tensor<64x64xf32>
    %126 = tensor.empty() : tensor<64x1xf32>
    %127 = tensor.empty() : tensor<64x64xf32>
    %128 = tensor.empty() : tensor<64x1xf32>
    %129 = tensor.empty() : tensor<128x64xf32>
    %130 = tensor.empty() : tensor<128x1xf32>
    %131 = tensor.empty() : tensor<64x64xf32>
    %132 = tensor.empty() : tensor<64x1xf32>
    %133 = tensor.empty() : tensor<64x64xf32>
    %134 = tensor.empty() : tensor<64x1xf32>
    %135 = tensor.empty() : tensor<512x64xf32>
    %136 = tensor.empty() : tensor<512x1xf32>
    %137 = tensor.empty() : tensor<64x512xf32>
    %138 = tensor.empty() : tensor<64x1xf32>
    %139 = tensor.empty() : tensor<512x64xf32>
    %140 = tensor.empty() : tensor<512x1xf32>
    %141 = tensor.empty() : tensor<64x512xf32>
    %142 = tensor.empty() : tensor<64x1xf32>
    %143 = tensor.empty() : tensor<512x4608xf32>
    %144 = tensor.empty() : tensor<512x1xf32>
    %145 = tensor.empty() : tensor<3x128xf32>
    %146 = tensor.empty() : tensor<3x1xf32>
    %147 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %148 = tensor.splat %147 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %149 = "tensor.insert_slice"(%98, %148) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %150 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%149 : tensor<1x1x66x96xf32>) outs(%150 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%152: f32, %153: f32):
      linalg.yield %152 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %154 = tensor.collapse_shape %151 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %155 = tensor.expand_shape %154 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %156 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %157 = tensor.expand_shape %156 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %158 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %159 = tensor.splat %158 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %160 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%157, %155 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%159 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %161 = tensor.collapse_shape %160 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %162 = tensor.expand_shape %161 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %163 = tensor.collapse_shape %162 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %164 = tensor.expand_shape %163 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %165 = tensor.empty() : tensor<1x32x15x23xf32>
    %166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%164, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%165 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%167: f32, %168: f32, %169: f32):
      %170 = arith.addf %167, %168 : f32
      linalg.yield %170 : f32
    } -> tensor<1x32x15x23xf32>
    %171 = tensor.collapse_shape %166 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %172 = tensor.expand_shape %171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %173 = tensor.empty() : tensor<1x345x32xf32>
    %174 = linalg.transpose ins(%172:tensor<1x32x345xf32>) outs(%173:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %175 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %176 = tensor.splat %175 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %177 = linalg.reduce ins(%174:tensor<1x345x32xf32>) outs(%176:tensor<1x345xf32>) dimensions = [2]
    (%178: f32, %179: f32) {
      %180 = arith.addf %178, %179 : f32
      linalg.yield %180 : f32
    }
    %181 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %182 = tensor.splat %181 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %183 = tensor.empty() : tensor<1x345xf32>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%177, %182 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%183 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%185: f32, %186: f32, %187: f32):
      %188 = arith.divf %185, %186 : f32
      linalg.yield %188 : f32
    } -> tensor<1x345xf32>
    %189 = tensor.collapse_shape %184 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %190 = tensor.expand_shape %189 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %191 = tensor.empty() : tensor<1x345x32xf32>
    %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174, %190 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%191 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%193: f32, %194: f32, %195: f32):
      %196 = arith.subf %193, %194 : f32
      linalg.yield %196 : f32
    } -> tensor<1x345x32xf32>
    %197 = tensor.empty() : tensor<1x345x32xf32>
    %198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %192 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%197 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%199: f32, %200: f32, %201: f32):
      %202 = arith.mulf %199, %200 : f32
      linalg.yield %202 : f32
    } -> tensor<1x345x32xf32>
    %203 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %204 = tensor.splat %203 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %205 = linalg.reduce ins(%198:tensor<1x345x32xf32>) outs(%204:tensor<1x345xf32>) dimensions = [2]
    (%206: f32, %207: f32) {
      %208 = arith.addf %206, %207 : f32
      linalg.yield %208 : f32
    }
    %209 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %210 = tensor.splat %209 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %211 = tensor.empty() : tensor<1x345xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%205, %210 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%211 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%213: f32, %214: f32, %215: f32):
      %216 = arith.divf %213, %214 : f32
      linalg.yield %216 : f32
    } -> tensor<1x345xf32>
    %217 = tensor.collapse_shape %212 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %218 = tensor.expand_shape %217 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %219 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %220 = tensor.splat %219 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %221 = tensor.empty() : tensor<1x345x1xf32>
    %222 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%218, %220 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%221 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%223: f32, %224: f32, %225: f32):
      %226 = arith.addf %223, %224 : f32
      linalg.yield %226 : f32
    } -> tensor<1x345x1xf32>
    %227 = tensor.empty() : tensor<1x345x1xf32>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222 : tensor<1x345x1xf32>) outs(%227 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%229: f32, %230: f32):
      %231 = math.rsqrt %229 : f32
      linalg.yield %231 : f32
    } -> tensor<1x345x1xf32>
    %232 = tensor.empty() : tensor<1x345x32xf32>
    %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %228 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%232 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%234: f32, %235: f32, %236: f32):
      %237 = arith.mulf %234, %235 : f32
      linalg.yield %237 : f32
    } -> tensor<1x345x32xf32>
    %238 = tensor.empty() : tensor<1x345x32xf32>
    %239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%233, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%238 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%240: f32, %241: f32, %242: f32):
      %243 = arith.mulf %240, %241 : f32
      linalg.yield %243 : f32
    } -> tensor<1x345x32xf32>
    %244 = tensor.empty() : tensor<1x345x32xf32>
    %245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%239, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%244 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%246: f32, %247: f32, %248: f32):
      %249 = arith.addf %246, %247 : f32
      linalg.yield %249 : f32
    } -> tensor<1x345x32xf32>
    %250 = tensor.empty() : tensor<1x32x345xf32>
    %251 = linalg.transpose ins(%245:tensor<1x345x32xf32>) outs(%250:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %252 = tensor.collapse_shape %251 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %253 = tensor.expand_shape %252 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %254 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%253 : tensor<1x32x15x23xf32>) outs(%254 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%256: f32, %257: f32):
      linalg.yield %256 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %258 = tensor.collapse_shape %255 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %259 = tensor.expand_shape %258 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %260 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %261 = tensor.expand_shape %260 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %262 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %263 = tensor.splat %262 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %264 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%261, %259 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%263 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %265 = tensor.collapse_shape %264 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %266 = tensor.expand_shape %265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %267 = tensor.collapse_shape %266 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %268 = tensor.expand_shape %267 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %269 = tensor.empty() : tensor<1x32x1x2xf32>
    %270 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%269 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%271: f32, %272: f32, %273: f32):
      %274 = arith.addf %271, %272 : f32
      linalg.yield %274 : f32
    } -> tensor<1x32x1x2xf32>
    %275 = tensor.collapse_shape %270 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %276 = tensor.expand_shape %275 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %277 = tensor.empty() : tensor<1x2x32xf32>
    %278 = linalg.transpose ins(%276:tensor<1x32x2xf32>) outs(%277:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %279 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %280 = tensor.splat %279 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %281 = linalg.reduce ins(%278:tensor<1x2x32xf32>) outs(%280:tensor<1x2xf32>) dimensions = [2]
    (%282: f32, %283: f32) {
      %284 = arith.addf %282, %283 : f32
      linalg.yield %284 : f32
    }
    %285 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %286 = tensor.splat %285 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %287 = tensor.empty() : tensor<1x2xf32>
    %288 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%281, %286 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%287 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%289: f32, %290: f32, %291: f32):
      %292 = arith.divf %289, %290 : f32
      linalg.yield %292 : f32
    } -> tensor<1x2xf32>
    %293 = tensor.collapse_shape %288 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %294 = tensor.expand_shape %293 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %295 = tensor.empty() : tensor<1x2x32xf32>
    %296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278, %294 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%295 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%297: f32, %298: f32, %299: f32):
      %300 = arith.subf %297, %298 : f32
      linalg.yield %300 : f32
    } -> tensor<1x2x32xf32>
    %301 = tensor.empty() : tensor<1x2x32xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%296, %296 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%301 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%303: f32, %304: f32, %305: f32):
      %306 = arith.mulf %303, %304 : f32
      linalg.yield %306 : f32
    } -> tensor<1x2x32xf32>
    %307 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %308 = tensor.splat %307 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %309 = linalg.reduce ins(%302:tensor<1x2x32xf32>) outs(%308:tensor<1x2xf32>) dimensions = [2]
    (%310: f32, %311: f32) {
      %312 = arith.addf %310, %311 : f32
      linalg.yield %312 : f32
    }
    %313 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %314 = tensor.splat %313 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %315 = tensor.empty() : tensor<1x2xf32>
    %316 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%309, %314 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%315 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%317: f32, %318: f32, %319: f32):
      %320 = arith.divf %317, %318 : f32
      linalg.yield %320 : f32
    } -> tensor<1x2xf32>
    %321 = tensor.collapse_shape %316 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %322 = tensor.expand_shape %321 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %323 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %324 = tensor.splat %323 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %325 = tensor.empty() : tensor<1x2x1xf32>
    %326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322, %324 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%325 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%327: f32, %328: f32, %329: f32):
      %330 = arith.addf %327, %328 : f32
      linalg.yield %330 : f32
    } -> tensor<1x2x1xf32>
    %331 = tensor.empty() : tensor<1x2x1xf32>
    %332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%326 : tensor<1x2x1xf32>) outs(%331 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%333: f32, %334: f32):
      %335 = math.rsqrt %333 : f32
      linalg.yield %335 : f32
    } -> tensor<1x2x1xf32>
    %336 = tensor.empty() : tensor<1x2x32xf32>
    %337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%296, %332 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%336 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%338: f32, %339: f32, %340: f32):
      %341 = arith.mulf %338, %339 : f32
      linalg.yield %341 : f32
    } -> tensor<1x2x32xf32>
    %342 = tensor.empty() : tensor<1x2x32xf32>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%342 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%344: f32, %345: f32, %346: f32):
      %347 = arith.mulf %344, %345 : f32
      linalg.yield %347 : f32
    } -> tensor<1x2x32xf32>
    %348 = tensor.empty() : tensor<1x2x32xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%343, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%348 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%350: f32, %351: f32, %352: f32):
      %353 = arith.addf %350, %351 : f32
      linalg.yield %353 : f32
    } -> tensor<1x2x32xf32>
    %354 = tensor.empty() : tensor<32x64xf32>
    %355 = linalg.transpose ins(%103:tensor<64x32xf32>) outs(%354:tensor<32x64xf32>) permutation = [1, 0]
    %356 = tensor.empty() : tensor<1x64xf32>
    %357 = linalg.transpose ins(%104:tensor<64x1xf32>) outs(%356:tensor<1x64xf32>) permutation = [1, 0]
    %358 = tensor.empty() : tensor<32x64xf32>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%355, %357 : tensor<32x64xf32>, tensor<1x64xf32>) outs(%358 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor", prov.quant_inner_0 = "net.encoder_blocks.0._attn.0.keyValueExtractor.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.0.keyValueExtractor.weight.scale"} {
    ^bb22(%360: f32, %361: f32, %362: f32):
      %363 = arith.mulf %360, %361 : f32
      linalg.yield %363 : f32
    } -> tensor<32x64xf32>
    %364 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} 0.000000e+00 : f32
    %365 = tensor.splat %364 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x64xf32>
    %366 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%349, %359 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%365 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb23(%367: f32, %368: f32, %369: f32):
      %370 = arith.mulf %367, %368 : f32
      %371 = arith.addf %369, %370 : f32
      linalg.yield %371 : f32
    } -> tensor<1x2x64xf32>
    %372 = tensor.empty() : tensor<1x2x64xf32>
    %373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%366, %9 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%372 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb24(%374: f32, %375: f32, %376: f32):
      %377 = arith.addf %374, %375 : f32
      linalg.yield %377 : f32
    } -> tensor<1x2x64xf32>
    %378 = tensor.collapse_shape %373 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %379 = tensor.expand_shape %378 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %380 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %381 = linalg.transpose ins(%379:tensor<1x2x2x1x32xf32>) outs(%380:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %382 = "tensor.extract_slice"(%381) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %383 = tensor.collapse_shape %382 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %384 = tensor.expand_shape %383 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %385 = "tensor.extract_slice"(%381) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %386 = tensor.collapse_shape %385 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %387 = tensor.expand_shape %386 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %388 = tensor.empty() : tensor<32x32xf32>
    %389 = linalg.transpose ins(%105:tensor<32x32xf32>) outs(%388:tensor<32x32xf32>) permutation = [1, 0]
    %390 = tensor.empty() : tensor<1x32xf32>
    %391 = linalg.transpose ins(%106:tensor<32x1xf32>) outs(%390:tensor<1x32xf32>) permutation = [1, 0]
    %392 = tensor.empty() : tensor<32x32xf32>
    %393 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%389, %391 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%392 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query", prov.quant_inner_0 = "net.encoder_blocks.0._attn.0.query.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.0.query.weight.scale"} {
    ^bb25(%394: f32, %395: f32, %396: f32):
      %397 = arith.mulf %394, %395 : f32
      linalg.yield %397 : f32
    } -> tensor<32x32xf32>
    %398 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} 0.000000e+00 : f32
    %399 = tensor.splat %398 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32>
    %400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%245, %393 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%399 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb26(%401: f32, %402: f32, %403: f32):
      %404 = arith.mulf %401, %402 : f32
      %405 = arith.addf %403, %404 : f32
      linalg.yield %405 : f32
    } -> tensor<1x345x32xf32>
    %406 = tensor.empty() : tensor<1x345x32xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%400, %11 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%406 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb27(%408: f32, %409: f32, %410: f32):
      %411 = arith.addf %408, %409 : f32
      linalg.yield %411 : f32
    } -> tensor<1x345x32xf32>
    %412 = tensor.collapse_shape %407 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %413 = tensor.expand_shape %412 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %414 = tensor.empty() : tensor<1x1x345x32xf32>
    %415 = linalg.transpose ins(%413:tensor<1x345x1x32xf32>) outs(%414:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %416 = tensor.empty() : tensor<1x1x32x2xf32>
    %417 = linalg.transpose ins(%384:tensor<1x1x2x32xf32>) outs(%416:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %418 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %419 = tensor.splat %418 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%415, %417 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%419 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb28(%421: f32, %422: f32, %423: f32):
      %424 = arith.mulf %421, %422 : f32
      %425 = arith.addf %423, %424 : f32
      linalg.yield %425 : f32
    } -> tensor<1x1x345x2xf32>
    %426 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %427 = tensor.splat %426 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %428 = tensor.empty() : tensor<1x1x345x2xf32>
    %429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420, %427 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%428 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb29(%430: f32, %431: f32, %432: f32):
      %433 = arith.divf %430, %431 : f32
      linalg.yield %433 : f32
    } -> tensor<1x1x345x2xf32>
    %434 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %435 = tensor.splat %434 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %436 = linalg.reduce ins(%429:tensor<1x1x345x2xf32>) outs(%435:tensor<1x1x345xf32>) dimensions = [3]
    (%437: f32, %438: f32) {
      %439 = arith.maximumf %437, %438 : f32
      linalg.yield %439 : f32
    }
    %440 = tensor.collapse_shape %436 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %441 = tensor.expand_shape %440 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %442 = tensor.empty() : tensor<1x1x345x2xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %441 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%442 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb30(%444: f32, %445: f32, %446: f32):
      %447 = arith.subf %444, %445 : f32
      linalg.yield %447 : f32
    } -> tensor<1x1x345x2xf32>
    %448 = tensor.empty() : tensor<1x1x345x2xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443 : tensor<1x1x345x2xf32>) outs(%448 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb31(%450: f32, %451: f32):
      %452 = math.exp %450 : f32
      linalg.yield %452 : f32
    } -> tensor<1x1x345x2xf32>
    %453 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %454 = tensor.splat %453 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %455 = linalg.reduce ins(%449:tensor<1x1x345x2xf32>) outs(%454:tensor<1x1x345xf32>) dimensions = [3]
    (%456: f32, %457: f32) {
      %458 = arith.addf %456, %457 : f32
      linalg.yield %458 : f32
    }
    %459 = tensor.collapse_shape %455 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %460 = tensor.expand_shape %459 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %461 = tensor.empty() : tensor<1x1x345x2xf32>
    %462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449, %460 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%461 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb32(%463: f32, %464: f32, %465: f32):
      %466 = arith.divf %463, %464 : f32
      linalg.yield %466 : f32
    } -> tensor<1x1x345x2xf32>
    %467 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %468 = tensor.splat %467 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%462, %387 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%468 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb33(%470: f32, %471: f32, %472: f32):
      %473 = arith.mulf %470, %471 : f32
      %474 = arith.addf %472, %473 : f32
      linalg.yield %474 : f32
    } -> tensor<1x1x345x32xf32>
    %475 = tensor.empty() : tensor<1x345x1x32xf32>
    %476 = linalg.transpose ins(%469:tensor<1x1x345x32xf32>) outs(%475:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %477 = tensor.collapse_shape %476 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %478 = tensor.expand_shape %477 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %479 = tensor.empty() : tensor<32x32xf32>
    %480 = linalg.transpose ins(%107:tensor<32x32xf32>) outs(%479:tensor<32x32xf32>) permutation = [1, 0]
    %481 = tensor.empty() : tensor<1x32xf32>
    %482 = linalg.transpose ins(%108:tensor<32x1xf32>) outs(%481:tensor<1x32xf32>) permutation = [1, 0]
    %483 = tensor.empty() : tensor<32x32xf32>
    %484 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%480, %482 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%483 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer", prov.quant_inner_0 = "net.encoder_blocks.0._attn.0.finalLayer.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.0.finalLayer.weight.scale"} {
    ^bb34(%485: f32, %486: f32, %487: f32):
      %488 = arith.mulf %485, %486 : f32
      linalg.yield %488 : f32
    } -> tensor<32x32xf32>
    %489 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} 0.000000e+00 : f32
    %490 = tensor.splat %489 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%478, %484 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%490 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb35(%492: f32, %493: f32, %494: f32):
      %495 = arith.mulf %492, %493 : f32
      %496 = arith.addf %494, %495 : f32
      linalg.yield %496 : f32
    } -> tensor<1x345x32xf32>
    %497 = tensor.empty() : tensor<1x345x32xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%491, %13 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%497 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb36(%499: f32, %500: f32, %501: f32):
      %502 = arith.addf %499, %500 : f32
      linalg.yield %502 : f32
    } -> tensor<1x345x32xf32>
    %503 = tensor.empty() : tensor<1x345x32xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%245, %498 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%503 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb37(%505: f32, %506: f32, %507: f32):
      %508 = arith.addf %505, %506 : f32
      linalg.yield %508 : f32
    } -> tensor<1x345x32xf32>
    %509 = tensor.empty() : tensor<32x256xf32>
    %510 = linalg.transpose ins(%115:tensor<256x32xf32>) outs(%509:tensor<32x256xf32>) permutation = [1, 0]
    %511 = tensor.empty() : tensor<1x256xf32>
    %512 = linalg.transpose ins(%116:tensor<256x1xf32>) outs(%511:tensor<1x256xf32>) permutation = [1, 0]
    %513 = tensor.empty() : tensor<32x256xf32>
    %514 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510, %512 : tensor<32x256xf32>, tensor<1x256xf32>) outs(%513 : tensor<32x256xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1", prov.quant_inner_0 = "net.encoder_blocks.0._ffn.0.mlp1.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._ffn.0.mlp1.weight.scale"} {
    ^bb38(%515: f32, %516: f32, %517: f32):
      %518 = arith.mulf %515, %516 : f32
      linalg.yield %518 : f32
    } -> tensor<32x256xf32>
    %519 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} 0.000000e+00 : f32
    %520 = tensor.splat %519 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x256xf32>
    %521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%504, %514 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%520 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb39(%522: f32, %523: f32, %524: f32):
      %525 = arith.mulf %522, %523 : f32
      %526 = arith.addf %524, %525 : f32
      linalg.yield %526 : f32
    } -> tensor<1x345x256xf32>
    %527 = tensor.empty() : tensor<1x345x256xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%521, %25 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%527 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb40(%529: f32, %530: f32, %531: f32):
      %532 = arith.addf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<1x345x256xf32>
    %533 = tensor.empty() : tensor<1x256x345xf32>
    %534 = linalg.transpose ins(%528:tensor<1x345x256xf32>) outs(%533:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %535 = tensor.collapse_shape %534 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %536 = tensor.expand_shape %535 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %537 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %538 = tensor.splat %537 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %539 = "tensor.insert_slice"(%536, %538) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %540 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%539 : tensor<1x256x17x25xf32>) outs(%540 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb41(%542: f32, %543: f32):
      linalg.yield %542 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %544 = tensor.collapse_shape %541 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %545 = tensor.expand_shape %544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %546 = tensor.collapse_shape %26 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %547 = tensor.expand_shape %546 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %548 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %549 = tensor.splat %548 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%547, %545 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%549 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb42(%551: f32, %552: f32, %553: f32):
      %554 = arith.mulf %551, %552 : f32
      %555 = arith.addf %553, %554 : f32
      linalg.yield %555 : f32
    } -> tensor<32x8x345xf32>
    %556 = tensor.collapse_shape %550 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %557 = tensor.expand_shape %556 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %558 = tensor.collapse_shape %557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %559 = tensor.expand_shape %558 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %560 = tensor.empty() : tensor<1x256x15x23xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%559, %27 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%560 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb43(%562: f32, %563: f32, %564: f32):
      %565 = arith.addf %562, %563 : f32
      linalg.yield %565 : f32
    } -> tensor<1x256x15x23xf32>
    %566 = tensor.collapse_shape %561 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %567 = tensor.expand_shape %566 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %568 = tensor.empty() : tensor<1x345x256xf32>
    %569 = linalg.transpose ins(%567:tensor<1x256x345xf32>) outs(%568:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %570 = tensor.empty() : tensor<1x345x256xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%569 : tensor<1x345x256xf32>) outs(%570 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb44(%572: f32, %573: f32):
      %574 = arith.constant 5.000000e-01 : f32
      %575 = arith.constant 1.000000e+00 : f32
      %576 = arith.constant 0.707106769 : f32
      %577 = arith.mulf %572, %576 : f32
      %578 = math.erf %577 : f32
      %579 = arith.addf %575, %578 : f32
      %580 = arith.mulf %574, %572 : f32
      %581 = arith.mulf %580, %579 : f32
      linalg.yield %581 : f32
    } -> tensor<1x345x256xf32>
    %582 = tensor.empty() : tensor<256x32xf32>
    %583 = linalg.transpose ins(%117:tensor<32x256xf32>) outs(%582:tensor<256x32xf32>) permutation = [1, 0]
    %584 = tensor.empty() : tensor<1x32xf32>
    %585 = linalg.transpose ins(%118:tensor<32x1xf32>) outs(%584:tensor<1x32xf32>) permutation = [1, 0]
    %586 = tensor.empty() : tensor<256x32xf32>
    %587 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%583, %585 : tensor<256x32xf32>, tensor<1x32xf32>) outs(%586 : tensor<256x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2", prov.quant_inner_0 = "net.encoder_blocks.0._ffn.0.mlp2.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._ffn.0.mlp2.weight.scale"} {
    ^bb45(%588: f32, %589: f32, %590: f32):
      %591 = arith.mulf %588, %589 : f32
      linalg.yield %591 : f32
    } -> tensor<256x32xf32>
    %592 = arith.constant {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} 0.000000e+00 : f32
    %593 = tensor.splat %592 {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x32xf32>
    %594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%571, %587 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%593 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb46(%595: f32, %596: f32, %597: f32):
      %598 = arith.mulf %595, %596 : f32
      %599 = arith.addf %597, %598 : f32
      linalg.yield %599 : f32
    } -> tensor<1x345x32xf32>
    %600 = tensor.empty() : tensor<1x345x32xf32>
    %601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%594, %29 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%600 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb47(%602: f32, %603: f32, %604: f32):
      %605 = arith.addf %602, %603 : f32
      linalg.yield %605 : f32
    } -> tensor<1x345x32xf32>
    %606 = tensor.empty() : tensor<1x345x32xf32>
    %607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504, %601 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%606 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb48(%608: f32, %609: f32, %610: f32):
      %611 = arith.addf %608, %609 : f32
      linalg.yield %611 : f32
    } -> tensor<1x345x32xf32>
    %612 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %613 = tensor.splat %612 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %614 = linalg.reduce ins(%607:tensor<1x345x32xf32>) outs(%613:tensor<1x345xf32>) dimensions = [2]
    (%615: f32, %616: f32) {
      %617 = arith.addf %615, %616 : f32
      linalg.yield %617 : f32
    }
    %618 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %619 = tensor.splat %618 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %620 = tensor.empty() : tensor<1x345xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%614, %619 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%620 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb49(%622: f32, %623: f32, %624: f32):
      %625 = arith.divf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<1x345xf32>
    %626 = tensor.collapse_shape %621 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %627 = tensor.expand_shape %626 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %628 = tensor.empty() : tensor<1x345x32xf32>
    %629 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%607, %627 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%628 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb50(%630: f32, %631: f32, %632: f32):
      %633 = arith.subf %630, %631 : f32
      linalg.yield %633 : f32
    } -> tensor<1x345x32xf32>
    %634 = tensor.empty() : tensor<1x345x32xf32>
    %635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629, %629 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%634 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb51(%636: f32, %637: f32, %638: f32):
      %639 = arith.mulf %636, %637 : f32
      linalg.yield %639 : f32
    } -> tensor<1x345x32xf32>
    %640 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %641 = tensor.splat %640 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %642 = linalg.reduce ins(%635:tensor<1x345x32xf32>) outs(%641:tensor<1x345xf32>) dimensions = [2]
    (%643: f32, %644: f32) {
      %645 = arith.addf %643, %644 : f32
      linalg.yield %645 : f32
    }
    %646 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %647 = tensor.splat %646 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %648 = tensor.empty() : tensor<1x345xf32>
    %649 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%642, %647 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%648 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb52(%650: f32, %651: f32, %652: f32):
      %653 = arith.divf %650, %651 : f32
      linalg.yield %653 : f32
    } -> tensor<1x345xf32>
    %654 = tensor.collapse_shape %649 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %655 = tensor.expand_shape %654 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %656 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %657 = tensor.splat %656 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %658 = tensor.empty() : tensor<1x345x1xf32>
    %659 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%655, %657 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%658 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb53(%660: f32, %661: f32, %662: f32):
      %663 = arith.addf %660, %661 : f32
      linalg.yield %663 : f32
    } -> tensor<1x345x1xf32>
    %664 = tensor.empty() : tensor<1x345x1xf32>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%659 : tensor<1x345x1xf32>) outs(%664 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb54(%666: f32, %667: f32):
      %668 = math.rsqrt %666 : f32
      linalg.yield %668 : f32
    } -> tensor<1x345x1xf32>
    %669 = tensor.empty() : tensor<1x345x32xf32>
    %670 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%629, %665 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%669 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb55(%671: f32, %672: f32, %673: f32):
      %674 = arith.mulf %671, %672 : f32
      linalg.yield %674 : f32
    } -> tensor<1x345x32xf32>
    %675 = tensor.empty() : tensor<1x345x32xf32>
    %676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670, %36 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%675 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb56(%677: f32, %678: f32, %679: f32):
      %680 = arith.mulf %677, %678 : f32
      linalg.yield %680 : f32
    } -> tensor<1x345x32xf32>
    %681 = tensor.empty() : tensor<1x345x32xf32>
    %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%676, %37 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%681 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb57(%683: f32, %684: f32, %685: f32):
      %686 = arith.addf %683, %684 : f32
      linalg.yield %686 : f32
    } -> tensor<1x345x32xf32>
    %687 = tensor.empty() : tensor<1x32x345xf32>
    %688 = linalg.transpose ins(%682:tensor<1x345x32xf32>) outs(%687:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %689 = tensor.collapse_shape %688 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %690 = tensor.expand_shape %689 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %691 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %692 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%690 : tensor<1x32x15x23xf32>) outs(%691 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb58(%693: f32, %694: f32):
      linalg.yield %693 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %695 = tensor.collapse_shape %692 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %696 = tensor.expand_shape %695 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %697 = tensor.collapse_shape %14 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %698 = tensor.expand_shape %697 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %699 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %700 = tensor.splat %699 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %701 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%698, %696 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%700 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %702 = tensor.collapse_shape %701 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %703 = tensor.expand_shape %702 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %704 = tensor.collapse_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %706 = tensor.empty() : tensor<1x32x1x2xf32>
    %707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%705, %15 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%706 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb59(%708: f32, %709: f32, %710: f32):
      %711 = arith.addf %708, %709 : f32
      linalg.yield %711 : f32
    } -> tensor<1x32x1x2xf32>
    %712 = tensor.collapse_shape %707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %713 = tensor.expand_shape %712 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %714 = tensor.empty() : tensor<1x2x32xf32>
    %715 = linalg.transpose ins(%713:tensor<1x32x2xf32>) outs(%714:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %716 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %717 = tensor.splat %716 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %718 = linalg.reduce ins(%715:tensor<1x2x32xf32>) outs(%717:tensor<1x2xf32>) dimensions = [2]
    (%719: f32, %720: f32) {
      %721 = arith.addf %719, %720 : f32
      linalg.yield %721 : f32
    }
    %722 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %723 = tensor.splat %722 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %724 = tensor.empty() : tensor<1x2xf32>
    %725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%718, %723 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%724 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb60(%726: f32, %727: f32, %728: f32):
      %729 = arith.divf %726, %727 : f32
      linalg.yield %729 : f32
    } -> tensor<1x2xf32>
    %730 = tensor.collapse_shape %725 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %731 = tensor.expand_shape %730 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %732 = tensor.empty() : tensor<1x2x32xf32>
    %733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%715, %731 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%732 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb61(%734: f32, %735: f32, %736: f32):
      %737 = arith.subf %734, %735 : f32
      linalg.yield %737 : f32
    } -> tensor<1x2x32xf32>
    %738 = tensor.empty() : tensor<1x2x32xf32>
    %739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%733, %733 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%738 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb62(%740: f32, %741: f32, %742: f32):
      %743 = arith.mulf %740, %741 : f32
      linalg.yield %743 : f32
    } -> tensor<1x2x32xf32>
    %744 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %745 = tensor.splat %744 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %746 = linalg.reduce ins(%739:tensor<1x2x32xf32>) outs(%745:tensor<1x2xf32>) dimensions = [2]
    (%747: f32, %748: f32) {
      %749 = arith.addf %747, %748 : f32
      linalg.yield %749 : f32
    }
    %750 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %751 = tensor.splat %750 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %752 = tensor.empty() : tensor<1x2xf32>
    %753 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%746, %751 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%752 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb63(%754: f32, %755: f32, %756: f32):
      %757 = arith.divf %754, %755 : f32
      linalg.yield %757 : f32
    } -> tensor<1x2xf32>
    %758 = tensor.collapse_shape %753 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %759 = tensor.expand_shape %758 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %760 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %761 = tensor.splat %760 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %762 = tensor.empty() : tensor<1x2x1xf32>
    %763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%759, %761 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%762 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb64(%764: f32, %765: f32, %766: f32):
      %767 = arith.addf %764, %765 : f32
      linalg.yield %767 : f32
    } -> tensor<1x2x1xf32>
    %768 = tensor.empty() : tensor<1x2x1xf32>
    %769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%763 : tensor<1x2x1xf32>) outs(%768 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb65(%770: f32, %771: f32):
      %772 = math.rsqrt %770 : f32
      linalg.yield %772 : f32
    } -> tensor<1x2x1xf32>
    %773 = tensor.empty() : tensor<1x2x32xf32>
    %774 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%733, %769 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%773 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb66(%775: f32, %776: f32, %777: f32):
      %778 = arith.mulf %775, %776 : f32
      linalg.yield %778 : f32
    } -> tensor<1x2x32xf32>
    %779 = tensor.empty() : tensor<1x2x32xf32>
    %780 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%774, %16 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%779 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb67(%781: f32, %782: f32, %783: f32):
      %784 = arith.mulf %781, %782 : f32
      linalg.yield %784 : f32
    } -> tensor<1x2x32xf32>
    %785 = tensor.empty() : tensor<1x2x32xf32>
    %786 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%780, %17 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%785 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb68(%787: f32, %788: f32, %789: f32):
      %790 = arith.addf %787, %788 : f32
      linalg.yield %790 : f32
    } -> tensor<1x2x32xf32>
    %791 = tensor.empty() : tensor<32x64xf32>
    %792 = linalg.transpose ins(%109:tensor<64x32xf32>) outs(%791:tensor<32x64xf32>) permutation = [1, 0]
    %793 = tensor.empty() : tensor<1x64xf32>
    %794 = linalg.transpose ins(%110:tensor<64x1xf32>) outs(%793:tensor<1x64xf32>) permutation = [1, 0]
    %795 = tensor.empty() : tensor<32x64xf32>
    %796 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%792, %794 : tensor<32x64xf32>, tensor<1x64xf32>) outs(%795 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor", prov.quant_inner_0 = "net.encoder_blocks.0._attn.1.keyValueExtractor.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.1.keyValueExtractor.weight.scale"} {
    ^bb69(%797: f32, %798: f32, %799: f32):
      %800 = arith.mulf %797, %798 : f32
      linalg.yield %800 : f32
    } -> tensor<32x64xf32>
    %801 = arith.constant {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} 0.000000e+00 : f32
    %802 = tensor.splat %801 {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x64xf32>
    %803 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%786, %796 : tensor<1x2x32xf32>, tensor<32x64xf32>) outs(%802 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb70(%804: f32, %805: f32, %806: f32):
      %807 = arith.mulf %804, %805 : f32
      %808 = arith.addf %806, %807 : f32
      linalg.yield %808 : f32
    } -> tensor<1x2x64xf32>
    %809 = tensor.empty() : tensor<1x2x64xf32>
    %810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%803, %19 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%809 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb71(%811: f32, %812: f32, %813: f32):
      %814 = arith.addf %811, %812 : f32
      linalg.yield %814 : f32
    } -> tensor<1x2x64xf32>
    %815 = tensor.collapse_shape %810 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %816 = tensor.expand_shape %815 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %817 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %818 = linalg.transpose ins(%816:tensor<1x2x2x1x32xf32>) outs(%817:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %819 = "tensor.extract_slice"(%818) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %820 = tensor.collapse_shape %819 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %821 = tensor.expand_shape %820 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %822 = "tensor.extract_slice"(%818) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %823 = tensor.collapse_shape %822 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %824 = tensor.expand_shape %823 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %825 = tensor.empty() : tensor<32x32xf32>
    %826 = linalg.transpose ins(%111:tensor<32x32xf32>) outs(%825:tensor<32x32xf32>) permutation = [1, 0]
    %827 = tensor.empty() : tensor<1x32xf32>
    %828 = linalg.transpose ins(%112:tensor<32x1xf32>) outs(%827:tensor<1x32xf32>) permutation = [1, 0]
    %829 = tensor.empty() : tensor<32x32xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%826, %828 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%829 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query", prov.quant_inner_0 = "net.encoder_blocks.0._attn.1.query.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.1.query.weight.scale"} {
    ^bb72(%831: f32, %832: f32, %833: f32):
      %834 = arith.mulf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<32x32xf32>
    %835 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} 0.000000e+00 : f32
    %836 = tensor.splat %835 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32>
    %837 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%682, %830 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%836 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb73(%838: f32, %839: f32, %840: f32):
      %841 = arith.mulf %838, %839 : f32
      %842 = arith.addf %840, %841 : f32
      linalg.yield %842 : f32
    } -> tensor<1x345x32xf32>
    %843 = tensor.empty() : tensor<1x345x32xf32>
    %844 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%837, %21 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%843 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb74(%845: f32, %846: f32, %847: f32):
      %848 = arith.addf %845, %846 : f32
      linalg.yield %848 : f32
    } -> tensor<1x345x32xf32>
    %849 = tensor.collapse_shape %844 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %850 = tensor.expand_shape %849 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %851 = tensor.empty() : tensor<1x1x345x32xf32>
    %852 = linalg.transpose ins(%850:tensor<1x345x1x32xf32>) outs(%851:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %853 = tensor.empty() : tensor<1x1x32x2xf32>
    %854 = linalg.transpose ins(%821:tensor<1x1x2x32xf32>) outs(%853:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %855 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %856 = tensor.splat %855 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %857 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%852, %854 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%856 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb75(%858: f32, %859: f32, %860: f32):
      %861 = arith.mulf %858, %859 : f32
      %862 = arith.addf %860, %861 : f32
      linalg.yield %862 : f32
    } -> tensor<1x1x345x2xf32>
    %863 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %864 = tensor.splat %863 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %865 = tensor.empty() : tensor<1x1x345x2xf32>
    %866 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%857, %864 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%865 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%867: f32, %868: f32, %869: f32):
      %870 = arith.divf %867, %868 : f32
      linalg.yield %870 : f32
    } -> tensor<1x1x345x2xf32>
    %871 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %872 = tensor.splat %871 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %873 = linalg.reduce ins(%866:tensor<1x1x345x2xf32>) outs(%872:tensor<1x1x345xf32>) dimensions = [3]
    (%874: f32, %875: f32) {
      %876 = arith.maximumf %874, %875 : f32
      linalg.yield %876 : f32
    }
    %877 = tensor.collapse_shape %873 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %878 = tensor.expand_shape %877 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %879 = tensor.empty() : tensor<1x1x345x2xf32>
    %880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%866, %878 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%879 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb77(%881: f32, %882: f32, %883: f32):
      %884 = arith.subf %881, %882 : f32
      linalg.yield %884 : f32
    } -> tensor<1x1x345x2xf32>
    %885 = tensor.empty() : tensor<1x1x345x2xf32>
    %886 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%880 : tensor<1x1x345x2xf32>) outs(%885 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb78(%887: f32, %888: f32):
      %889 = math.exp %887 : f32
      linalg.yield %889 : f32
    } -> tensor<1x1x345x2xf32>
    %890 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %891 = tensor.splat %890 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %892 = linalg.reduce ins(%886:tensor<1x1x345x2xf32>) outs(%891:tensor<1x1x345xf32>) dimensions = [3]
    (%893: f32, %894: f32) {
      %895 = arith.addf %893, %894 : f32
      linalg.yield %895 : f32
    }
    %896 = tensor.collapse_shape %892 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %897 = tensor.expand_shape %896 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %898 = tensor.empty() : tensor<1x1x345x2xf32>
    %899 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%886, %897 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%898 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb79(%900: f32, %901: f32, %902: f32):
      %903 = arith.divf %900, %901 : f32
      linalg.yield %903 : f32
    } -> tensor<1x1x345x2xf32>
    %904 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %905 = tensor.splat %904 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %906 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%899, %824 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%905 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb80(%907: f32, %908: f32, %909: f32):
      %910 = arith.mulf %907, %908 : f32
      %911 = arith.addf %909, %910 : f32
      linalg.yield %911 : f32
    } -> tensor<1x1x345x32xf32>
    %912 = tensor.empty() : tensor<1x345x1x32xf32>
    %913 = linalg.transpose ins(%906:tensor<1x1x345x32xf32>) outs(%912:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %914 = tensor.collapse_shape %913 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %915 = tensor.expand_shape %914 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %916 = tensor.empty() : tensor<32x32xf32>
    %917 = linalg.transpose ins(%113:tensor<32x32xf32>) outs(%916:tensor<32x32xf32>) permutation = [1, 0]
    %918 = tensor.empty() : tensor<1x32xf32>
    %919 = linalg.transpose ins(%114:tensor<32x1xf32>) outs(%918:tensor<1x32xf32>) permutation = [1, 0]
    %920 = tensor.empty() : tensor<32x32xf32>
    %921 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%917, %919 : tensor<32x32xf32>, tensor<1x32xf32>) outs(%920 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer", prov.quant_inner_0 = "net.encoder_blocks.0._attn.1.finalLayer.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._attn.1.finalLayer.weight.scale"} {
    ^bb81(%922: f32, %923: f32, %924: f32):
      %925 = arith.mulf %922, %923 : f32
      linalg.yield %925 : f32
    } -> tensor<32x32xf32>
    %926 = arith.constant {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} 0.000000e+00 : f32
    %927 = tensor.splat %926 {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32>
    %928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%915, %921 : tensor<1x345x32xf32>, tensor<32x32xf32>) outs(%927 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb82(%929: f32, %930: f32, %931: f32):
      %932 = arith.mulf %929, %930 : f32
      %933 = arith.addf %931, %932 : f32
      linalg.yield %933 : f32
    } -> tensor<1x345x32xf32>
    %934 = tensor.empty() : tensor<1x345x32xf32>
    %935 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%928, %23 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%934 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb83(%936: f32, %937: f32, %938: f32):
      %939 = arith.addf %936, %937 : f32
      linalg.yield %939 : f32
    } -> tensor<1x345x32xf32>
    %940 = tensor.empty() : tensor<1x345x32xf32>
    %941 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%682, %935 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%940 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%942: f32, %943: f32, %944: f32):
      %945 = arith.addf %942, %943 : f32
      linalg.yield %945 : f32
    } -> tensor<1x345x32xf32>
    %946 = tensor.empty() : tensor<32x256xf32>
    %947 = linalg.transpose ins(%119:tensor<256x32xf32>) outs(%946:tensor<32x256xf32>) permutation = [1, 0]
    %948 = tensor.empty() : tensor<1x256xf32>
    %949 = linalg.transpose ins(%120:tensor<256x1xf32>) outs(%948:tensor<1x256xf32>) permutation = [1, 0]
    %950 = tensor.empty() : tensor<32x256xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%947, %949 : tensor<32x256xf32>, tensor<1x256xf32>) outs(%950 : tensor<32x256xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1", prov.quant_inner_0 = "net.encoder_blocks.0._ffn.1.mlp1.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._ffn.1.mlp1.weight.scale"} {
    ^bb85(%952: f32, %953: f32, %954: f32):
      %955 = arith.mulf %952, %953 : f32
      linalg.yield %955 : f32
    } -> tensor<32x256xf32>
    %956 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} 0.000000e+00 : f32
    %957 = tensor.splat %956 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x256xf32>
    %958 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%941, %951 : tensor<1x345x32xf32>, tensor<32x256xf32>) outs(%957 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb86(%959: f32, %960: f32, %961: f32):
      %962 = arith.mulf %959, %960 : f32
      %963 = arith.addf %961, %962 : f32
      linalg.yield %963 : f32
    } -> tensor<1x345x256xf32>
    %964 = tensor.empty() : tensor<1x345x256xf32>
    %965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958, %31 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%964 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb87(%966: f32, %967: f32, %968: f32):
      %969 = arith.addf %966, %967 : f32
      linalg.yield %969 : f32
    } -> tensor<1x345x256xf32>
    %970 = tensor.empty() : tensor<1x256x345xf32>
    %971 = linalg.transpose ins(%965:tensor<1x345x256xf32>) outs(%970:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %972 = tensor.collapse_shape %971 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %973 = tensor.expand_shape %972 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %974 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %975 = tensor.splat %974 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %976 = "tensor.insert_slice"(%973, %975) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %977 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %978 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%976 : tensor<1x256x17x25xf32>) outs(%977 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb88(%979: f32, %980: f32):
      linalg.yield %979 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %981 = tensor.collapse_shape %978 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %982 = tensor.expand_shape %981 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %983 = tensor.collapse_shape %32 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %984 = tensor.expand_shape %983 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %985 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %986 = tensor.splat %985 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %987 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%984, %982 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%986 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb89(%988: f32, %989: f32, %990: f32):
      %991 = arith.mulf %988, %989 : f32
      %992 = arith.addf %990, %991 : f32
      linalg.yield %992 : f32
    } -> tensor<32x8x345xf32>
    %993 = tensor.collapse_shape %987 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %994 = tensor.expand_shape %993 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %995 = tensor.collapse_shape %994 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %996 = tensor.expand_shape %995 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %997 = tensor.empty() : tensor<1x256x15x23xf32>
    %998 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%996, %33 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%997 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb90(%999: f32, %1000: f32, %1001: f32):
      %1002 = arith.addf %999, %1000 : f32
      linalg.yield %1002 : f32
    } -> tensor<1x256x15x23xf32>
    %1003 = tensor.collapse_shape %998 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %1004 = tensor.expand_shape %1003 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %1005 = tensor.empty() : tensor<1x345x256xf32>
    %1006 = linalg.transpose ins(%1004:tensor<1x256x345xf32>) outs(%1005:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %1007 = tensor.empty() : tensor<1x345x256xf32>
    %1008 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1006 : tensor<1x345x256xf32>) outs(%1007 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb91(%1009: f32, %1010: f32):
      %1011 = arith.constant 5.000000e-01 : f32
      %1012 = arith.constant 1.000000e+00 : f32
      %1013 = arith.constant 0.707106769 : f32
      %1014 = arith.mulf %1009, %1013 : f32
      %1015 = math.erf %1014 : f32
      %1016 = arith.addf %1012, %1015 : f32
      %1017 = arith.mulf %1011, %1009 : f32
      %1018 = arith.mulf %1017, %1016 : f32
      linalg.yield %1018 : f32
    } -> tensor<1x345x256xf32>
    %1019 = tensor.empty() : tensor<256x32xf32>
    %1020 = linalg.transpose ins(%121:tensor<32x256xf32>) outs(%1019:tensor<256x32xf32>) permutation = [1, 0]
    %1021 = tensor.empty() : tensor<1x32xf32>
    %1022 = linalg.transpose ins(%122:tensor<32x1xf32>) outs(%1021:tensor<1x32xf32>) permutation = [1, 0]
    %1023 = tensor.empty() : tensor<256x32xf32>
    %1024 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1020, %1022 : tensor<256x32xf32>, tensor<1x32xf32>) outs(%1023 : tensor<256x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2", prov.quant_inner_0 = "net.encoder_blocks.0._ffn.1.mlp2.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.0._ffn.1.mlp2.weight.scale"} {
    ^bb92(%1025: f32, %1026: f32, %1027: f32):
      %1028 = arith.mulf %1025, %1026 : f32
      linalg.yield %1028 : f32
    } -> tensor<256x32xf32>
    %1029 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} 0.000000e+00 : f32
    %1030 = tensor.splat %1029 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x32xf32>
    %1031 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1008, %1024 : tensor<1x345x256xf32>, tensor<256x32xf32>) outs(%1030 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb93(%1032: f32, %1033: f32, %1034: f32):
      %1035 = arith.mulf %1032, %1033 : f32
      %1036 = arith.addf %1034, %1035 : f32
      linalg.yield %1036 : f32
    } -> tensor<1x345x32xf32>
    %1037 = tensor.empty() : tensor<1x345x32xf32>
    %1038 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1031, %35 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1037 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb94(%1039: f32, %1040: f32, %1041: f32):
      %1042 = arith.addf %1039, %1040 : f32
      linalg.yield %1042 : f32
    } -> tensor<1x345x32xf32>
    %1043 = tensor.empty() : tensor<1x345x32xf32>
    %1044 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%941, %1038 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1043 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb95(%1045: f32, %1046: f32, %1047: f32):
      %1048 = arith.addf %1045, %1046 : f32
      linalg.yield %1048 : f32
    } -> tensor<1x345x32xf32>
    %1049 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1050 = tensor.splat %1049 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1051 = linalg.reduce ins(%1044:tensor<1x345x32xf32>) outs(%1050:tensor<1x345xf32>) dimensions = [2]
    (%1052: f32, %1053: f32) {
      %1054 = arith.addf %1052, %1053 : f32
      linalg.yield %1054 : f32
    }
    %1055 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1056 = tensor.splat %1055 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1057 = tensor.empty() : tensor<1x345xf32>
    %1058 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1051, %1056 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1057 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb96(%1059: f32, %1060: f32, %1061: f32):
      %1062 = arith.divf %1059, %1060 : f32
      linalg.yield %1062 : f32
    } -> tensor<1x345xf32>
    %1063 = tensor.collapse_shape %1058 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1064 = tensor.expand_shape %1063 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1065 = tensor.empty() : tensor<1x345x32xf32>
    %1066 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1044, %1064 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1065 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb97(%1067: f32, %1068: f32, %1069: f32):
      %1070 = arith.subf %1067, %1068 : f32
      linalg.yield %1070 : f32
    } -> tensor<1x345x32xf32>
    %1071 = tensor.empty() : tensor<1x345x32xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1066, %1066 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1071 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb98(%1073: f32, %1074: f32, %1075: f32):
      %1076 = arith.mulf %1073, %1074 : f32
      linalg.yield %1076 : f32
    } -> tensor<1x345x32xf32>
    %1077 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1078 = tensor.splat %1077 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1079 = linalg.reduce ins(%1072:tensor<1x345x32xf32>) outs(%1078:tensor<1x345xf32>) dimensions = [2]
    (%1080: f32, %1081: f32) {
      %1082 = arith.addf %1080, %1081 : f32
      linalg.yield %1082 : f32
    }
    %1083 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1084 = tensor.splat %1083 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1085 = tensor.empty() : tensor<1x345xf32>
    %1086 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1079, %1084 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1085 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb99(%1087: f32, %1088: f32, %1089: f32):
      %1090 = arith.divf %1087, %1088 : f32
      linalg.yield %1090 : f32
    } -> tensor<1x345xf32>
    %1091 = tensor.collapse_shape %1086 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1092 = tensor.expand_shape %1091 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1093 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1094 = tensor.splat %1093 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1095 = tensor.empty() : tensor<1x345x1xf32>
    %1096 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1092, %1094 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1095 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb100(%1097: f32, %1098: f32, %1099: f32):
      %1100 = arith.addf %1097, %1098 : f32
      linalg.yield %1100 : f32
    } -> tensor<1x345x1xf32>
    %1101 = tensor.empty() : tensor<1x345x1xf32>
    %1102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1096 : tensor<1x345x1xf32>) outs(%1101 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb101(%1103: f32, %1104: f32):
      %1105 = math.rsqrt %1103 : f32
      linalg.yield %1105 : f32
    } -> tensor<1x345x1xf32>
    %1106 = tensor.empty() : tensor<1x345x32xf32>
    %1107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1066, %1102 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1106 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb102(%1108: f32, %1109: f32, %1110: f32):
      %1111 = arith.mulf %1108, %1109 : f32
      linalg.yield %1111 : f32
    } -> tensor<1x345x32xf32>
    %1112 = tensor.empty() : tensor<1x345x32xf32>
    %1113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1107, %38 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1112 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb103(%1114: f32, %1115: f32, %1116: f32):
      %1117 = arith.mulf %1114, %1115 : f32
      linalg.yield %1117 : f32
    } -> tensor<1x345x32xf32>
    %1118 = tensor.empty() : tensor<1x345x32xf32>
    %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1113, %39 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1118 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb104(%1120: f32, %1121: f32, %1122: f32):
      %1123 = arith.addf %1120, %1121 : f32
      linalg.yield %1123 : f32
    } -> tensor<1x345x32xf32>
    %1124 = tensor.collapse_shape %1119 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1125 = tensor.expand_shape %1124 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1126 = tensor.empty() : tensor<1x32x15x23xf32>
    %1127 = linalg.transpose ins(%1125:tensor<1x15x23x32xf32>) outs(%1126:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1128 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1129 = tensor.splat %1128 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1130 = "tensor.insert_slice"(%1127, %1129) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1131 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1130 : tensor<1x32x17x25xf32>) outs(%1131 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb105(%1133: f32, %1134: f32):
      linalg.yield %1133 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1135 = tensor.collapse_shape %1132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1136 = tensor.expand_shape %1135 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1137 = tensor.collapse_shape %40 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1138 = tensor.expand_shape %1137 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1139 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1140 = tensor.splat %1139 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1141 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1138, %1136 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1140 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1142 = tensor.collapse_shape %1141 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1143 = tensor.expand_shape %1142 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1144 = tensor.collapse_shape %1143 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1145 = tensor.expand_shape %1144 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1146 = tensor.empty() : tensor<1x64x8x12xf32>
    %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1145, %41 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1146 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb106(%1148: f32, %1149: f32, %1150: f32):
      %1151 = arith.addf %1148, %1149 : f32
      linalg.yield %1151 : f32
    } -> tensor<1x64x8x12xf32>
    %1152 = tensor.collapse_shape %1147 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1153 = tensor.expand_shape %1152 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1154 = tensor.empty() : tensor<1x96x64xf32>
    %1155 = linalg.transpose ins(%1153:tensor<1x64x96xf32>) outs(%1154:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1156 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1157 = tensor.splat %1156 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1158 = linalg.reduce ins(%1155:tensor<1x96x64xf32>) outs(%1157:tensor<1x96xf32>) dimensions = [2]
    (%1159: f32, %1160: f32) {
      %1161 = arith.addf %1159, %1160 : f32
      linalg.yield %1161 : f32
    }
    %1162 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1163 = tensor.splat %1162 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1164 = tensor.empty() : tensor<1x96xf32>
    %1165 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1158, %1163 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1164 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb107(%1166: f32, %1167: f32, %1168: f32):
      %1169 = arith.divf %1166, %1167 : f32
      linalg.yield %1169 : f32
    } -> tensor<1x96xf32>
    %1170 = tensor.collapse_shape %1165 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1171 = tensor.expand_shape %1170 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1172 = tensor.empty() : tensor<1x96x64xf32>
    %1173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1155, %1171 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1172 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb108(%1174: f32, %1175: f32, %1176: f32):
      %1177 = arith.subf %1174, %1175 : f32
      linalg.yield %1177 : f32
    } -> tensor<1x96x64xf32>
    %1178 = tensor.empty() : tensor<1x96x64xf32>
    %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1173, %1173 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1178 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb109(%1180: f32, %1181: f32, %1182: f32):
      %1183 = arith.mulf %1180, %1181 : f32
      linalg.yield %1183 : f32
    } -> tensor<1x96x64xf32>
    %1184 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1185 = tensor.splat %1184 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1186 = linalg.reduce ins(%1179:tensor<1x96x64xf32>) outs(%1185:tensor<1x96xf32>) dimensions = [2]
    (%1187: f32, %1188: f32) {
      %1189 = arith.addf %1187, %1188 : f32
      linalg.yield %1189 : f32
    }
    %1190 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1191 = tensor.splat %1190 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1192 = tensor.empty() : tensor<1x96xf32>
    %1193 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1186, %1191 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1192 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb110(%1194: f32, %1195: f32, %1196: f32):
      %1197 = arith.divf %1194, %1195 : f32
      linalg.yield %1197 : f32
    } -> tensor<1x96xf32>
    %1198 = tensor.collapse_shape %1193 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1199 = tensor.expand_shape %1198 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1200 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1201 = tensor.splat %1200 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1202 = tensor.empty() : tensor<1x96x1xf32>
    %1203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1199, %1201 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1202 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb111(%1204: f32, %1205: f32, %1206: f32):
      %1207 = arith.addf %1204, %1205 : f32
      linalg.yield %1207 : f32
    } -> tensor<1x96x1xf32>
    %1208 = tensor.empty() : tensor<1x96x1xf32>
    %1209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1203 : tensor<1x96x1xf32>) outs(%1208 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb112(%1210: f32, %1211: f32):
      %1212 = math.rsqrt %1210 : f32
      linalg.yield %1212 : f32
    } -> tensor<1x96x1xf32>
    %1213 = tensor.empty() : tensor<1x96x64xf32>
    %1214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1173, %1209 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1213 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb113(%1215: f32, %1216: f32, %1217: f32):
      %1218 = arith.mulf %1215, %1216 : f32
      linalg.yield %1218 : f32
    } -> tensor<1x96x64xf32>
    %1219 = tensor.empty() : tensor<1x96x64xf32>
    %1220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1214, %42 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1219 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb114(%1221: f32, %1222: f32, %1223: f32):
      %1224 = arith.mulf %1221, %1222 : f32
      linalg.yield %1224 : f32
    } -> tensor<1x96x64xf32>
    %1225 = tensor.empty() : tensor<1x96x64xf32>
    %1226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1220, %43 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1225 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb115(%1227: f32, %1228: f32, %1229: f32):
      %1230 = arith.addf %1227, %1228 : f32
      linalg.yield %1230 : f32
    } -> tensor<1x96x64xf32>
    %1231 = tensor.empty() : tensor<1x64x96xf32>
    %1232 = linalg.transpose ins(%1226:tensor<1x96x64xf32>) outs(%1231:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1233 = tensor.collapse_shape %1232 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1235 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1234 : tensor<1x64x8x12xf32>) outs(%1235 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb116(%1237: f32, %1238: f32):
      linalg.yield %1237 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1239 = tensor.collapse_shape %1236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1240 = tensor.expand_shape %1239 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1241 = tensor.collapse_shape %44 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1242 = tensor.expand_shape %1241 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1243 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1244 = tensor.splat %1243 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1245 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1242, %1240 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1244 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1246 = tensor.collapse_shape %1245 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1247 = tensor.expand_shape %1246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1248 = tensor.collapse_shape %1247 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1249 = tensor.expand_shape %1248 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1250 = tensor.empty() : tensor<1x64x2x3xf32>
    %1251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1249, %45 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1250 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb117(%1252: f32, %1253: f32, %1254: f32):
      %1255 = arith.addf %1252, %1253 : f32
      linalg.yield %1255 : f32
    } -> tensor<1x64x2x3xf32>
    %1256 = tensor.collapse_shape %1251 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1257 = tensor.expand_shape %1256 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1258 = tensor.empty() : tensor<1x6x64xf32>
    %1259 = linalg.transpose ins(%1257:tensor<1x64x6xf32>) outs(%1258:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1260 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1261 = tensor.splat %1260 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1262 = linalg.reduce ins(%1259:tensor<1x6x64xf32>) outs(%1261:tensor<1x6xf32>) dimensions = [2]
    (%1263: f32, %1264: f32) {
      %1265 = arith.addf %1263, %1264 : f32
      linalg.yield %1265 : f32
    }
    %1266 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1267 = tensor.splat %1266 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1268 = tensor.empty() : tensor<1x6xf32>
    %1269 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1262, %1267 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1268 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb118(%1270: f32, %1271: f32, %1272: f32):
      %1273 = arith.divf %1270, %1271 : f32
      linalg.yield %1273 : f32
    } -> tensor<1x6xf32>
    %1274 = tensor.collapse_shape %1269 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1275 = tensor.expand_shape %1274 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1276 = tensor.empty() : tensor<1x6x64xf32>
    %1277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1259, %1275 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1276 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb119(%1278: f32, %1279: f32, %1280: f32):
      %1281 = arith.subf %1278, %1279 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x6x64xf32>
    %1282 = tensor.empty() : tensor<1x6x64xf32>
    %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277, %1277 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1282 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb120(%1284: f32, %1285: f32, %1286: f32):
      %1287 = arith.mulf %1284, %1285 : f32
      linalg.yield %1287 : f32
    } -> tensor<1x6x64xf32>
    %1288 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1289 = tensor.splat %1288 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1290 = linalg.reduce ins(%1283:tensor<1x6x64xf32>) outs(%1289:tensor<1x6xf32>) dimensions = [2]
    (%1291: f32, %1292: f32) {
      %1293 = arith.addf %1291, %1292 : f32
      linalg.yield %1293 : f32
    }
    %1294 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1295 = tensor.splat %1294 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1296 = tensor.empty() : tensor<1x6xf32>
    %1297 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1290, %1295 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1296 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb121(%1298: f32, %1299: f32, %1300: f32):
      %1301 = arith.divf %1298, %1299 : f32
      linalg.yield %1301 : f32
    } -> tensor<1x6xf32>
    %1302 = tensor.collapse_shape %1297 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1303 = tensor.expand_shape %1302 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1304 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1305 = tensor.splat %1304 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1306 = tensor.empty() : tensor<1x6x1xf32>
    %1307 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1303, %1305 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1306 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb122(%1308: f32, %1309: f32, %1310: f32):
      %1311 = arith.addf %1308, %1309 : f32
      linalg.yield %1311 : f32
    } -> tensor<1x6x1xf32>
    %1312 = tensor.empty() : tensor<1x6x1xf32>
    %1313 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1307 : tensor<1x6x1xf32>) outs(%1312 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb123(%1314: f32, %1315: f32):
      %1316 = math.rsqrt %1314 : f32
      linalg.yield %1316 : f32
    } -> tensor<1x6x1xf32>
    %1317 = tensor.empty() : tensor<1x6x64xf32>
    %1318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277, %1313 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1317 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb124(%1319: f32, %1320: f32, %1321: f32):
      %1322 = arith.mulf %1319, %1320 : f32
      linalg.yield %1322 : f32
    } -> tensor<1x6x64xf32>
    %1323 = tensor.empty() : tensor<1x6x64xf32>
    %1324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1318, %46 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1323 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb125(%1325: f32, %1326: f32, %1327: f32):
      %1328 = arith.mulf %1325, %1326 : f32
      linalg.yield %1328 : f32
    } -> tensor<1x6x64xf32>
    %1329 = tensor.empty() : tensor<1x6x64xf32>
    %1330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1324, %47 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1329 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb126(%1331: f32, %1332: f32, %1333: f32):
      %1334 = arith.addf %1331, %1332 : f32
      linalg.yield %1334 : f32
    } -> tensor<1x6x64xf32>
    %1335 = tensor.empty() : tensor<64x128xf32>
    %1336 = linalg.transpose ins(%123:tensor<128x64xf32>) outs(%1335:tensor<64x128xf32>) permutation = [1, 0]
    %1337 = tensor.empty() : tensor<1x128xf32>
    %1338 = linalg.transpose ins(%124:tensor<128x1xf32>) outs(%1337:tensor<1x128xf32>) permutation = [1, 0]
    %1339 = tensor.empty() : tensor<64x128xf32>
    %1340 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1336, %1338 : tensor<64x128xf32>, tensor<1x128xf32>) outs(%1339 : tensor<64x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor", prov.quant_inner_0 = "net.encoder_blocks.1._attn.0.keyValueExtractor.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.0.keyValueExtractor.weight.scale"} {
    ^bb127(%1341: f32, %1342: f32, %1343: f32):
      %1344 = arith.mulf %1341, %1342 : f32
      linalg.yield %1344 : f32
    } -> tensor<64x128xf32>
    %1345 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} 0.000000e+00 : f32
    %1346 = tensor.splat %1345 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x128xf32>
    %1347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1330, %1340 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1346 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb128(%1348: f32, %1349: f32, %1350: f32):
      %1351 = arith.mulf %1348, %1349 : f32
      %1352 = arith.addf %1350, %1351 : f32
      linalg.yield %1352 : f32
    } -> tensor<1x6x128xf32>
    %1353 = tensor.empty() : tensor<1x6x128xf32>
    %1354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1347, %49 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1353 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb129(%1355: f32, %1356: f32, %1357: f32):
      %1358 = arith.addf %1355, %1356 : f32
      linalg.yield %1358 : f32
    } -> tensor<1x6x128xf32>
    %1359 = tensor.collapse_shape %1354 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1360 = tensor.expand_shape %1359 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1361 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1362 = linalg.transpose ins(%1360:tensor<1x6x2x2x32xf32>) outs(%1361:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1363 = "tensor.extract_slice"(%1362) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1364 = tensor.collapse_shape %1363 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1365 = tensor.expand_shape %1364 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1366 = "tensor.extract_slice"(%1362) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1367 = tensor.collapse_shape %1366 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1368 = tensor.expand_shape %1367 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1369 = tensor.empty() : tensor<64x64xf32>
    %1370 = linalg.transpose ins(%125:tensor<64x64xf32>) outs(%1369:tensor<64x64xf32>) permutation = [1, 0]
    %1371 = tensor.empty() : tensor<1x64xf32>
    %1372 = linalg.transpose ins(%126:tensor<64x1xf32>) outs(%1371:tensor<1x64xf32>) permutation = [1, 0]
    %1373 = tensor.empty() : tensor<64x64xf32>
    %1374 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1370, %1372 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1373 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query", prov.quant_inner_0 = "net.encoder_blocks.1._attn.0.query.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.0.query.weight.scale"} {
    ^bb130(%1375: f32, %1376: f32, %1377: f32):
      %1378 = arith.mulf %1375, %1376 : f32
      linalg.yield %1378 : f32
    } -> tensor<64x64xf32>
    %1379 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} 0.000000e+00 : f32
    %1380 = tensor.splat %1379 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32>
    %1381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1226, %1374 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1380 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb131(%1382: f32, %1383: f32, %1384: f32):
      %1385 = arith.mulf %1382, %1383 : f32
      %1386 = arith.addf %1384, %1385 : f32
      linalg.yield %1386 : f32
    } -> tensor<1x96x64xf32>
    %1387 = tensor.empty() : tensor<1x96x64xf32>
    %1388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1381, %51 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1387 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb132(%1389: f32, %1390: f32, %1391: f32):
      %1392 = arith.addf %1389, %1390 : f32
      linalg.yield %1392 : f32
    } -> tensor<1x96x64xf32>
    %1393 = tensor.collapse_shape %1388 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1394 = tensor.expand_shape %1393 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1395 = tensor.empty() : tensor<1x2x96x32xf32>
    %1396 = linalg.transpose ins(%1394:tensor<1x96x2x32xf32>) outs(%1395:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1397 = tensor.empty() : tensor<1x2x32x6xf32>
    %1398 = linalg.transpose ins(%1365:tensor<1x2x6x32xf32>) outs(%1397:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1399 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1400 = tensor.splat %1399 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1396, %1398 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1400 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb133(%1402: f32, %1403: f32, %1404: f32):
      %1405 = arith.mulf %1402, %1403 : f32
      %1406 = arith.addf %1404, %1405 : f32
      linalg.yield %1406 : f32
    } -> tensor<1x2x96x6xf32>
    %1407 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1408 = tensor.splat %1407 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1409 = tensor.empty() : tensor<1x2x96x6xf32>
    %1410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1401, %1408 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1409 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb134(%1411: f32, %1412: f32, %1413: f32):
      %1414 = arith.divf %1411, %1412 : f32
      linalg.yield %1414 : f32
    } -> tensor<1x2x96x6xf32>
    %1415 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1416 = tensor.splat %1415 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1417 = linalg.reduce ins(%1410:tensor<1x2x96x6xf32>) outs(%1416:tensor<1x2x96xf32>) dimensions = [3]
    (%1418: f32, %1419: f32) {
      %1420 = arith.maximumf %1418, %1419 : f32
      linalg.yield %1420 : f32
    }
    %1421 = tensor.collapse_shape %1417 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1422 = tensor.expand_shape %1421 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1423 = tensor.empty() : tensor<1x2x96x6xf32>
    %1424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1410, %1422 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1423 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb135(%1425: f32, %1426: f32, %1427: f32):
      %1428 = arith.subf %1425, %1426 : f32
      linalg.yield %1428 : f32
    } -> tensor<1x2x96x6xf32>
    %1429 = tensor.empty() : tensor<1x2x96x6xf32>
    %1430 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1424 : tensor<1x2x96x6xf32>) outs(%1429 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb136(%1431: f32, %1432: f32):
      %1433 = math.exp %1431 : f32
      linalg.yield %1433 : f32
    } -> tensor<1x2x96x6xf32>
    %1434 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1435 = tensor.splat %1434 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1436 = linalg.reduce ins(%1430:tensor<1x2x96x6xf32>) outs(%1435:tensor<1x2x96xf32>) dimensions = [3]
    (%1437: f32, %1438: f32) {
      %1439 = arith.addf %1437, %1438 : f32
      linalg.yield %1439 : f32
    }
    %1440 = tensor.collapse_shape %1436 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1441 = tensor.expand_shape %1440 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1442 = tensor.empty() : tensor<1x2x96x6xf32>
    %1443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1430, %1441 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1442 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb137(%1444: f32, %1445: f32, %1446: f32):
      %1447 = arith.divf %1444, %1445 : f32
      linalg.yield %1447 : f32
    } -> tensor<1x2x96x6xf32>
    %1448 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1449 = tensor.splat %1448 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1450 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1443, %1368 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1449 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb138(%1451: f32, %1452: f32, %1453: f32):
      %1454 = arith.mulf %1451, %1452 : f32
      %1455 = arith.addf %1453, %1454 : f32
      linalg.yield %1455 : f32
    } -> tensor<1x2x96x32xf32>
    %1456 = tensor.empty() : tensor<1x96x2x32xf32>
    %1457 = linalg.transpose ins(%1450:tensor<1x2x96x32xf32>) outs(%1456:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1458 = tensor.collapse_shape %1457 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1459 = tensor.expand_shape %1458 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1460 = tensor.empty() : tensor<64x64xf32>
    %1461 = linalg.transpose ins(%127:tensor<64x64xf32>) outs(%1460:tensor<64x64xf32>) permutation = [1, 0]
    %1462 = tensor.empty() : tensor<1x64xf32>
    %1463 = linalg.transpose ins(%128:tensor<64x1xf32>) outs(%1462:tensor<1x64xf32>) permutation = [1, 0]
    %1464 = tensor.empty() : tensor<64x64xf32>
    %1465 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1461, %1463 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1464 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer", prov.quant_inner_0 = "net.encoder_blocks.1._attn.0.finalLayer.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.0.finalLayer.weight.scale"} {
    ^bb139(%1466: f32, %1467: f32, %1468: f32):
      %1469 = arith.mulf %1466, %1467 : f32
      linalg.yield %1469 : f32
    } -> tensor<64x64xf32>
    %1470 = arith.constant {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} 0.000000e+00 : f32
    %1471 = tensor.splat %1470 {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32>
    %1472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1459, %1465 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1471 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb140(%1473: f32, %1474: f32, %1475: f32):
      %1476 = arith.mulf %1473, %1474 : f32
      %1477 = arith.addf %1475, %1476 : f32
      linalg.yield %1477 : f32
    } -> tensor<1x96x64xf32>
    %1478 = tensor.empty() : tensor<1x96x64xf32>
    %1479 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1472, %53 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1478 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb141(%1480: f32, %1481: f32, %1482: f32):
      %1483 = arith.addf %1480, %1481 : f32
      linalg.yield %1483 : f32
    } -> tensor<1x96x64xf32>
    %1484 = tensor.empty() : tensor<1x96x64xf32>
    %1485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1226, %1479 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1484 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb142(%1486: f32, %1487: f32, %1488: f32):
      %1489 = arith.addf %1486, %1487 : f32
      linalg.yield %1489 : f32
    } -> tensor<1x96x64xf32>
    %1490 = tensor.empty() : tensor<64x512xf32>
    %1491 = linalg.transpose ins(%135:tensor<512x64xf32>) outs(%1490:tensor<64x512xf32>) permutation = [1, 0]
    %1492 = tensor.empty() : tensor<1x512xf32>
    %1493 = linalg.transpose ins(%136:tensor<512x1xf32>) outs(%1492:tensor<1x512xf32>) permutation = [1, 0]
    %1494 = tensor.empty() : tensor<64x512xf32>
    %1495 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1491, %1493 : tensor<64x512xf32>, tensor<1x512xf32>) outs(%1494 : tensor<64x512xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1", prov.quant_inner_0 = "net.encoder_blocks.1._ffn.0.mlp1.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._ffn.0.mlp1.weight.scale"} {
    ^bb143(%1496: f32, %1497: f32, %1498: f32):
      %1499 = arith.mulf %1496, %1497 : f32
      linalg.yield %1499 : f32
    } -> tensor<64x512xf32>
    %1500 = arith.constant {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} 0.000000e+00 : f32
    %1501 = tensor.splat %1500 {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x512xf32>
    %1502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1485, %1495 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%1501 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb144(%1503: f32, %1504: f32, %1505: f32):
      %1506 = arith.mulf %1503, %1504 : f32
      %1507 = arith.addf %1505, %1506 : f32
      linalg.yield %1507 : f32
    } -> tensor<1x96x512xf32>
    %1508 = tensor.empty() : tensor<1x96x512xf32>
    %1509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1502, %65 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1508 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb145(%1510: f32, %1511: f32, %1512: f32):
      %1513 = arith.addf %1510, %1511 : f32
      linalg.yield %1513 : f32
    } -> tensor<1x96x512xf32>
    %1514 = tensor.empty() : tensor<1x512x96xf32>
    %1515 = linalg.transpose ins(%1509:tensor<1x96x512xf32>) outs(%1514:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1516 = tensor.collapse_shape %1515 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1517 = tensor.expand_shape %1516 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1518 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1519 = tensor.splat %1518 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1520 = "tensor.insert_slice"(%1517, %1519) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1521 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1522 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1520 : tensor<1x512x10x14xf32>) outs(%1521 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb146(%1523: f32, %1524: f32):
      linalg.yield %1523 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1525 = tensor.collapse_shape %1522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1526 = tensor.expand_shape %1525 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1527 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1528 = tensor.expand_shape %1527 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1529 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1530 = tensor.splat %1529 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1528, %1526 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1530 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb147(%1532: f32, %1533: f32, %1534: f32):
      %1535 = arith.mulf %1532, %1533 : f32
      %1536 = arith.addf %1534, %1535 : f32
      linalg.yield %1536 : f32
    } -> tensor<64x8x96xf32>
    %1537 = tensor.collapse_shape %1531 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1538 = tensor.expand_shape %1537 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1539 = tensor.collapse_shape %1538 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1540 = tensor.expand_shape %1539 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1541 = tensor.empty() : tensor<1x512x8x12xf32>
    %1542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1540, %67 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1541 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb148(%1543: f32, %1544: f32, %1545: f32):
      %1546 = arith.addf %1543, %1544 : f32
      linalg.yield %1546 : f32
    } -> tensor<1x512x8x12xf32>
    %1547 = tensor.collapse_shape %1542 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1548 = tensor.expand_shape %1547 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1549 = tensor.empty() : tensor<1x96x512xf32>
    %1550 = linalg.transpose ins(%1548:tensor<1x512x96xf32>) outs(%1549:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1551 = tensor.empty() : tensor<1x96x512xf32>
    %1552 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1550 : tensor<1x96x512xf32>) outs(%1551 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb149(%1553: f32, %1554: f32):
      %1555 = arith.constant 5.000000e-01 : f32
      %1556 = arith.constant 1.000000e+00 : f32
      %1557 = arith.constant 0.707106769 : f32
      %1558 = arith.mulf %1553, %1557 : f32
      %1559 = math.erf %1558 : f32
      %1560 = arith.addf %1556, %1559 : f32
      %1561 = arith.mulf %1555, %1553 : f32
      %1562 = arith.mulf %1561, %1560 : f32
      linalg.yield %1562 : f32
    } -> tensor<1x96x512xf32>
    %1563 = tensor.empty() : tensor<512x64xf32>
    %1564 = linalg.transpose ins(%137:tensor<64x512xf32>) outs(%1563:tensor<512x64xf32>) permutation = [1, 0]
    %1565 = tensor.empty() : tensor<1x64xf32>
    %1566 = linalg.transpose ins(%138:tensor<64x1xf32>) outs(%1565:tensor<1x64xf32>) permutation = [1, 0]
    %1567 = tensor.empty() : tensor<512x64xf32>
    %1568 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1564, %1566 : tensor<512x64xf32>, tensor<1x64xf32>) outs(%1567 : tensor<512x64xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2", prov.quant_inner_0 = "net.encoder_blocks.1._ffn.0.mlp2.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._ffn.0.mlp2.weight.scale"} {
    ^bb150(%1569: f32, %1570: f32, %1571: f32):
      %1572 = arith.mulf %1569, %1570 : f32
      linalg.yield %1572 : f32
    } -> tensor<512x64xf32>
    %1573 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} 0.000000e+00 : f32
    %1574 = tensor.splat %1573 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x64xf32>
    %1575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1552, %1568 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%1574 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb151(%1576: f32, %1577: f32, %1578: f32):
      %1579 = arith.mulf %1576, %1577 : f32
      %1580 = arith.addf %1578, %1579 : f32
      linalg.yield %1580 : f32
    } -> tensor<1x96x64xf32>
    %1581 = tensor.empty() : tensor<1x96x64xf32>
    %1582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1575, %69 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1581 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb152(%1583: f32, %1584: f32, %1585: f32):
      %1586 = arith.addf %1583, %1584 : f32
      linalg.yield %1586 : f32
    } -> tensor<1x96x64xf32>
    %1587 = tensor.empty() : tensor<1x96x64xf32>
    %1588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1485, %1582 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1587 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb153(%1589: f32, %1590: f32, %1591: f32):
      %1592 = arith.addf %1589, %1590 : f32
      linalg.yield %1592 : f32
    } -> tensor<1x96x64xf32>
    %1593 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1594 = tensor.splat %1593 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1595 = linalg.reduce ins(%1588:tensor<1x96x64xf32>) outs(%1594:tensor<1x96xf32>) dimensions = [2]
    (%1596: f32, %1597: f32) {
      %1598 = arith.addf %1596, %1597 : f32
      linalg.yield %1598 : f32
    }
    %1599 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1600 = tensor.splat %1599 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1601 = tensor.empty() : tensor<1x96xf32>
    %1602 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1595, %1600 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1601 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb154(%1603: f32, %1604: f32, %1605: f32):
      %1606 = arith.divf %1603, %1604 : f32
      linalg.yield %1606 : f32
    } -> tensor<1x96xf32>
    %1607 = tensor.collapse_shape %1602 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1608 = tensor.expand_shape %1607 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1609 = tensor.empty() : tensor<1x96x64xf32>
    %1610 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1588, %1608 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1609 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb155(%1611: f32, %1612: f32, %1613: f32):
      %1614 = arith.subf %1611, %1612 : f32
      linalg.yield %1614 : f32
    } -> tensor<1x96x64xf32>
    %1615 = tensor.empty() : tensor<1x96x64xf32>
    %1616 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1610, %1610 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1615 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb156(%1617: f32, %1618: f32, %1619: f32):
      %1620 = arith.mulf %1617, %1618 : f32
      linalg.yield %1620 : f32
    } -> tensor<1x96x64xf32>
    %1621 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1622 = tensor.splat %1621 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1623 = linalg.reduce ins(%1616:tensor<1x96x64xf32>) outs(%1622:tensor<1x96xf32>) dimensions = [2]
    (%1624: f32, %1625: f32) {
      %1626 = arith.addf %1624, %1625 : f32
      linalg.yield %1626 : f32
    }
    %1627 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1628 = tensor.splat %1627 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1629 = tensor.empty() : tensor<1x96xf32>
    %1630 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1623, %1628 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1629 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb157(%1631: f32, %1632: f32, %1633: f32):
      %1634 = arith.divf %1631, %1632 : f32
      linalg.yield %1634 : f32
    } -> tensor<1x96xf32>
    %1635 = tensor.collapse_shape %1630 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1636 = tensor.expand_shape %1635 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1637 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1638 = tensor.splat %1637 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1639 = tensor.empty() : tensor<1x96x1xf32>
    %1640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1636, %1638 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1639 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb158(%1641: f32, %1642: f32, %1643: f32):
      %1644 = arith.addf %1641, %1642 : f32
      linalg.yield %1644 : f32
    } -> tensor<1x96x1xf32>
    %1645 = tensor.empty() : tensor<1x96x1xf32>
    %1646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1640 : tensor<1x96x1xf32>) outs(%1645 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb159(%1647: f32, %1648: f32):
      %1649 = math.rsqrt %1647 : f32
      linalg.yield %1649 : f32
    } -> tensor<1x96x1xf32>
    %1650 = tensor.empty() : tensor<1x96x64xf32>
    %1651 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1610, %1646 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1650 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%1652: f32, %1653: f32, %1654: f32):
      %1655 = arith.mulf %1652, %1653 : f32
      linalg.yield %1655 : f32
    } -> tensor<1x96x64xf32>
    %1656 = tensor.empty() : tensor<1x96x64xf32>
    %1657 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1651, %76 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1656 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%1658: f32, %1659: f32, %1660: f32):
      %1661 = arith.mulf %1658, %1659 : f32
      linalg.yield %1661 : f32
    } -> tensor<1x96x64xf32>
    %1662 = tensor.empty() : tensor<1x96x64xf32>
    %1663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1657, %77 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1662 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb162(%1664: f32, %1665: f32, %1666: f32):
      %1667 = arith.addf %1664, %1665 : f32
      linalg.yield %1667 : f32
    } -> tensor<1x96x64xf32>
    %1668 = tensor.empty() : tensor<1x64x96xf32>
    %1669 = linalg.transpose ins(%1663:tensor<1x96x64xf32>) outs(%1668:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1670 = tensor.collapse_shape %1669 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1671 = tensor.expand_shape %1670 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1672 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1671 : tensor<1x64x8x12xf32>) outs(%1672 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb163(%1674: f32, %1675: f32):
      linalg.yield %1674 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1676 = tensor.collapse_shape %1673 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1677 = tensor.expand_shape %1676 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1678 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1679 = tensor.expand_shape %1678 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1680 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1681 = tensor.splat %1680 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1682 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1679, %1677 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1681 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1683 = tensor.collapse_shape %1682 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1684 = tensor.expand_shape %1683 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1685 = tensor.collapse_shape %1684 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1686 = tensor.expand_shape %1685 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1687 = tensor.empty() : tensor<1x64x2x3xf32>
    %1688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1686, %55 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1687 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb164(%1689: f32, %1690: f32, %1691: f32):
      %1692 = arith.addf %1689, %1690 : f32
      linalg.yield %1692 : f32
    } -> tensor<1x64x2x3xf32>
    %1693 = tensor.collapse_shape %1688 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1694 = tensor.expand_shape %1693 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1695 = tensor.empty() : tensor<1x6x64xf32>
    %1696 = linalg.transpose ins(%1694:tensor<1x64x6xf32>) outs(%1695:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1697 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1698 = tensor.splat %1697 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1699 = linalg.reduce ins(%1696:tensor<1x6x64xf32>) outs(%1698:tensor<1x6xf32>) dimensions = [2]
    (%1700: f32, %1701: f32) {
      %1702 = arith.addf %1700, %1701 : f32
      linalg.yield %1702 : f32
    }
    %1703 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1704 = tensor.splat %1703 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1705 = tensor.empty() : tensor<1x6xf32>
    %1706 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1699, %1704 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1705 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb165(%1707: f32, %1708: f32, %1709: f32):
      %1710 = arith.divf %1707, %1708 : f32
      linalg.yield %1710 : f32
    } -> tensor<1x6xf32>
    %1711 = tensor.collapse_shape %1706 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1712 = tensor.expand_shape %1711 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1713 = tensor.empty() : tensor<1x6x64xf32>
    %1714 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1696, %1712 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1713 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb166(%1715: f32, %1716: f32, %1717: f32):
      %1718 = arith.subf %1715, %1716 : f32
      linalg.yield %1718 : f32
    } -> tensor<1x6x64xf32>
    %1719 = tensor.empty() : tensor<1x6x64xf32>
    %1720 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1714, %1714 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1719 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb167(%1721: f32, %1722: f32, %1723: f32):
      %1724 = arith.mulf %1721, %1722 : f32
      linalg.yield %1724 : f32
    } -> tensor<1x6x64xf32>
    %1725 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1726 = tensor.splat %1725 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1727 = linalg.reduce ins(%1720:tensor<1x6x64xf32>) outs(%1726:tensor<1x6xf32>) dimensions = [2]
    (%1728: f32, %1729: f32) {
      %1730 = arith.addf %1728, %1729 : f32
      linalg.yield %1730 : f32
    }
    %1731 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1732 = tensor.splat %1731 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1733 = tensor.empty() : tensor<1x6xf32>
    %1734 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1727, %1732 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1733 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb168(%1735: f32, %1736: f32, %1737: f32):
      %1738 = arith.divf %1735, %1736 : f32
      linalg.yield %1738 : f32
    } -> tensor<1x6xf32>
    %1739 = tensor.collapse_shape %1734 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1740 = tensor.expand_shape %1739 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1741 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1742 = tensor.splat %1741 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1743 = tensor.empty() : tensor<1x6x1xf32>
    %1744 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1740, %1742 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1743 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb169(%1745: f32, %1746: f32, %1747: f32):
      %1748 = arith.addf %1745, %1746 : f32
      linalg.yield %1748 : f32
    } -> tensor<1x6x1xf32>
    %1749 = tensor.empty() : tensor<1x6x1xf32>
    %1750 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1744 : tensor<1x6x1xf32>) outs(%1749 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb170(%1751: f32, %1752: f32):
      %1753 = math.rsqrt %1751 : f32
      linalg.yield %1753 : f32
    } -> tensor<1x6x1xf32>
    %1754 = tensor.empty() : tensor<1x6x64xf32>
    %1755 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1714, %1750 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1754 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb171(%1756: f32, %1757: f32, %1758: f32):
      %1759 = arith.mulf %1756, %1757 : f32
      linalg.yield %1759 : f32
    } -> tensor<1x6x64xf32>
    %1760 = tensor.empty() : tensor<1x6x64xf32>
    %1761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1755, %56 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1760 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb172(%1762: f32, %1763: f32, %1764: f32):
      %1765 = arith.mulf %1762, %1763 : f32
      linalg.yield %1765 : f32
    } -> tensor<1x6x64xf32>
    %1766 = tensor.empty() : tensor<1x6x64xf32>
    %1767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1761, %57 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1766 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb173(%1768: f32, %1769: f32, %1770: f32):
      %1771 = arith.addf %1768, %1769 : f32
      linalg.yield %1771 : f32
    } -> tensor<1x6x64xf32>
    %1772 = tensor.empty() : tensor<64x128xf32>
    %1773 = linalg.transpose ins(%129:tensor<128x64xf32>) outs(%1772:tensor<64x128xf32>) permutation = [1, 0]
    %1774 = tensor.empty() : tensor<1x128xf32>
    %1775 = linalg.transpose ins(%130:tensor<128x1xf32>) outs(%1774:tensor<1x128xf32>) permutation = [1, 0]
    %1776 = tensor.empty() : tensor<64x128xf32>
    %1777 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1773, %1775 : tensor<64x128xf32>, tensor<1x128xf32>) outs(%1776 : tensor<64x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor", prov.quant_inner_0 = "net.encoder_blocks.1._attn.1.keyValueExtractor.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.1.keyValueExtractor.weight.scale"} {
    ^bb174(%1778: f32, %1779: f32, %1780: f32):
      %1781 = arith.mulf %1778, %1779 : f32
      linalg.yield %1781 : f32
    } -> tensor<64x128xf32>
    %1782 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} 0.000000e+00 : f32
    %1783 = tensor.splat %1782 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x128xf32>
    %1784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1767, %1777 : tensor<1x6x64xf32>, tensor<64x128xf32>) outs(%1783 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb175(%1785: f32, %1786: f32, %1787: f32):
      %1788 = arith.mulf %1785, %1786 : f32
      %1789 = arith.addf %1787, %1788 : f32
      linalg.yield %1789 : f32
    } -> tensor<1x6x128xf32>
    %1790 = tensor.empty() : tensor<1x6x128xf32>
    %1791 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1784, %59 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1790 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb176(%1792: f32, %1793: f32, %1794: f32):
      %1795 = arith.addf %1792, %1793 : f32
      linalg.yield %1795 : f32
    } -> tensor<1x6x128xf32>
    %1796 = tensor.collapse_shape %1791 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1797 = tensor.expand_shape %1796 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1798 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1799 = linalg.transpose ins(%1797:tensor<1x6x2x2x32xf32>) outs(%1798:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1800 = "tensor.extract_slice"(%1799) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1801 = tensor.collapse_shape %1800 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1802 = tensor.expand_shape %1801 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1803 = "tensor.extract_slice"(%1799) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1804 = tensor.collapse_shape %1803 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1805 = tensor.expand_shape %1804 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1806 = tensor.empty() : tensor<64x64xf32>
    %1807 = linalg.transpose ins(%131:tensor<64x64xf32>) outs(%1806:tensor<64x64xf32>) permutation = [1, 0]
    %1808 = tensor.empty() : tensor<1x64xf32>
    %1809 = linalg.transpose ins(%132:tensor<64x1xf32>) outs(%1808:tensor<1x64xf32>) permutation = [1, 0]
    %1810 = tensor.empty() : tensor<64x64xf32>
    %1811 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1807, %1809 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1810 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query", prov.quant_inner_0 = "net.encoder_blocks.1._attn.1.query.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.1.query.weight.scale"} {
    ^bb177(%1812: f32, %1813: f32, %1814: f32):
      %1815 = arith.mulf %1812, %1813 : f32
      linalg.yield %1815 : f32
    } -> tensor<64x64xf32>
    %1816 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} 0.000000e+00 : f32
    %1817 = tensor.splat %1816 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32>
    %1818 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1663, %1811 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1817 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb178(%1819: f32, %1820: f32, %1821: f32):
      %1822 = arith.mulf %1819, %1820 : f32
      %1823 = arith.addf %1821, %1822 : f32
      linalg.yield %1823 : f32
    } -> tensor<1x96x64xf32>
    %1824 = tensor.empty() : tensor<1x96x64xf32>
    %1825 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1818, %61 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1824 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb179(%1826: f32, %1827: f32, %1828: f32):
      %1829 = arith.addf %1826, %1827 : f32
      linalg.yield %1829 : f32
    } -> tensor<1x96x64xf32>
    %1830 = tensor.collapse_shape %1825 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1831 = tensor.expand_shape %1830 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1832 = tensor.empty() : tensor<1x2x96x32xf32>
    %1833 = linalg.transpose ins(%1831:tensor<1x96x2x32xf32>) outs(%1832:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1834 = tensor.empty() : tensor<1x2x32x6xf32>
    %1835 = linalg.transpose ins(%1802:tensor<1x2x6x32xf32>) outs(%1834:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1836 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1837 = tensor.splat %1836 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1833, %1835 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1837 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb180(%1839: f32, %1840: f32, %1841: f32):
      %1842 = arith.mulf %1839, %1840 : f32
      %1843 = arith.addf %1841, %1842 : f32
      linalg.yield %1843 : f32
    } -> tensor<1x2x96x6xf32>
    %1844 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1845 = tensor.splat %1844 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1846 = tensor.empty() : tensor<1x2x96x6xf32>
    %1847 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1838, %1845 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1846 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb181(%1848: f32, %1849: f32, %1850: f32):
      %1851 = arith.divf %1848, %1849 : f32
      linalg.yield %1851 : f32
    } -> tensor<1x2x96x6xf32>
    %1852 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1853 = tensor.splat %1852 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1854 = linalg.reduce ins(%1847:tensor<1x2x96x6xf32>) outs(%1853:tensor<1x2x96xf32>) dimensions = [3]
    (%1855: f32, %1856: f32) {
      %1857 = arith.maximumf %1855, %1856 : f32
      linalg.yield %1857 : f32
    }
    %1858 = tensor.collapse_shape %1854 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1859 = tensor.expand_shape %1858 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1860 = tensor.empty() : tensor<1x2x96x6xf32>
    %1861 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1847, %1859 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1860 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb182(%1862: f32, %1863: f32, %1864: f32):
      %1865 = arith.subf %1862, %1863 : f32
      linalg.yield %1865 : f32
    } -> tensor<1x2x96x6xf32>
    %1866 = tensor.empty() : tensor<1x2x96x6xf32>
    %1867 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1861 : tensor<1x2x96x6xf32>) outs(%1866 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb183(%1868: f32, %1869: f32):
      %1870 = math.exp %1868 : f32
      linalg.yield %1870 : f32
    } -> tensor<1x2x96x6xf32>
    %1871 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1872 = tensor.splat %1871 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1873 = linalg.reduce ins(%1867:tensor<1x2x96x6xf32>) outs(%1872:tensor<1x2x96xf32>) dimensions = [3]
    (%1874: f32, %1875: f32) {
      %1876 = arith.addf %1874, %1875 : f32
      linalg.yield %1876 : f32
    }
    %1877 = tensor.collapse_shape %1873 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1878 = tensor.expand_shape %1877 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1879 = tensor.empty() : tensor<1x2x96x6xf32>
    %1880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1867, %1878 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1879 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb184(%1881: f32, %1882: f32, %1883: f32):
      %1884 = arith.divf %1881, %1882 : f32
      linalg.yield %1884 : f32
    } -> tensor<1x2x96x6xf32>
    %1885 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1886 = tensor.splat %1885 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1887 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1880, %1805 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1886 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb185(%1888: f32, %1889: f32, %1890: f32):
      %1891 = arith.mulf %1888, %1889 : f32
      %1892 = arith.addf %1890, %1891 : f32
      linalg.yield %1892 : f32
    } -> tensor<1x2x96x32xf32>
    %1893 = tensor.empty() : tensor<1x96x2x32xf32>
    %1894 = linalg.transpose ins(%1887:tensor<1x2x96x32xf32>) outs(%1893:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1895 = tensor.collapse_shape %1894 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1896 = tensor.expand_shape %1895 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1897 = tensor.empty() : tensor<64x64xf32>
    %1898 = linalg.transpose ins(%133:tensor<64x64xf32>) outs(%1897:tensor<64x64xf32>) permutation = [1, 0]
    %1899 = tensor.empty() : tensor<1x64xf32>
    %1900 = linalg.transpose ins(%134:tensor<64x1xf32>) outs(%1899:tensor<1x64xf32>) permutation = [1, 0]
    %1901 = tensor.empty() : tensor<64x64xf32>
    %1902 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1898, %1900 : tensor<64x64xf32>, tensor<1x64xf32>) outs(%1901 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer", prov.quant_inner_0 = "net.encoder_blocks.1._attn.1.finalLayer.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._attn.1.finalLayer.weight.scale"} {
    ^bb186(%1903: f32, %1904: f32, %1905: f32):
      %1906 = arith.mulf %1903, %1904 : f32
      linalg.yield %1906 : f32
    } -> tensor<64x64xf32>
    %1907 = arith.constant {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} 0.000000e+00 : f32
    %1908 = tensor.splat %1907 {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32>
    %1909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1896, %1902 : tensor<1x96x64xf32>, tensor<64x64xf32>) outs(%1908 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb187(%1910: f32, %1911: f32, %1912: f32):
      %1913 = arith.mulf %1910, %1911 : f32
      %1914 = arith.addf %1912, %1913 : f32
      linalg.yield %1914 : f32
    } -> tensor<1x96x64xf32>
    %1915 = tensor.empty() : tensor<1x96x64xf32>
    %1916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1909, %63 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1915 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb188(%1917: f32, %1918: f32, %1919: f32):
      %1920 = arith.addf %1917, %1918 : f32
      linalg.yield %1920 : f32
    } -> tensor<1x96x64xf32>
    %1921 = tensor.empty() : tensor<1x96x64xf32>
    %1922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1663, %1916 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1921 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb189(%1923: f32, %1924: f32, %1925: f32):
      %1926 = arith.addf %1923, %1924 : f32
      linalg.yield %1926 : f32
    } -> tensor<1x96x64xf32>
    %1927 = tensor.empty() : tensor<64x512xf32>
    %1928 = linalg.transpose ins(%139:tensor<512x64xf32>) outs(%1927:tensor<64x512xf32>) permutation = [1, 0]
    %1929 = tensor.empty() : tensor<1x512xf32>
    %1930 = linalg.transpose ins(%140:tensor<512x1xf32>) outs(%1929:tensor<1x512xf32>) permutation = [1, 0]
    %1931 = tensor.empty() : tensor<64x512xf32>
    %1932 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1928, %1930 : tensor<64x512xf32>, tensor<1x512xf32>) outs(%1931 : tensor<64x512xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1", prov.quant_inner_0 = "net.encoder_blocks.1._ffn.1.mlp1.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._ffn.1.mlp1.weight.scale"} {
    ^bb190(%1933: f32, %1934: f32, %1935: f32):
      %1936 = arith.mulf %1933, %1934 : f32
      linalg.yield %1936 : f32
    } -> tensor<64x512xf32>
    %1937 = arith.constant {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} 0.000000e+00 : f32
    %1938 = tensor.splat %1937 {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x512xf32>
    %1939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1922, %1932 : tensor<1x96x64xf32>, tensor<64x512xf32>) outs(%1938 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb191(%1940: f32, %1941: f32, %1942: f32):
      %1943 = arith.mulf %1940, %1941 : f32
      %1944 = arith.addf %1942, %1943 : f32
      linalg.yield %1944 : f32
    } -> tensor<1x96x512xf32>
    %1945 = tensor.empty() : tensor<1x96x512xf32>
    %1946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1939, %71 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1945 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb192(%1947: f32, %1948: f32, %1949: f32):
      %1950 = arith.addf %1947, %1948 : f32
      linalg.yield %1950 : f32
    } -> tensor<1x96x512xf32>
    %1951 = tensor.empty() : tensor<1x512x96xf32>
    %1952 = linalg.transpose ins(%1946:tensor<1x96x512xf32>) outs(%1951:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1953 = tensor.collapse_shape %1952 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1954 = tensor.expand_shape %1953 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1955 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1956 = tensor.splat %1955 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %1957 = "tensor.insert_slice"(%1954, %1956) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1958 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1957 : tensor<1x512x10x14xf32>) outs(%1958 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb193(%1960: f32, %1961: f32):
      linalg.yield %1960 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1962 = tensor.collapse_shape %1959 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1963 = tensor.expand_shape %1962 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1964 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1965 = tensor.expand_shape %1964 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1966 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1967 = tensor.splat %1966 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %1968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1965, %1963 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1967 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb194(%1969: f32, %1970: f32, %1971: f32):
      %1972 = arith.mulf %1969, %1970 : f32
      %1973 = arith.addf %1971, %1972 : f32
      linalg.yield %1973 : f32
    } -> tensor<64x8x96xf32>
    %1974 = tensor.collapse_shape %1968 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1975 = tensor.expand_shape %1974 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1976 = tensor.collapse_shape %1975 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1977 = tensor.expand_shape %1976 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1978 = tensor.empty() : tensor<1x512x8x12xf32>
    %1979 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1977, %73 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1978 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb195(%1980: f32, %1981: f32, %1982: f32):
      %1983 = arith.addf %1980, %1981 : f32
      linalg.yield %1983 : f32
    } -> tensor<1x512x8x12xf32>
    %1984 = tensor.collapse_shape %1979 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1985 = tensor.expand_shape %1984 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1986 = tensor.empty() : tensor<1x96x512xf32>
    %1987 = linalg.transpose ins(%1985:tensor<1x512x96xf32>) outs(%1986:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1988 = tensor.empty() : tensor<1x96x512xf32>
    %1989 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1987 : tensor<1x96x512xf32>) outs(%1988 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb196(%1990: f32, %1991: f32):
      %1992 = arith.constant 5.000000e-01 : f32
      %1993 = arith.constant 1.000000e+00 : f32
      %1994 = arith.constant 0.707106769 : f32
      %1995 = arith.mulf %1990, %1994 : f32
      %1996 = math.erf %1995 : f32
      %1997 = arith.addf %1993, %1996 : f32
      %1998 = arith.mulf %1992, %1990 : f32
      %1999 = arith.mulf %1998, %1997 : f32
      linalg.yield %1999 : f32
    } -> tensor<1x96x512xf32>
    %2000 = tensor.empty() : tensor<512x64xf32>
    %2001 = linalg.transpose ins(%141:tensor<64x512xf32>) outs(%2000:tensor<512x64xf32>) permutation = [1, 0]
    %2002 = tensor.empty() : tensor<1x64xf32>
    %2003 = linalg.transpose ins(%142:tensor<64x1xf32>) outs(%2002:tensor<1x64xf32>) permutation = [1, 0]
    %2004 = tensor.empty() : tensor<512x64xf32>
    %2005 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2001, %2003 : tensor<512x64xf32>, tensor<1x64xf32>) outs(%2004 : tensor<512x64xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2", prov.quant_inner_0 = "net.encoder_blocks.1._ffn.1.mlp2.weight.qdata", prov.quant_inner_1 = "net.encoder_blocks.1._ffn.1.mlp2.weight.scale"} {
    ^bb197(%2006: f32, %2007: f32, %2008: f32):
      %2009 = arith.mulf %2006, %2007 : f32
      linalg.yield %2009 : f32
    } -> tensor<512x64xf32>
    %2010 = arith.constant {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} 0.000000e+00 : f32
    %2011 = tensor.splat %2010 {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x64xf32>
    %2012 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1989, %2005 : tensor<1x96x512xf32>, tensor<512x64xf32>) outs(%2011 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb198(%2013: f32, %2014: f32, %2015: f32):
      %2016 = arith.mulf %2013, %2014 : f32
      %2017 = arith.addf %2015, %2016 : f32
      linalg.yield %2017 : f32
    } -> tensor<1x96x64xf32>
    %2018 = tensor.empty() : tensor<1x96x64xf32>
    %2019 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2012, %75 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2018 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb199(%2020: f32, %2021: f32, %2022: f32):
      %2023 = arith.addf %2020, %2021 : f32
      linalg.yield %2023 : f32
    } -> tensor<1x96x64xf32>
    %2024 = tensor.empty() : tensor<1x96x64xf32>
    %2025 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1922, %2019 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2024 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb200(%2026: f32, %2027: f32, %2028: f32):
      %2029 = arith.addf %2026, %2027 : f32
      linalg.yield %2029 : f32
    } -> tensor<1x96x64xf32>
    %2030 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2031 = tensor.splat %2030 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2032 = linalg.reduce ins(%2025:tensor<1x96x64xf32>) outs(%2031:tensor<1x96xf32>) dimensions = [2]
    (%2033: f32, %2034: f32) {
      %2035 = arith.addf %2033, %2034 : f32
      linalg.yield %2035 : f32
    }
    %2036 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2037 = tensor.splat %2036 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2038 = tensor.empty() : tensor<1x96xf32>
    %2039 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2032, %2037 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2038 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb201(%2040: f32, %2041: f32, %2042: f32):
      %2043 = arith.divf %2040, %2041 : f32
      linalg.yield %2043 : f32
    } -> tensor<1x96xf32>
    %2044 = tensor.collapse_shape %2039 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2045 = tensor.expand_shape %2044 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2046 = tensor.empty() : tensor<1x96x64xf32>
    %2047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2025, %2045 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2046 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb202(%2048: f32, %2049: f32, %2050: f32):
      %2051 = arith.subf %2048, %2049 : f32
      linalg.yield %2051 : f32
    } -> tensor<1x96x64xf32>
    %2052 = tensor.empty() : tensor<1x96x64xf32>
    %2053 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2047, %2047 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2052 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb203(%2054: f32, %2055: f32, %2056: f32):
      %2057 = arith.mulf %2054, %2055 : f32
      linalg.yield %2057 : f32
    } -> tensor<1x96x64xf32>
    %2058 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2059 = tensor.splat %2058 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2060 = linalg.reduce ins(%2053:tensor<1x96x64xf32>) outs(%2059:tensor<1x96xf32>) dimensions = [2]
    (%2061: f32, %2062: f32) {
      %2063 = arith.addf %2061, %2062 : f32
      linalg.yield %2063 : f32
    }
    %2064 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2065 = tensor.splat %2064 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2066 = tensor.empty() : tensor<1x96xf32>
    %2067 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2060, %2065 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2066 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb204(%2068: f32, %2069: f32, %2070: f32):
      %2071 = arith.divf %2068, %2069 : f32
      linalg.yield %2071 : f32
    } -> tensor<1x96xf32>
    %2072 = tensor.collapse_shape %2067 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2073 = tensor.expand_shape %2072 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2074 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %2075 = tensor.splat %2074 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %2076 = tensor.empty() : tensor<1x96x1xf32>
    %2077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2073, %2075 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%2076 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb205(%2078: f32, %2079: f32, %2080: f32):
      %2081 = arith.addf %2078, %2079 : f32
      linalg.yield %2081 : f32
    } -> tensor<1x96x1xf32>
    %2082 = tensor.empty() : tensor<1x96x1xf32>
    %2083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2077 : tensor<1x96x1xf32>) outs(%2082 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb206(%2084: f32, %2085: f32):
      %2086 = math.rsqrt %2084 : f32
      linalg.yield %2086 : f32
    } -> tensor<1x96x1xf32>
    %2087 = tensor.empty() : tensor<1x96x64xf32>
    %2088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2047, %2083 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2087 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb207(%2089: f32, %2090: f32, %2091: f32):
      %2092 = arith.mulf %2089, %2090 : f32
      linalg.yield %2092 : f32
    } -> tensor<1x96x64xf32>
    %2093 = tensor.empty() : tensor<1x96x64xf32>
    %2094 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2088, %78 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2093 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb208(%2095: f32, %2096: f32, %2097: f32):
      %2098 = arith.mulf %2095, %2096 : f32
      linalg.yield %2098 : f32
    } -> tensor<1x96x64xf32>
    %2099 = tensor.empty() : tensor<1x96x64xf32>
    %2100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2094, %79 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2099 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb209(%2101: f32, %2102: f32, %2103: f32):
      %2104 = arith.addf %2101, %2102 : f32
      linalg.yield %2104 : f32
    } -> tensor<1x96x64xf32>
    %2105 = tensor.collapse_shape %2100 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2106 = tensor.expand_shape %2105 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2107 = tensor.empty() : tensor<1x64x8x12xf32>
    %2108 = linalg.transpose ins(%2106:tensor<1x8x12x64xf32>) outs(%2107:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2109 = tensor.collapse_shape %2108 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2110 = tensor.expand_shape %2109 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2111 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2112 = linalg.transpose ins(%2110:tensor<1x16x2x2x8x12xf32>) outs(%2111:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2113 = tensor.collapse_shape %2112 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2114 = tensor.expand_shape %2113 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2115 = tensor.empty() : tensor<1x32x23x15xf32>
    %2116 = linalg.transpose ins(%1127:tensor<1x32x15x23xf32>) outs(%2115:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2117 = tensor.collapse_shape %2116 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2118 = tensor.expand_shape %2117 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2119 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2120 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2121 = tensor.splat %2120 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2122 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2118, %2119 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2121 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2123 = tensor.collapse_shape %2122 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2124 = tensor.expand_shape %2123 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2125 = tensor.empty() : tensor<1x32x16x23xf32>
    %2126 = linalg.transpose ins(%2124:tensor<1x32x23x16xf32>) outs(%2125:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2127 = tensor.collapse_shape %2126 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2128 = tensor.expand_shape %2127 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2129 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2130 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2131 = tensor.splat %2130 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2132 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2128, %2129 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2131 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2133 = tensor.collapse_shape %2132 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2134 = tensor.expand_shape %2133 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2135 = tensor.concat dim(1) %2114, %2134 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2136 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2137 = tensor.splat %2136 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2138 = "tensor.insert_slice"(%2135, %2137) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2139 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2138 : tensor<1x48x18x26xf32>) outs(%2139 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb210(%2141: f32, %2142: f32):
      linalg.yield %2141 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2143 = tensor.collapse_shape %2140 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2144 = tensor.expand_shape %2143 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2145 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2146 = tensor.expand_shape %2145 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2147 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2148 = tensor.splat %2147 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2149 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2146, %2144 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2148 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2150 = tensor.collapse_shape %2149 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2151 = tensor.expand_shape %2150 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2152 = tensor.collapse_shape %2151 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2153 = tensor.expand_shape %2152 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2154 = tensor.empty() : tensor<1x12x16x24xf32>
    %2155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2153, %97 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2154 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb211(%2156: f32, %2157: f32, %2158: f32):
      %2159 = arith.addf %2156, %2157 : f32
      linalg.yield %2159 : f32
    } -> tensor<1x12x16x24xf32>
    %2160 = tensor.collapse_shape %2155 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2161 = tensor.expand_shape %2160 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2162 = tensor.empty() : tensor<4608x512xf32>
    %2163 = linalg.transpose ins(%143:tensor<512x4608xf32>) outs(%2162:tensor<4608x512xf32>) permutation = [1, 0]
    %2164 = tensor.empty() : tensor<1x512xf32>
    %2165 = linalg.transpose ins(%144:tensor<512x1xf32>) outs(%2164:tensor<1x512xf32>) permutation = [1, 0]
    %2166 = tensor.empty() : tensor<4608x512xf32>
    %2167 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2163, %2165 : tensor<4608x512xf32>, tensor<1x512xf32>) outs(%2166 : tensor<4608x512xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder", prov.quant_inner_0 = "net.decoder.weight.qdata", prov.quant_inner_1 = "net.decoder.weight.scale"} {
    ^bb212(%2168: f32, %2169: f32, %2170: f32):
      %2171 = arith.mulf %2168, %2169 : f32
      linalg.yield %2171 : f32
    } -> tensor<4608x512xf32>
    %2172 = tensor.empty() : tensor<1x512xf32>
    %2173 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2174 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2173 : f32) outs(%2172 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2175 = linalg.matmul {prov.region_id = "matmul_28", prov._pattern_hint = "matmul", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%2161, %2167 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2174 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2176 = tensor.empty() : tensor<1x512xf32>
    %2177 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2175, %80 : tensor<1x512xf32>, tensor<512xf32>) outs(%2176 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb213(%2178: f32, %2179: f32, %2180: f32):
      %2181 = arith.addf %2178, %2179 : f32
      linalg.yield %2181 : f32
    } -> tensor<1x512xf32>
    %2182 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2183 = tensor.splat %2182 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2184 = tensor.empty() : tensor<1x1xf32>
    %2185 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %2183 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2184 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb214(%2186: f32, %2187: f32, %2188: f32):
      %2189 = arith.divf %2186, %2187 : f32
      linalg.yield %2189 : f32
    } -> tensor<1x1xf32>
    %2190 = tensor.concat dim(1) %2177, %2185, %100 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2191 = tensor.collapse_shape %2190 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2192 = tensor.expand_shape %2191 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2193 = tensor.collapse_shape %101 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2194 = tensor.expand_shape %2193 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2195 = tensor.collapse_shape %102 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2196 = tensor.expand_shape %2195 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2197 = "tensor.extract_slice"(%2192) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 517>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x517xf32>) -> tensor<1x1x517xf32>
    %2198 = tensor.collapse_shape %2197 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2199 = tensor.expand_shape %2198 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2200 = "tensor.extract_slice"(%2194) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2201 = tensor.collapse_shape %2200 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2202 = tensor.expand_shape %2201 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2203 = "tensor.extract_slice"(%2196) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2204 = tensor.collapse_shape %2203 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2205 = tensor.expand_shape %2204 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2206 = tensor.empty() : tensor<517x512xf32>
    %2207 = linalg.transpose ins(%82:tensor<512x517xf32>) outs(%2206:tensor<517x512xf32>) permutation = [1, 0]
    %2208 = tensor.empty() : tensor<1x512xf32>
    %2209 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2210 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2209 : f32) outs(%2208 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2211 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2199, %2207 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2210 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2212 = tensor.empty() : tensor<128x512xf32>
    %2213 = linalg.transpose ins(%83:tensor<512x128xf32>) outs(%2212:tensor<128x512xf32>) permutation = [1, 0]
    %2214 = tensor.empty() : tensor<1x512xf32>
    %2215 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2216 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2215 : f32) outs(%2214 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2217 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2202, %2213 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2216 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2218 = tensor.empty() : tensor<1x512xf32>
    %2219 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2211, %2217, %84, %85 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2218 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb215(%2220: f32, %2221: f32, %2222: f32, %2223: f32, %2224: f32):
      %2225 = arith.addf %2220, %2221 : f32
      %2226 = arith.addf %2225, %2222 : f32
      %2227 = arith.addf %2226, %2223 : f32
      linalg.yield %2227 : f32
    } -> tensor<1x512xf32>
    %2228 = "tensor.extract_slice"(%2219) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2229 = "tensor.extract_slice"(%2219) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2230 = "tensor.extract_slice"(%2219) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2231 = "tensor.extract_slice"(%2219) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2232 = tensor.empty() : tensor<1x128xf32>
    %2233 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2228, %2229, %2230, %2205 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2232 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb216(%2234: f32, %2235: f32, %2236: f32, %2237: f32, %2238: f32):
      %2239 = arith.constant 1.000000e+00 : f32
      %2240 = arith.negf %2235 : f32
      %2241 = math.exp %2240 : f32
      %2242 = arith.addf %2239, %2241 : f32
      %2243 = arith.divf %2239, %2242 : f32
      %2244 = arith.constant 1.000000e+00 : f32
      %2245 = arith.negf %2234 : f32
      %2246 = math.exp %2245 : f32
      %2247 = arith.addf %2244, %2246 : f32
      %2248 = arith.divf %2244, %2247 : f32
      %2249 = math.tanh %2236 : f32
      %2250 = arith.mulf %2243, %2237 : f32
      %2251 = arith.mulf %2248, %2249 : f32
      %2252 = arith.addf %2250, %2251 : f32
      linalg.yield %2252 : f32
    } -> tensor<1x128xf32>
    %2253 = tensor.empty() : tensor<1x128xf32>
    %2254 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2231, %2233 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2253 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb217(%2255: f32, %2256: f32, %2257: f32):
      %2258 = arith.constant 1.000000e+00 : f32
      %2259 = arith.negf %2255 : f32
      %2260 = math.exp %2259 : f32
      %2261 = arith.addf %2258, %2260 : f32
      %2262 = arith.divf %2258, %2261 : f32
      %2263 = math.tanh %2256 : f32
      %2264 = arith.mulf %2262, %2263 : f32
      linalg.yield %2264 : f32
    } -> tensor<1x128xf32>
    %2265 = "tensor.extract_slice"(%2194) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2266 = tensor.collapse_shape %2265 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2267 = tensor.expand_shape %2266 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2268 = "tensor.extract_slice"(%2196) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2269 = tensor.collapse_shape %2268 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2270 = tensor.expand_shape %2269 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2271 = tensor.empty() : tensor<128x512xf32>
    %2272 = linalg.transpose ins(%86:tensor<512x128xf32>) outs(%2271:tensor<128x512xf32>) permutation = [1, 0]
    %2273 = tensor.empty() : tensor<1x512xf32>
    %2274 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2275 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2274 : f32) outs(%2273 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2276 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2254, %2272 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2275 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2277 = tensor.empty() : tensor<128x512xf32>
    %2278 = linalg.transpose ins(%87:tensor<512x128xf32>) outs(%2277:tensor<128x512xf32>) permutation = [1, 0]
    %2279 = tensor.empty() : tensor<1x512xf32>
    %2280 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2281 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2280 : f32) outs(%2279 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2282 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2267, %2278 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2281 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2283 = tensor.empty() : tensor<1x512xf32>
    %2284 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2276, %2282, %88, %89 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2283 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb218(%2285: f32, %2286: f32, %2287: f32, %2288: f32, %2289: f32):
      %2290 = arith.addf %2285, %2286 : f32
      %2291 = arith.addf %2290, %2287 : f32
      %2292 = arith.addf %2291, %2288 : f32
      linalg.yield %2292 : f32
    } -> tensor<1x512xf32>
    %2293 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2294 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2295 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2296 = "tensor.extract_slice"(%2284) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2297 = tensor.empty() : tensor<1x128xf32>
    %2298 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2293, %2294, %2295, %2270 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2297 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb219(%2299: f32, %2300: f32, %2301: f32, %2302: f32, %2303: f32):
      %2304 = arith.constant 1.000000e+00 : f32
      %2305 = arith.negf %2300 : f32
      %2306 = math.exp %2305 : f32
      %2307 = arith.addf %2304, %2306 : f32
      %2308 = arith.divf %2304, %2307 : f32
      %2309 = arith.constant 1.000000e+00 : f32
      %2310 = arith.negf %2299 : f32
      %2311 = math.exp %2310 : f32
      %2312 = arith.addf %2309, %2311 : f32
      %2313 = arith.divf %2309, %2312 : f32
      %2314 = math.tanh %2301 : f32
      %2315 = arith.mulf %2308, %2302 : f32
      %2316 = arith.mulf %2313, %2314 : f32
      %2317 = arith.addf %2315, %2316 : f32
      linalg.yield %2317 : f32
    } -> tensor<1x128xf32>
    %2318 = tensor.empty() : tensor<1x128xf32>
    %2319 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2296, %2298 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2318 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb220(%2320: f32, %2321: f32, %2322: f32):
      %2323 = arith.constant 1.000000e+00 : f32
      %2324 = arith.negf %2320 : f32
      %2325 = math.exp %2324 : f32
      %2326 = arith.addf %2323, %2325 : f32
      %2327 = arith.divf %2323, %2326 : f32
      %2328 = math.tanh %2321 : f32
      %2329 = arith.mulf %2327, %2328 : f32
      linalg.yield %2329 : f32
    } -> tensor<1x128xf32>
    %2330 = "tensor.extract_slice"(%2194) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2331 = tensor.collapse_shape %2330 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2332 = tensor.expand_shape %2331 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2333 = "tensor.extract_slice"(%2196) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2334 = tensor.collapse_shape %2333 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2335 = tensor.expand_shape %2334 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2336 = tensor.empty() : tensor<128x512xf32>
    %2337 = linalg.transpose ins(%90:tensor<512x128xf32>) outs(%2336:tensor<128x512xf32>) permutation = [1, 0]
    %2338 = tensor.empty() : tensor<1x512xf32>
    %2339 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2340 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2339 : f32) outs(%2338 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2341 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2319, %2337 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2340 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2342 = tensor.empty() : tensor<128x512xf32>
    %2343 = linalg.transpose ins(%91:tensor<512x128xf32>) outs(%2342:tensor<128x512xf32>) permutation = [1, 0]
    %2344 = tensor.empty() : tensor<1x512xf32>
    %2345 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2346 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2345 : f32) outs(%2344 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2347 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2332, %2343 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2346 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2348 = tensor.empty() : tensor<1x512xf32>
    %2349 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2341, %2347, %92, %93 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2348 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb221(%2350: f32, %2351: f32, %2352: f32, %2353: f32, %2354: f32):
      %2355 = arith.addf %2350, %2351 : f32
      %2356 = arith.addf %2355, %2352 : f32
      %2357 = arith.addf %2356, %2353 : f32
      linalg.yield %2357 : f32
    } -> tensor<1x512xf32>
    %2358 = "tensor.extract_slice"(%2349) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2359 = "tensor.extract_slice"(%2349) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2360 = "tensor.extract_slice"(%2349) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2361 = "tensor.extract_slice"(%2349) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2362 = tensor.empty() : tensor<1x128xf32>
    %2363 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2358, %2359, %2360, %2335 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2362 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb222(%2364: f32, %2365: f32, %2366: f32, %2367: f32, %2368: f32):
      %2369 = arith.constant 1.000000e+00 : f32
      %2370 = arith.negf %2365 : f32
      %2371 = math.exp %2370 : f32
      %2372 = arith.addf %2369, %2371 : f32
      %2373 = arith.divf %2369, %2372 : f32
      %2374 = arith.constant 1.000000e+00 : f32
      %2375 = arith.negf %2364 : f32
      %2376 = math.exp %2375 : f32
      %2377 = arith.addf %2374, %2376 : f32
      %2378 = arith.divf %2374, %2377 : f32
      %2379 = math.tanh %2366 : f32
      %2380 = arith.mulf %2373, %2367 : f32
      %2381 = arith.mulf %2378, %2379 : f32
      %2382 = arith.addf %2380, %2381 : f32
      linalg.yield %2382 : f32
    } -> tensor<1x128xf32>
    %2383 = tensor.empty() : tensor<1x128xf32>
    %2384 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2361, %2363 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2383 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb223(%2385: f32, %2386: f32, %2387: f32):
      %2388 = arith.constant 1.000000e+00 : f32
      %2389 = arith.negf %2385 : f32
      %2390 = math.exp %2389 : f32
      %2391 = arith.addf %2388, %2390 : f32
      %2392 = arith.divf %2388, %2391 : f32
      %2393 = math.tanh %2386 : f32
      %2394 = arith.mulf %2392, %2393 : f32
      linalg.yield %2394 : f32
    } -> tensor<1x128xf32>
    %2395 = tensor.empty() : tensor<1x1x128xf32>
    %2396 = tensor.collapse_shape %2384 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2397 = tensor.expand_shape %2396 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2398 = "tensor.insert_slice"(%2397, %2395) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2399 = tensor.empty() : tensor<3x1x128xf32>
    %2400 = tensor.collapse_shape %2254 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2401 = tensor.expand_shape %2400 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2402 = "tensor.insert_slice"(%2401, %2399) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2403 = tensor.collapse_shape %2319 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2404 = tensor.expand_shape %2403 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2405 = "tensor.insert_slice"(%2404, %2402) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2406 = tensor.collapse_shape %2384 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2407 = tensor.expand_shape %2406 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2408 = "tensor.insert_slice"(%2407, %2405) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2409 = tensor.empty() : tensor<3x1x128xf32>
    %2410 = tensor.collapse_shape %2233 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2411 = tensor.expand_shape %2410 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2412 = "tensor.insert_slice"(%2411, %2409) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2413 = tensor.collapse_shape %2298 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2414 = tensor.expand_shape %2413 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2415 = "tensor.insert_slice"(%2414, %2412) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2416 = tensor.collapse_shape %2363 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2417 = tensor.expand_shape %2416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2418 = "tensor.insert_slice"(%2417, %2415) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2419 = tensor.collapse_shape %2398 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2420 = tensor.expand_shape %2419 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2421 = tensor.collapse_shape %2408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2422 = tensor.expand_shape %2421 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2423 = tensor.collapse_shape %2418 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2424 = tensor.expand_shape %2423 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2425 = tensor.empty() : tensor<128x3xf32>
    %2426 = linalg.transpose ins(%145:tensor<3x128xf32>) outs(%2425:tensor<128x3xf32>) permutation = [1, 0]
    %2427 = tensor.empty() : tensor<1x3xf32>
    %2428 = linalg.transpose ins(%146:tensor<3x1xf32>) outs(%2427:tensor<1x3xf32>) permutation = [1, 0]
    %2429 = tensor.empty() : tensor<128x3xf32>
    %2430 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2426, %2428 : tensor<128x3xf32>, tensor<1x3xf32>) outs(%2429 : tensor<128x3xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2", prov.quant_inner_0 = "net.nn_fc2.weight.qdata", prov.quant_inner_1 = "net.nn_fc2.weight.scale"} {
    ^bb224(%2431: f32, %2432: f32, %2433: f32):
      %2434 = arith.mulf %2431, %2432 : f32
      linalg.yield %2434 : f32
    } -> tensor<128x3xf32>
    %2435 = tensor.empty() : tensor<1x3xf32>
    %2436 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2437 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2436 : f32) outs(%2435 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2438 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "matmul", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2420, %2430 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2437 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2439 = tensor.empty() : tensor<1x3xf32>
    %2440 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2438, %94 : tensor<1x3xf32>, tensor<3xf32>) outs(%2439 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb225(%2441: f32, %2442: f32, %2443: f32):
      %2444 = arith.addf %2441, %2442 : f32
      linalg.yield %2444 : f32
    } -> tensor<1x3xf32>
    func.return %2440, %2422, %2424 : tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>
  }
}
