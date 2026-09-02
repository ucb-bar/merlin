builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32x1x7x7xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<32x32x8x8xf32>, %5: tensor<32xf32>, %6: tensor<32xf32>, %7: tensor<32xf32>, %8: tensor<64x32xf32>, %9: tensor<64xf32>, %10: tensor<32x32xf32>, %11: tensor<32xf32>, %12: tensor<32x32xf32>, %13: tensor<32xf32>, %14: tensor<32x32x8x8xf32>, %15: tensor<32xf32>, %16: tensor<32xf32>, %17: tensor<32xf32>, %18: tensor<64x32xf32>, %19: tensor<64xf32>, %20: tensor<32x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32xf32>, %24: tensor<256x32xf32>, %25: tensor<256xf32>, %26: tensor<256x8x3x3xf32>, %27: tensor<256xf32>, %28: tensor<32x256xf32>, %29: tensor<32xf32>, %30: tensor<256x32xf32>, %31: tensor<256xf32>, %32: tensor<256x8x3x3xf32>, %33: tensor<256xf32>, %34: tensor<32x256xf32>, %35: tensor<32xf32>, %36: tensor<32xf32>, %37: tensor<32xf32>, %38: tensor<32xf32>, %39: tensor<32xf32>, %40: tensor<64x32x3x3xf32>, %41: tensor<64xf32>, %42: tensor<64xf32>, %43: tensor<64xf32>, %44: tensor<64x64x4x4xf32>, %45: tensor<64xf32>, %46: tensor<64xf32>, %47: tensor<64xf32>, %48: tensor<128x64xf32>, %49: tensor<128xf32>, %50: tensor<64x64xf32>, %51: tensor<64xf32>, %52: tensor<64x64xf32>, %53: tensor<64xf32>, %54: tensor<64x64x4x4xf32>, %55: tensor<64xf32>, %56: tensor<64xf32>, %57: tensor<64xf32>, %58: tensor<128x64xf32>, %59: tensor<128xf32>, %60: tensor<64x64xf32>, %61: tensor<64xf32>, %62: tensor<64x64xf32>, %63: tensor<64xf32>, %64: tensor<512x64xf32>, %65: tensor<512xf32>, %66: tensor<512x8x3x3xf32>, %67: tensor<512xf32>, %68: tensor<64x512xf32>, %69: tensor<64xf32>, %70: tensor<512x64xf32>, %71: tensor<512xf32>, %72: tensor<512x8x3x3xf32>, %73: tensor<512xf32>, %74: tensor<64x512xf32>, %75: tensor<64xf32>, %76: tensor<64xf32>, %77: tensor<64xf32>, %78: tensor<64xf32>, %79: tensor<64xf32>, %80: tensor<512xf32>, %81: tensor<512x4608xf32>, %82: tensor<512x517xf32>, %83: tensor<512x128xf32>, %84: tensor<512xf32>, %85: tensor<512xf32>, %86: tensor<512x128xf32>, %87: tensor<512x128xf32>, %88: tensor<512xf32>, %89: tensor<512xf32>, %90: tensor<512x128xf32>, %91: tensor<512x128xf32>, %92: tensor<512xf32>, %93: tensor<512xf32>, %94: tensor<3xf32>, %95: tensor<3x128xf32>, %96: tensor<12x48x3x3xf32>, %97: tensor<12xf32>, %98: tensor<1x1x60x90xf32>, %99: tensor<1x1xf32>, %100: tensor<1x4xf32>, %101: tensor<3x128xf32>, %102: tensor<3x128xf32>) -> (tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>) {
    %103 = tensor.empty() : tensor<64x32xi8>
    %104 = tensor.empty() : tensor<64x32xi8>
    %105 = tensor.empty() : tensor<64xf32>
    %106 = tensor.empty() : tensor<32x32xi8>
    %107 = tensor.empty() : tensor<32x32xi8>
    %108 = tensor.empty() : tensor<32xf32>
    %109 = tensor.empty() : tensor<32x32xi8>
    %110 = tensor.empty() : tensor<32x32xi8>
    %111 = tensor.empty() : tensor<32xf32>
    %112 = tensor.empty() : tensor<64x32xi8>
    %113 = tensor.empty() : tensor<64x32xi8>
    %114 = tensor.empty() : tensor<64xf32>
    %115 = tensor.empty() : tensor<32x32xi8>
    %116 = tensor.empty() : tensor<32x32xi8>
    %117 = tensor.empty() : tensor<32xf32>
    %118 = tensor.empty() : tensor<32x32xi8>
    %119 = tensor.empty() : tensor<32x32xi8>
    %120 = tensor.empty() : tensor<32xf32>
    %121 = tensor.empty() : tensor<256x32xi8>
    %122 = tensor.empty() : tensor<256x32xi8>
    %123 = tensor.empty() : tensor<256xf32>
    %124 = tensor.empty() : tensor<32x256xi8>
    %125 = tensor.empty() : tensor<32x256xi8>
    %126 = tensor.empty() : tensor<32xf32>
    %127 = tensor.empty() : tensor<256x32xi8>
    %128 = tensor.empty() : tensor<256x32xi8>
    %129 = tensor.empty() : tensor<256xf32>
    %130 = tensor.empty() : tensor<32x256xi8>
    %131 = tensor.empty() : tensor<32x256xi8>
    %132 = tensor.empty() : tensor<32xf32>
    %133 = tensor.empty() : tensor<128x64xi8>
    %134 = tensor.empty() : tensor<128x64xi8>
    %135 = tensor.empty() : tensor<128xf32>
    %136 = tensor.empty() : tensor<64x64xi8>
    %137 = tensor.empty() : tensor<64x64xi8>
    %138 = tensor.empty() : tensor<64xf32>
    %139 = tensor.empty() : tensor<64x64xi8>
    %140 = tensor.empty() : tensor<64x64xi8>
    %141 = tensor.empty() : tensor<64xf32>
    %142 = tensor.empty() : tensor<128x64xi8>
    %143 = tensor.empty() : tensor<128x64xi8>
    %144 = tensor.empty() : tensor<128xf32>
    %145 = tensor.empty() : tensor<64x64xi8>
    %146 = tensor.empty() : tensor<64x64xi8>
    %147 = tensor.empty() : tensor<64xf32>
    %148 = tensor.empty() : tensor<64x64xi8>
    %149 = tensor.empty() : tensor<64x64xi8>
    %150 = tensor.empty() : tensor<64xf32>
    %151 = tensor.empty() : tensor<512x64xi8>
    %152 = tensor.empty() : tensor<512x64xi8>
    %153 = tensor.empty() : tensor<512xf32>
    %154 = tensor.empty() : tensor<64x512xi8>
    %155 = tensor.empty() : tensor<64x512xi8>
    %156 = tensor.empty() : tensor<64xf32>
    %157 = tensor.empty() : tensor<512x64xi8>
    %158 = tensor.empty() : tensor<512x64xi8>
    %159 = tensor.empty() : tensor<512xf32>
    %160 = tensor.empty() : tensor<64x512xi8>
    %161 = tensor.empty() : tensor<64x512xi8>
    %162 = tensor.empty() : tensor<64xf32>
    %163 = tensor.empty() : tensor<512x4608xi8>
    %164 = tensor.empty() : tensor<512x4608xi8>
    %165 = tensor.empty() : tensor<512xf32>
    %166 = tensor.empty() : tensor<3x128xi8>
    %167 = tensor.empty() : tensor<3x128xi8>
    %168 = tensor.empty() : tensor<3xf32>
    %169 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %170 = tensor.splat %169 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x1x66x96xf32>
    %171 = "tensor.insert_slice"(%98, %170) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 1, 60, 90>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : (tensor<1x1x60x90xf32>, tensor<1x1x66x96xf32>) -> tensor<1x1x66x96xf32>
    %172 = tensor.empty() : tensor<1x7x7x1x15x23xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%171 : tensor<1x1x66x96xf32>) outs(%172 : tensor<1x7x7x1x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb0(%174: f32, %175: f32):
      linalg.yield %174 : f32
    } -> tensor<1x7x7x1x15x23xf32>
    %176 = tensor.collapse_shape %173 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1x7x7x1x15x23xf32> into tensor<16905xf32>
    %177 = tensor.expand_shape %176 [[0 : i64, 1 : i64]] output_shape [49, 345] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<16905xf32> into tensor<49x345xf32>
    %178 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x7x7xf32> into tensor<1568xf32>
    %179 = tensor.expand_shape %178 [[0 : i64, 1 : i64]] output_shape [32, 49] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<1568xf32> into tensor<32x49xf32>
    %180 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} 0.000000e+00 : f32
    %181 = tensor.splat %180 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32>
    %182 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} ins(%179, %177 : tensor<32x49xf32>, tensor<49x345xf32>) outs(%181 : tensor<32x345xf32>) -> tensor<32x345xf32>
    %183 = tensor.collapse_shape %182 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x345xf32> into tensor<11040xf32>
    %184 = tensor.expand_shape %183 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<32x1x15x23xf32>
    %185 = tensor.collapse_shape %184 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<32x1x15x23xf32> into tensor<11040xf32>
    %186 = tensor.expand_shape %185 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %187 = tensor.empty() : tensor<1x32x15x23xf32>
    %188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186, %1 : tensor<1x32x15x23xf32>, tensor<32xf32>) outs(%187 : tensor<1x32x15x23xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.cn1"} {
    ^bb1(%189: f32, %190: f32, %191: f32):
      %192 = arith.addf %189, %190 : f32
      linalg.yield %192 : f32
    } -> tensor<1x32x15x23xf32>
    %193 = tensor.collapse_shape %188 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<1x32x15x23xf32> into tensor<11040xf32>
    %194 = tensor.expand_shape %193 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 345] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge"} : tensor<11040xf32> into tensor<1x32x345xf32>
    %195 = tensor.empty() : tensor<1x345x32xf32>
    %196 = linalg.transpose ins(%194:tensor<1x32x345xf32>) outs(%195:tensor<1x345x32xf32>) permutation = [0, 2, 1]
    %197 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %198 = tensor.splat %197 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %199 = linalg.reduce ins(%196:tensor<1x345x32xf32>) outs(%198:tensor<1x345xf32>) dimensions = [2]
    (%200: f32, %201: f32) {
      %202 = arith.addf %200, %201 : f32
      linalg.yield %202 : f32
    }
    %203 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %204 = tensor.splat %203 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %205 = tensor.empty() : tensor<1x345xf32>
    %206 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%199, %204 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%205 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb2(%207: f32, %208: f32, %209: f32):
      %210 = arith.divf %207, %208 : f32
      linalg.yield %210 : f32
    } -> tensor<1x345xf32>
    %211 = tensor.collapse_shape %206 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %212 = tensor.expand_shape %211 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %213 = tensor.empty() : tensor<1x345x32xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%196, %212 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%213 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb3(%215: f32, %216: f32, %217: f32):
      %218 = arith.subf %215, %216 : f32
      linalg.yield %218 : f32
    } -> tensor<1x345x32xf32>
    %219 = tensor.empty() : tensor<1x345x32xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %214 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%219 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb4(%221: f32, %222: f32, %223: f32):
      %224 = arith.mulf %221, %222 : f32
      linalg.yield %224 : f32
    } -> tensor<1x345x32xf32>
    %225 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 0.000000e+00 : f32
    %226 = tensor.splat %225 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %227 = linalg.reduce ins(%220:tensor<1x345x32xf32>) outs(%226:tensor<1x345xf32>) dimensions = [2]
    (%228: f32, %229: f32) {
      %230 = arith.addf %228, %229 : f32
      linalg.yield %230 : f32
    }
    %231 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 3.200000e+01 : f32
    %232 = tensor.splat %231 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32>
    %233 = tensor.empty() : tensor<1x345xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%227, %232 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%233 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb5(%235: f32, %236: f32, %237: f32):
      %238 = arith.divf %235, %236 : f32
      linalg.yield %238 : f32
    } -> tensor<1x345xf32>
    %239 = tensor.collapse_shape %234 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345xf32> into tensor<345xf32>
    %240 = tensor.expand_shape %239 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<345xf32> into tensor<1x345x1xf32>
    %241 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} 1.000000e-05 : f32
    %242 = tensor.splat %241 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} : tensor<1x345x1xf32>
    %243 = tensor.empty() : tensor<1x345x1xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%240, %242 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%243 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb6(%245: f32, %246: f32, %247: f32):
      %248 = arith.addf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x345x1xf32>
    %249 = tensor.empty() : tensor<1x345x1xf32>
    %250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%244 : tensor<1x345x1xf32>) outs(%249 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb7(%251: f32, %252: f32):
      %253 = math.rsqrt %251 : f32
      linalg.yield %253 : f32
    } -> tensor<1x345x1xf32>
    %254 = tensor.empty() : tensor<1x345x32xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %250 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%254 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb8(%256: f32, %257: f32, %258: f32):
      %259 = arith.mulf %256, %257 : f32
      linalg.yield %259 : f32
    } -> tensor<1x345x32xf32>
    %260 = tensor.empty() : tensor<1x345x32xf32>
    %261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%255, %2 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%260 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb9(%262: f32, %263: f32, %264: f32):
      %265 = arith.mulf %262, %263 : f32
      linalg.yield %265 : f32
    } -> tensor<1x345x32xf32>
    %266 = tensor.empty() : tensor<1x345x32xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%261, %3 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%266 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0.patchMerge.layerNorm"} {
    ^bb10(%268: f32, %269: f32, %270: f32):
      %271 = arith.addf %268, %269 : f32
      linalg.yield %271 : f32
    } -> tensor<1x345x32xf32>
    %272 = tensor.empty() : tensor<1x32x345xf32>
    %273 = linalg.transpose ins(%267:tensor<1x345x32xf32>) outs(%272:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %274 = tensor.collapse_shape %273 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %275 = tensor.expand_shape %274 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %276 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%275 : tensor<1x32x15x23xf32>) outs(%276 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb11(%278: f32, %279: f32):
      linalg.yield %278 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %280 = tensor.collapse_shape %277 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %281 = tensor.expand_shape %280 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %282 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %283 = tensor.expand_shape %282 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %284 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} 0.000000e+00 : f32
    %285 = tensor.splat %284 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32>
    %286 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} ins(%283, %281 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%285 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %287 = tensor.collapse_shape %286 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %288 = tensor.expand_shape %287 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %289 = tensor.collapse_shape %288 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %290 = tensor.expand_shape %289 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %291 = tensor.empty() : tensor<1x32x1x2xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290, %5 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%291 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.cn1"} {
    ^bb12(%293: f32, %294: f32, %295: f32):
      %296 = arith.addf %293, %294 : f32
      linalg.yield %296 : f32
    } -> tensor<1x32x1x2xf32>
    %297 = tensor.collapse_shape %292 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %298 = tensor.expand_shape %297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %299 = tensor.empty() : tensor<1x2x32xf32>
    %300 = linalg.transpose ins(%298:tensor<1x32x2xf32>) outs(%299:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %301 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %302 = tensor.splat %301 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %303 = linalg.reduce ins(%300:tensor<1x2x32xf32>) outs(%302:tensor<1x2xf32>) dimensions = [2]
    (%304: f32, %305: f32) {
      %306 = arith.addf %304, %305 : f32
      linalg.yield %306 : f32
    }
    %307 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %308 = tensor.splat %307 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %309 = tensor.empty() : tensor<1x2xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%303, %308 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%309 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb13(%311: f32, %312: f32, %313: f32):
      %314 = arith.divf %311, %312 : f32
      linalg.yield %314 : f32
    } -> tensor<1x2xf32>
    %315 = tensor.collapse_shape %310 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %316 = tensor.expand_shape %315 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %317 = tensor.empty() : tensor<1x2x32xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%300, %316 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%317 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb14(%319: f32, %320: f32, %321: f32):
      %322 = arith.subf %319, %320 : f32
      linalg.yield %322 : f32
    } -> tensor<1x2x32xf32>
    %323 = tensor.empty() : tensor<1x2x32xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%318, %318 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%323 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb15(%325: f32, %326: f32, %327: f32):
      %328 = arith.mulf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x2x32xf32>
    %329 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 0.000000e+00 : f32
    %330 = tensor.splat %329 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %331 = linalg.reduce ins(%324:tensor<1x2x32xf32>) outs(%330:tensor<1x2xf32>) dimensions = [2]
    (%332: f32, %333: f32) {
      %334 = arith.addf %332, %333 : f32
      linalg.yield %334 : f32
    }
    %335 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 3.200000e+01 : f32
    %336 = tensor.splat %335 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32>
    %337 = tensor.empty() : tensor<1x2xf32>
    %338 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%331, %336 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%337 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb16(%339: f32, %340: f32, %341: f32):
      %342 = arith.divf %339, %340 : f32
      linalg.yield %342 : f32
    } -> tensor<1x2xf32>
    %343 = tensor.collapse_shape %338 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %344 = tensor.expand_shape %343 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %345 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} 1.000000e-05 : f32
    %346 = tensor.splat %345 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} : tensor<1x2x1xf32>
    %347 = tensor.empty() : tensor<1x2x1xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%344, %346 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%347 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb17(%349: f32, %350: f32, %351: f32):
      %352 = arith.addf %349, %350 : f32
      linalg.yield %352 : f32
    } -> tensor<1x2x1xf32>
    %353 = tensor.empty() : tensor<1x2x1xf32>
    %354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%348 : tensor<1x2x1xf32>) outs(%353 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb18(%355: f32, %356: f32):
      %357 = math.rsqrt %355 : f32
      linalg.yield %357 : f32
    } -> tensor<1x2x1xf32>
    %358 = tensor.empty() : tensor<1x2x32xf32>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%318, %354 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%358 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb19(%360: f32, %361: f32, %362: f32):
      %363 = arith.mulf %360, %361 : f32
      linalg.yield %363 : f32
    } -> tensor<1x2x32xf32>
    %364 = tensor.empty() : tensor<1x2x32xf32>
    %365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%359, %6 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%364 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb20(%366: f32, %367: f32, %368: f32):
      %369 = arith.mulf %366, %367 : f32
      linalg.yield %369 : f32
    } -> tensor<1x2x32xf32>
    %370 = tensor.empty() : tensor<1x2x32xf32>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%365, %7 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%370 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.ln1"} {
    ^bb21(%372: f32, %373: f32, %374: f32):
      %375 = arith.addf %372, %373 : f32
      linalg.yield %375 : f32
    } -> tensor<1x2x32xf32>
    %376 = tensor.empty() : tensor<32x64xi8>
    %377 = linalg.transpose ins(%104:tensor<64x32xi8>) outs(%376:tensor<32x64xi8>) permutation = [1, 0]
    %378 = tensor.collapse_shape %371 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %379 = tensor.expand_shape %378 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %380 = tensor.empty() : tensor<32x64xf32>
    %381 = arith.constant 0 : i32
    %382 = tensor.splat %381 : tensor<64xi32>
    %383 = "quant_ext.dequantize_per_channel"(%377, %105, %382) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.0.keyValueExtractor.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.0.keyValueExtractor.weight.tensor_impl.scale"} : (tensor<32x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<32x64xf32>
    %384 = tensor.empty() : tensor<2x64xf32>
    %385 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %386 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%385 : f32) outs(%384 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %387 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} ins(%379, %383 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%386 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %388 = tensor.empty() : tensor<2x64xf32>
    %389 = tensor.collapse_shape %387 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %390 = tensor.expand_shape %389 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %391 = tensor.empty() : tensor<1x2x64xf32>
    %392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%390, %9 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%391 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.keyValueExtractor"} {
    ^bb22(%393: f32, %394: f32, %395: f32):
      %396 = arith.addf %393, %394 : f32
      linalg.yield %396 : f32
    } -> tensor<1x2x64xf32>
    %397 = tensor.collapse_shape %392 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %398 = tensor.expand_shape %397 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %399 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %400 = linalg.transpose ins(%398:tensor<1x2x2x1x32xf32>) outs(%399:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %401 = "tensor.extract_slice"(%400) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %402 = tensor.collapse_shape %401 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %404 = "tensor.extract_slice"(%400) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %405 = tensor.collapse_shape %404 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %406 = tensor.expand_shape %405 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %407 = tensor.empty() : tensor<32x32xi8>
    %408 = linalg.transpose ins(%107:tensor<32x32xi8>) outs(%407:tensor<32x32xi8>) permutation = [1, 0]
    %409 = tensor.collapse_shape %267 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %410 = tensor.expand_shape %409 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %411 = tensor.empty() : tensor<32x32xf32>
    %412 = arith.constant 0 : i32
    %413 = tensor.splat %412 : tensor<32xi32>
    %414 = "quant_ext.dequantize_per_channel"(%408, %108, %413) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.0.query.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.0.query.weight.tensor_impl.scale"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %415 = tensor.empty() : tensor<345x32xf32>
    %416 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %417 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%416 : f32) outs(%415 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %418 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} ins(%410, %414 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%417 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %419 = tensor.empty() : tensor<345x32xf32>
    %420 = tensor.collapse_shape %418 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %421 = tensor.expand_shape %420 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %422 = tensor.empty() : tensor<1x345x32xf32>
    %423 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%421, %11 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%422 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.query"} {
    ^bb23(%424: f32, %425: f32, %426: f32):
      %427 = arith.addf %424, %425 : f32
      linalg.yield %427 : f32
    } -> tensor<1x345x32xf32>
    %428 = tensor.collapse_shape %423 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %429 = tensor.expand_shape %428 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %430 = tensor.empty() : tensor<1x1x345x32xf32>
    %431 = linalg.transpose ins(%429:tensor<1x345x1x32xf32>) outs(%430:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %432 = tensor.empty() : tensor<1x1x32x2xf32>
    %433 = linalg.transpose ins(%403:tensor<1x1x2x32xf32>) outs(%432:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %434 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %435 = tensor.splat %434 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%431, %433 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%435 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb24(%437: f32, %438: f32, %439: f32):
      %440 = arith.mulf %437, %438 : f32
      %441 = arith.addf %439, %440 : f32
      linalg.yield %441 : f32
    } -> tensor<1x1x345x2xf32>
    %442 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %443 = tensor.splat %442 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %444 = tensor.empty() : tensor<1x1x345x2xf32>
    %445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%436, %443 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%444 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb25(%446: f32, %447: f32, %448: f32):
      %449 = arith.divf %446, %447 : f32
      linalg.yield %449 : f32
    } -> tensor<1x1x345x2xf32>
    %450 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0xff800000 : f32
    %451 = tensor.splat %450 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %452 = linalg.reduce ins(%445:tensor<1x1x345x2xf32>) outs(%451:tensor<1x1x345xf32>) dimensions = [3]
    (%453: f32, %454: f32) {
      %455 = arith.maximumf %453, %454 : f32
      linalg.yield %455 : f32
    }
    %456 = tensor.collapse_shape %452 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %457 = tensor.expand_shape %456 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %458 = tensor.empty() : tensor<1x1x345x2xf32>
    %459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%445, %457 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%458 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb26(%460: f32, %461: f32, %462: f32):
      %463 = arith.subf %460, %461 : f32
      linalg.yield %463 : f32
    } -> tensor<1x1x345x2xf32>
    %464 = tensor.empty() : tensor<1x1x345x2xf32>
    %465 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%459 : tensor<1x1x345x2xf32>) outs(%464 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb27(%466: f32, %467: f32):
      %468 = math.exp %466 : f32
      linalg.yield %468 : f32
    } -> tensor<1x1x345x2xf32>
    %469 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} 0.000000e+00 : f32
    %470 = tensor.splat %469 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32>
    %471 = linalg.reduce ins(%465:tensor<1x1x345x2xf32>) outs(%470:tensor<1x1x345xf32>) dimensions = [3]
    (%472: f32, %473: f32) {
      %474 = arith.addf %472, %473 : f32
      linalg.yield %474 : f32
    }
    %475 = tensor.collapse_shape %471 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %476 = tensor.expand_shape %475 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %477 = tensor.empty() : tensor<1x1x345x2xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%465, %476 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%477 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.smax"} {
    ^bb28(%479: f32, %480: f32, %481: f32):
      %482 = arith.divf %479, %480 : f32
      linalg.yield %482 : f32
    } -> tensor<1x1x345x2xf32>
    %483 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %484 = tensor.splat %483 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%478, %406 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%484 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb29(%486: f32, %487: f32, %488: f32):
      %489 = arith.mulf %486, %487 : f32
      %490 = arith.addf %488, %489 : f32
      linalg.yield %490 : f32
    } -> tensor<1x1x345x32xf32>
    %491 = tensor.empty() : tensor<1x345x1x32xf32>
    %492 = linalg.transpose ins(%485:tensor<1x1x345x32xf32>) outs(%491:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %493 = tensor.collapse_shape %492 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %494 = tensor.expand_shape %493 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %495 = tensor.empty() : tensor<32x32xi8>
    %496 = linalg.transpose ins(%110:tensor<32x32xi8>) outs(%495:tensor<32x32xi8>) permutation = [1, 0]
    %497 = tensor.collapse_shape %494 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %498 = tensor.expand_shape %497 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %499 = tensor.empty() : tensor<32x32xf32>
    %500 = arith.constant 0 : i32
    %501 = tensor.splat %500 : tensor<32xi32>
    %502 = "quant_ext.dequantize_per_channel"(%496, %111, %501) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.0.finalLayer.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.0.finalLayer.weight.tensor_impl.scale"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %503 = tensor.empty() : tensor<345x32xf32>
    %504 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %505 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%504 : f32) outs(%503 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %506 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} ins(%498, %502 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%505 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %507 = tensor.empty() : tensor<345x32xf32>
    %508 = tensor.collapse_shape %506 [[0 : i64, 1 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %509 = tensor.expand_shape %508 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %510 = tensor.empty() : tensor<1x345x32xf32>
    %511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509, %13 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%510 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.0.finalLayer"} {
    ^bb30(%512: f32, %513: f32, %514: f32):
      %515 = arith.addf %512, %513 : f32
      linalg.yield %515 : f32
    } -> tensor<1x345x32xf32>
    %516 = tensor.empty() : tensor<1x345x32xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%267, %511 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%516 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb31(%518: f32, %519: f32, %520: f32):
      %521 = arith.addf %518, %519 : f32
      linalg.yield %521 : f32
    } -> tensor<1x345x32xf32>
    %522 = tensor.empty() : tensor<32x256xi8>
    %523 = linalg.transpose ins(%122:tensor<256x32xi8>) outs(%522:tensor<32x256xi8>) permutation = [1, 0]
    %524 = tensor.collapse_shape %517 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %525 = tensor.expand_shape %524 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %526 = tensor.empty() : tensor<32x256xf32>
    %527 = arith.constant 0 : i32
    %528 = tensor.splat %527 : tensor<256xi32>
    %529 = "quant_ext.dequantize_per_channel"(%523, %123, %528) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._ffn.0.mlp1.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._ffn.0.mlp1.weight.tensor_impl.scale"} : (tensor<32x256xi8>, tensor<256xf32>, tensor<256xi32>) -> tensor<32x256xf32>
    %530 = tensor.empty() : tensor<345x256xf32>
    %531 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %532 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%531 : f32) outs(%530 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %533 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} ins(%525, %529 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%532 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %534 = tensor.empty() : tensor<345x256xf32>
    %535 = tensor.collapse_shape %533 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %536 = tensor.expand_shape %535 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %537 = tensor.empty() : tensor<1x345x256xf32>
    %538 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%536, %25 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%537 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp1"} {
    ^bb32(%539: f32, %540: f32, %541: f32):
      %542 = arith.addf %539, %540 : f32
      linalg.yield %542 : f32
    } -> tensor<1x345x256xf32>
    %543 = tensor.empty() : tensor<1x256x345xf32>
    %544 = linalg.transpose ins(%538:tensor<1x345x256xf32>) outs(%543:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %545 = tensor.collapse_shape %544 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %546 = tensor.expand_shape %545 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %547 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %548 = tensor.splat %547 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<1x256x17x25xf32>
    %549 = "tensor.insert_slice"(%546, %548) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %550 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%549 : tensor<1x256x17x25xf32>) outs(%550 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb33(%552: f32, %553: f32):
      linalg.yield %552 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %554 = tensor.collapse_shape %551 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %555 = tensor.expand_shape %554 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %556 = tensor.collapse_shape %26 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %557 = tensor.expand_shape %556 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %558 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} 0.000000e+00 : f32
    %559 = tensor.splat %558 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32>
    %560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%557, %555 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%559 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb34(%561: f32, %562: f32, %563: f32):
      %564 = arith.mulf %561, %562 : f32
      %565 = arith.addf %563, %564 : f32
      linalg.yield %565 : f32
    } -> tensor<32x8x345xf32>
    %566 = tensor.collapse_shape %560 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %567 = tensor.expand_shape %566 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %568 = tensor.collapse_shape %567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %569 = tensor.expand_shape %568 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %570 = tensor.empty() : tensor<1x256x15x23xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%569, %27 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%570 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.depthwise"} {
    ^bb35(%572: f32, %573: f32, %574: f32):
      %575 = arith.addf %572, %573 : f32
      linalg.yield %575 : f32
    } -> tensor<1x256x15x23xf32>
    %576 = tensor.collapse_shape %571 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %577 = tensor.expand_shape %576 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %578 = tensor.empty() : tensor<1x345x256xf32>
    %579 = linalg.transpose ins(%577:tensor<1x256x345xf32>) outs(%578:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %580 = tensor.empty() : tensor<1x345x256xf32>
    %581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%579 : tensor<1x345x256xf32>) outs(%580 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.gelu"} {
    ^bb36(%582: f32, %583: f32):
      %584 = arith.constant 5.000000e-01 : f32
      %585 = arith.constant 1.000000e+00 : f32
      %586 = arith.constant 0.707106769 : f32
      %587 = arith.mulf %582, %586 : f32
      %588 = math.erf %587 : f32
      %589 = arith.addf %585, %588 : f32
      %590 = arith.mulf %584, %582 : f32
      %591 = arith.mulf %590, %589 : f32
      linalg.yield %591 : f32
    } -> tensor<1x345x256xf32>
    %592 = tensor.empty() : tensor<256x32xi8>
    %593 = linalg.transpose ins(%125:tensor<32x256xi8>) outs(%592:tensor<256x32xi8>) permutation = [1, 0]
    %594 = tensor.collapse_shape %581 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %595 = tensor.expand_shape %594 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %596 = tensor.empty() : tensor<256x32xf32>
    %597 = arith.constant 0 : i32
    %598 = tensor.splat %597 : tensor<32xi32>
    %599 = "quant_ext.dequantize_per_channel"(%593, %126, %598) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._ffn.0.mlp2.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._ffn.0.mlp2.weight.tensor_impl.scale"} : (tensor<256x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<256x32xf32>
    %600 = tensor.empty() : tensor<345x32xf32>
    %601 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %602 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%601 : f32) outs(%600 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %603 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} ins(%595, %599 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%602 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %604 = tensor.empty() : tensor<345x32xf32>
    %605 = tensor.collapse_shape %603 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %606 = tensor.expand_shape %605 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %607 = tensor.empty() : tensor<1x345x32xf32>
    %608 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606, %29 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%607 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.0.mlp2"} {
    ^bb37(%609: f32, %610: f32, %611: f32):
      %612 = arith.addf %609, %610 : f32
      linalg.yield %612 : f32
    } -> tensor<1x345x32xf32>
    %613 = tensor.empty() : tensor<1x345x32xf32>
    %614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%517, %608 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%613 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb38(%615: f32, %616: f32, %617: f32):
      %618 = arith.addf %615, %616 : f32
      linalg.yield %618 : f32
    } -> tensor<1x345x32xf32>
    %619 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %620 = tensor.splat %619 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %621 = linalg.reduce ins(%614:tensor<1x345x32xf32>) outs(%620:tensor<1x345xf32>) dimensions = [2]
    (%622: f32, %623: f32) {
      %624 = arith.addf %622, %623 : f32
      linalg.yield %624 : f32
    }
    %625 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %626 = tensor.splat %625 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %627 = tensor.empty() : tensor<1x345xf32>
    %628 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%621, %626 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%627 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb39(%629: f32, %630: f32, %631: f32):
      %632 = arith.divf %629, %630 : f32
      linalg.yield %632 : f32
    } -> tensor<1x345xf32>
    %633 = tensor.collapse_shape %628 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %634 = tensor.expand_shape %633 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %635 = tensor.empty() : tensor<1x345x32xf32>
    %636 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614, %634 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%635 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb40(%637: f32, %638: f32, %639: f32):
      %640 = arith.subf %637, %638 : f32
      linalg.yield %640 : f32
    } -> tensor<1x345x32xf32>
    %641 = tensor.empty() : tensor<1x345x32xf32>
    %642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%636, %636 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%641 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb41(%643: f32, %644: f32, %645: f32):
      %646 = arith.mulf %643, %644 : f32
      linalg.yield %646 : f32
    } -> tensor<1x345x32xf32>
    %647 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %648 = tensor.splat %647 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %649 = linalg.reduce ins(%642:tensor<1x345x32xf32>) outs(%648:tensor<1x345xf32>) dimensions = [2]
    (%650: f32, %651: f32) {
      %652 = arith.addf %650, %651 : f32
      linalg.yield %652 : f32
    }
    %653 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %654 = tensor.splat %653 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %655 = tensor.empty() : tensor<1x345xf32>
    %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%649, %654 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%655 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb42(%657: f32, %658: f32, %659: f32):
      %660 = arith.divf %657, %658 : f32
      linalg.yield %660 : f32
    } -> tensor<1x345xf32>
    %661 = tensor.collapse_shape %656 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %662 = tensor.expand_shape %661 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %663 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %664 = tensor.splat %663 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %665 = tensor.empty() : tensor<1x345x1xf32>
    %666 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%662, %664 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%665 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb43(%667: f32, %668: f32, %669: f32):
      %670 = arith.addf %667, %668 : f32
      linalg.yield %670 : f32
    } -> tensor<1x345x1xf32>
    %671 = tensor.empty() : tensor<1x345x1xf32>
    %672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%666 : tensor<1x345x1xf32>) outs(%671 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb44(%673: f32, %674: f32):
      %675 = math.rsqrt %673 : f32
      linalg.yield %675 : f32
    } -> tensor<1x345x1xf32>
    %676 = tensor.empty() : tensor<1x345x32xf32>
    %677 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%636, %672 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%676 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb45(%678: f32, %679: f32, %680: f32):
      %681 = arith.mulf %678, %679 : f32
      linalg.yield %681 : f32
    } -> tensor<1x345x32xf32>
    %682 = tensor.empty() : tensor<1x345x32xf32>
    %683 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677, %36 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%682 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb46(%684: f32, %685: f32, %686: f32):
      %687 = arith.mulf %684, %685 : f32
      linalg.yield %687 : f32
    } -> tensor<1x345x32xf32>
    %688 = tensor.empty() : tensor<1x345x32xf32>
    %689 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%683, %37 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%688 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb47(%690: f32, %691: f32, %692: f32):
      %693 = arith.addf %690, %691 : f32
      linalg.yield %693 : f32
    } -> tensor<1x345x32xf32>
    %694 = tensor.empty() : tensor<1x32x345xf32>
    %695 = linalg.transpose ins(%689:tensor<1x345x32xf32>) outs(%694:tensor<1x32x345xf32>) permutation = [0, 2, 1]
    %696 = tensor.collapse_shape %695 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x345xf32> into tensor<11040xf32>
    %697 = tensor.expand_shape %696 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 23] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x32x15x23xf32>
    %698 = tensor.empty() : tensor<32x8x8x1x1x2xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 8) + d1), ((d5 * 8) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%697 : tensor<1x32x15x23xf32>) outs(%698 : tensor<32x8x8x1x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb48(%700: f32, %701: f32):
      linalg.yield %700 : f32
    } -> tensor<32x8x8x1x1x2xf32>
    %702 = tensor.collapse_shape %699 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x8x8x1x1x2xf32> into tensor<4096xf32>
    %703 = tensor.expand_shape %702 [[0 : i64, 1 : i64]] output_shape [2048, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<4096xf32> into tensor<2048x2xf32>
    %704 = tensor.collapse_shape %14 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x32x8x8xf32> into tensor<65536xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<65536xf32> into tensor<32x2048xf32>
    %706 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} 0.000000e+00 : f32
    %707 = tensor.splat %706 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32>
    %708 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} ins(%705, %703 : tensor<32x2048xf32>, tensor<2048x2xf32>) outs(%707 : tensor<32x2xf32>) -> tensor<32x2xf32>
    %709 = tensor.collapse_shape %708 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x2xf32> into tensor<64xf32>
    %710 = tensor.expand_shape %709 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [32, 1, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<32x1x1x2xf32>
    %711 = tensor.collapse_shape %710 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<32x1x1x2xf32> into tensor<64xf32>
    %712 = tensor.expand_shape %711 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 2] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} : tensor<64xf32> into tensor<1x32x1x2xf32>
    %713 = tensor.empty() : tensor<1x32x1x2xf32>
    %714 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%712, %15 : tensor<1x32x1x2xf32>, tensor<32xf32>) outs(%713 : tensor<1x32x1x2xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.cn1"} {
    ^bb49(%715: f32, %716: f32, %717: f32):
      %718 = arith.addf %715, %716 : f32
      linalg.yield %718 : f32
    } -> tensor<1x32x1x2xf32>
    %719 = tensor.collapse_shape %714 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x32x1x2xf32> into tensor<64xf32>
    %720 = tensor.expand_shape %719 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 2] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x32x2xf32>
    %721 = tensor.empty() : tensor<1x2x32xf32>
    %722 = linalg.transpose ins(%720:tensor<1x32x2xf32>) outs(%721:tensor<1x2x32xf32>) permutation = [0, 2, 1]
    %723 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %724 = tensor.splat %723 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %725 = linalg.reduce ins(%722:tensor<1x2x32xf32>) outs(%724:tensor<1x2xf32>) dimensions = [2]
    (%726: f32, %727: f32) {
      %728 = arith.addf %726, %727 : f32
      linalg.yield %728 : f32
    }
    %729 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %730 = tensor.splat %729 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %731 = tensor.empty() : tensor<1x2xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%725, %730 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%731 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb50(%733: f32, %734: f32, %735: f32):
      %736 = arith.divf %733, %734 : f32
      linalg.yield %736 : f32
    } -> tensor<1x2xf32>
    %737 = tensor.collapse_shape %732 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %738 = tensor.expand_shape %737 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %739 = tensor.empty() : tensor<1x2x32xf32>
    %740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722, %738 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%739 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb51(%741: f32, %742: f32, %743: f32):
      %744 = arith.subf %741, %742 : f32
      linalg.yield %744 : f32
    } -> tensor<1x2x32xf32>
    %745 = tensor.empty() : tensor<1x2x32xf32>
    %746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%740, %740 : tensor<1x2x32xf32>, tensor<1x2x32xf32>) outs(%745 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb52(%747: f32, %748: f32, %749: f32):
      %750 = arith.mulf %747, %748 : f32
      linalg.yield %750 : f32
    } -> tensor<1x2x32xf32>
    %751 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 0.000000e+00 : f32
    %752 = tensor.splat %751 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %753 = linalg.reduce ins(%746:tensor<1x2x32xf32>) outs(%752:tensor<1x2xf32>) dimensions = [2]
    (%754: f32, %755: f32) {
      %756 = arith.addf %754, %755 : f32
      linalg.yield %756 : f32
    }
    %757 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 3.200000e+01 : f32
    %758 = tensor.splat %757 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32>
    %759 = tensor.empty() : tensor<1x2xf32>
    %760 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%753, %758 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%759 : tensor<1x2xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb53(%761: f32, %762: f32, %763: f32):
      %764 = arith.divf %761, %762 : f32
      linalg.yield %764 : f32
    } -> tensor<1x2xf32>
    %765 = tensor.collapse_shape %760 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2xf32> into tensor<2xf32>
    %766 = tensor.expand_shape %765 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<2xf32> into tensor<1x2x1xf32>
    %767 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} 1.000000e-05 : f32
    %768 = tensor.splat %767 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} : tensor<1x2x1xf32>
    %769 = tensor.empty() : tensor<1x2x1xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%766, %768 : tensor<1x2x1xf32>, tensor<1x2x1xf32>) outs(%769 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb54(%771: f32, %772: f32, %773: f32):
      %774 = arith.addf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x2x1xf32>
    %775 = tensor.empty() : tensor<1x2x1xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%770 : tensor<1x2x1xf32>) outs(%775 : tensor<1x2x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb55(%777: f32, %778: f32):
      %779 = math.rsqrt %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x2x1xf32>
    %780 = tensor.empty() : tensor<1x2x32xf32>
    %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%740, %776 : tensor<1x2x32xf32>, tensor<1x2x1xf32>) outs(%780 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb56(%782: f32, %783: f32, %784: f32):
      %785 = arith.mulf %782, %783 : f32
      linalg.yield %785 : f32
    } -> tensor<1x2x32xf32>
    %786 = tensor.empty() : tensor<1x2x32xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781, %16 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%786 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb57(%788: f32, %789: f32, %790: f32):
      %791 = arith.mulf %788, %789 : f32
      linalg.yield %791 : f32
    } -> tensor<1x2x32xf32>
    %792 = tensor.empty() : tensor<1x2x32xf32>
    %793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%787, %17 : tensor<1x2x32xf32>, tensor<32xf32>) outs(%792 : tensor<1x2x32xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.ln1"} {
    ^bb58(%794: f32, %795: f32, %796: f32):
      %797 = arith.addf %794, %795 : f32
      linalg.yield %797 : f32
    } -> tensor<1x2x32xf32>
    %798 = tensor.empty() : tensor<32x64xi8>
    %799 = linalg.transpose ins(%113:tensor<64x32xi8>) outs(%798:tensor<32x64xi8>) permutation = [1, 0]
    %800 = tensor.collapse_shape %793 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<1x2x32xf32> into tensor<64xf32>
    %801 = tensor.expand_shape %800 [[0 : i64, 1 : i64]] output_shape [2, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<64xf32> into tensor<2x32xf32>
    %802 = tensor.empty() : tensor<32x64xf32>
    %803 = arith.constant 0 : i32
    %804 = tensor.splat %803 : tensor<64xi32>
    %805 = "quant_ext.dequantize_per_channel"(%799, %114, %804) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.1.keyValueExtractor.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.1.keyValueExtractor.weight.tensor_impl.scale"} : (tensor<32x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<32x64xf32>
    %806 = tensor.empty() : tensor<2x64xf32>
    %807 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %808 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%807 : f32) outs(%806 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %809 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} ins(%801, %805 : tensor<2x32xf32>, tensor<32x64xf32>) outs(%808 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %810 = tensor.empty() : tensor<2x64xf32>
    %811 = tensor.collapse_shape %809 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<2x64xf32> into tensor<128xf32>
    %812 = tensor.expand_shape %811 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 2, 64] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} : tensor<128xf32> into tensor<1x2x64xf32>
    %813 = tensor.empty() : tensor<1x2x64xf32>
    %814 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%812, %19 : tensor<1x2x64xf32>, tensor<64xf32>) outs(%813 : tensor<1x2x64xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.keyValueExtractor"} {
    ^bb59(%815: f32, %816: f32, %817: f32):
      %818 = arith.addf %815, %816 : f32
      linalg.yield %818 : f32
    } -> tensor<1x2x64xf32>
    %819 = tensor.collapse_shape %814 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x2x64xf32> into tensor<128xf32>
    %820 = tensor.expand_shape %819 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 2, 2, 1, 32] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<128xf32> into tensor<1x2x2x1x32xf32>
    %821 = tensor.empty() : tensor<2x1x1x2x32xf32>
    %822 = linalg.transpose ins(%820:tensor<1x2x2x1x32xf32>) outs(%821:tensor<2x1x1x2x32xf32>) permutation = [2, 0, 3, 1, 4]
    %823 = "tensor.extract_slice"(%822) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %824 = tensor.collapse_shape %823 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %825 = tensor.expand_shape %824 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %826 = "tensor.extract_slice"(%822) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 2, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : (tensor<2x1x1x2x32xf32>) -> tensor<1x1x1x2x32xf32>
    %827 = tensor.collapse_shape %826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x1x2x32xf32> into tensor<64xf32>
    %828 = tensor.expand_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 2, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<64xf32> into tensor<1x1x2x32xf32>
    %829 = tensor.empty() : tensor<32x32xi8>
    %830 = linalg.transpose ins(%116:tensor<32x32xi8>) outs(%829:tensor<32x32xi8>) permutation = [1, 0]
    %831 = tensor.collapse_shape %689 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %832 = tensor.expand_shape %831 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<345x32xf32>
    %833 = tensor.empty() : tensor<32x32xf32>
    %834 = arith.constant 0 : i32
    %835 = tensor.splat %834 : tensor<32xi32>
    %836 = "quant_ext.dequantize_per_channel"(%830, %117, %835) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.1.query.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.1.query.weight.tensor_impl.scale"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %837 = tensor.empty() : tensor<345x32xf32>
    %838 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %839 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%838 : f32) outs(%837 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %840 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} ins(%832, %836 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%839 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %841 = tensor.empty() : tensor<345x32xf32>
    %842 = tensor.collapse_shape %840 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<345x32xf32> into tensor<11040xf32>
    %843 = tensor.expand_shape %842 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %844 = tensor.empty() : tensor<1x345x32xf32>
    %845 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%843, %21 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%844 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.query"} {
    ^bb60(%846: f32, %847: f32, %848: f32):
      %849 = arith.addf %846, %847 : f32
      linalg.yield %849 : f32
    } -> tensor<1x345x32xf32>
    %850 = tensor.collapse_shape %845 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %851 = tensor.expand_shape %850 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 345, 1, 32] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x1x32xf32>
    %852 = tensor.empty() : tensor<1x1x345x32xf32>
    %853 = linalg.transpose ins(%851:tensor<1x345x1x32xf32>) outs(%852:tensor<1x1x345x32xf32>) permutation = [0, 2, 1, 3]
    %854 = tensor.empty() : tensor<1x1x32x2xf32>
    %855 = linalg.transpose ins(%825:tensor<1x1x2x32xf32>) outs(%854:tensor<1x1x32x2xf32>) permutation = [0, 1, 3, 2]
    %856 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %857 = tensor.splat %856 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%853, %855 : tensor<1x1x345x32xf32>, tensor<1x1x32x2xf32>) outs(%857 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb61(%859: f32, %860: f32, %861: f32):
      %862 = arith.mulf %859, %860 : f32
      %863 = arith.addf %861, %862 : f32
      linalg.yield %863 : f32
    } -> tensor<1x1x345x2xf32>
    %864 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 5.65685415 : f32
    %865 = tensor.splat %864 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x2xf32>
    %866 = tensor.empty() : tensor<1x1x345x2xf32>
    %867 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858, %865 : tensor<1x1x345x2xf32>, tensor<1x1x345x2xf32>) outs(%866 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb62(%868: f32, %869: f32, %870: f32):
      %871 = arith.divf %868, %869 : f32
      linalg.yield %871 : f32
    } -> tensor<1x1x345x2xf32>
    %872 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0xff800000 : f32
    %873 = tensor.splat %872 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %874 = linalg.reduce ins(%867:tensor<1x1x345x2xf32>) outs(%873:tensor<1x1x345xf32>) dimensions = [3]
    (%875: f32, %876: f32) {
      %877 = arith.maximumf %875, %876 : f32
      linalg.yield %877 : f32
    }
    %878 = tensor.collapse_shape %874 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %880 = tensor.empty() : tensor<1x1x345x2xf32>
    %881 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%867, %879 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%880 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb63(%882: f32, %883: f32, %884: f32):
      %885 = arith.subf %882, %883 : f32
      linalg.yield %885 : f32
    } -> tensor<1x1x345x2xf32>
    %886 = tensor.empty() : tensor<1x1x345x2xf32>
    %887 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%881 : tensor<1x1x345x2xf32>) outs(%886 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb64(%888: f32, %889: f32):
      %890 = math.exp %888 : f32
      linalg.yield %890 : f32
    } -> tensor<1x1x345x2xf32>
    %891 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} 0.000000e+00 : f32
    %892 = tensor.splat %891 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32>
    %893 = linalg.reduce ins(%887:tensor<1x1x345x2xf32>) outs(%892:tensor<1x1x345xf32>) dimensions = [3]
    (%894: f32, %895: f32) {
      %896 = arith.addf %894, %895 : f32
      linalg.yield %896 : f32
    }
    %897 = tensor.collapse_shape %893 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<1x1x345xf32> into tensor<345xf32>
    %898 = tensor.expand_shape %897 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 345, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} : tensor<345xf32> into tensor<1x1x345x1xf32>
    %899 = tensor.empty() : tensor<1x1x345x2xf32>
    %900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%887, %898 : tensor<1x1x345x2xf32>, tensor<1x1x345x1xf32>) outs(%899 : tensor<1x1x345x2xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.smax"} {
    ^bb65(%901: f32, %902: f32, %903: f32):
      %904 = arith.divf %901, %902 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1x345x2xf32>
    %905 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %906 = tensor.splat %905 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x1x345x32xf32>
    %907 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%900, %828 : tensor<1x1x345x2xf32>, tensor<1x1x2x32xf32>) outs(%906 : tensor<1x1x345x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb66(%908: f32, %909: f32, %910: f32):
      %911 = arith.mulf %908, %909 : f32
      %912 = arith.addf %910, %911 : f32
      linalg.yield %912 : f32
    } -> tensor<1x1x345x32xf32>
    %913 = tensor.empty() : tensor<1x345x1x32xf32>
    %914 = linalg.transpose ins(%907:tensor<1x1x345x32xf32>) outs(%913:tensor<1x345x1x32xf32>) permutation = [0, 2, 1, 3]
    %915 = tensor.collapse_shape %914 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1x32xf32> into tensor<11040xf32>
    %916 = tensor.expand_shape %915 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %917 = tensor.empty() : tensor<32x32xi8>
    %918 = linalg.transpose ins(%119:tensor<32x32xi8>) outs(%917:tensor<32x32xi8>) permutation = [1, 0]
    %919 = tensor.collapse_shape %916 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %920 = tensor.expand_shape %919 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<345x32xf32>
    %921 = tensor.empty() : tensor<32x32xf32>
    %922 = arith.constant 0 : i32
    %923 = tensor.splat %922 : tensor<32xi32>
    %924 = "quant_ext.dequantize_per_channel"(%918, %120, %923) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._attn.1.finalLayer.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._attn.1.finalLayer.weight.tensor_impl.scale"} : (tensor<32x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<32x32xf32>
    %925 = tensor.empty() : tensor<345x32xf32>
    %926 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %927 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%926 : f32) outs(%925 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %928 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} ins(%920, %924 : tensor<345x32xf32>, tensor<32x32xf32>) outs(%927 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %929 = tensor.empty() : tensor<345x32xf32>
    %930 = tensor.collapse_shape %928 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<345x32xf32> into tensor<11040xf32>
    %931 = tensor.expand_shape %930 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %932 = tensor.empty() : tensor<1x345x32xf32>
    %933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%931, %23 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%932 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._attn.1.finalLayer"} {
    ^bb67(%934: f32, %935: f32, %936: f32):
      %937 = arith.addf %934, %935 : f32
      linalg.yield %937 : f32
    } -> tensor<1x345x32xf32>
    %938 = tensor.empty() : tensor<1x345x32xf32>
    %939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%689, %933 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%938 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb68(%940: f32, %941: f32, %942: f32):
      %943 = arith.addf %940, %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x345x32xf32>
    %944 = tensor.empty() : tensor<32x256xi8>
    %945 = linalg.transpose ins(%128:tensor<256x32xi8>) outs(%944:tensor<32x256xi8>) permutation = [1, 0]
    %946 = tensor.collapse_shape %939 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %947 = tensor.expand_shape %946 [[0 : i64, 1 : i64]] output_shape [345, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<11040xf32> into tensor<345x32xf32>
    %948 = tensor.empty() : tensor<32x256xf32>
    %949 = arith.constant 0 : i32
    %950 = tensor.splat %949 : tensor<256xi32>
    %951 = "quant_ext.dequantize_per_channel"(%945, %129, %950) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._ffn.1.mlp1.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._ffn.1.mlp1.weight.tensor_impl.scale"} : (tensor<32x256xi8>, tensor<256xf32>, tensor<256xi32>) -> tensor<32x256xf32>
    %952 = tensor.empty() : tensor<345x256xf32>
    %953 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %954 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%953 : f32) outs(%952 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %955 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} ins(%947, %951 : tensor<345x32xf32>, tensor<32x256xf32>) outs(%954 : tensor<345x256xf32>) -> tensor<345x256xf32>
    %956 = tensor.empty() : tensor<345x256xf32>
    %957 = tensor.collapse_shape %955 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<345x256xf32> into tensor<88320xf32>
    %958 = tensor.expand_shape %957 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 256] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} : tensor<88320xf32> into tensor<1x345x256xf32>
    %959 = tensor.empty() : tensor<1x345x256xf32>
    %960 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958, %31 : tensor<1x345x256xf32>, tensor<256xf32>) outs(%959 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp1"} {
    ^bb69(%961: f32, %962: f32, %963: f32):
      %964 = arith.addf %961, %962 : f32
      linalg.yield %964 : f32
    } -> tensor<1x345x256xf32>
    %965 = tensor.empty() : tensor<1x256x345xf32>
    %966 = linalg.transpose ins(%960:tensor<1x345x256xf32>) outs(%965:tensor<1x256x345xf32>) permutation = [0, 2, 1]
    %967 = tensor.collapse_shape %966 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x345xf32> into tensor<88320xf32>
    %968 = tensor.expand_shape %967 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %969 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %970 = tensor.splat %969 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<1x256x17x25xf32>
    %971 = "tensor.insert_slice"(%968, %970) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : (tensor<1x256x15x23xf32>, tensor<1x256x17x25xf32>) -> tensor<1x256x17x25xf32>
    %972 = tensor.empty() : tensor<32x8x3x3x1x15x23xf32>
    %973 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%971 : tensor<1x256x17x25xf32>) outs(%972 : tensor<32x8x3x3x1x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb70(%974: f32, %975: f32):
      linalg.yield %974 : f32
    } -> tensor<32x8x3x3x1x15x23xf32>
    %976 = tensor.collapse_shape %973 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x3x3x1x15x23xf32> into tensor<794880xf32>
    %977 = tensor.expand_shape %976 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 72, 345] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<794880xf32> into tensor<32x72x345xf32>
    %978 = tensor.collapse_shape %32 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x8x3x3xf32> into tensor<18432xf32>
    %979 = tensor.expand_shape %978 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 72] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<18432xf32> into tensor<32x8x72xf32>
    %980 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} 0.000000e+00 : f32
    %981 = tensor.splat %980 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32>
    %982 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%979, %977 : tensor<32x8x72xf32>, tensor<32x72x345xf32>) outs(%981 : tensor<32x8x345xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb71(%983: f32, %984: f32, %985: f32):
      %986 = arith.mulf %983, %984 : f32
      %987 = arith.addf %985, %986 : f32
      linalg.yield %987 : f32
    } -> tensor<32x8x345xf32>
    %988 = tensor.collapse_shape %982 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<32x8x345xf32> into tensor<88320xf32>
    %989 = tensor.expand_shape %988 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<256x1x15x23xf32>
    %990 = tensor.collapse_shape %989 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<256x1x15x23xf32> into tensor<88320xf32>
    %991 = tensor.expand_shape %990 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 15, 23] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} : tensor<88320xf32> into tensor<1x256x15x23xf32>
    %992 = tensor.empty() : tensor<1x256x15x23xf32>
    %993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%991, %33 : tensor<1x256x15x23xf32>, tensor<256xf32>) outs(%992 : tensor<1x256x15x23xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.depthwise"} {
    ^bb72(%994: f32, %995: f32, %996: f32):
      %997 = arith.addf %994, %995 : f32
      linalg.yield %997 : f32
    } -> tensor<1x256x15x23xf32>
    %998 = tensor.collapse_shape %993 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x256x15x23xf32> into tensor<88320xf32>
    %999 = tensor.expand_shape %998 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 345] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<88320xf32> into tensor<1x256x345xf32>
    %1000 = tensor.empty() : tensor<1x345x256xf32>
    %1001 = linalg.transpose ins(%999:tensor<1x256x345xf32>) outs(%1000:tensor<1x345x256xf32>) permutation = [0, 2, 1]
    %1002 = tensor.empty() : tensor<1x345x256xf32>
    %1003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1001 : tensor<1x345x256xf32>) outs(%1002 : tensor<1x345x256xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.gelu"} {
    ^bb73(%1004: f32, %1005: f32):
      %1006 = arith.constant 5.000000e-01 : f32
      %1007 = arith.constant 1.000000e+00 : f32
      %1008 = arith.constant 0.707106769 : f32
      %1009 = arith.mulf %1004, %1008 : f32
      %1010 = math.erf %1009 : f32
      %1011 = arith.addf %1007, %1010 : f32
      %1012 = arith.mulf %1006, %1004 : f32
      %1013 = arith.mulf %1012, %1011 : f32
      linalg.yield %1013 : f32
    } -> tensor<1x345x256xf32>
    %1014 = tensor.empty() : tensor<256x32xi8>
    %1015 = linalg.transpose ins(%131:tensor<32x256xi8>) outs(%1014:tensor<256x32xi8>) permutation = [1, 0]
    %1016 = tensor.collapse_shape %1003 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<1x345x256xf32> into tensor<88320xf32>
    %1017 = tensor.expand_shape %1016 [[0 : i64, 1 : i64]] output_shape [345, 256] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<88320xf32> into tensor<345x256xf32>
    %1018 = tensor.empty() : tensor<256x32xf32>
    %1019 = arith.constant 0 : i32
    %1020 = tensor.splat %1019 : tensor<32xi32>
    %1021 = "quant_ext.dequantize_per_channel"(%1015, %132, %1020) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.0._ffn.1.mlp2.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.0._ffn.1.mlp2.weight.tensor_impl.scale"} : (tensor<256x32xi8>, tensor<32xf32>, tensor<32xi32>) -> tensor<256x32xf32>
    %1022 = tensor.empty() : tensor<345x32xf32>
    %1023 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1024 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1023 : f32) outs(%1022 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1025 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} ins(%1017, %1021 : tensor<345x256xf32>, tensor<256x32xf32>) outs(%1024 : tensor<345x32xf32>) -> tensor<345x32xf32>
    %1026 = tensor.empty() : tensor<345x32xf32>
    %1027 = tensor.collapse_shape %1025 [[0 : i64, 1 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<345x32xf32> into tensor<11040xf32>
    %1028 = tensor.expand_shape %1027 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 32] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} : tensor<11040xf32> into tensor<1x345x32xf32>
    %1029 = tensor.empty() : tensor<1x345x32xf32>
    %1030 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1028, %35 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1029 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0._ffn.1.mlp2"} {
    ^bb74(%1031: f32, %1032: f32, %1033: f32):
      %1034 = arith.addf %1031, %1032 : f32
      linalg.yield %1034 : f32
    } -> tensor<1x345x32xf32>
    %1035 = tensor.empty() : tensor<1x345x32xf32>
    %1036 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%939, %1030 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1035 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb75(%1037: f32, %1038: f32, %1039: f32):
      %1040 = arith.addf %1037, %1038 : f32
      linalg.yield %1040 : f32
    } -> tensor<1x345x32xf32>
    %1041 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1042 = tensor.splat %1041 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1043 = linalg.reduce ins(%1036:tensor<1x345x32xf32>) outs(%1042:tensor<1x345xf32>) dimensions = [2]
    (%1044: f32, %1045: f32) {
      %1046 = arith.addf %1044, %1045 : f32
      linalg.yield %1046 : f32
    }
    %1047 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1048 = tensor.splat %1047 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1049 = tensor.empty() : tensor<1x345xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1043, %1048 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1049 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb76(%1051: f32, %1052: f32, %1053: f32):
      %1054 = arith.divf %1051, %1052 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x345xf32>
    %1055 = tensor.collapse_shape %1050 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1056 = tensor.expand_shape %1055 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1057 = tensor.empty() : tensor<1x345x32xf32>
    %1058 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1036, %1056 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1057 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb77(%1059: f32, %1060: f32, %1061: f32):
      %1062 = arith.subf %1059, %1060 : f32
      linalg.yield %1062 : f32
    } -> tensor<1x345x32xf32>
    %1063 = tensor.empty() : tensor<1x345x32xf32>
    %1064 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1058, %1058 : tensor<1x345x32xf32>, tensor<1x345x32xf32>) outs(%1063 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb78(%1065: f32, %1066: f32, %1067: f32):
      %1068 = arith.mulf %1065, %1066 : f32
      linalg.yield %1068 : f32
    } -> tensor<1x345x32xf32>
    %1069 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 0.000000e+00 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1071 = linalg.reduce ins(%1064:tensor<1x345x32xf32>) outs(%1070:tensor<1x345xf32>) dimensions = [2]
    (%1072: f32, %1073: f32) {
      %1074 = arith.addf %1072, %1073 : f32
      linalg.yield %1074 : f32
    }
    %1075 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 3.200000e+01 : f32
    %1076 = tensor.splat %1075 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32>
    %1077 = tensor.empty() : tensor<1x345xf32>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1071, %1076 : tensor<1x345xf32>, tensor<1x345xf32>) outs(%1077 : tensor<1x345xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb79(%1079: f32, %1080: f32, %1081: f32):
      %1082 = arith.divf %1079, %1080 : f32
      linalg.yield %1082 : f32
    } -> tensor<1x345xf32>
    %1083 = tensor.collapse_shape %1078 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345xf32> into tensor<345xf32>
    %1084 = tensor.expand_shape %1083 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 345, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<345xf32> into tensor<1x345x1xf32>
    %1085 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} 1.000000e-05 : f32
    %1086 = tensor.splat %1085 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x1xf32>
    %1087 = tensor.empty() : tensor<1x345x1xf32>
    %1088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1084, %1086 : tensor<1x345x1xf32>, tensor<1x345x1xf32>) outs(%1087 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb80(%1089: f32, %1090: f32, %1091: f32):
      %1092 = arith.addf %1089, %1090 : f32
      linalg.yield %1092 : f32
    } -> tensor<1x345x1xf32>
    %1093 = tensor.empty() : tensor<1x345x1xf32>
    %1094 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1088 : tensor<1x345x1xf32>) outs(%1093 : tensor<1x345x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb81(%1095: f32, %1096: f32):
      %1097 = math.rsqrt %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x345x1xf32>
    %1098 = tensor.empty() : tensor<1x345x32xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1058, %1094 : tensor<1x345x32xf32>, tensor<1x345x1xf32>) outs(%1098 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb82(%1100: f32, %1101: f32, %1102: f32):
      %1103 = arith.mulf %1100, %1101 : f32
      linalg.yield %1103 : f32
    } -> tensor<1x345x32xf32>
    %1104 = tensor.empty() : tensor<1x345x32xf32>
    %1105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099, %38 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1104 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb83(%1106: f32, %1107: f32, %1108: f32):
      %1109 = arith.mulf %1106, %1107 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x345x32xf32>
    %1110 = tensor.empty() : tensor<1x345x32xf32>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1105, %39 : tensor<1x345x32xf32>, tensor<32xf32>) outs(%1110 : tensor<1x345x32xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} {
    ^bb84(%1112: f32, %1113: f32, %1114: f32):
      %1115 = arith.addf %1112, %1113 : f32
      linalg.yield %1115 : f32
    } -> tensor<1x345x32xf32>
    %1116 = tensor.collapse_shape %1111 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<1x345x32xf32> into tensor<11040xf32>
    %1117 = tensor.expand_shape %1116 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 15, 23, 32] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.0"} : tensor<11040xf32> into tensor<1x15x23x32xf32>
    %1118 = tensor.empty() : tensor<1x32x15x23xf32>
    %1119 = linalg.transpose ins(%1117:tensor<1x15x23x32xf32>) outs(%1118:tensor<1x32x15x23xf32>) permutation = [0, 3, 1, 2]
    %1120 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1121 = tensor.splat %1120 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<1x32x17x25xf32>
    %1122 = "tensor.insert_slice"(%1119, %1121) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 32, 15, 23>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : (tensor<1x32x15x23xf32>, tensor<1x32x17x25xf32>) -> tensor<1x32x17x25xf32>
    %1123 = tensor.empty() : tensor<32x3x3x1x8x12xf32>
    %1124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1122 : tensor<1x32x17x25xf32>) outs(%1123 : tensor<32x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb85(%1125: f32, %1126: f32):
      linalg.yield %1125 : f32
    } -> tensor<32x3x3x1x8x12xf32>
    %1127 = tensor.collapse_shape %1124 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<32x3x3x1x8x12xf32> into tensor<27648xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64]] output_shape [288, 96] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<27648xf32> into tensor<288x96xf32>
    %1129 = tensor.collapse_shape %40 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x32x3x3xf32> into tensor<18432xf32>
    %1130 = tensor.expand_shape %1129 [[0 : i64, 1 : i64]] output_shape [64, 288] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<18432xf32> into tensor<64x288xf32>
    %1131 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} 0.000000e+00 : f32
    %1132 = tensor.splat %1131 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32>
    %1133 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} ins(%1130, %1128 : tensor<64x288xf32>, tensor<288x96xf32>) outs(%1132 : tensor<64x96xf32>) -> tensor<64x96xf32>
    %1134 = tensor.collapse_shape %1133 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x96xf32> into tensor<6144xf32>
    %1135 = tensor.expand_shape %1134 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<64x1x8x12xf32>
    %1136 = tensor.collapse_shape %1135 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<64x1x8x12xf32> into tensor<6144xf32>
    %1137 = tensor.expand_shape %1136 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1138 = tensor.empty() : tensor<1x64x8x12xf32>
    %1139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1137, %41 : tensor<1x64x8x12xf32>, tensor<64xf32>) outs(%1138 : tensor<1x64x8x12xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.cn1"} {
    ^bb86(%1140: f32, %1141: f32, %1142: f32):
      %1143 = arith.addf %1140, %1141 : f32
      linalg.yield %1143 : f32
    } -> tensor<1x64x8x12xf32>
    %1144 = tensor.collapse_shape %1139 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %1145 = tensor.expand_shape %1144 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 96] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge"} : tensor<6144xf32> into tensor<1x64x96xf32>
    %1146 = tensor.empty() : tensor<1x96x64xf32>
    %1147 = linalg.transpose ins(%1145:tensor<1x64x96xf32>) outs(%1146:tensor<1x96x64xf32>) permutation = [0, 2, 1]
    %1148 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1149 = tensor.splat %1148 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1150 = linalg.reduce ins(%1147:tensor<1x96x64xf32>) outs(%1149:tensor<1x96xf32>) dimensions = [2]
    (%1151: f32, %1152: f32) {
      %1153 = arith.addf %1151, %1152 : f32
      linalg.yield %1153 : f32
    }
    %1154 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1155 = tensor.splat %1154 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1156 = tensor.empty() : tensor<1x96xf32>
    %1157 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1150, %1155 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1156 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb87(%1158: f32, %1159: f32, %1160: f32):
      %1161 = arith.divf %1158, %1159 : f32
      linalg.yield %1161 : f32
    } -> tensor<1x96xf32>
    %1162 = tensor.collapse_shape %1157 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1163 = tensor.expand_shape %1162 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1164 = tensor.empty() : tensor<1x96x64xf32>
    %1165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %1163 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1164 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb88(%1166: f32, %1167: f32, %1168: f32):
      %1169 = arith.subf %1166, %1167 : f32
      linalg.yield %1169 : f32
    } -> tensor<1x96x64xf32>
    %1170 = tensor.empty() : tensor<1x96x64xf32>
    %1171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1165, %1165 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1170 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb89(%1172: f32, %1173: f32, %1174: f32):
      %1175 = arith.mulf %1172, %1173 : f32
      linalg.yield %1175 : f32
    } -> tensor<1x96x64xf32>
    %1176 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 0.000000e+00 : f32
    %1177 = tensor.splat %1176 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1178 = linalg.reduce ins(%1171:tensor<1x96x64xf32>) outs(%1177:tensor<1x96xf32>) dimensions = [2]
    (%1179: f32, %1180: f32) {
      %1181 = arith.addf %1179, %1180 : f32
      linalg.yield %1181 : f32
    }
    %1182 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 6.400000e+01 : f32
    %1183 = tensor.splat %1182 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32>
    %1184 = tensor.empty() : tensor<1x96xf32>
    %1185 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1178, %1183 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1184 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb90(%1186: f32, %1187: f32, %1188: f32):
      %1189 = arith.divf %1186, %1187 : f32
      linalg.yield %1189 : f32
    } -> tensor<1x96xf32>
    %1190 = tensor.collapse_shape %1185 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96xf32> into tensor<96xf32>
    %1191 = tensor.expand_shape %1190 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1192 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} 1.000000e-05 : f32
    %1193 = tensor.splat %1192 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} : tensor<1x96x1xf32>
    %1194 = tensor.empty() : tensor<1x96x1xf32>
    %1195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1191, %1193 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1194 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb91(%1196: f32, %1197: f32, %1198: f32):
      %1199 = arith.addf %1196, %1197 : f32
      linalg.yield %1199 : f32
    } -> tensor<1x96x1xf32>
    %1200 = tensor.empty() : tensor<1x96x1xf32>
    %1201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1195 : tensor<1x96x1xf32>) outs(%1200 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb92(%1202: f32, %1203: f32):
      %1204 = math.rsqrt %1202 : f32
      linalg.yield %1204 : f32
    } -> tensor<1x96x1xf32>
    %1205 = tensor.empty() : tensor<1x96x64xf32>
    %1206 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1165, %1201 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1205 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb93(%1207: f32, %1208: f32, %1209: f32):
      %1210 = arith.mulf %1207, %1208 : f32
      linalg.yield %1210 : f32
    } -> tensor<1x96x64xf32>
    %1211 = tensor.empty() : tensor<1x96x64xf32>
    %1212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1206, %42 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1211 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb94(%1213: f32, %1214: f32, %1215: f32):
      %1216 = arith.mulf %1213, %1214 : f32
      linalg.yield %1216 : f32
    } -> tensor<1x96x64xf32>
    %1217 = tensor.empty() : tensor<1x96x64xf32>
    %1218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1212, %43 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1217 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1.patchMerge.layerNorm"} {
    ^bb95(%1219: f32, %1220: f32, %1221: f32):
      %1222 = arith.addf %1219, %1220 : f32
      linalg.yield %1222 : f32
    } -> tensor<1x96x64xf32>
    %1223 = tensor.empty() : tensor<1x64x96xf32>
    %1224 = linalg.transpose ins(%1218:tensor<1x96x64xf32>) outs(%1223:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1225 = tensor.collapse_shape %1224 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1226 = tensor.expand_shape %1225 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1227 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1226 : tensor<1x64x8x12xf32>) outs(%1227 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb96(%1229: f32, %1230: f32):
      linalg.yield %1229 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1231 = tensor.collapse_shape %1228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1232 = tensor.expand_shape %1231 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1233 = tensor.collapse_shape %44 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1235 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} 0.000000e+00 : f32
    %1236 = tensor.splat %1235 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32>
    %1237 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} ins(%1234, %1232 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1236 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1238 = tensor.collapse_shape %1237 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1239 = tensor.expand_shape %1238 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1240 = tensor.collapse_shape %1239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1241 = tensor.expand_shape %1240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1242 = tensor.empty() : tensor<1x64x2x3xf32>
    %1243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1241, %45 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1242 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.cn1"} {
    ^bb97(%1244: f32, %1245: f32, %1246: f32):
      %1247 = arith.addf %1244, %1245 : f32
      linalg.yield %1247 : f32
    } -> tensor<1x64x2x3xf32>
    %1248 = tensor.collapse_shape %1243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1249 = tensor.expand_shape %1248 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1250 = tensor.empty() : tensor<1x6x64xf32>
    %1251 = linalg.transpose ins(%1249:tensor<1x64x6xf32>) outs(%1250:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1252 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1253 = tensor.splat %1252 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1254 = linalg.reduce ins(%1251:tensor<1x6x64xf32>) outs(%1253:tensor<1x6xf32>) dimensions = [2]
    (%1255: f32, %1256: f32) {
      %1257 = arith.addf %1255, %1256 : f32
      linalg.yield %1257 : f32
    }
    %1258 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1259 = tensor.splat %1258 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1260 = tensor.empty() : tensor<1x6xf32>
    %1261 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1254, %1259 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1260 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb98(%1262: f32, %1263: f32, %1264: f32):
      %1265 = arith.divf %1262, %1263 : f32
      linalg.yield %1265 : f32
    } -> tensor<1x6xf32>
    %1266 = tensor.collapse_shape %1261 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1267 = tensor.expand_shape %1266 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1268 = tensor.empty() : tensor<1x6x64xf32>
    %1269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251, %1267 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1268 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb99(%1270: f32, %1271: f32, %1272: f32):
      %1273 = arith.subf %1270, %1271 : f32
      linalg.yield %1273 : f32
    } -> tensor<1x6x64xf32>
    %1274 = tensor.empty() : tensor<1x6x64xf32>
    %1275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1269, %1269 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1274 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb100(%1276: f32, %1277: f32, %1278: f32):
      %1279 = arith.mulf %1276, %1277 : f32
      linalg.yield %1279 : f32
    } -> tensor<1x6x64xf32>
    %1280 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 0.000000e+00 : f32
    %1281 = tensor.splat %1280 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1282 = linalg.reduce ins(%1275:tensor<1x6x64xf32>) outs(%1281:tensor<1x6xf32>) dimensions = [2]
    (%1283: f32, %1284: f32) {
      %1285 = arith.addf %1283, %1284 : f32
      linalg.yield %1285 : f32
    }
    %1286 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 6.400000e+01 : f32
    %1287 = tensor.splat %1286 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32>
    %1288 = tensor.empty() : tensor<1x6xf32>
    %1289 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1282, %1287 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1288 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb101(%1290: f32, %1291: f32, %1292: f32):
      %1293 = arith.divf %1290, %1291 : f32
      linalg.yield %1293 : f32
    } -> tensor<1x6xf32>
    %1294 = tensor.collapse_shape %1289 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1295 = tensor.expand_shape %1294 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1296 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} 1.000000e-05 : f32
    %1297 = tensor.splat %1296 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} : tensor<1x6x1xf32>
    %1298 = tensor.empty() : tensor<1x6x1xf32>
    %1299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295, %1297 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1298 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb102(%1300: f32, %1301: f32, %1302: f32):
      %1303 = arith.addf %1300, %1301 : f32
      linalg.yield %1303 : f32
    } -> tensor<1x6x1xf32>
    %1304 = tensor.empty() : tensor<1x6x1xf32>
    %1305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1299 : tensor<1x6x1xf32>) outs(%1304 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb103(%1306: f32, %1307: f32):
      %1308 = math.rsqrt %1306 : f32
      linalg.yield %1308 : f32
    } -> tensor<1x6x1xf32>
    %1309 = tensor.empty() : tensor<1x6x64xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1269, %1305 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1309 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb104(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.mulf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x6x64xf32>
    %1315 = tensor.empty() : tensor<1x6x64xf32>
    %1316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1310, %46 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1315 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb105(%1317: f32, %1318: f32, %1319: f32):
      %1320 = arith.mulf %1317, %1318 : f32
      linalg.yield %1320 : f32
    } -> tensor<1x6x64xf32>
    %1321 = tensor.empty() : tensor<1x6x64xf32>
    %1322 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316, %47 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1321 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.ln1"} {
    ^bb106(%1323: f32, %1324: f32, %1325: f32):
      %1326 = arith.addf %1323, %1324 : f32
      linalg.yield %1326 : f32
    } -> tensor<1x6x64xf32>
    %1327 = tensor.empty() : tensor<64x128xi8>
    %1328 = linalg.transpose ins(%134:tensor<128x64xi8>) outs(%1327:tensor<64x128xi8>) permutation = [1, 0]
    %1329 = tensor.collapse_shape %1322 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1330 = tensor.expand_shape %1329 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1331 = tensor.empty() : tensor<64x128xf32>
    %1332 = arith.constant 0 : i32
    %1333 = tensor.splat %1332 : tensor<128xi32>
    %1334 = "quant_ext.dequantize_per_channel"(%1328, %135, %1333) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.0.keyValueExtractor.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.0.keyValueExtractor.weight.tensor_impl.scale"} : (tensor<64x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<64x128xf32>
    %1335 = tensor.empty() : tensor<6x128xf32>
    %1336 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1337 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1336 : f32) outs(%1335 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1338 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} ins(%1330, %1334 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1337 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1339 = tensor.empty() : tensor<6x128xf32>
    %1340 = tensor.collapse_shape %1338 [[0 : i64, 1 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1341 = tensor.expand_shape %1340 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1342 = tensor.empty() : tensor<1x6x128xf32>
    %1343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1341, %49 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1342 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.keyValueExtractor"} {
    ^bb107(%1344: f32, %1345: f32, %1346: f32):
      %1347 = arith.addf %1344, %1345 : f32
      linalg.yield %1347 : f32
    } -> tensor<1x6x128xf32>
    %1348 = tensor.collapse_shape %1343 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1349 = tensor.expand_shape %1348 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1350 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1351 = linalg.transpose ins(%1349:tensor<1x6x2x2x32xf32>) outs(%1350:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1352 = "tensor.extract_slice"(%1351) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1353 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1354 = tensor.expand_shape %1353 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1355 = "tensor.extract_slice"(%1351) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1356 = tensor.collapse_shape %1355 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1357 = tensor.expand_shape %1356 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1358 = tensor.empty() : tensor<64x64xi8>
    %1359 = linalg.transpose ins(%137:tensor<64x64xi8>) outs(%1358:tensor<64x64xi8>) permutation = [1, 0]
    %1360 = tensor.collapse_shape %1218 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1361 = tensor.expand_shape %1360 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1362 = tensor.empty() : tensor<64x64xf32>
    %1363 = arith.constant 0 : i32
    %1364 = tensor.splat %1363 : tensor<64xi32>
    %1365 = "quant_ext.dequantize_per_channel"(%1359, %138, %1364) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.0.query.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.0.query.weight.tensor_impl.scale"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1366 = tensor.empty() : tensor<96x64xf32>
    %1367 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1368 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1367 : f32) outs(%1366 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1369 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} ins(%1361, %1365 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1368 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1370 = tensor.empty() : tensor<96x64xf32>
    %1371 = tensor.collapse_shape %1369 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1372 = tensor.expand_shape %1371 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1373 = tensor.empty() : tensor<1x96x64xf32>
    %1374 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1372, %51 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1373 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.query"} {
    ^bb108(%1375: f32, %1376: f32, %1377: f32):
      %1378 = arith.addf %1375, %1376 : f32
      linalg.yield %1378 : f32
    } -> tensor<1x96x64xf32>
    %1379 = tensor.collapse_shape %1374 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1380 = tensor.expand_shape %1379 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1381 = tensor.empty() : tensor<1x2x96x32xf32>
    %1382 = linalg.transpose ins(%1380:tensor<1x96x2x32xf32>) outs(%1381:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1383 = tensor.empty() : tensor<1x2x32x6xf32>
    %1384 = linalg.transpose ins(%1354:tensor<1x2x6x32xf32>) outs(%1383:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1385 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1386 = tensor.splat %1385 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1382, %1384 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1386 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb109(%1388: f32, %1389: f32, %1390: f32):
      %1391 = arith.mulf %1388, %1389 : f32
      %1392 = arith.addf %1390, %1391 : f32
      linalg.yield %1392 : f32
    } -> tensor<1x2x96x6xf32>
    %1393 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1394 = tensor.splat %1393 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1395 = tensor.empty() : tensor<1x2x96x6xf32>
    %1396 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1387, %1394 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1395 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb110(%1397: f32, %1398: f32, %1399: f32):
      %1400 = arith.divf %1397, %1398 : f32
      linalg.yield %1400 : f32
    } -> tensor<1x2x96x6xf32>
    %1401 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0xff800000 : f32
    %1402 = tensor.splat %1401 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1403 = linalg.reduce ins(%1396:tensor<1x2x96x6xf32>) outs(%1402:tensor<1x2x96xf32>) dimensions = [3]
    (%1404: f32, %1405: f32) {
      %1406 = arith.maximumf %1404, %1405 : f32
      linalg.yield %1406 : f32
    }
    %1407 = tensor.collapse_shape %1403 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1408 = tensor.expand_shape %1407 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1409 = tensor.empty() : tensor<1x2x96x6xf32>
    %1410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1396, %1408 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1409 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb111(%1411: f32, %1412: f32, %1413: f32):
      %1414 = arith.subf %1411, %1412 : f32
      linalg.yield %1414 : f32
    } -> tensor<1x2x96x6xf32>
    %1415 = tensor.empty() : tensor<1x2x96x6xf32>
    %1416 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1410 : tensor<1x2x96x6xf32>) outs(%1415 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb112(%1417: f32, %1418: f32):
      %1419 = math.exp %1417 : f32
      linalg.yield %1419 : f32
    } -> tensor<1x2x96x6xf32>
    %1420 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} 0.000000e+00 : f32
    %1421 = tensor.splat %1420 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32>
    %1422 = linalg.reduce ins(%1416:tensor<1x2x96x6xf32>) outs(%1421:tensor<1x2x96xf32>) dimensions = [3]
    (%1423: f32, %1424: f32) {
      %1425 = arith.addf %1423, %1424 : f32
      linalg.yield %1425 : f32
    }
    %1426 = tensor.collapse_shape %1422 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1427 = tensor.expand_shape %1426 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1428 = tensor.empty() : tensor<1x2x96x6xf32>
    %1429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1416, %1427 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1428 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.smax"} {
    ^bb113(%1430: f32, %1431: f32, %1432: f32):
      %1433 = arith.divf %1430, %1431 : f32
      linalg.yield %1433 : f32
    } -> tensor<1x2x96x6xf32>
    %1434 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1435 = tensor.splat %1434 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1429, %1357 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1435 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb114(%1437: f32, %1438: f32, %1439: f32):
      %1440 = arith.mulf %1437, %1438 : f32
      %1441 = arith.addf %1439, %1440 : f32
      linalg.yield %1441 : f32
    } -> tensor<1x2x96x32xf32>
    %1442 = tensor.empty() : tensor<1x96x2x32xf32>
    %1443 = linalg.transpose ins(%1436:tensor<1x2x96x32xf32>) outs(%1442:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1444 = tensor.collapse_shape %1443 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1445 = tensor.expand_shape %1444 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1446 = tensor.empty() : tensor<64x64xi8>
    %1447 = linalg.transpose ins(%140:tensor<64x64xi8>) outs(%1446:tensor<64x64xi8>) permutation = [1, 0]
    %1448 = tensor.collapse_shape %1445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1449 = tensor.expand_shape %1448 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1450 = tensor.empty() : tensor<64x64xf32>
    %1451 = arith.constant 0 : i32
    %1452 = tensor.splat %1451 : tensor<64xi32>
    %1453 = "quant_ext.dequantize_per_channel"(%1447, %141, %1452) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.0.finalLayer.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.0.finalLayer.weight.tensor_impl.scale"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1454 = tensor.empty() : tensor<96x64xf32>
    %1455 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1456 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1455 : f32) outs(%1454 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1457 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} ins(%1449, %1453 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1456 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1458 = tensor.empty() : tensor<96x64xf32>
    %1459 = tensor.collapse_shape %1457 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1460 = tensor.expand_shape %1459 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1461 = tensor.empty() : tensor<1x96x64xf32>
    %1462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1460, %53 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1461 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.0.finalLayer"} {
    ^bb115(%1463: f32, %1464: f32, %1465: f32):
      %1466 = arith.addf %1463, %1464 : f32
      linalg.yield %1466 : f32
    } -> tensor<1x96x64xf32>
    %1467 = tensor.empty() : tensor<1x96x64xf32>
    %1468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1218, %1462 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1467 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb116(%1469: f32, %1470: f32, %1471: f32):
      %1472 = arith.addf %1469, %1470 : f32
      linalg.yield %1472 : f32
    } -> tensor<1x96x64xf32>
    %1473 = tensor.empty() : tensor<64x512xi8>
    %1474 = linalg.transpose ins(%152:tensor<512x64xi8>) outs(%1473:tensor<64x512xi8>) permutation = [1, 0]
    %1475 = tensor.collapse_shape %1468 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1476 = tensor.expand_shape %1475 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1477 = tensor.empty() : tensor<64x512xf32>
    %1478 = arith.constant 0 : i32
    %1479 = tensor.splat %1478 : tensor<512xi32>
    %1480 = "quant_ext.dequantize_per_channel"(%1474, %153, %1479) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._ffn.0.mlp1.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._ffn.0.mlp1.weight.tensor_impl.scale"} : (tensor<64x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<64x512xf32>
    %1481 = tensor.empty() : tensor<96x512xf32>
    %1482 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1483 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1482 : f32) outs(%1481 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1484 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} ins(%1476, %1480 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1483 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1485 = tensor.empty() : tensor<96x512xf32>
    %1486 = tensor.collapse_shape %1484 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1487 = tensor.expand_shape %1486 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1488 = tensor.empty() : tensor<1x96x512xf32>
    %1489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1487, %65 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1488 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp1"} {
    ^bb117(%1490: f32, %1491: f32, %1492: f32):
      %1493 = arith.addf %1490, %1491 : f32
      linalg.yield %1493 : f32
    } -> tensor<1x96x512xf32>
    %1494 = tensor.empty() : tensor<1x512x96xf32>
    %1495 = linalg.transpose ins(%1489:tensor<1x96x512xf32>) outs(%1494:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1496 = tensor.collapse_shape %1495 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1497 = tensor.expand_shape %1496 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1498 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1499 = tensor.splat %1498 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<1x512x10x14xf32>
    %1500 = "tensor.insert_slice"(%1497, %1499) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1501 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1500 : tensor<1x512x10x14xf32>) outs(%1501 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb118(%1503: f32, %1504: f32):
      linalg.yield %1503 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1505 = tensor.collapse_shape %1502 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1506 = tensor.expand_shape %1505 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1507 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1508 = tensor.expand_shape %1507 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1509 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} 0.000000e+00 : f32
    %1510 = tensor.splat %1509 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32>
    %1511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1508, %1506 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1510 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb119(%1512: f32, %1513: f32, %1514: f32):
      %1515 = arith.mulf %1512, %1513 : f32
      %1516 = arith.addf %1514, %1515 : f32
      linalg.yield %1516 : f32
    } -> tensor<64x8x96xf32>
    %1517 = tensor.collapse_shape %1511 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1518 = tensor.expand_shape %1517 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1519 = tensor.collapse_shape %1518 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1520 = tensor.expand_shape %1519 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1521 = tensor.empty() : tensor<1x512x8x12xf32>
    %1522 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1520, %67 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1521 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.depthwise"} {
    ^bb120(%1523: f32, %1524: f32, %1525: f32):
      %1526 = arith.addf %1523, %1524 : f32
      linalg.yield %1526 : f32
    } -> tensor<1x512x8x12xf32>
    %1527 = tensor.collapse_shape %1522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1528 = tensor.expand_shape %1527 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1529 = tensor.empty() : tensor<1x96x512xf32>
    %1530 = linalg.transpose ins(%1528:tensor<1x512x96xf32>) outs(%1529:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1531 = tensor.empty() : tensor<1x96x512xf32>
    %1532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1530 : tensor<1x96x512xf32>) outs(%1531 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.gelu"} {
    ^bb121(%1533: f32, %1534: f32):
      %1535 = arith.constant 5.000000e-01 : f32
      %1536 = arith.constant 1.000000e+00 : f32
      %1537 = arith.constant 0.707106769 : f32
      %1538 = arith.mulf %1533, %1537 : f32
      %1539 = math.erf %1538 : f32
      %1540 = arith.addf %1536, %1539 : f32
      %1541 = arith.mulf %1535, %1533 : f32
      %1542 = arith.mulf %1541, %1540 : f32
      linalg.yield %1542 : f32
    } -> tensor<1x96x512xf32>
    %1543 = tensor.empty() : tensor<512x64xi8>
    %1544 = linalg.transpose ins(%155:tensor<64x512xi8>) outs(%1543:tensor<512x64xi8>) permutation = [1, 0]
    %1545 = tensor.collapse_shape %1532 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1546 = tensor.expand_shape %1545 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1547 = tensor.empty() : tensor<512x64xf32>
    %1548 = arith.constant 0 : i32
    %1549 = tensor.splat %1548 : tensor<64xi32>
    %1550 = "quant_ext.dequantize_per_channel"(%1544, %156, %1549) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._ffn.0.mlp2.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._ffn.0.mlp2.weight.tensor_impl.scale"} : (tensor<512x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<512x64xf32>
    %1551 = tensor.empty() : tensor<96x64xf32>
    %1552 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1553 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1552 : f32) outs(%1551 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1554 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} ins(%1546, %1550 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1553 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1555 = tensor.empty() : tensor<96x64xf32>
    %1556 = tensor.collapse_shape %1554 [[0 : i64, 1 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1557 = tensor.expand_shape %1556 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1558 = tensor.empty() : tensor<1x96x64xf32>
    %1559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1557, %69 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1558 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.0.mlp2"} {
    ^bb122(%1560: f32, %1561: f32, %1562: f32):
      %1563 = arith.addf %1560, %1561 : f32
      linalg.yield %1563 : f32
    } -> tensor<1x96x64xf32>
    %1564 = tensor.empty() : tensor<1x96x64xf32>
    %1565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1468, %1559 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1564 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb123(%1566: f32, %1567: f32, %1568: f32):
      %1569 = arith.addf %1566, %1567 : f32
      linalg.yield %1569 : f32
    } -> tensor<1x96x64xf32>
    %1570 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1571 = tensor.splat %1570 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1572 = linalg.reduce ins(%1565:tensor<1x96x64xf32>) outs(%1571:tensor<1x96xf32>) dimensions = [2]
    (%1573: f32, %1574: f32) {
      %1575 = arith.addf %1573, %1574 : f32
      linalg.yield %1575 : f32
    }
    %1576 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1577 = tensor.splat %1576 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1578 = tensor.empty() : tensor<1x96xf32>
    %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1572, %1577 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1578 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb124(%1580: f32, %1581: f32, %1582: f32):
      %1583 = arith.divf %1580, %1581 : f32
      linalg.yield %1583 : f32
    } -> tensor<1x96xf32>
    %1584 = tensor.collapse_shape %1579 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1585 = tensor.expand_shape %1584 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1586 = tensor.empty() : tensor<1x96x64xf32>
    %1587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1565, %1585 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1586 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb125(%1588: f32, %1589: f32, %1590: f32):
      %1591 = arith.subf %1588, %1589 : f32
      linalg.yield %1591 : f32
    } -> tensor<1x96x64xf32>
    %1592 = tensor.empty() : tensor<1x96x64xf32>
    %1593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1587, %1587 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1592 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb126(%1594: f32, %1595: f32, %1596: f32):
      %1597 = arith.mulf %1594, %1595 : f32
      linalg.yield %1597 : f32
    } -> tensor<1x96x64xf32>
    %1598 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1599 = tensor.splat %1598 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1600 = linalg.reduce ins(%1593:tensor<1x96x64xf32>) outs(%1599:tensor<1x96xf32>) dimensions = [2]
    (%1601: f32, %1602: f32) {
      %1603 = arith.addf %1601, %1602 : f32
      linalg.yield %1603 : f32
    }
    %1604 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1605 = tensor.splat %1604 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1606 = tensor.empty() : tensor<1x96xf32>
    %1607 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1600, %1605 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%1606 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb127(%1608: f32, %1609: f32, %1610: f32):
      %1611 = arith.divf %1608, %1609 : f32
      linalg.yield %1611 : f32
    } -> tensor<1x96xf32>
    %1612 = tensor.collapse_shape %1607 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %1613 = tensor.expand_shape %1612 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %1614 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %1615 = tensor.splat %1614 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %1616 = tensor.empty() : tensor<1x96x1xf32>
    %1617 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1613, %1615 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%1616 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb128(%1618: f32, %1619: f32, %1620: f32):
      %1621 = arith.addf %1618, %1619 : f32
      linalg.yield %1621 : f32
    } -> tensor<1x96x1xf32>
    %1622 = tensor.empty() : tensor<1x96x1xf32>
    %1623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1617 : tensor<1x96x1xf32>) outs(%1622 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb129(%1624: f32, %1625: f32):
      %1626 = math.rsqrt %1624 : f32
      linalg.yield %1626 : f32
    } -> tensor<1x96x1xf32>
    %1627 = tensor.empty() : tensor<1x96x64xf32>
    %1628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1587, %1623 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%1627 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb130(%1629: f32, %1630: f32, %1631: f32):
      %1632 = arith.mulf %1629, %1630 : f32
      linalg.yield %1632 : f32
    } -> tensor<1x96x64xf32>
    %1633 = tensor.empty() : tensor<1x96x64xf32>
    %1634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1628, %76 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1633 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb131(%1635: f32, %1636: f32, %1637: f32):
      %1638 = arith.mulf %1635, %1636 : f32
      linalg.yield %1638 : f32
    } -> tensor<1x96x64xf32>
    %1639 = tensor.empty() : tensor<1x96x64xf32>
    %1640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1634, %77 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1639 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb132(%1641: f32, %1642: f32, %1643: f32):
      %1644 = arith.addf %1641, %1642 : f32
      linalg.yield %1644 : f32
    } -> tensor<1x96x64xf32>
    %1645 = tensor.empty() : tensor<1x64x96xf32>
    %1646 = linalg.transpose ins(%1640:tensor<1x96x64xf32>) outs(%1645:tensor<1x64x96xf32>) permutation = [0, 2, 1]
    %1647 = tensor.collapse_shape %1646 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x96xf32> into tensor<6144xf32>
    %1648 = tensor.expand_shape %1647 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 12] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x64x8x12xf32>
    %1649 = tensor.empty() : tensor<64x4x4x1x2x3xf32>
    %1650 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1648 : tensor<1x64x8x12xf32>) outs(%1649 : tensor<64x4x4x1x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb133(%1651: f32, %1652: f32):
      linalg.yield %1651 : f32
    } -> tensor<64x4x4x1x2x3xf32>
    %1653 = tensor.collapse_shape %1650 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x4x4x1x2x3xf32> into tensor<6144xf32>
    %1654 = tensor.expand_shape %1653 [[0 : i64, 1 : i64]] output_shape [1024, 6] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<6144xf32> into tensor<1024x6xf32>
    %1655 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x64x4x4xf32> into tensor<65536xf32>
    %1656 = tensor.expand_shape %1655 [[0 : i64, 1 : i64]] output_shape [64, 1024] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<65536xf32> into tensor<64x1024xf32>
    %1657 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} 0.000000e+00 : f32
    %1658 = tensor.splat %1657 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32>
    %1659 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} ins(%1656, %1654 : tensor<64x1024xf32>, tensor<1024x6xf32>) outs(%1658 : tensor<64x6xf32>) -> tensor<64x6xf32>
    %1660 = tensor.collapse_shape %1659 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x6xf32> into tensor<384xf32>
    %1661 = tensor.expand_shape %1660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<64x1x2x3xf32>
    %1662 = tensor.collapse_shape %1661 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<64x1x2x3xf32> into tensor<384xf32>
    %1663 = tensor.expand_shape %1662 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 2, 3] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} : tensor<384xf32> into tensor<1x64x2x3xf32>
    %1664 = tensor.empty() : tensor<1x64x2x3xf32>
    %1665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1663, %55 : tensor<1x64x2x3xf32>, tensor<64xf32>) outs(%1664 : tensor<1x64x2x3xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.cn1"} {
    ^bb134(%1666: f32, %1667: f32, %1668: f32):
      %1669 = arith.addf %1666, %1667 : f32
      linalg.yield %1669 : f32
    } -> tensor<1x64x2x3xf32>
    %1670 = tensor.collapse_shape %1665 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x64x2x3xf32> into tensor<384xf32>
    %1671 = tensor.expand_shape %1670 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 6] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x64x6xf32>
    %1672 = tensor.empty() : tensor<1x6x64xf32>
    %1673 = linalg.transpose ins(%1671:tensor<1x64x6xf32>) outs(%1672:tensor<1x6x64xf32>) permutation = [0, 2, 1]
    %1674 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1675 = tensor.splat %1674 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1676 = linalg.reduce ins(%1673:tensor<1x6x64xf32>) outs(%1675:tensor<1x6xf32>) dimensions = [2]
    (%1677: f32, %1678: f32) {
      %1679 = arith.addf %1677, %1678 : f32
      linalg.yield %1679 : f32
    }
    %1680 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1681 = tensor.splat %1680 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1682 = tensor.empty() : tensor<1x6xf32>
    %1683 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1676, %1681 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1682 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb135(%1684: f32, %1685: f32, %1686: f32):
      %1687 = arith.divf %1684, %1685 : f32
      linalg.yield %1687 : f32
    } -> tensor<1x6xf32>
    %1688 = tensor.collapse_shape %1683 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1689 = tensor.expand_shape %1688 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1690 = tensor.empty() : tensor<1x6x64xf32>
    %1691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673, %1689 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1690 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb136(%1692: f32, %1693: f32, %1694: f32):
      %1695 = arith.subf %1692, %1693 : f32
      linalg.yield %1695 : f32
    } -> tensor<1x6x64xf32>
    %1696 = tensor.empty() : tensor<1x6x64xf32>
    %1697 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1691, %1691 : tensor<1x6x64xf32>, tensor<1x6x64xf32>) outs(%1696 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb137(%1698: f32, %1699: f32, %1700: f32):
      %1701 = arith.mulf %1698, %1699 : f32
      linalg.yield %1701 : f32
    } -> tensor<1x6x64xf32>
    %1702 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 0.000000e+00 : f32
    %1703 = tensor.splat %1702 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1704 = linalg.reduce ins(%1697:tensor<1x6x64xf32>) outs(%1703:tensor<1x6xf32>) dimensions = [2]
    (%1705: f32, %1706: f32) {
      %1707 = arith.addf %1705, %1706 : f32
      linalg.yield %1707 : f32
    }
    %1708 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 6.400000e+01 : f32
    %1709 = tensor.splat %1708 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32>
    %1710 = tensor.empty() : tensor<1x6xf32>
    %1711 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1704, %1709 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%1710 : tensor<1x6xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb138(%1712: f32, %1713: f32, %1714: f32):
      %1715 = arith.divf %1712, %1713 : f32
      linalg.yield %1715 : f32
    } -> tensor<1x6xf32>
    %1716 = tensor.collapse_shape %1711 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6xf32> into tensor<6xf32>
    %1717 = tensor.expand_shape %1716 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<6xf32> into tensor<1x6x1xf32>
    %1718 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} 1.000000e-05 : f32
    %1719 = tensor.splat %1718 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} : tensor<1x6x1xf32>
    %1720 = tensor.empty() : tensor<1x6x1xf32>
    %1721 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1717, %1719 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%1720 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb139(%1722: f32, %1723: f32, %1724: f32):
      %1725 = arith.addf %1722, %1723 : f32
      linalg.yield %1725 : f32
    } -> tensor<1x6x1xf32>
    %1726 = tensor.empty() : tensor<1x6x1xf32>
    %1727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1721 : tensor<1x6x1xf32>) outs(%1726 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb140(%1728: f32, %1729: f32):
      %1730 = math.rsqrt %1728 : f32
      linalg.yield %1730 : f32
    } -> tensor<1x6x1xf32>
    %1731 = tensor.empty() : tensor<1x6x64xf32>
    %1732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1691, %1727 : tensor<1x6x64xf32>, tensor<1x6x1xf32>) outs(%1731 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb141(%1733: f32, %1734: f32, %1735: f32):
      %1736 = arith.mulf %1733, %1734 : f32
      linalg.yield %1736 : f32
    } -> tensor<1x6x64xf32>
    %1737 = tensor.empty() : tensor<1x6x64xf32>
    %1738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1732, %56 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1737 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb142(%1739: f32, %1740: f32, %1741: f32):
      %1742 = arith.mulf %1739, %1740 : f32
      linalg.yield %1742 : f32
    } -> tensor<1x6x64xf32>
    %1743 = tensor.empty() : tensor<1x6x64xf32>
    %1744 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1738, %57 : tensor<1x6x64xf32>, tensor<64xf32>) outs(%1743 : tensor<1x6x64xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.ln1"} {
    ^bb143(%1745: f32, %1746: f32, %1747: f32):
      %1748 = arith.addf %1745, %1746 : f32
      linalg.yield %1748 : f32
    } -> tensor<1x6x64xf32>
    %1749 = tensor.empty() : tensor<64x128xi8>
    %1750 = linalg.transpose ins(%143:tensor<128x64xi8>) outs(%1749:tensor<64x128xi8>) permutation = [1, 0]
    %1751 = tensor.collapse_shape %1744 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<1x6x64xf32> into tensor<384xf32>
    %1752 = tensor.expand_shape %1751 [[0 : i64, 1 : i64]] output_shape [6, 64] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<384xf32> into tensor<6x64xf32>
    %1753 = tensor.empty() : tensor<64x128xf32>
    %1754 = arith.constant 0 : i32
    %1755 = tensor.splat %1754 : tensor<128xi32>
    %1756 = "quant_ext.dequantize_per_channel"(%1750, %144, %1755) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.1.keyValueExtractor.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.1.keyValueExtractor.weight.tensor_impl.scale"} : (tensor<64x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<64x128xf32>
    %1757 = tensor.empty() : tensor<6x128xf32>
    %1758 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1759 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1758 : f32) outs(%1757 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1760 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} ins(%1752, %1756 : tensor<6x64xf32>, tensor<64x128xf32>) outs(%1759 : tensor<6x128xf32>) -> tensor<6x128xf32>
    %1761 = tensor.empty() : tensor<6x128xf32>
    %1762 = tensor.collapse_shape %1760 [[0 : i64, 1 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<6x128xf32> into tensor<768xf32>
    %1763 = tensor.expand_shape %1762 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 128] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} : tensor<768xf32> into tensor<1x6x128xf32>
    %1764 = tensor.empty() : tensor<1x6x128xf32>
    %1765 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763, %59 : tensor<1x6x128xf32>, tensor<128xf32>) outs(%1764 : tensor<1x6x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.keyValueExtractor"} {
    ^bb144(%1766: f32, %1767: f32, %1768: f32):
      %1769 = arith.addf %1766, %1767 : f32
      linalg.yield %1769 : f32
    } -> tensor<1x6x128xf32>
    %1770 = tensor.collapse_shape %1765 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x6x128xf32> into tensor<768xf32>
    %1771 = tensor.expand_shape %1770 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 6, 2, 2, 32] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<768xf32> into tensor<1x6x2x2x32xf32>
    %1772 = tensor.empty() : tensor<2x1x2x6x32xf32>
    %1773 = linalg.transpose ins(%1771:tensor<1x6x2x2x32xf32>) outs(%1772:tensor<2x1x2x6x32xf32>) permutation = [2, 0, 3, 1, 4]
    %1774 = "tensor.extract_slice"(%1773) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1775 = tensor.collapse_shape %1774 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1776 = tensor.expand_shape %1775 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1777 = "tensor.extract_slice"(%1773) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 2, 6, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : (tensor<2x1x2x6x32xf32>) -> tensor<1x1x2x6x32xf32>
    %1778 = tensor.collapse_shape %1777 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x1x2x6x32xf32> into tensor<384xf32>
    %1779 = tensor.expand_shape %1778 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 6, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<384xf32> into tensor<1x2x6x32xf32>
    %1780 = tensor.empty() : tensor<64x64xi8>
    %1781 = linalg.transpose ins(%146:tensor<64x64xi8>) outs(%1780:tensor<64x64xi8>) permutation = [1, 0]
    %1782 = tensor.collapse_shape %1640 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1783 = tensor.expand_shape %1782 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<96x64xf32>
    %1784 = tensor.empty() : tensor<64x64xf32>
    %1785 = arith.constant 0 : i32
    %1786 = tensor.splat %1785 : tensor<64xi32>
    %1787 = "quant_ext.dequantize_per_channel"(%1781, %147, %1786) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.1.query.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.1.query.weight.tensor_impl.scale"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1788 = tensor.empty() : tensor<96x64xf32>
    %1789 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1790 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1789 : f32) outs(%1788 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1791 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} ins(%1783, %1787 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1790 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1792 = tensor.empty() : tensor<96x64xf32>
    %1793 = tensor.collapse_shape %1791 [[0 : i64, 1 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<96x64xf32> into tensor<6144xf32>
    %1794 = tensor.expand_shape %1793 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1795 = tensor.empty() : tensor<1x96x64xf32>
    %1796 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1794, %61 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1795 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.query"} {
    ^bb145(%1797: f32, %1798: f32, %1799: f32):
      %1800 = arith.addf %1797, %1798 : f32
      linalg.yield %1800 : f32
    } -> tensor<1x96x64xf32>
    %1801 = tensor.collapse_shape %1796 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1802 = tensor.expand_shape %1801 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 96, 2, 32] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x2x32xf32>
    %1803 = tensor.empty() : tensor<1x2x96x32xf32>
    %1804 = linalg.transpose ins(%1802:tensor<1x96x2x32xf32>) outs(%1803:tensor<1x2x96x32xf32>) permutation = [0, 2, 1, 3]
    %1805 = tensor.empty() : tensor<1x2x32x6xf32>
    %1806 = linalg.transpose ins(%1776:tensor<1x2x6x32xf32>) outs(%1805:tensor<1x2x32x6xf32>) permutation = [0, 1, 3, 2]
    %1807 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1808 = tensor.splat %1807 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1809 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1804, %1806 : tensor<1x2x96x32xf32>, tensor<1x2x32x6xf32>) outs(%1808 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb146(%1810: f32, %1811: f32, %1812: f32):
      %1813 = arith.mulf %1810, %1811 : f32
      %1814 = arith.addf %1812, %1813 : f32
      linalg.yield %1814 : f32
    } -> tensor<1x2x96x6xf32>
    %1815 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 5.65685415 : f32
    %1816 = tensor.splat %1815 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x6xf32>
    %1817 = tensor.empty() : tensor<1x2x96x6xf32>
    %1818 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1809, %1816 : tensor<1x2x96x6xf32>, tensor<1x2x96x6xf32>) outs(%1817 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb147(%1819: f32, %1820: f32, %1821: f32):
      %1822 = arith.divf %1819, %1820 : f32
      linalg.yield %1822 : f32
    } -> tensor<1x2x96x6xf32>
    %1823 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0xff800000 : f32
    %1824 = tensor.splat %1823 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1825 = linalg.reduce ins(%1818:tensor<1x2x96x6xf32>) outs(%1824:tensor<1x2x96xf32>) dimensions = [3]
    (%1826: f32, %1827: f32) {
      %1828 = arith.maximumf %1826, %1827 : f32
      linalg.yield %1828 : f32
    }
    %1829 = tensor.collapse_shape %1825 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1830 = tensor.expand_shape %1829 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1831 = tensor.empty() : tensor<1x2x96x6xf32>
    %1832 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1818, %1830 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1831 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb148(%1833: f32, %1834: f32, %1835: f32):
      %1836 = arith.subf %1833, %1834 : f32
      linalg.yield %1836 : f32
    } -> tensor<1x2x96x6xf32>
    %1837 = tensor.empty() : tensor<1x2x96x6xf32>
    %1838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1832 : tensor<1x2x96x6xf32>) outs(%1837 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb149(%1839: f32, %1840: f32):
      %1841 = math.exp %1839 : f32
      linalg.yield %1841 : f32
    } -> tensor<1x2x96x6xf32>
    %1842 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} 0.000000e+00 : f32
    %1843 = tensor.splat %1842 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32>
    %1844 = linalg.reduce ins(%1838:tensor<1x2x96x6xf32>) outs(%1843:tensor<1x2x96xf32>) dimensions = [3]
    (%1845: f32, %1846: f32) {
      %1847 = arith.addf %1845, %1846 : f32
      linalg.yield %1847 : f32
    }
    %1848 = tensor.collapse_shape %1844 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<1x2x96xf32> into tensor<192xf32>
    %1849 = tensor.expand_shape %1848 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 96, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} : tensor<192xf32> into tensor<1x2x96x1xf32>
    %1850 = tensor.empty() : tensor<1x2x96x6xf32>
    %1851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1838, %1849 : tensor<1x2x96x6xf32>, tensor<1x2x96x1xf32>) outs(%1850 : tensor<1x2x96x6xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.smax"} {
    ^bb150(%1852: f32, %1853: f32, %1854: f32):
      %1855 = arith.divf %1852, %1853 : f32
      linalg.yield %1855 : f32
    } -> tensor<1x2x96x6xf32>
    %1856 = arith.constant {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1857 = tensor.splat %1856 {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x2x96x32xf32>
    %1858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1851, %1779 : tensor<1x2x96x6xf32>, tensor<1x2x6x32xf32>) outs(%1857 : tensor<1x2x96x32xf32>) attrs =  {prov.region_id = "matmul_24", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb151(%1859: f32, %1860: f32, %1861: f32):
      %1862 = arith.mulf %1859, %1860 : f32
      %1863 = arith.addf %1861, %1862 : f32
      linalg.yield %1863 : f32
    } -> tensor<1x2x96x32xf32>
    %1864 = tensor.empty() : tensor<1x96x2x32xf32>
    %1865 = linalg.transpose ins(%1858:tensor<1x2x96x32xf32>) outs(%1864:tensor<1x96x2x32xf32>) permutation = [0, 2, 1, 3]
    %1866 = tensor.collapse_shape %1865 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x2x32xf32> into tensor<6144xf32>
    %1867 = tensor.expand_shape %1866 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1868 = tensor.empty() : tensor<64x64xi8>
    %1869 = linalg.transpose ins(%149:tensor<64x64xi8>) outs(%1868:tensor<64x64xi8>) permutation = [1, 0]
    %1870 = tensor.collapse_shape %1867 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1871 = tensor.expand_shape %1870 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<96x64xf32>
    %1872 = tensor.empty() : tensor<64x64xf32>
    %1873 = arith.constant 0 : i32
    %1874 = tensor.splat %1873 : tensor<64xi32>
    %1875 = "quant_ext.dequantize_per_channel"(%1869, %150, %1874) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._attn.1.finalLayer.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._attn.1.finalLayer.weight.tensor_impl.scale"} : (tensor<64x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<64x64xf32>
    %1876 = tensor.empty() : tensor<96x64xf32>
    %1877 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1878 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1877 : f32) outs(%1876 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1879 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} ins(%1871, %1875 : tensor<96x64xf32>, tensor<64x64xf32>) outs(%1878 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1880 = tensor.empty() : tensor<96x64xf32>
    %1881 = tensor.collapse_shape %1879 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<96x64xf32> into tensor<6144xf32>
    %1882 = tensor.expand_shape %1881 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1883 = tensor.empty() : tensor<1x96x64xf32>
    %1884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1882, %63 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1883 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._attn.1.finalLayer"} {
    ^bb152(%1885: f32, %1886: f32, %1887: f32):
      %1888 = arith.addf %1885, %1886 : f32
      linalg.yield %1888 : f32
    } -> tensor<1x96x64xf32>
    %1889 = tensor.empty() : tensor<1x96x64xf32>
    %1890 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1640, %1884 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1889 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb153(%1891: f32, %1892: f32, %1893: f32):
      %1894 = arith.addf %1891, %1892 : f32
      linalg.yield %1894 : f32
    } -> tensor<1x96x64xf32>
    %1895 = tensor.empty() : tensor<64x512xi8>
    %1896 = linalg.transpose ins(%158:tensor<512x64xi8>) outs(%1895:tensor<64x512xi8>) permutation = [1, 0]
    %1897 = tensor.collapse_shape %1890 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %1898 = tensor.expand_shape %1897 [[0 : i64, 1 : i64]] output_shape [96, 64] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<6144xf32> into tensor<96x64xf32>
    %1899 = tensor.empty() : tensor<64x512xf32>
    %1900 = arith.constant 0 : i32
    %1901 = tensor.splat %1900 : tensor<512xi32>
    %1902 = "quant_ext.dequantize_per_channel"(%1896, %159, %1901) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._ffn.1.mlp1.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._ffn.1.mlp1.weight.tensor_impl.scale"} : (tensor<64x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<64x512xf32>
    %1903 = tensor.empty() : tensor<96x512xf32>
    %1904 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1905 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1904 : f32) outs(%1903 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1906 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} ins(%1898, %1902 : tensor<96x64xf32>, tensor<64x512xf32>) outs(%1905 : tensor<96x512xf32>) -> tensor<96x512xf32>
    %1907 = tensor.empty() : tensor<96x512xf32>
    %1908 = tensor.collapse_shape %1906 [[0 : i64, 1 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<96x512xf32> into tensor<49152xf32>
    %1909 = tensor.expand_shape %1908 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 512] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} : tensor<49152xf32> into tensor<1x96x512xf32>
    %1910 = tensor.empty() : tensor<1x96x512xf32>
    %1911 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1909, %71 : tensor<1x96x512xf32>, tensor<512xf32>) outs(%1910 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp1"} {
    ^bb154(%1912: f32, %1913: f32, %1914: f32):
      %1915 = arith.addf %1912, %1913 : f32
      linalg.yield %1915 : f32
    } -> tensor<1x96x512xf32>
    %1916 = tensor.empty() : tensor<1x512x96xf32>
    %1917 = linalg.transpose ins(%1911:tensor<1x96x512xf32>) outs(%1916:tensor<1x512x96xf32>) permutation = [0, 2, 1]
    %1918 = tensor.collapse_shape %1917 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x96xf32> into tensor<49152xf32>
    %1919 = tensor.expand_shape %1918 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1920 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1921 = tensor.splat %1920 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<1x512x10x14xf32>
    %1922 = "tensor.insert_slice"(%1919, %1921) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 8, 12>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : (tensor<1x512x8x12xf32>, tensor<1x512x10x14xf32>) -> tensor<1x512x10x14xf32>
    %1923 = tensor.empty() : tensor<64x8x3x3x1x8x12xf32>
    %1924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, ((d0 * 8) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1922 : tensor<1x512x10x14xf32>) outs(%1923 : tensor<64x8x3x3x1x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb155(%1925: f32, %1926: f32):
      linalg.yield %1925 : f32
    } -> tensor<64x8x3x3x1x8x12xf32>
    %1927 = tensor.collapse_shape %1924 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x3x3x1x8x12xf32> into tensor<442368xf32>
    %1928 = tensor.expand_shape %1927 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 72, 96] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<442368xf32> into tensor<64x72x96xf32>
    %1929 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x8x3x3xf32> into tensor<36864xf32>
    %1930 = tensor.expand_shape %1929 [[0 : i64, 1 : i64, 2 : i64]] output_shape [64, 8, 72] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<36864xf32> into tensor<64x8x72xf32>
    %1931 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} 0.000000e+00 : f32
    %1932 = tensor.splat %1931 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32>
    %1933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1930, %1928 : tensor<64x8x72xf32>, tensor<64x72x96xf32>) outs(%1932 : tensor<64x8x96xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb156(%1934: f32, %1935: f32, %1936: f32):
      %1937 = arith.mulf %1934, %1935 : f32
      %1938 = arith.addf %1936, %1937 : f32
      linalg.yield %1938 : f32
    } -> tensor<64x8x96xf32>
    %1939 = tensor.collapse_shape %1933 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<64x8x96xf32> into tensor<49152xf32>
    %1940 = tensor.expand_shape %1939 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<512x1x8x12xf32>
    %1941 = tensor.collapse_shape %1940 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<512x1x8x12xf32> into tensor<49152xf32>
    %1942 = tensor.expand_shape %1941 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 8, 12] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} : tensor<49152xf32> into tensor<1x512x8x12xf32>
    %1943 = tensor.empty() : tensor<1x512x8x12xf32>
    %1944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1942, %73 : tensor<1x512x8x12xf32>, tensor<512xf32>) outs(%1943 : tensor<1x512x8x12xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.padding", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.depthwise"} {
    ^bb157(%1945: f32, %1946: f32, %1947: f32):
      %1948 = arith.addf %1945, %1946 : f32
      linalg.yield %1948 : f32
    } -> tensor<1x512x8x12xf32>
    %1949 = tensor.collapse_shape %1944 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x512x8x12xf32> into tensor<49152xf32>
    %1950 = tensor.expand_shape %1949 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 512, 96] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<49152xf32> into tensor<1x512x96xf32>
    %1951 = tensor.empty() : tensor<1x96x512xf32>
    %1952 = linalg.transpose ins(%1950:tensor<1x512x96xf32>) outs(%1951:tensor<1x96x512xf32>) permutation = [0, 2, 1]
    %1953 = tensor.empty() : tensor<1x96x512xf32>
    %1954 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1952 : tensor<1x96x512xf32>) outs(%1953 : tensor<1x96x512xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.gelu"} {
    ^bb158(%1955: f32, %1956: f32):
      %1957 = arith.constant 5.000000e-01 : f32
      %1958 = arith.constant 1.000000e+00 : f32
      %1959 = arith.constant 0.707106769 : f32
      %1960 = arith.mulf %1955, %1959 : f32
      %1961 = math.erf %1960 : f32
      %1962 = arith.addf %1958, %1961 : f32
      %1963 = arith.mulf %1957, %1955 : f32
      %1964 = arith.mulf %1963, %1962 : f32
      linalg.yield %1964 : f32
    } -> tensor<1x96x512xf32>
    %1965 = tensor.empty() : tensor<512x64xi8>
    %1966 = linalg.transpose ins(%161:tensor<64x512xi8>) outs(%1965:tensor<512x64xi8>) permutation = [1, 0]
    %1967 = tensor.collapse_shape %1954 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<1x96x512xf32> into tensor<49152xf32>
    %1968 = tensor.expand_shape %1967 [[0 : i64, 1 : i64]] output_shape [96, 512] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<49152xf32> into tensor<96x512xf32>
    %1969 = tensor.empty() : tensor<512x64xf32>
    %1970 = arith.constant 0 : i32
    %1971 = tensor.splat %1970 : tensor<64xi32>
    %1972 = "quant_ext.dequantize_per_channel"(%1966, %162, %1971) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.encoder_blocks.1._ffn.1.mlp2.weight.tensor_impl.int_data", prov.quant_inner_s = "net.encoder_blocks.1._ffn.1.mlp2.weight.tensor_impl.scale"} : (tensor<512x64xi8>, tensor<64xf32>, tensor<64xi32>) -> tensor<512x64xf32>
    %1973 = tensor.empty() : tensor<96x64xf32>
    %1974 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %1975 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%1974 : f32) outs(%1973 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1976 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} ins(%1968, %1972 : tensor<96x512xf32>, tensor<512x64xf32>) outs(%1975 : tensor<96x64xf32>) -> tensor<96x64xf32>
    %1977 = tensor.empty() : tensor<96x64xf32>
    %1978 = tensor.collapse_shape %1976 [[0 : i64, 1 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<96x64xf32> into tensor<6144xf32>
    %1979 = tensor.expand_shape %1978 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 64] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} : tensor<6144xf32> into tensor<1x96x64xf32>
    %1980 = tensor.empty() : tensor<1x96x64xf32>
    %1981 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1979, %75 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%1980 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1._ffn.1.mlp2"} {
    ^bb159(%1982: f32, %1983: f32, %1984: f32):
      %1985 = arith.addf %1982, %1983 : f32
      linalg.yield %1985 : f32
    } -> tensor<1x96x64xf32>
    %1986 = tensor.empty() : tensor<1x96x64xf32>
    %1987 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1890, %1981 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%1986 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb160(%1988: f32, %1989: f32, %1990: f32):
      %1991 = arith.addf %1988, %1989 : f32
      linalg.yield %1991 : f32
    } -> tensor<1x96x64xf32>
    %1992 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %1993 = tensor.splat %1992 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %1994 = linalg.reduce ins(%1987:tensor<1x96x64xf32>) outs(%1993:tensor<1x96xf32>) dimensions = [2]
    (%1995: f32, %1996: f32) {
      %1997 = arith.addf %1995, %1996 : f32
      linalg.yield %1997 : f32
    }
    %1998 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %1999 = tensor.splat %1998 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2000 = tensor.empty() : tensor<1x96xf32>
    %2001 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1994, %1999 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2000 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb161(%2002: f32, %2003: f32, %2004: f32):
      %2005 = arith.divf %2002, %2003 : f32
      linalg.yield %2005 : f32
    } -> tensor<1x96xf32>
    %2006 = tensor.collapse_shape %2001 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2007 = tensor.expand_shape %2006 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2008 = tensor.empty() : tensor<1x96x64xf32>
    %2009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1987, %2007 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2008 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb162(%2010: f32, %2011: f32, %2012: f32):
      %2013 = arith.subf %2010, %2011 : f32
      linalg.yield %2013 : f32
    } -> tensor<1x96x64xf32>
    %2014 = tensor.empty() : tensor<1x96x64xf32>
    %2015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2009, %2009 : tensor<1x96x64xf32>, tensor<1x96x64xf32>) outs(%2014 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb163(%2016: f32, %2017: f32, %2018: f32):
      %2019 = arith.mulf %2016, %2017 : f32
      linalg.yield %2019 : f32
    } -> tensor<1x96x64xf32>
    %2020 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 0.000000e+00 : f32
    %2021 = tensor.splat %2020 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2022 = linalg.reduce ins(%2015:tensor<1x96x64xf32>) outs(%2021:tensor<1x96xf32>) dimensions = [2]
    (%2023: f32, %2024: f32) {
      %2025 = arith.addf %2023, %2024 : f32
      linalg.yield %2025 : f32
    }
    %2026 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 6.400000e+01 : f32
    %2027 = tensor.splat %2026 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32>
    %2028 = tensor.empty() : tensor<1x96xf32>
    %2029 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2022, %2027 : tensor<1x96xf32>, tensor<1x96xf32>) outs(%2028 : tensor<1x96xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb164(%2030: f32, %2031: f32, %2032: f32):
      %2033 = arith.divf %2030, %2031 : f32
      linalg.yield %2033 : f32
    } -> tensor<1x96xf32>
    %2034 = tensor.collapse_shape %2029 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96xf32> into tensor<96xf32>
    %2035 = tensor.expand_shape %2034 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 96, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<96xf32> into tensor<1x96x1xf32>
    %2036 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} 1.000000e-05 : f32
    %2037 = tensor.splat %2036 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x1xf32>
    %2038 = tensor.empty() : tensor<1x96x1xf32>
    %2039 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2035, %2037 : tensor<1x96x1xf32>, tensor<1x96x1xf32>) outs(%2038 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb165(%2040: f32, %2041: f32, %2042: f32):
      %2043 = arith.addf %2040, %2041 : f32
      linalg.yield %2043 : f32
    } -> tensor<1x96x1xf32>
    %2044 = tensor.empty() : tensor<1x96x1xf32>
    %2045 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2039 : tensor<1x96x1xf32>) outs(%2044 : tensor<1x96x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb166(%2046: f32, %2047: f32):
      %2048 = math.rsqrt %2046 : f32
      linalg.yield %2048 : f32
    } -> tensor<1x96x1xf32>
    %2049 = tensor.empty() : tensor<1x96x64xf32>
    %2050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2009, %2045 : tensor<1x96x64xf32>, tensor<1x96x1xf32>) outs(%2049 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb167(%2051: f32, %2052: f32, %2053: f32):
      %2054 = arith.mulf %2051, %2052 : f32
      linalg.yield %2054 : f32
    } -> tensor<1x96x64xf32>
    %2055 = tensor.empty() : tensor<1x96x64xf32>
    %2056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2050, %78 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2055 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb168(%2057: f32, %2058: f32, %2059: f32):
      %2060 = arith.mulf %2057, %2058 : f32
      linalg.yield %2060 : f32
    } -> tensor<1x96x64xf32>
    %2061 = tensor.empty() : tensor<1x96x64xf32>
    %2062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2056, %79 : tensor<1x96x64xf32>, tensor<64xf32>) outs(%2061 : tensor<1x96x64xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.layer_norm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} {
    ^bb169(%2063: f32, %2064: f32, %2065: f32):
      %2066 = arith.addf %2063, %2064 : f32
      linalg.yield %2066 : f32
    } -> tensor<1x96x64xf32>
    %2067 = tensor.collapse_shape %2062 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<1x96x64xf32> into tensor<6144xf32>
    %2068 = tensor.expand_shape %2067 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 12, 64] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.encoder_blocks.1"} : tensor<6144xf32> into tensor<1x8x12x64xf32>
    %2069 = tensor.empty() : tensor<1x64x8x12xf32>
    %2070 = linalg.transpose ins(%2068:tensor<1x8x12x64xf32>) outs(%2069:tensor<1x64x8x12xf32>) permutation = [0, 3, 1, 2]
    %2071 = tensor.collapse_shape %2070 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x64x8x12xf32> into tensor<6144xf32>
    %2072 = tensor.expand_shape %2071 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 16, 2, 2, 8, 12] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x2x2x8x12xf32>
    %2073 = tensor.empty() : tensor<1x16x8x2x12x2xf32>
    %2074 = linalg.transpose ins(%2072:tensor<1x16x2x2x8x12xf32>) outs(%2073:tensor<1x16x8x2x12x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %2075 = tensor.collapse_shape %2074 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<1x16x8x2x12x2xf32> into tensor<6144xf32>
    %2076 = tensor.expand_shape %2075 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 16, 16, 24] {prov._pattern_hint = "pixel_shuffle", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.pixel_shuffle.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.pxShuffle"} : tensor<6144xf32> into tensor<1x16x16x24xf32>
    %2077 = tensor.empty() : tensor<1x32x23x15xf32>
    %2078 = linalg.transpose ins(%1119:tensor<1x32x15x23xf32>) outs(%2077:tensor<1x32x23x15xf32>) permutation = [0, 1, 3, 2]
    %2079 = tensor.collapse_shape %2078 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x23x15xf32> into tensor<11040xf32>
    %2080 = tensor.expand_shape %2079 [[0 : i64, 1 : i64]] output_shape [736, 15] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11040xf32> into tensor<736x15xf32>
    %2081 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F8988883D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEE6E3F8988083E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DEDD5D3FCDCC4C3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3F8988883E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000BCBB3B3FABAAAA3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAA2A3FCDCCCC3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009A99193FEFEEEE3E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083F8988083F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000EFEEEE3E9A99193F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCCCC3EABAA2A3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ABAAAA3EBCBB3B3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883ECDCC4C3F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDCC4C3EDEDD5D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988083EEFEE6E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008988883D0000803F"> : tensor<15x16xf32>
    %2082 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2083 = tensor.splat %2082 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32>
    %2084 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2080, %2081 : tensor<736x15xf32>, tensor<15x16xf32>) outs(%2083 : tensor<736x16xf32>) -> tensor<736x16xf32>
    %2085 = tensor.collapse_shape %2084 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<736x16xf32> into tensor<11776xf32>
    %2086 = tensor.expand_shape %2085 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 23, 16] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<1x32x23x16xf32>
    %2087 = tensor.empty() : tensor<1x32x16x23xf32>
    %2088 = linalg.transpose ins(%2086:tensor<1x32x23x16xf32>) outs(%2087:tensor<1x32x16x23xf32>) permutation = [0, 1, 3, 2]
    %2089 = tensor.collapse_shape %2088 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<1x32x16x23xf32> into tensor<11776xf32>
    %2090 = tensor.expand_shape %2089 [[0 : i64, 1 : i64]] output_shape [512, 23] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<11776xf32> into tensor<512x23xf32>
    %2091 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} dense<"0x0000803F4316323D00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDE743F4316B23D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000038BD693FB290053E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3F4316323E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006F7A533FD39B5E3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59483FB290853E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A7373D3F7AD39B3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323F4316B23E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000DFF4263F0B59C83E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD31B3FD39BDE3E000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000016B2103F9CDEF43E0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053FB290053F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009CDEF43E16B2103F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39BDE3E7AD31B3F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B59C83EDFF4263F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23E4316323F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007AD39B3EA7373D3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290853E0B59483F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000D39B5E3E6F7A533F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323ED39B5E3F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000B290053E38BD693F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316B23D9CDE743F00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004316323D0000803F"> : tensor<23x24xf32>
    %2092 = arith.constant {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} 0.000000e+00 : f32
    %2093 = tensor.splat %2092 {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32>
    %2094 = linalg.matmul {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} ins(%2090, %2091 : tensor<512x23xf32>, tensor<23x24xf32>) outs(%2093 : tensor<512x24xf32>) -> tensor<512x24xf32>
    %2095 = tensor.collapse_shape %2094 [[0 : i64, 1 : i64]] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<512x24xf32> into tensor<12288xf32>
    %2096 = tensor.expand_shape %2095 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 16, 24] {prov.region_id = "resize_0", prov.family = "resize", prov._pattern_hint = "resize", prov.op = "resize", prov.aten = "aten.upsample_bilinear2d.vec", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.up_sample"} : tensor<12288xf32> into tensor<1x32x16x24xf32>
    %2097 = tensor.concat dim(1) %2076, %2096 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x16x16x24xf32>, tensor<1x32x16x24xf32>) -> tensor<1x48x16x24xf32>
    %2098 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2099 = tensor.splat %2098 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<1x48x18x26xf32>
    %2100 = "tensor.insert_slice"(%2097, %2099) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 16, 24>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : (tensor<1x48x16x24xf32>, tensor<1x48x18x26xf32>) -> tensor<1x48x18x26xf32>
    %2101 = tensor.empty() : tensor<48x3x3x1x16x24xf32>
    %2102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2100 : tensor<1x48x18x26xf32>) outs(%2101 : tensor<48x3x3x1x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb170(%2103: f32, %2104: f32):
      linalg.yield %2103 : f32
    } -> tensor<48x3x3x1x16x24xf32>
    %2105 = tensor.collapse_shape %2102 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<48x3x3x1x16x24xf32> into tensor<165888xf32>
    %2106 = tensor.expand_shape %2105 [[0 : i64, 1 : i64]] output_shape [432, 384] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<165888xf32> into tensor<432x384xf32>
    %2107 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x48x3x3xf32> into tensor<5184xf32>
    %2108 = tensor.expand_shape %2107 [[0 : i64, 1 : i64]] output_shape [12, 432] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<5184xf32> into tensor<12x432xf32>
    %2109 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} 0.000000e+00 : f32
    %2110 = tensor.splat %2109 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32>
    %2111 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} ins(%2108, %2106 : tensor<12x432xf32>, tensor<432x384xf32>) outs(%2110 : tensor<12x384xf32>) -> tensor<12x384xf32>
    %2112 = tensor.collapse_shape %2111 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x384xf32> into tensor<4608xf32>
    %2113 = tensor.expand_shape %2112 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [12, 1, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<12x1x16x24xf32>
    %2114 = tensor.collapse_shape %2113 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<12x1x16x24xf32> into tensor<4608xf32>
    %2115 = tensor.expand_shape %2114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 12, 16, 24] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} : tensor<4608xf32> into tensor<1x12x16x24xf32>
    %2116 = tensor.empty() : tensor<1x12x16x24xf32>
    %2117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2115, %97 : tensor<1x12x16x24xf32>, tensor<12xf32>) outs(%2116 : tensor<1x12x16x24xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.down_sample"} {
    ^bb171(%2118: f32, %2119: f32, %2120: f32):
      %2121 = arith.addf %2118, %2119 : f32
      linalg.yield %2121 : f32
    } -> tensor<1x12x16x24xf32>
    %2122 = tensor.collapse_shape %2117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x12x16x24xf32> into tensor<4608xf32>
    %2123 = tensor.expand_shape %2122 [[0 : i64, 1 : i64]] output_shape [1, 4608] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<4608xf32> into tensor<1x4608xf32>
    %2124 = tensor.empty() : tensor<4608x512xi8>
    %2125 = linalg.transpose ins(%164:tensor<512x4608xi8>) outs(%2124:tensor<4608x512xi8>) permutation = [1, 0]
    %2126 = tensor.empty() : tensor<4608x512xf32>
    %2127 = arith.constant 0 : i32
    %2128 = tensor.splat %2127 : tensor<512xi32>
    %2129 = "quant_ext.dequantize_per_channel"(%2125, %165, %2128) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.decoder.weight.tensor_impl.int_data", prov.quant_inner_s = "net.decoder.weight.tensor_impl.scale"} : (tensor<4608x512xi8>, tensor<512xf32>, tensor<512xi32>) -> tensor<4608x512xf32>
    %2130 = tensor.empty() : tensor<1x512xf32>
    %2131 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2132 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2131 : f32) outs(%2130 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2133 = linalg.matmul {prov.region_id = "matmul_28", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} ins(%2123, %2129 : tensor<1x4608xf32>, tensor<4608x512xf32>) outs(%2132 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2134 = tensor.empty() : tensor<1x512xf32>
    %2135 = tensor.empty() : tensor<1x512xf32>
    %2136 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2133, %80 : tensor<1x512xf32>, tensor<512xf32>) outs(%2135 : tensor<1x512xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.decoder"} {
    ^bb172(%2137: f32, %2138: f32, %2139: f32):
      %2140 = arith.addf %2137, %2138 : f32
      linalg.yield %2140 : f32
    } -> tensor<1x512xf32>
    %2141 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} 1.000000e+01 : f32
    %2142 = tensor.splat %2141 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : tensor<1x1xf32>
    %2143 = tensor.empty() : tensor<1x1xf32>
    %2144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %2142 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2143 : tensor<1x1xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} {
    ^bb173(%2145: f32, %2146: f32, %2147: f32):
      %2148 = arith.divf %2145, %2146 : f32
      linalg.yield %2148 : f32
    } -> tensor<1x1xf32>
    %2149 = tensor.concat dim(1) %2136, %2144, %100 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net"} : (tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x517xf32>
    %2150 = tensor.collapse_shape %2149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x517xf32> into tensor<517xf32>
    %2151 = tensor.expand_shape %2150 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 517] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x1x517xf32>
    %2152 = tensor.collapse_shape %101 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2153 = tensor.expand_shape %2152 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2154 = tensor.collapse_shape %102 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x128xf32> into tensor<384xf32>
    %2155 = tensor.expand_shape %2154 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 1, 128] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x1x128xf32>
    %2156 = "tensor.extract_slice"(%2151) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 517>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x517xf32>) -> tensor<1x1x517xf32>
    %2157 = tensor.collapse_shape %2156 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x517xf32> into tensor<517xf32>
    %2158 = tensor.expand_shape %2157 [[0 : i64, 1 : i64]] output_shape [1, 517] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<517xf32> into tensor<1x517xf32>
    %2159 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2160 = tensor.collapse_shape %2159 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2161 = tensor.expand_shape %2160 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2162 = "tensor.extract_slice"(%2155) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2163 = tensor.collapse_shape %2162 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2164 = tensor.expand_shape %2163 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2165 = tensor.empty() : tensor<517x512xf32>
    %2166 = linalg.transpose ins(%82:tensor<512x517xf32>) outs(%2165:tensor<517x512xf32>) permutation = [1, 0]
    %2167 = tensor.empty() : tensor<1x512xf32>
    %2168 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2169 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2168 : f32) outs(%2167 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2170 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2158, %2166 : tensor<1x517xf32>, tensor<517x512xf32>) outs(%2169 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2171 = tensor.empty() : tensor<128x512xf32>
    %2172 = linalg.transpose ins(%83:tensor<512x128xf32>) outs(%2171:tensor<128x512xf32>) permutation = [1, 0]
    %2173 = tensor.empty() : tensor<1x512xf32>
    %2174 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2175 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2174 : f32) outs(%2173 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2176 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2161, %2172 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2175 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2177 = tensor.empty() : tensor<1x512xf32>
    %2178 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2170, %2176, %84, %85 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2177 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb174(%2179: f32, %2180: f32, %2181: f32, %2182: f32, %2183: f32):
      %2184 = arith.addf %2179, %2180 : f32
      %2185 = arith.addf %2184, %2181 : f32
      %2186 = arith.addf %2185, %2182 : f32
      linalg.yield %2186 : f32
    } -> tensor<1x512xf32>
    %2187 = "tensor.extract_slice"(%2178) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2188 = "tensor.extract_slice"(%2178) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2189 = "tensor.extract_slice"(%2178) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2190 = "tensor.extract_slice"(%2178) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2191 = tensor.empty() : tensor<1x128xf32>
    %2192 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2187, %2188, %2189, %2164 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2191 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb175(%2193: f32, %2194: f32, %2195: f32, %2196: f32, %2197: f32):
      %2198 = arith.constant 1.000000e+00 : f32
      %2199 = arith.negf %2194 : f32
      %2200 = math.exp %2199 : f32
      %2201 = arith.addf %2198, %2200 : f32
      %2202 = arith.divf %2198, %2201 : f32
      %2203 = arith.constant 1.000000e+00 : f32
      %2204 = arith.negf %2193 : f32
      %2205 = math.exp %2204 : f32
      %2206 = arith.addf %2203, %2205 : f32
      %2207 = arith.divf %2203, %2206 : f32
      %2208 = math.tanh %2195 : f32
      %2209 = arith.mulf %2202, %2196 : f32
      %2210 = arith.mulf %2207, %2208 : f32
      %2211 = arith.addf %2209, %2210 : f32
      linalg.yield %2211 : f32
    } -> tensor<1x128xf32>
    %2212 = tensor.empty() : tensor<1x128xf32>
    %2213 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2190, %2192 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2212 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb176(%2214: f32, %2215: f32, %2216: f32):
      %2217 = arith.constant 1.000000e+00 : f32
      %2218 = arith.negf %2214 : f32
      %2219 = math.exp %2218 : f32
      %2220 = arith.addf %2217, %2219 : f32
      %2221 = arith.divf %2217, %2220 : f32
      %2222 = math.tanh %2215 : f32
      %2223 = arith.mulf %2221, %2222 : f32
      linalg.yield %2223 : f32
    } -> tensor<1x128xf32>
    %2224 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2225 = tensor.collapse_shape %2224 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2226 = tensor.expand_shape %2225 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2227 = "tensor.extract_slice"(%2155) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2228 = tensor.collapse_shape %2227 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2229 = tensor.expand_shape %2228 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2230 = tensor.empty() : tensor<128x512xf32>
    %2231 = linalg.transpose ins(%86:tensor<512x128xf32>) outs(%2230:tensor<128x512xf32>) permutation = [1, 0]
    %2232 = tensor.empty() : tensor<1x512xf32>
    %2233 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2234 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2233 : f32) outs(%2232 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2235 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2213, %2231 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2234 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2236 = tensor.empty() : tensor<128x512xf32>
    %2237 = linalg.transpose ins(%87:tensor<512x128xf32>) outs(%2236:tensor<128x512xf32>) permutation = [1, 0]
    %2238 = tensor.empty() : tensor<1x512xf32>
    %2239 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2240 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2239 : f32) outs(%2238 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2241 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2226, %2237 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2240 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2242 = tensor.empty() : tensor<1x512xf32>
    %2243 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2235, %2241, %88, %89 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2242 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb177(%2244: f32, %2245: f32, %2246: f32, %2247: f32, %2248: f32):
      %2249 = arith.addf %2244, %2245 : f32
      %2250 = arith.addf %2249, %2246 : f32
      %2251 = arith.addf %2250, %2247 : f32
      linalg.yield %2251 : f32
    } -> tensor<1x512xf32>
    %2252 = "tensor.extract_slice"(%2243) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2253 = "tensor.extract_slice"(%2243) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2254 = "tensor.extract_slice"(%2243) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2255 = "tensor.extract_slice"(%2243) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2256 = tensor.empty() : tensor<1x128xf32>
    %2257 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2252, %2253, %2254, %2229 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2256 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb178(%2258: f32, %2259: f32, %2260: f32, %2261: f32, %2262: f32):
      %2263 = arith.constant 1.000000e+00 : f32
      %2264 = arith.negf %2259 : f32
      %2265 = math.exp %2264 : f32
      %2266 = arith.addf %2263, %2265 : f32
      %2267 = arith.divf %2263, %2266 : f32
      %2268 = arith.constant 1.000000e+00 : f32
      %2269 = arith.negf %2258 : f32
      %2270 = math.exp %2269 : f32
      %2271 = arith.addf %2268, %2270 : f32
      %2272 = arith.divf %2268, %2271 : f32
      %2273 = math.tanh %2260 : f32
      %2274 = arith.mulf %2267, %2261 : f32
      %2275 = arith.mulf %2272, %2273 : f32
      %2276 = arith.addf %2274, %2275 : f32
      linalg.yield %2276 : f32
    } -> tensor<1x128xf32>
    %2277 = tensor.empty() : tensor<1x128xf32>
    %2278 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2255, %2257 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2277 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb179(%2279: f32, %2280: f32, %2281: f32):
      %2282 = arith.constant 1.000000e+00 : f32
      %2283 = arith.negf %2279 : f32
      %2284 = math.exp %2283 : f32
      %2285 = arith.addf %2282, %2284 : f32
      %2286 = arith.divf %2282, %2285 : f32
      %2287 = math.tanh %2280 : f32
      %2288 = arith.mulf %2286, %2287 : f32
      linalg.yield %2288 : f32
    } -> tensor<1x128xf32>
    %2289 = "tensor.extract_slice"(%2153) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2290 = tensor.collapse_shape %2289 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2291 = tensor.expand_shape %2290 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2292 = "tensor.extract_slice"(%2155) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<3x1x128xf32>) -> tensor<1x1x128xf32>
    %2293 = tensor.collapse_shape %2292 [[0 : i64, 1 : i64, 2 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2294 = tensor.expand_shape %2293 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2295 = tensor.empty() : tensor<128x512xf32>
    %2296 = linalg.transpose ins(%90:tensor<512x128xf32>) outs(%2295:tensor<128x512xf32>) permutation = [1, 0]
    %2297 = tensor.empty() : tensor<1x512xf32>
    %2298 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2299 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2298 : f32) outs(%2297 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2300 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2278, %2296 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2299 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2301 = tensor.empty() : tensor<128x512xf32>
    %2302 = linalg.transpose ins(%91:tensor<512x128xf32>) outs(%2301:tensor<128x512xf32>) permutation = [1, 0]
    %2303 = tensor.empty() : tensor<1x512xf32>
    %2304 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2305 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2304 : f32) outs(%2303 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2306 = linalg.matmul {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm", prov.transposed_b = "true"} ins(%2291, %2302 : tensor<1x128xf32>, tensor<128x512xf32>) outs(%2305 : tensor<1x512xf32>) -> tensor<1x512xf32>
    %2307 = tensor.empty() : tensor<1x512xf32>
    %2308 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2300, %2306, %92, %93 : tensor<1x512xf32>, tensor<1x512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%2307 : tensor<1x512xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb180(%2309: f32, %2310: f32, %2311: f32, %2312: f32, %2313: f32):
      %2314 = arith.addf %2309, %2310 : f32
      %2315 = arith.addf %2314, %2311 : f32
      %2316 = arith.addf %2315, %2312 : f32
      linalg.yield %2316 : f32
    } -> tensor<1x512xf32>
    %2317 = "tensor.extract_slice"(%2308) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2318 = "tensor.extract_slice"(%2308) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2319 = "tensor.extract_slice"(%2308) <{static_offsets = array<i64: 0, 256>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2320 = "tensor.extract_slice"(%2308) <{static_offsets = array<i64: 0, 384>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x512xf32>) -> tensor<1x128xf32>
    %2321 = tensor.empty() : tensor<1x128xf32>
    %2322 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2317, %2318, %2319, %2294 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) outs(%2321 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb181(%2323: f32, %2324: f32, %2325: f32, %2326: f32, %2327: f32):
      %2328 = arith.constant 1.000000e+00 : f32
      %2329 = arith.negf %2324 : f32
      %2330 = math.exp %2329 : f32
      %2331 = arith.addf %2328, %2330 : f32
      %2332 = arith.divf %2328, %2331 : f32
      %2333 = arith.constant 1.000000e+00 : f32
      %2334 = arith.negf %2323 : f32
      %2335 = math.exp %2334 : f32
      %2336 = arith.addf %2333, %2335 : f32
      %2337 = arith.divf %2333, %2336 : f32
      %2338 = math.tanh %2325 : f32
      %2339 = arith.mulf %2332, %2326 : f32
      %2340 = arith.mulf %2337, %2338 : f32
      %2341 = arith.addf %2339, %2340 : f32
      linalg.yield %2341 : f32
    } -> tensor<1x128xf32>
    %2342 = tensor.empty() : tensor<1x128xf32>
    %2343 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2320, %2322 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%2342 : tensor<1x128xf32>) attrs =  {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "lstm", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} {
    ^bb182(%2344: f32, %2345: f32, %2346: f32):
      %2347 = arith.constant 1.000000e+00 : f32
      %2348 = arith.negf %2344 : f32
      %2349 = math.exp %2348 : f32
      %2350 = arith.addf %2347, %2349 : f32
      %2351 = arith.divf %2347, %2350 : f32
      %2352 = math.tanh %2345 : f32
      %2353 = arith.mulf %2351, %2352 : f32
      linalg.yield %2353 : f32
    } -> tensor<1x128xf32>
    %2354 = tensor.empty() : tensor<1x1x128xf32>
    %2355 = tensor.collapse_shape %2343 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2356 = tensor.expand_shape %2355 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2357 = "tensor.insert_slice"(%2356, %2354) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %2358 = tensor.empty() : tensor<3x1x128xf32>
    %2359 = tensor.collapse_shape %2213 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2360 = tensor.expand_shape %2359 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2361 = "tensor.insert_slice"(%2360, %2358) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2362 = tensor.collapse_shape %2278 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2363 = tensor.expand_shape %2362 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2364 = "tensor.insert_slice"(%2363, %2361) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2365 = tensor.collapse_shape %2343 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2366 = tensor.expand_shape %2365 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2367 = "tensor.insert_slice"(%2366, %2364) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2368 = tensor.empty() : tensor<3x1x128xf32>
    %2369 = tensor.collapse_shape %2192 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2370 = tensor.expand_shape %2369 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2371 = "tensor.insert_slice"(%2370, %2368) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2372 = tensor.collapse_shape %2257 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2373 = tensor.expand_shape %2372 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2374 = "tensor.insert_slice"(%2373, %2371) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2375 = tensor.collapse_shape %2322 [[0 : i64, 1 : i64]] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x128xf32> into tensor<128xf32>
    %2376 = tensor.expand_shape %2375 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov._pattern_hint = "lstm", prov.op = "reshape", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x1x128xf32>
    %2377 = "tensor.insert_slice"(%2376, %2374) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "lstm_0", prov._pattern_hint = "lstm", prov.op = "slice_scatter", prov.family = "layout", prov.aten = "aten.lstm.input", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : (tensor<1x1x128xf32>, tensor<3x1x128xf32>) -> tensor<3x1x128xf32>
    %2378 = tensor.collapse_shape %2357 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<1x1x128xf32> into tensor<128xf32>
    %2379 = tensor.expand_shape %2378 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<128xf32> into tensor<1x128xf32>
    %2380 = tensor.collapse_shape %2367 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2381 = tensor.expand_shape %2380 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2382 = tensor.collapse_shape %2377 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<3x1x128xf32> into tensor<384xf32>
    %2383 = tensor.expand_shape %2382 [[0 : i64, 1 : i64]] output_shape [3, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dim", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.lstm"} : tensor<384xf32> into tensor<3x128xf32>
    %2384 = tensor.empty() : tensor<128x3xi8>
    %2385 = linalg.transpose ins(%167:tensor<3x128xi8>) outs(%2384:tensor<128x3xi8>) permutation = [1, 0]
    %2386 = tensor.empty() : tensor<128x3xf32>
    %2387 = arith.constant 0 : i32
    %2388 = tensor.splat %2387 : tensor<3xi32>
    %2389 = "quant_ext.dequantize_per_channel"(%2385, %168, %2388) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize", prov.quant_inner_w = "net.nn_fc2.weight.tensor_impl.int_data", prov.quant_inner_s = "net.nn_fc2.weight.tensor_impl.scale"} : (tensor<128x3xi8>, tensor<3xf32>, tensor<3xi32>) -> tensor<128x3xf32>
    %2390 = tensor.empty() : tensor<1x3xf32>
    %2391 = arith.constant {prov.module = "net"} 0.000000e+00 : f32
    %2392 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "net"} ins(%2391 : f32) outs(%2390 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2393 = linalg.matmul {prov.region_id = "matmul_29", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} ins(%2379, %2389 : tensor<1x128xf32>, tensor<128x3xf32>) outs(%2392 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %2394 = tensor.empty() : tensor<1x3xf32>
    %2395 = tensor.empty() : tensor<1x3xf32>
    %2396 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2393, %94 : tensor<1x3xf32>, tensor<3xf32>) outs(%2395 : tensor<1x3xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "net", prov.fqn = "net.nn_fc2"} {
    ^bb183(%2397: f32, %2398: f32, %2399: f32):
      %2400 = arith.addf %2397, %2398 : f32
      linalg.yield %2400 : f32
    } -> tensor<1x3xf32>
    func.return %2396, %2381, %2383 : tensor<1x3xf32>, tensor<3x128xf32>, tensor<3x128xf32>
  }
}
