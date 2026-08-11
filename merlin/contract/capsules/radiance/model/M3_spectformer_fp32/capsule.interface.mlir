builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x196x256xf32>, %1: tensor<256x3x16x16xf32>, %2: tensor<256xf32>, %3: tensor<256xf32>, %4: tensor<256xf32>, %5: tensor<14x8x256x2xf32>, %6: tensor<256xf32>, %7: tensor<256xf32>, %8: tensor<1024x256xf32>, %9: tensor<1024xf32>, %10: tensor<256x1024xf32>, %11: tensor<256xf32>, %12: tensor<256xf32>, %13: tensor<256xf32>, %14: tensor<14x8x256x2xf32>, %15: tensor<256xf32>, %16: tensor<256xf32>, %17: tensor<1024x256xf32>, %18: tensor<1024xf32>, %19: tensor<256x1024xf32>, %20: tensor<256xf32>, %21: tensor<256xf32>, %22: tensor<256xf32>, %23: tensor<14x8x256x2xf32>, %24: tensor<256xf32>, %25: tensor<256xf32>, %26: tensor<1024x256xf32>, %27: tensor<1024xf32>, %28: tensor<256x1024xf32>, %29: tensor<256xf32>, %30: tensor<256xf32>, %31: tensor<256xf32>, %32: tensor<14x8x256x2xf32>, %33: tensor<256xf32>, %34: tensor<256xf32>, %35: tensor<1024x256xf32>, %36: tensor<1024xf32>, %37: tensor<256x1024xf32>, %38: tensor<256xf32>, %39: tensor<256xf32>, %40: tensor<256xf32>, %41: tensor<256xf32>, %42: tensor<256xf32>, %43: tensor<1024x256xf32>, %44: tensor<1024xf32>, %45: tensor<256x1024xf32>, %46: tensor<256xf32>, %47: tensor<768x256xf32>, %48: tensor<768xf32>, %49: tensor<256x256xf32>, %50: tensor<256xf32>, %51: tensor<256xf32>, %52: tensor<256xf32>, %53: tensor<256xf32>, %54: tensor<256xf32>, %55: tensor<1024x256xf32>, %56: tensor<1024xf32>, %57: tensor<256x1024xf32>, %58: tensor<256xf32>, %59: tensor<768x256xf32>, %60: tensor<768xf32>, %61: tensor<256x256xf32>, %62: tensor<256xf32>, %63: tensor<256xf32>, %64: tensor<256xf32>, %65: tensor<256xf32>, %66: tensor<256xf32>, %67: tensor<1024x256xf32>, %68: tensor<1024xf32>, %69: tensor<256x1024xf32>, %70: tensor<256xf32>, %71: tensor<768x256xf32>, %72: tensor<768xf32>, %73: tensor<256x256xf32>, %74: tensor<256xf32>, %75: tensor<256xf32>, %76: tensor<256xf32>, %77: tensor<256xf32>, %78: tensor<256xf32>, %79: tensor<1024x256xf32>, %80: tensor<1024xf32>, %81: tensor<256x1024xf32>, %82: tensor<256xf32>, %83: tensor<768x256xf32>, %84: tensor<768xf32>, %85: tensor<256x256xf32>, %86: tensor<256xf32>, %87: tensor<256xf32>, %88: tensor<256xf32>, %89: tensor<256xf32>, %90: tensor<256xf32>, %91: tensor<1024x256xf32>, %92: tensor<1024xf32>, %93: tensor<256x1024xf32>, %94: tensor<256xf32>, %95: tensor<768x256xf32>, %96: tensor<768xf32>, %97: tensor<256x256xf32>, %98: tensor<256xf32>, %99: tensor<256xf32>, %100: tensor<256xf32>, %101: tensor<256xf32>, %102: tensor<256xf32>, %103: tensor<1024x256xf32>, %104: tensor<1024xf32>, %105: tensor<256x1024xf32>, %106: tensor<256xf32>, %107: tensor<768x256xf32>, %108: tensor<768xf32>, %109: tensor<256x256xf32>, %110: tensor<256xf32>, %111: tensor<256xf32>, %112: tensor<256xf32>, %113: tensor<256xf32>, %114: tensor<256xf32>, %115: tensor<1024x256xf32>, %116: tensor<1024xf32>, %117: tensor<256x1024xf32>, %118: tensor<256xf32>, %119: tensor<768x256xf32>, %120: tensor<768xf32>, %121: tensor<256x256xf32>, %122: tensor<256xf32>, %123: tensor<256xf32>, %124: tensor<256xf32>, %125: tensor<256xf32>, %126: tensor<256xf32>, %127: tensor<1024x256xf32>, %128: tensor<1024xf32>, %129: tensor<256x1024xf32>, %130: tensor<256xf32>, %131: tensor<768x256xf32>, %132: tensor<768xf32>, %133: tensor<256x256xf32>, %134: tensor<256xf32>, %135: tensor<256xf32>, %136: tensor<256xf32>, %137: tensor<1000x256xf32>, %138: tensor<1000xf32>, %139: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %140 = tensor.empty() : tensor<3x16x16x1x14x14xf32>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 16) + d1), ((d5 * 16) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%139 : tensor<1x3x224x224xf32>) outs(%140 : tensor<3x16x16x1x14x14xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} {
    ^bb0(%142: f32, %143: f32):
      linalg.yield %142 : f32
    } -> tensor<3x16x16x1x14x14xf32>
    %144 = tensor.collapse_shape %141 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<3x16x16x1x14x14xf32> into tensor<150528xf32>
    %145 = tensor.expand_shape %144 [[0 : i64, 1 : i64]] output_shape [768, 196] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<150528xf32> into tensor<768x196xf32>
    %146 = tensor.collapse_shape %1 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<256x3x16x16xf32> into tensor<196608xf32>
    %147 = tensor.expand_shape %146 [[0 : i64, 1 : i64]] output_shape [256, 768] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<196608xf32> into tensor<256x768xf32>
    %148 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} 0.000000e+00 : f32
    %149 = tensor.splat %148 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<256x196xf32>
    %150 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} ins(%147, %145 : tensor<256x768xf32>, tensor<768x196xf32>) outs(%149 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %151 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<256x196xf32> into tensor<50176xf32>
    %152 = tensor.expand_shape %151 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %153 = tensor.collapse_shape %152 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %154 = tensor.expand_shape %153 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %155 = tensor.empty() : tensor<1x256x14x14xf32>
    %156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154, %2 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%155 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed.proj"} {
    ^bb1(%157: f32, %158: f32, %159: f32):
      %160 = arith.addf %157, %158 : f32
      linalg.yield %160 : f32
    } -> tensor<1x256x14x14xf32>
    %161 = tensor.collapse_shape %156 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed"} : tensor<1x256x14x14xf32> into tensor<50176xf32>
    %162 = tensor.expand_shape %161 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 256, 196] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "patch_embed", prov.fqn = "patch_embed"} : tensor<50176xf32> into tensor<1x256x196xf32>
    %163 = tensor.empty() : tensor<1x196x256xf32>
    %164 = linalg.transpose ins(%162:tensor<1x256x196xf32>) outs(%163:tensor<1x196x256xf32>) permutation = [0, 2, 1]
    %165 = tensor.empty() : tensor<1x196x256xf32>
    %166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%164, %0 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%165 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%167: f32, %168: f32, %169: f32):
      %170 = arith.addf %167, %168 : f32
      linalg.yield %170 : f32
    } -> tensor<1x196x256xf32>
    %171 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} 0.000000e+00 : f32
    %172 = tensor.splat %171 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32>
    %173 = linalg.reduce ins(%166:tensor<1x196x256xf32>) outs(%172:tensor<1x196xf32>) dimensions = [2]
    (%174: f32, %175: f32) {
      %176 = arith.addf %174, %175 : f32
      linalg.yield %176 : f32
    }
    %177 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} 2.560000e+02 : f32
    %178 = tensor.splat %177 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32>
    %179 = tensor.empty() : tensor<1x196xf32>
    %180 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%173, %178 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%179 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb3(%181: f32, %182: f32, %183: f32):
      %184 = arith.divf %181, %182 : f32
      linalg.yield %184 : f32
    } -> tensor<1x196xf32>
    %185 = tensor.collapse_shape %180 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %186 = tensor.expand_shape %185 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %187 = tensor.empty() : tensor<1x196x256xf32>
    %188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %186 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%187 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb4(%189: f32, %190: f32, %191: f32):
      %192 = arith.subf %189, %190 : f32
      linalg.yield %192 : f32
    } -> tensor<1x196x256xf32>
    %193 = tensor.empty() : tensor<1x196x256xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %188 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%193 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb5(%195: f32, %196: f32, %197: f32):
      %198 = arith.mulf %195, %196 : f32
      linalg.yield %198 : f32
    } -> tensor<1x196x256xf32>
    %199 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} 0.000000e+00 : f32
    %200 = tensor.splat %199 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32>
    %201 = linalg.reduce ins(%194:tensor<1x196x256xf32>) outs(%200:tensor<1x196xf32>) dimensions = [2]
    (%202: f32, %203: f32) {
      %204 = arith.addf %202, %203 : f32
      linalg.yield %204 : f32
    }
    %205 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} 2.560000e+02 : f32
    %206 = tensor.splat %205 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32>
    %207 = tensor.empty() : tensor<1x196xf32>
    %208 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%201, %206 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%207 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb6(%209: f32, %210: f32, %211: f32):
      %212 = arith.divf %209, %210 : f32
      linalg.yield %212 : f32
    } -> tensor<1x196xf32>
    %213 = tensor.collapse_shape %208 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %214 = tensor.expand_shape %213 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %215 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} 1.000000e-06 : f32
    %216 = tensor.splat %215 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} : tensor<1x196x1xf32>
    %217 = tensor.empty() : tensor<1x196x1xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %216 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%217 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb7(%219: f32, %220: f32, %221: f32):
      %222 = arith.addf %219, %220 : f32
      linalg.yield %222 : f32
    } -> tensor<1x196x1xf32>
    %223 = tensor.empty() : tensor<1x196x1xf32>
    %224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%218 : tensor<1x196x1xf32>) outs(%223 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb8(%225: f32, %226: f32):
      %227 = math.rsqrt %225 : f32
      linalg.yield %227 : f32
    } -> tensor<1x196x1xf32>
    %228 = tensor.empty() : tensor<1x196x256xf32>
    %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %224 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%228 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb9(%230: f32, %231: f32, %232: f32):
      %233 = arith.mulf %230, %231 : f32
      linalg.yield %233 : f32
    } -> tensor<1x196x256xf32>
    %234 = tensor.empty() : tensor<1x196x256xf32>
    %235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%229, %3 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%234 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb10(%236: f32, %237: f32, %238: f32):
      %239 = arith.mulf %236, %237 : f32
      linalg.yield %239 : f32
    } -> tensor<1x196x256xf32>
    %240 = tensor.empty() : tensor<1x196x256xf32>
    %241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%235, %4 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%240 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm1"} {
    ^bb11(%242: f32, %243: f32, %244: f32):
      %245 = arith.addf %242, %243 : f32
      linalg.yield %245 : f32
    } -> tensor<1x196x256xf32>
    %246 = tensor.collapse_shape %241 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %247 = tensor.expand_shape %246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 14, 256] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<50176xf32> into tensor<1x14x14x256xf32>
    %248 = tensor.empty() : tensor<1x14x256x14xf32>
    %249 = linalg.transpose ins(%247:tensor<1x14x14x256xf32>) outs(%248:tensor<1x14x256x14xf32>) permutation = [0, 1, 3, 2]
    %250 = tensor.collapse_shape %249 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x256x14xf32> into tensor<50176xf32>
    %251 = tensor.expand_shape %250 [[0 : i64, 1 : i64]] output_shape [3584, 14] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<50176xf32> into tensor<3584x14xf32>
    %252 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD"> : tensor<14x8xf32>
    %253 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000804CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCCB5C21A300000080D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643DCB5CA12300000080379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD300BF2A300000080379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DCB5C212400000080D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BDFEB349A4000000804CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C300B722400000080CB5C21A3CB5CA123300BF2A3CB5C2124FEB349A4300B722432318DA4000000804CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBCCB5CA12400000080D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D6488B5A400000080379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDFEB3C92400000080379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D7EA2352500000080D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD300BF224000000804CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CD7D6D3A5"> : tensor<14x8xf32>
    %254 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %255 = tensor.splat %254 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x8xf32>
    %256 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%251, %252 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%255 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %257 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %258 = tensor.splat %257 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x8xf32>
    %259 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%251, %253 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%258 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %260 = tensor.collapse_shape %256 [[0 : i64, 1 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %261 = tensor.expand_shape %260 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %262 = tensor.empty() : tensor<1x14x8x256xf32>
    %263 = linalg.transpose ins(%261:tensor<1x14x256x8xf32>) outs(%262:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %264 = tensor.collapse_shape %259 [[0 : i64, 1 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %265 = tensor.expand_shape %264 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %266 = tensor.empty() : tensor<1x14x8x256xf32>
    %267 = linalg.transpose ins(%265:tensor<1x14x256x8xf32>) outs(%266:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %268 = tensor.empty() : tensor<1x8x256x14xf32>
    %269 = linalg.transpose ins(%263:tensor<1x14x8x256xf32>) outs(%268:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %270 = tensor.collapse_shape %269 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %271 = tensor.expand_shape %270 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %272 = tensor.empty() : tensor<1x8x256x14xf32>
    %273 = linalg.transpose ins(%267:tensor<1x14x8x256xf32>) outs(%272:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %274 = tensor.collapse_shape %273 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %275 = tensor.expand_shape %274 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %276 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F"> : tensor<14x14xf32>
    %277 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE32310DA50226DE3E1C26483FE094793FE094793F1C26483F0226DE3E000000801C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F32318D251C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F00000080E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BFCAC9D3A5E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F00000080E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F32310D26E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F000000801C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF7E7D30A61C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F000000800226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3ECAC953260226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3E0000008032310DA532318D25CAC9D3A532310D267E7D30A6CAC95326171677A632318D2658D79EA67E7DB026E988B0A7CAC9D32602522328000000800226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE32318D260226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE000000801C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F58D79EA61C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF00000080E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF7E7DB026E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF00000080E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F2EEE1E27E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BF000000801C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BFCAC9D3261C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BF000000800226DE3E1C26483FE094793FE094793F1C26483F0226DE3EFC5BB9A70226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE"> : tensor<14x14xf32>
    %278 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %279 = tensor.splat %278 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %280 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%271, %276 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%279 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %281 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %282 = tensor.splat %281 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %283 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%275, %277 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%282 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %284 = tensor.empty() : tensor<2048x14xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280, %283 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%284 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb12(%286: f32, %287: f32, %288: f32):
      %289 = arith.subf %286, %287 : f32
      linalg.yield %289 : f32
    } -> tensor<2048x14xf32>
    %290 = tensor.collapse_shape %285 [[0 : i64, 1 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %291 = tensor.expand_shape %290 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %292 = tensor.empty() : tensor<1x14x8x256xf32>
    %293 = linalg.transpose ins(%291:tensor<1x8x256x14xf32>) outs(%292:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %294 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %295 = tensor.splat %294 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %296 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%275, %276 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%295 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %297 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %298 = tensor.splat %297 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %299 = linalg.matmul {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%271, %277 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%298 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %300 = tensor.empty() : tensor<2048x14xf32>
    %301 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%296, %299 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%300 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb13(%302: f32, %303: f32, %304: f32):
      %305 = arith.addf %302, %303 : f32
      linalg.yield %305 : f32
    } -> tensor<2048x14xf32>
    %306 = tensor.collapse_shape %301 [[0 : i64, 1 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %307 = tensor.expand_shape %306 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %308 = tensor.empty() : tensor<1x14x8x256xf32>
    %309 = linalg.transpose ins(%307:tensor<1x8x256x14xf32>) outs(%308:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %310 = arith.constant {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %311 = tensor.splat %310 {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x2xf32>
    %312 = tensor.collapse_shape %293 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %313 = tensor.expand_shape %312 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %314 = "tensor.insert_slice"(%313, %311) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %315 = tensor.collapse_shape %309 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %316 = tensor.expand_shape %315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %317 = "tensor.insert_slice"(%316, %314) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_0", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %318 = "tensor.extract_slice"(%317) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %319 = tensor.collapse_shape %318 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %320 = tensor.expand_shape %319 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %321 = "tensor.extract_slice"(%317) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %322 = tensor.collapse_shape %321 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %323 = tensor.expand_shape %322 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %324 = "tensor.extract_slice"(%5) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %325 = tensor.collapse_shape %324 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %326 = tensor.expand_shape %325 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %327 = "tensor.extract_slice"(%5) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %328 = tensor.collapse_shape %327 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %329 = tensor.expand_shape %328 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %330 = tensor.empty() : tensor<1x14x8x256xf32>
    %331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %326 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%330 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb14(%332: f32, %333: f32, %334: f32):
      %335 = arith.mulf %332, %333 : f32
      linalg.yield %335 : f32
    } -> tensor<1x14x8x256xf32>
    %336 = tensor.empty() : tensor<1x14x8x256xf32>
    %337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %329 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%336 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb15(%338: f32, %339: f32, %340: f32):
      %341 = arith.mulf %338, %339 : f32
      linalg.yield %341 : f32
    } -> tensor<1x14x8x256xf32>
    %342 = tensor.empty() : tensor<1x14x8x256xf32>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320, %329 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%342 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb16(%344: f32, %345: f32, %346: f32):
      %347 = arith.mulf %344, %345 : f32
      linalg.yield %347 : f32
    } -> tensor<1x14x8x256xf32>
    %348 = tensor.empty() : tensor<1x14x8x256xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %326 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%348 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb17(%350: f32, %351: f32, %352: f32):
      %353 = arith.mulf %350, %351 : f32
      linalg.yield %353 : f32
    } -> tensor<1x14x8x256xf32>
    %354 = tensor.empty() : tensor<1x14x8x256xf32>
    %355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%331, %337 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%354 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb18(%356: f32, %357: f32, %358: f32):
      %359 = arith.subf %356, %357 : f32
      linalg.yield %359 : f32
    } -> tensor<1x14x8x256xf32>
    %360 = tensor.empty() : tensor<1x14x8x256xf32>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %349 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%360 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb19(%362: f32, %363: f32, %364: f32):
      %365 = arith.addf %362, %363 : f32
      linalg.yield %365 : f32
    } -> tensor<1x14x8x256xf32>
    %366 = arith.constant {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %367 = tensor.splat %366 {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x2xf32>
    %368 = tensor.collapse_shape %355 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %369 = tensor.expand_shape %368 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %370 = "tensor.insert_slice"(%369, %367) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %371 = tensor.collapse_shape %361 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %372 = tensor.expand_shape %371 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %373 = "tensor.insert_slice"(%372, %370) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_0", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %374 = "tensor.extract_slice"(%373) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %375 = tensor.collapse_shape %374 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %376 = tensor.expand_shape %375 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %377 = "tensor.extract_slice"(%373) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %378 = tensor.collapse_shape %377 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %379 = tensor.expand_shape %378 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %380 = tensor.empty() : tensor<1x8x256x14xf32>
    %381 = linalg.transpose ins(%376:tensor<1x14x8x256xf32>) outs(%380:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %382 = tensor.collapse_shape %381 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %383 = tensor.expand_shape %382 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %384 = tensor.empty() : tensor<1x8x256x14xf32>
    %385 = linalg.transpose ins(%379:tensor<1x14x8x256xf32>) outs(%384:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %386 = tensor.collapse_shape %385 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %387 = tensor.expand_shape %386 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %388 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D"> : tensor<14x14xf32>
    %389 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CCB5C21234CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBC00000000D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BDCB5CA1A3D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD00000000379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D300BF223379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD00000000379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDCB5C21A4379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBD00000000D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643DFEB34924D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD000000004CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC300B72A44CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC00000000CB5C2123CB5CA1A3300BF223CB5C21A4FEB34924300B72A432318D24CB5CA1A46488B524FEB3C9A40AC1C925300BF2A4DEA63AA6000000004CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3CCB5CA1A44CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C00000000D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD6488B524D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D00000000379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DFEB3C9A4379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3D00000000379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD7EA235A5379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D00000000D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D300BF2A4D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D000000004CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCD7D6D3254CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3C"> : tensor<14x14xf32>
    %390 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %391 = tensor.splat %390 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %392 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%383, %388 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%391 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %393 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %394 = tensor.splat %393 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %395 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%387, %389 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%394 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %396 = tensor.empty() : tensor<2048x14xf32>
    %397 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%392, %395 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%396 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb20(%398: f32, %399: f32, %400: f32):
      %401 = arith.subf %398, %399 : f32
      linalg.yield %401 : f32
    } -> tensor<2048x14xf32>
    %402 = tensor.collapse_shape %397 [[0 : i64, 1 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %404 = tensor.empty() : tensor<1x14x8x256xf32>
    %405 = linalg.transpose ins(%403:tensor<1x8x256x14xf32>) outs(%404:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %406 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %407 = tensor.splat %406 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %408 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%387, %388 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%407 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %409 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %410 = tensor.splat %409 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32>
    %411 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%383, %389 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%410 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %412 = tensor.empty() : tensor<2048x14xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%408, %411 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%412 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb21(%414: f32, %415: f32, %416: f32):
      %417 = arith.addf %414, %415 : f32
      linalg.yield %417 : f32
    } -> tensor<2048x14xf32>
    %418 = tensor.collapse_shape %413 [[0 : i64, 1 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %419 = tensor.expand_shape %418 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %420 = tensor.empty() : tensor<1x14x8x256xf32>
    %421 = linalg.transpose ins(%419:tensor<1x8x256x14xf32>) outs(%420:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %422 = tensor.empty() : tensor<1x14x256x8xf32>
    %423 = linalg.transpose ins(%405:tensor<1x14x8x256xf32>) outs(%422:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %424 = tensor.collapse_shape %423 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %425 = tensor.expand_shape %424 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %426 = tensor.empty() : tensor<1x14x256x8xf32>
    %427 = linalg.transpose ins(%421:tensor<1x14x8x256xf32>) outs(%426:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %428 = tensor.collapse_shape %427 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %429 = tensor.expand_shape %428 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %430 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F00000040E5A5E63F079D9F3F87DCE33E87DCE3BE079D9FBFE5A5E6BF000000C0E5A5E6BF079D9FBF87DCE3BE87DCE33E079D9F3FE5A5E63F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F0000004087DCE33EE5A5E6BF079D9FBF079D9F3FE5A5E63F87DCE3BE000000C087DCE3BEE5A5E63F079D9F3F079D9FBFE5A5E6BF87DCE33E0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE00000040079D9FBF87DCE3BEE5A5E63FE5A5E6BF87DCE33E079D9F3F000000C0079D9F3F87DCE33EE5A5E6BFE5A5E63F87DCE3BE079D9FBF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF"> : tensor<8x14xf32>
    %431 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002265E3F1C26C83FE094F93FE094F93F1C26C83F02265E3F32318D2502265EBF1C26C8BFE094F9BFE094F9BF1C26C8BF02265EBF000000001C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF32310DA61C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF00000000E094F93F02265E3F1C26C8BF1C26C8BF02265E3FE094F93FCAC95326E094F9BF02265EBF1C26C83F1C26C83F02265EBFE094F9BF00000000E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF32318DA6E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF000000001C26C83FE094F9BF02265E3F02265E3FE094F9BF1C26C83F7E7DB0261C26C8BFE094F93F02265EBF02265EBFE094F93F1C26C8BF0000000002265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBFCAC9D3A602265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBF0000000032310D2532318DA5CAC9D32532310DA67E7D3026CAC953A61716772632318DA658D79E267E7DB0A6E988B027CAC9D3A6025223A8"> : tensor<8x14xf32>
    %432 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %433 = tensor.splat %432 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x14xf32>
    %434 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%425, %430 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%433 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %435 = arith.constant {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} 0.000000e+00 : f32
    %436 = tensor.splat %435 {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x14xf32>
    %437 = linalg.matmul {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} ins(%429, %431 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%436 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %438 = tensor.empty() : tensor<3584x14xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%434, %437 : tensor<3584x14xf32>, tensor<3584x14xf32>) outs(%438 : tensor<3584x14xf32>) attrs =  {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} {
    ^bb22(%440: f32, %441: f32, %442: f32):
      %443 = arith.subf %440, %441 : f32
      linalg.yield %443 : f32
    } -> tensor<3584x14xf32>
    %444 = tensor.collapse_shape %439 [[0 : i64, 1 : i64]] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<3584x14xf32> into tensor<50176xf32>
    %445 = tensor.expand_shape %444 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 14] {prov.region_id = "fft_1", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<50176xf32> into tensor<1x14x256x14xf32>
    %446 = tensor.empty() : tensor<1x14x14x256xf32>
    %447 = linalg.transpose ins(%445:tensor<1x14x256x14xf32>) outs(%446:tensor<1x14x14x256xf32>) permutation = [0, 1, 3, 2]
    %448 = tensor.collapse_shape %447 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<1x14x14x256xf32> into tensor<50176xf32>
    %449 = tensor.expand_shape %448 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.filter"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %450 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} 0.000000e+00 : f32
    %451 = tensor.splat %450 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32>
    %452 = linalg.reduce ins(%449:tensor<1x196x256xf32>) outs(%451:tensor<1x196xf32>) dimensions = [2]
    (%453: f32, %454: f32) {
      %455 = arith.addf %453, %454 : f32
      linalg.yield %455 : f32
    }
    %456 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} 2.560000e+02 : f32
    %457 = tensor.splat %456 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32>
    %458 = tensor.empty() : tensor<1x196xf32>
    %459 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%452, %457 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%458 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb23(%460: f32, %461: f32, %462: f32):
      %463 = arith.divf %460, %461 : f32
      linalg.yield %463 : f32
    } -> tensor<1x196xf32>
    %464 = tensor.collapse_shape %459 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %465 = tensor.expand_shape %464 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %466 = tensor.empty() : tensor<1x196x256xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%449, %465 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%466 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb24(%468: f32, %469: f32, %470: f32):
      %471 = arith.subf %468, %469 : f32
      linalg.yield %471 : f32
    } -> tensor<1x196x256xf32>
    %472 = tensor.empty() : tensor<1x196x256xf32>
    %473 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%467, %467 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%472 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb25(%474: f32, %475: f32, %476: f32):
      %477 = arith.mulf %474, %475 : f32
      linalg.yield %477 : f32
    } -> tensor<1x196x256xf32>
    %478 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} 0.000000e+00 : f32
    %479 = tensor.splat %478 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32>
    %480 = linalg.reduce ins(%473:tensor<1x196x256xf32>) outs(%479:tensor<1x196xf32>) dimensions = [2]
    (%481: f32, %482: f32) {
      %483 = arith.addf %481, %482 : f32
      linalg.yield %483 : f32
    }
    %484 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} 2.560000e+02 : f32
    %485 = tensor.splat %484 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32>
    %486 = tensor.empty() : tensor<1x196xf32>
    %487 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%480, %485 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%486 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb26(%488: f32, %489: f32, %490: f32):
      %491 = arith.divf %488, %489 : f32
      linalg.yield %491 : f32
    } -> tensor<1x196xf32>
    %492 = tensor.collapse_shape %487 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %493 = tensor.expand_shape %492 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %494 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} 1.000000e-06 : f32
    %495 = tensor.splat %494 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} : tensor<1x196x1xf32>
    %496 = tensor.empty() : tensor<1x196x1xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%493, %495 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%496 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb27(%498: f32, %499: f32, %500: f32):
      %501 = arith.addf %498, %499 : f32
      linalg.yield %501 : f32
    } -> tensor<1x196x1xf32>
    %502 = tensor.empty() : tensor<1x196x1xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%497 : tensor<1x196x1xf32>) outs(%502 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb28(%504: f32, %505: f32):
      %506 = math.rsqrt %504 : f32
      linalg.yield %506 : f32
    } -> tensor<1x196x1xf32>
    %507 = tensor.empty() : tensor<1x196x256xf32>
    %508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%467, %503 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%507 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb29(%509: f32, %510: f32, %511: f32):
      %512 = arith.mulf %509, %510 : f32
      linalg.yield %512 : f32
    } -> tensor<1x196x256xf32>
    %513 = tensor.empty() : tensor<1x196x256xf32>
    %514 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%508, %6 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%513 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb30(%515: f32, %516: f32, %517: f32):
      %518 = arith.mulf %515, %516 : f32
      linalg.yield %518 : f32
    } -> tensor<1x196x256xf32>
    %519 = tensor.empty() : tensor<1x196x256xf32>
    %520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%514, %7 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%519 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.norm2"} {
    ^bb31(%521: f32, %522: f32, %523: f32):
      %524 = arith.addf %521, %522 : f32
      linalg.yield %524 : f32
    } -> tensor<1x196x256xf32>
    %525 = tensor.collapse_shape %520 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %526 = tensor.expand_shape %525 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %527 = tensor.empty() : tensor<256x1024xf32>
    %528 = linalg.transpose ins(%8:tensor<1024x256xf32>) outs(%527:tensor<256x1024xf32>) permutation = [1, 0]
    %529 = tensor.empty() : tensor<196x1024xf32>
    %530 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %531 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%530 : f32) outs(%529 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %532 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1", prov.transposed_b = "true"} ins(%526, %528 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%531 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %533 = tensor.empty() : tensor<196x1024xf32>
    %534 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%532, %9 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%533 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1"} {
    ^bb32(%535: f32, %536: f32, %537: f32):
      %538 = arith.addf %535, %536 : f32
      linalg.yield %538 : f32
    } -> tensor<196x1024xf32>
    %539 = tensor.collapse_shape %534 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %540 = tensor.expand_shape %539 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %541 = tensor.empty() : tensor<1x196x1024xf32>
    %542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%540 : tensor<1x196x1024xf32>) outs(%541 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.act"} {
    ^bb33(%543: f32, %544: f32):
      %545 = arith.constant 5.000000e-01 : f32
      %546 = arith.constant 1.000000e+00 : f32
      %547 = arith.constant 0.707106769 : f32
      %548 = arith.mulf %543, %547 : f32
      %549 = math.erf %548 : f32
      %550 = arith.addf %546, %549 : f32
      %551 = arith.mulf %545, %543 : f32
      %552 = arith.mulf %551, %550 : f32
      linalg.yield %552 : f32
    } -> tensor<1x196x1024xf32>
    %553 = tensor.collapse_shape %542 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %554 = tensor.expand_shape %553 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %555 = tensor.empty() : tensor<1024x256xf32>
    %556 = linalg.transpose ins(%10:tensor<256x1024xf32>) outs(%555:tensor<1024x256xf32>) permutation = [1, 0]
    %557 = tensor.empty() : tensor<196x256xf32>
    %558 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %559 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%558 : f32) outs(%557 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %560 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2", prov.transposed_b = "true"} ins(%554, %556 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%559 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %561 = tensor.empty() : tensor<196x256xf32>
    %562 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%560, %11 : tensor<196x256xf32>, tensor<256xf32>) outs(%561 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2"} {
    ^bb34(%563: f32, %564: f32, %565: f32):
      %566 = arith.addf %563, %564 : f32
      linalg.yield %566 : f32
    } -> tensor<196x256xf32>
    %567 = tensor.collapse_shape %562 [[0 : i64, 1 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %568 = tensor.expand_shape %567 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %569 = tensor.empty() : tensor<1x196x256xf32>
    %570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %568 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%569 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb35(%571: f32, %572: f32, %573: f32):
      %574 = arith.addf %571, %572 : f32
      linalg.yield %574 : f32
    } -> tensor<1x196x256xf32>
    %575 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} 0.000000e+00 : f32
    %576 = tensor.splat %575 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32>
    %577 = linalg.reduce ins(%570:tensor<1x196x256xf32>) outs(%576:tensor<1x196xf32>) dimensions = [2]
    (%578: f32, %579: f32) {
      %580 = arith.addf %578, %579 : f32
      linalg.yield %580 : f32
    }
    %581 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} 2.560000e+02 : f32
    %582 = tensor.splat %581 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32>
    %583 = tensor.empty() : tensor<1x196xf32>
    %584 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%577, %582 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%583 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb36(%585: f32, %586: f32, %587: f32):
      %588 = arith.divf %585, %586 : f32
      linalg.yield %588 : f32
    } -> tensor<1x196xf32>
    %589 = tensor.collapse_shape %584 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %590 = tensor.expand_shape %589 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %591 = tensor.empty() : tensor<1x196x256xf32>
    %592 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%570, %590 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%591 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb37(%593: f32, %594: f32, %595: f32):
      %596 = arith.subf %593, %594 : f32
      linalg.yield %596 : f32
    } -> tensor<1x196x256xf32>
    %597 = tensor.empty() : tensor<1x196x256xf32>
    %598 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592, %592 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%597 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb38(%599: f32, %600: f32, %601: f32):
      %602 = arith.mulf %599, %600 : f32
      linalg.yield %602 : f32
    } -> tensor<1x196x256xf32>
    %603 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} 0.000000e+00 : f32
    %604 = tensor.splat %603 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32>
    %605 = linalg.reduce ins(%598:tensor<1x196x256xf32>) outs(%604:tensor<1x196xf32>) dimensions = [2]
    (%606: f32, %607: f32) {
      %608 = arith.addf %606, %607 : f32
      linalg.yield %608 : f32
    }
    %609 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} 2.560000e+02 : f32
    %610 = tensor.splat %609 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32>
    %611 = tensor.empty() : tensor<1x196xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%605, %610 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%611 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb39(%613: f32, %614: f32, %615: f32):
      %616 = arith.divf %613, %614 : f32
      linalg.yield %616 : f32
    } -> tensor<1x196xf32>
    %617 = tensor.collapse_shape %612 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %618 = tensor.expand_shape %617 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %619 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} 1.000000e-06 : f32
    %620 = tensor.splat %619 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} : tensor<1x196x1xf32>
    %621 = tensor.empty() : tensor<1x196x1xf32>
    %622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%618, %620 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%621 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb40(%623: f32, %624: f32, %625: f32):
      %626 = arith.addf %623, %624 : f32
      linalg.yield %626 : f32
    } -> tensor<1x196x1xf32>
    %627 = tensor.empty() : tensor<1x196x1xf32>
    %628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%622 : tensor<1x196x1xf32>) outs(%627 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb41(%629: f32, %630: f32):
      %631 = math.rsqrt %629 : f32
      linalg.yield %631 : f32
    } -> tensor<1x196x1xf32>
    %632 = tensor.empty() : tensor<1x196x256xf32>
    %633 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592, %628 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%632 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb42(%634: f32, %635: f32, %636: f32):
      %637 = arith.mulf %634, %635 : f32
      linalg.yield %637 : f32
    } -> tensor<1x196x256xf32>
    %638 = tensor.empty() : tensor<1x196x256xf32>
    %639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%633, %12 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%638 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb43(%640: f32, %641: f32, %642: f32):
      %643 = arith.mulf %640, %641 : f32
      linalg.yield %643 : f32
    } -> tensor<1x196x256xf32>
    %644 = tensor.empty() : tensor<1x196x256xf32>
    %645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639, %13 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%644 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm1"} {
    ^bb44(%646: f32, %647: f32, %648: f32):
      %649 = arith.addf %646, %647 : f32
      linalg.yield %649 : f32
    } -> tensor<1x196x256xf32>
    %650 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %651 = tensor.expand_shape %650 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 14, 256] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<50176xf32> into tensor<1x14x14x256xf32>
    %652 = tensor.empty() : tensor<1x14x256x14xf32>
    %653 = linalg.transpose ins(%651:tensor<1x14x14x256xf32>) outs(%652:tensor<1x14x256x14xf32>) permutation = [0, 1, 3, 2]
    %654 = tensor.collapse_shape %653 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x256x14xf32> into tensor<50176xf32>
    %655 = tensor.expand_shape %654 [[0 : i64, 1 : i64]] output_shape [3584, 14] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<50176xf32> into tensor<3584x14xf32>
    %656 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD"> : tensor<14x8xf32>
    %657 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000804CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCCB5C21A300000080D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643DCB5CA12300000080379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD300BF2A300000080379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DCB5C212400000080D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BDFEB349A4000000804CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C300B722400000080CB5C21A3CB5CA123300BF2A3CB5C2124FEB349A4300B722432318DA4000000804CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBCCB5CA12400000080D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D6488B5A400000080379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDFEB3C92400000080379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D7EA2352500000080D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD300BF224000000804CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CD7D6D3A5"> : tensor<14x8xf32>
    %658 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %659 = tensor.splat %658 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x8xf32>
    %660 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%655, %656 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%659 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %661 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %662 = tensor.splat %661 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x8xf32>
    %663 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%655, %657 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%662 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %664 = tensor.collapse_shape %660 [[0 : i64, 1 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %665 = tensor.expand_shape %664 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %666 = tensor.empty() : tensor<1x14x8x256xf32>
    %667 = linalg.transpose ins(%665:tensor<1x14x256x8xf32>) outs(%666:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %668 = tensor.collapse_shape %663 [[0 : i64, 1 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %669 = tensor.expand_shape %668 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %670 = tensor.empty() : tensor<1x14x8x256xf32>
    %671 = linalg.transpose ins(%669:tensor<1x14x256x8xf32>) outs(%670:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %672 = tensor.empty() : tensor<1x8x256x14xf32>
    %673 = linalg.transpose ins(%667:tensor<1x14x8x256xf32>) outs(%672:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %674 = tensor.collapse_shape %673 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %675 = tensor.expand_shape %674 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %676 = tensor.empty() : tensor<1x8x256x14xf32>
    %677 = linalg.transpose ins(%671:tensor<1x14x8x256xf32>) outs(%676:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %678 = tensor.collapse_shape %677 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %679 = tensor.expand_shape %678 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %680 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F"> : tensor<14x14xf32>
    %681 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE32310DA50226DE3E1C26483FE094793FE094793F1C26483F0226DE3E000000801C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F32318D251C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F00000080E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BFCAC9D3A5E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F00000080E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F32310D26E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F000000801C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF7E7D30A61C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F000000800226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3ECAC953260226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3E0000008032310DA532318D25CAC9D3A532310D267E7D30A6CAC95326171677A632318D2658D79EA67E7DB026E988B0A7CAC9D32602522328000000800226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE32318D260226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE000000801C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F58D79EA61C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF00000080E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF7E7DB026E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF00000080E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F2EEE1E27E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BF000000801C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BFCAC9D3261C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BF000000800226DE3E1C26483FE094793FE094793F1C26483F0226DE3EFC5BB9A70226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE"> : tensor<14x14xf32>
    %682 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %683 = tensor.splat %682 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %684 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%675, %680 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%683 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %685 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %686 = tensor.splat %685 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %687 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%679, %681 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%686 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %688 = tensor.empty() : tensor<2048x14xf32>
    %689 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%684, %687 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%688 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb45(%690: f32, %691: f32, %692: f32):
      %693 = arith.subf %690, %691 : f32
      linalg.yield %693 : f32
    } -> tensor<2048x14xf32>
    %694 = tensor.collapse_shape %689 [[0 : i64, 1 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %695 = tensor.expand_shape %694 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %696 = tensor.empty() : tensor<1x14x8x256xf32>
    %697 = linalg.transpose ins(%695:tensor<1x8x256x14xf32>) outs(%696:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %698 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %699 = tensor.splat %698 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %700 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%679, %680 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%699 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %701 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %702 = tensor.splat %701 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %703 = linalg.matmul {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%675, %681 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%702 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %704 = tensor.empty() : tensor<2048x14xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%700, %703 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%704 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb46(%706: f32, %707: f32, %708: f32):
      %709 = arith.addf %706, %707 : f32
      linalg.yield %709 : f32
    } -> tensor<2048x14xf32>
    %710 = tensor.collapse_shape %705 [[0 : i64, 1 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %711 = tensor.expand_shape %710 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %712 = tensor.empty() : tensor<1x14x8x256xf32>
    %713 = linalg.transpose ins(%711:tensor<1x8x256x14xf32>) outs(%712:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %714 = arith.constant {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %715 = tensor.splat %714 {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x2xf32>
    %716 = tensor.collapse_shape %697 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %717 = tensor.expand_shape %716 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %718 = "tensor.insert_slice"(%717, %715) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %719 = tensor.collapse_shape %713 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %720 = tensor.expand_shape %719 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %721 = "tensor.insert_slice"(%720, %718) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_2", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %722 = "tensor.extract_slice"(%721) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %723 = tensor.collapse_shape %722 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %724 = tensor.expand_shape %723 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %725 = "tensor.extract_slice"(%721) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %726 = tensor.collapse_shape %725 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %727 = tensor.expand_shape %726 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %728 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %729 = tensor.collapse_shape %728 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %730 = tensor.expand_shape %729 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %731 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %732 = tensor.collapse_shape %731 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %733 = tensor.expand_shape %732 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %734 = tensor.empty() : tensor<1x14x8x256xf32>
    %735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%724, %730 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%734 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb47(%736: f32, %737: f32, %738: f32):
      %739 = arith.mulf %736, %737 : f32
      linalg.yield %739 : f32
    } -> tensor<1x14x8x256xf32>
    %740 = tensor.empty() : tensor<1x14x8x256xf32>
    %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %733 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%740 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb48(%742: f32, %743: f32, %744: f32):
      %745 = arith.mulf %742, %743 : f32
      linalg.yield %745 : f32
    } -> tensor<1x14x8x256xf32>
    %746 = tensor.empty() : tensor<1x14x8x256xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%724, %733 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%746 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb49(%748: f32, %749: f32, %750: f32):
      %751 = arith.mulf %748, %749 : f32
      linalg.yield %751 : f32
    } -> tensor<1x14x8x256xf32>
    %752 = tensor.empty() : tensor<1x14x8x256xf32>
    %753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %730 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%752 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb50(%754: f32, %755: f32, %756: f32):
      %757 = arith.mulf %754, %755 : f32
      linalg.yield %757 : f32
    } -> tensor<1x14x8x256xf32>
    %758 = tensor.empty() : tensor<1x14x8x256xf32>
    %759 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%735, %741 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%758 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb51(%760: f32, %761: f32, %762: f32):
      %763 = arith.subf %760, %761 : f32
      linalg.yield %763 : f32
    } -> tensor<1x14x8x256xf32>
    %764 = tensor.empty() : tensor<1x14x8x256xf32>
    %765 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %753 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%764 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb52(%766: f32, %767: f32, %768: f32):
      %769 = arith.addf %766, %767 : f32
      linalg.yield %769 : f32
    } -> tensor<1x14x8x256xf32>
    %770 = arith.constant {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %771 = tensor.splat %770 {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x2xf32>
    %772 = tensor.collapse_shape %759 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %773 = tensor.expand_shape %772 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %774 = "tensor.insert_slice"(%773, %771) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %775 = tensor.collapse_shape %765 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %776 = tensor.expand_shape %775 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %777 = "tensor.insert_slice"(%776, %774) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_1", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %778 = "tensor.extract_slice"(%777) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %779 = tensor.collapse_shape %778 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %780 = tensor.expand_shape %779 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %781 = "tensor.extract_slice"(%777) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %782 = tensor.collapse_shape %781 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %783 = tensor.expand_shape %782 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %784 = tensor.empty() : tensor<1x8x256x14xf32>
    %785 = linalg.transpose ins(%780:tensor<1x14x8x256xf32>) outs(%784:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %786 = tensor.collapse_shape %785 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %787 = tensor.expand_shape %786 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %788 = tensor.empty() : tensor<1x8x256x14xf32>
    %789 = linalg.transpose ins(%783:tensor<1x14x8x256xf32>) outs(%788:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %790 = tensor.collapse_shape %789 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %791 = tensor.expand_shape %790 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %792 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D"> : tensor<14x14xf32>
    %793 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CCB5C21234CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBC00000000D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BDCB5CA1A3D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD00000000379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D300BF223379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD00000000379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDCB5C21A4379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBD00000000D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643DFEB34924D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD000000004CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC300B72A44CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC00000000CB5C2123CB5CA1A3300BF223CB5C21A4FEB34924300B72A432318D24CB5CA1A46488B524FEB3C9A40AC1C925300BF2A4DEA63AA6000000004CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3CCB5CA1A44CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C00000000D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD6488B524D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D00000000379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DFEB3C9A4379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3D00000000379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD7EA235A5379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D00000000D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D300BF2A4D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D000000004CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCD7D6D3254CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3C"> : tensor<14x14xf32>
    %794 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %795 = tensor.splat %794 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %796 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%787, %792 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%795 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %797 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %798 = tensor.splat %797 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %799 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%791, %793 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%798 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %800 = tensor.empty() : tensor<2048x14xf32>
    %801 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%796, %799 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%800 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb53(%802: f32, %803: f32, %804: f32):
      %805 = arith.subf %802, %803 : f32
      linalg.yield %805 : f32
    } -> tensor<2048x14xf32>
    %806 = tensor.collapse_shape %801 [[0 : i64, 1 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %807 = tensor.expand_shape %806 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %808 = tensor.empty() : tensor<1x14x8x256xf32>
    %809 = linalg.transpose ins(%807:tensor<1x8x256x14xf32>) outs(%808:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %810 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %811 = tensor.splat %810 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %812 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%791, %792 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%811 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %813 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %814 = tensor.splat %813 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32>
    %815 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%787, %793 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%814 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %816 = tensor.empty() : tensor<2048x14xf32>
    %817 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%812, %815 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%816 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb54(%818: f32, %819: f32, %820: f32):
      %821 = arith.addf %818, %819 : f32
      linalg.yield %821 : f32
    } -> tensor<2048x14xf32>
    %822 = tensor.collapse_shape %817 [[0 : i64, 1 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %823 = tensor.expand_shape %822 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %824 = tensor.empty() : tensor<1x14x8x256xf32>
    %825 = linalg.transpose ins(%823:tensor<1x8x256x14xf32>) outs(%824:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %826 = tensor.empty() : tensor<1x14x256x8xf32>
    %827 = linalg.transpose ins(%809:tensor<1x14x8x256xf32>) outs(%826:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %828 = tensor.collapse_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %829 = tensor.expand_shape %828 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %830 = tensor.empty() : tensor<1x14x256x8xf32>
    %831 = linalg.transpose ins(%825:tensor<1x14x8x256xf32>) outs(%830:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %832 = tensor.collapse_shape %831 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %833 = tensor.expand_shape %832 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %834 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F00000040E5A5E63F079D9F3F87DCE33E87DCE3BE079D9FBFE5A5E6BF000000C0E5A5E6BF079D9FBF87DCE3BE87DCE33E079D9F3FE5A5E63F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F0000004087DCE33EE5A5E6BF079D9FBF079D9F3FE5A5E63F87DCE3BE000000C087DCE3BEE5A5E63F079D9F3F079D9FBFE5A5E6BF87DCE33E0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE00000040079D9FBF87DCE3BEE5A5E63FE5A5E6BF87DCE33E079D9F3F000000C0079D9F3F87DCE33EE5A5E6BFE5A5E63F87DCE3BE079D9FBF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF"> : tensor<8x14xf32>
    %835 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002265E3F1C26C83FE094F93FE094F93F1C26C83F02265E3F32318D2502265EBF1C26C8BFE094F9BFE094F9BF1C26C8BF02265EBF000000001C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF32310DA61C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF00000000E094F93F02265E3F1C26C8BF1C26C8BF02265E3FE094F93FCAC95326E094F9BF02265EBF1C26C83F1C26C83F02265EBFE094F9BF00000000E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF32318DA6E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF000000001C26C83FE094F9BF02265E3F02265E3FE094F9BF1C26C83F7E7DB0261C26C8BFE094F93F02265EBF02265EBFE094F93F1C26C8BF0000000002265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBFCAC9D3A602265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBF0000000032310D2532318DA5CAC9D32532310DA67E7D3026CAC953A61716772632318DA658D79E267E7DB0A6E988B027CAC9D3A6025223A8"> : tensor<8x14xf32>
    %836 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %837 = tensor.splat %836 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x14xf32>
    %838 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%829, %834 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%837 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %839 = arith.constant {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} 0.000000e+00 : f32
    %840 = tensor.splat %839 {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x14xf32>
    %841 = linalg.matmul {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} ins(%833, %835 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%840 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %842 = tensor.empty() : tensor<3584x14xf32>
    %843 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%838, %841 : tensor<3584x14xf32>, tensor<3584x14xf32>) outs(%842 : tensor<3584x14xf32>) attrs =  {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} {
    ^bb55(%844: f32, %845: f32, %846: f32):
      %847 = arith.subf %844, %845 : f32
      linalg.yield %847 : f32
    } -> tensor<3584x14xf32>
    %848 = tensor.collapse_shape %843 [[0 : i64, 1 : i64]] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<3584x14xf32> into tensor<50176xf32>
    %849 = tensor.expand_shape %848 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 14] {prov.region_id = "fft_3", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<50176xf32> into tensor<1x14x256x14xf32>
    %850 = tensor.empty() : tensor<1x14x14x256xf32>
    %851 = linalg.transpose ins(%849:tensor<1x14x256x14xf32>) outs(%850:tensor<1x14x14x256xf32>) permutation = [0, 1, 3, 2]
    %852 = tensor.collapse_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<1x14x14x256xf32> into tensor<50176xf32>
    %853 = tensor.expand_shape %852 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.filter"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %854 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} 0.000000e+00 : f32
    %855 = tensor.splat %854 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32>
    %856 = linalg.reduce ins(%853:tensor<1x196x256xf32>) outs(%855:tensor<1x196xf32>) dimensions = [2]
    (%857: f32, %858: f32) {
      %859 = arith.addf %857, %858 : f32
      linalg.yield %859 : f32
    }
    %860 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} 2.560000e+02 : f32
    %861 = tensor.splat %860 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32>
    %862 = tensor.empty() : tensor<1x196xf32>
    %863 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%856, %861 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%862 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb56(%864: f32, %865: f32, %866: f32):
      %867 = arith.divf %864, %865 : f32
      linalg.yield %867 : f32
    } -> tensor<1x196xf32>
    %868 = tensor.collapse_shape %863 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %869 = tensor.expand_shape %868 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %870 = tensor.empty() : tensor<1x196x256xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%853, %869 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%870 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb57(%872: f32, %873: f32, %874: f32):
      %875 = arith.subf %872, %873 : f32
      linalg.yield %875 : f32
    } -> tensor<1x196x256xf32>
    %876 = tensor.empty() : tensor<1x196x256xf32>
    %877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%871, %871 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%876 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb58(%878: f32, %879: f32, %880: f32):
      %881 = arith.mulf %878, %879 : f32
      linalg.yield %881 : f32
    } -> tensor<1x196x256xf32>
    %882 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} 0.000000e+00 : f32
    %883 = tensor.splat %882 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32>
    %884 = linalg.reduce ins(%877:tensor<1x196x256xf32>) outs(%883:tensor<1x196xf32>) dimensions = [2]
    (%885: f32, %886: f32) {
      %887 = arith.addf %885, %886 : f32
      linalg.yield %887 : f32
    }
    %888 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} 2.560000e+02 : f32
    %889 = tensor.splat %888 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32>
    %890 = tensor.empty() : tensor<1x196xf32>
    %891 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%884, %889 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%890 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb59(%892: f32, %893: f32, %894: f32):
      %895 = arith.divf %892, %893 : f32
      linalg.yield %895 : f32
    } -> tensor<1x196xf32>
    %896 = tensor.collapse_shape %891 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %897 = tensor.expand_shape %896 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %898 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} 1.000000e-06 : f32
    %899 = tensor.splat %898 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} : tensor<1x196x1xf32>
    %900 = tensor.empty() : tensor<1x196x1xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%897, %899 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%900 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb60(%902: f32, %903: f32, %904: f32):
      %905 = arith.addf %902, %903 : f32
      linalg.yield %905 : f32
    } -> tensor<1x196x1xf32>
    %906 = tensor.empty() : tensor<1x196x1xf32>
    %907 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901 : tensor<1x196x1xf32>) outs(%906 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb61(%908: f32, %909: f32):
      %910 = math.rsqrt %908 : f32
      linalg.yield %910 : f32
    } -> tensor<1x196x1xf32>
    %911 = tensor.empty() : tensor<1x196x256xf32>
    %912 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%871, %907 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%911 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb62(%913: f32, %914: f32, %915: f32):
      %916 = arith.mulf %913, %914 : f32
      linalg.yield %916 : f32
    } -> tensor<1x196x256xf32>
    %917 = tensor.empty() : tensor<1x196x256xf32>
    %918 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%912, %15 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%917 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb63(%919: f32, %920: f32, %921: f32):
      %922 = arith.mulf %919, %920 : f32
      linalg.yield %922 : f32
    } -> tensor<1x196x256xf32>
    %923 = tensor.empty() : tensor<1x196x256xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%918, %16 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%923 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.norm2"} {
    ^bb64(%925: f32, %926: f32, %927: f32):
      %928 = arith.addf %925, %926 : f32
      linalg.yield %928 : f32
    } -> tensor<1x196x256xf32>
    %929 = tensor.collapse_shape %924 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %930 = tensor.expand_shape %929 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %931 = tensor.empty() : tensor<256x1024xf32>
    %932 = linalg.transpose ins(%17:tensor<1024x256xf32>) outs(%931:tensor<256x1024xf32>) permutation = [1, 0]
    %933 = tensor.empty() : tensor<196x1024xf32>
    %934 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %935 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%934 : f32) outs(%933 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %936 = linalg.matmul {prov.region_id = "matmul_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1", prov.transposed_b = "true"} ins(%930, %932 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%935 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %937 = tensor.empty() : tensor<196x1024xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%936, %18 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%937 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1"} {
    ^bb65(%939: f32, %940: f32, %941: f32):
      %942 = arith.addf %939, %940 : f32
      linalg.yield %942 : f32
    } -> tensor<196x1024xf32>
    %943 = tensor.collapse_shape %938 [[0 : i64, 1 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %944 = tensor.expand_shape %943 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %945 = tensor.empty() : tensor<1x196x1024xf32>
    %946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944 : tensor<1x196x1024xf32>) outs(%945 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.act"} {
    ^bb66(%947: f32, %948: f32):
      %949 = arith.constant 5.000000e-01 : f32
      %950 = arith.constant 1.000000e+00 : f32
      %951 = arith.constant 0.707106769 : f32
      %952 = arith.mulf %947, %951 : f32
      %953 = math.erf %952 : f32
      %954 = arith.addf %950, %953 : f32
      %955 = arith.mulf %949, %947 : f32
      %956 = arith.mulf %955, %954 : f32
      linalg.yield %956 : f32
    } -> tensor<1x196x1024xf32>
    %957 = tensor.collapse_shape %946 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %958 = tensor.expand_shape %957 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %959 = tensor.empty() : tensor<1024x256xf32>
    %960 = linalg.transpose ins(%19:tensor<256x1024xf32>) outs(%959:tensor<1024x256xf32>) permutation = [1, 0]
    %961 = tensor.empty() : tensor<196x256xf32>
    %962 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %963 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%962 : f32) outs(%961 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %964 = linalg.matmul {prov.region_id = "matmul_3", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2", prov.transposed_b = "true"} ins(%958, %960 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%963 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %965 = tensor.empty() : tensor<196x256xf32>
    %966 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%964, %20 : tensor<196x256xf32>, tensor<256xf32>) outs(%965 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2"} {
    ^bb67(%967: f32, %968: f32, %969: f32):
      %970 = arith.addf %967, %968 : f32
      linalg.yield %970 : f32
    } -> tensor<196x256xf32>
    %971 = tensor.collapse_shape %966 [[0 : i64, 1 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %972 = tensor.expand_shape %971 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %973 = tensor.empty() : tensor<1x196x256xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%570, %972 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%973 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb68(%975: f32, %976: f32, %977: f32):
      %978 = arith.addf %975, %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x196x256xf32>
    %979 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} 0.000000e+00 : f32
    %980 = tensor.splat %979 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32>
    %981 = linalg.reduce ins(%974:tensor<1x196x256xf32>) outs(%980:tensor<1x196xf32>) dimensions = [2]
    (%982: f32, %983: f32) {
      %984 = arith.addf %982, %983 : f32
      linalg.yield %984 : f32
    }
    %985 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} 2.560000e+02 : f32
    %986 = tensor.splat %985 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32>
    %987 = tensor.empty() : tensor<1x196xf32>
    %988 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%981, %986 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%987 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb69(%989: f32, %990: f32, %991: f32):
      %992 = arith.divf %989, %990 : f32
      linalg.yield %992 : f32
    } -> tensor<1x196xf32>
    %993 = tensor.collapse_shape %988 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %994 = tensor.expand_shape %993 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %995 = tensor.empty() : tensor<1x196x256xf32>
    %996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %994 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%995 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb70(%997: f32, %998: f32, %999: f32):
      %1000 = arith.subf %997, %998 : f32
      linalg.yield %1000 : f32
    } -> tensor<1x196x256xf32>
    %1001 = tensor.empty() : tensor<1x196x256xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%996, %996 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1001 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb71(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.mulf %1003, %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x196x256xf32>
    %1007 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} 0.000000e+00 : f32
    %1008 = tensor.splat %1007 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32>
    %1009 = linalg.reduce ins(%1002:tensor<1x196x256xf32>) outs(%1008:tensor<1x196xf32>) dimensions = [2]
    (%1010: f32, %1011: f32) {
      %1012 = arith.addf %1010, %1011 : f32
      linalg.yield %1012 : f32
    }
    %1013 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} 2.560000e+02 : f32
    %1014 = tensor.splat %1013 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32>
    %1015 = tensor.empty() : tensor<1x196xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1009, %1014 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1015 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb72(%1017: f32, %1018: f32, %1019: f32):
      %1020 = arith.divf %1017, %1018 : f32
      linalg.yield %1020 : f32
    } -> tensor<1x196xf32>
    %1021 = tensor.collapse_shape %1016 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %1022 = tensor.expand_shape %1021 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1023 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} 1.000000e-06 : f32
    %1024 = tensor.splat %1023 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} : tensor<1x196x1xf32>
    %1025 = tensor.empty() : tensor<1x196x1xf32>
    %1026 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1022, %1024 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%1025 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb73(%1027: f32, %1028: f32, %1029: f32):
      %1030 = arith.addf %1027, %1028 : f32
      linalg.yield %1030 : f32
    } -> tensor<1x196x1xf32>
    %1031 = tensor.empty() : tensor<1x196x1xf32>
    %1032 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1026 : tensor<1x196x1xf32>) outs(%1031 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb74(%1033: f32, %1034: f32):
      %1035 = math.rsqrt %1033 : f32
      linalg.yield %1035 : f32
    } -> tensor<1x196x1xf32>
    %1036 = tensor.empty() : tensor<1x196x256xf32>
    %1037 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%996, %1032 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1036 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb75(%1038: f32, %1039: f32, %1040: f32):
      %1041 = arith.mulf %1038, %1039 : f32
      linalg.yield %1041 : f32
    } -> tensor<1x196x256xf32>
    %1042 = tensor.empty() : tensor<1x196x256xf32>
    %1043 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %21 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1042 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb76(%1044: f32, %1045: f32, %1046: f32):
      %1047 = arith.mulf %1044, %1045 : f32
      linalg.yield %1047 : f32
    } -> tensor<1x196x256xf32>
    %1048 = tensor.empty() : tensor<1x196x256xf32>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1043, %22 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1048 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm1"} {
    ^bb77(%1050: f32, %1051: f32, %1052: f32):
      %1053 = arith.addf %1050, %1051 : f32
      linalg.yield %1053 : f32
    } -> tensor<1x196x256xf32>
    %1054 = tensor.collapse_shape %1049 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1055 = tensor.expand_shape %1054 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 14, 256] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<50176xf32> into tensor<1x14x14x256xf32>
    %1056 = tensor.empty() : tensor<1x14x256x14xf32>
    %1057 = linalg.transpose ins(%1055:tensor<1x14x14x256xf32>) outs(%1056:tensor<1x14x256x14xf32>) permutation = [0, 1, 3, 2]
    %1058 = tensor.collapse_shape %1057 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x256x14xf32> into tensor<50176xf32>
    %1059 = tensor.expand_shape %1058 [[0 : i64, 1 : i64]] output_shape [3584, 14] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<50176xf32> into tensor<3584x14xf32>
    %1060 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD"> : tensor<14x8xf32>
    %1061 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000804CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCCB5C21A300000080D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643DCB5CA12300000080379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD300BF2A300000080379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DCB5C212400000080D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BDFEB349A4000000804CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C300B722400000080CB5C21A3CB5CA123300BF2A3CB5C2124FEB349A4300B722432318DA4000000804CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBCCB5CA12400000080D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D6488B5A400000080379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDFEB3C92400000080379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D7EA2352500000080D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD300BF224000000804CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CD7D6D3A5"> : tensor<14x8xf32>
    %1062 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1063 = tensor.splat %1062 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x8xf32>
    %1064 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1059, %1060 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%1063 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %1065 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1066 = tensor.splat %1065 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x8xf32>
    %1067 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1059, %1061 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%1066 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %1068 = tensor.collapse_shape %1064 [[0 : i64, 1 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %1069 = tensor.expand_shape %1068 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %1070 = tensor.empty() : tensor<1x14x8x256xf32>
    %1071 = linalg.transpose ins(%1069:tensor<1x14x256x8xf32>) outs(%1070:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %1072 = tensor.collapse_shape %1067 [[0 : i64, 1 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %1073 = tensor.expand_shape %1072 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %1074 = tensor.empty() : tensor<1x14x8x256xf32>
    %1075 = linalg.transpose ins(%1073:tensor<1x14x256x8xf32>) outs(%1074:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %1076 = tensor.empty() : tensor<1x8x256x14xf32>
    %1077 = linalg.transpose ins(%1071:tensor<1x14x8x256xf32>) outs(%1076:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1078 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1079 = tensor.expand_shape %1078 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1080 = tensor.empty() : tensor<1x8x256x14xf32>
    %1081 = linalg.transpose ins(%1075:tensor<1x14x8x256xf32>) outs(%1080:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1082 = tensor.collapse_shape %1081 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1083 = tensor.expand_shape %1082 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1084 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F"> : tensor<14x14xf32>
    %1085 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE32310DA50226DE3E1C26483FE094793FE094793F1C26483F0226DE3E000000801C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F32318D251C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F00000080E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BFCAC9D3A5E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F00000080E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F32310D26E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F000000801C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF7E7D30A61C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F000000800226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3ECAC953260226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3E0000008032310DA532318D25CAC9D3A532310D267E7D30A6CAC95326171677A632318D2658D79EA67E7DB026E988B0A7CAC9D32602522328000000800226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE32318D260226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE000000801C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F58D79EA61C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF00000080E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF7E7DB026E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF00000080E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F2EEE1E27E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BF000000801C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BFCAC9D3261C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BF000000800226DE3E1C26483FE094793FE094793F1C26483F0226DE3EFC5BB9A70226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE"> : tensor<14x14xf32>
    %1086 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1087 = tensor.splat %1086 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1088 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1079, %1084 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1087 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1089 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1090 = tensor.splat %1089 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1091 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1083, %1085 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1090 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1092 = tensor.empty() : tensor<2048x14xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1088, %1091 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1092 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb78(%1094: f32, %1095: f32, %1096: f32):
      %1097 = arith.subf %1094, %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<2048x14xf32>
    %1098 = tensor.collapse_shape %1093 [[0 : i64, 1 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1099 = tensor.expand_shape %1098 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1100 = tensor.empty() : tensor<1x14x8x256xf32>
    %1101 = linalg.transpose ins(%1099:tensor<1x8x256x14xf32>) outs(%1100:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1102 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1103 = tensor.splat %1102 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1104 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1083, %1084 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1103 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1105 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1106 = tensor.splat %1105 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1107 = linalg.matmul {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1079, %1085 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1106 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1108 = tensor.empty() : tensor<2048x14xf32>
    %1109 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1104, %1107 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1108 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb79(%1110: f32, %1111: f32, %1112: f32):
      %1113 = arith.addf %1110, %1111 : f32
      linalg.yield %1113 : f32
    } -> tensor<2048x14xf32>
    %1114 = tensor.collapse_shape %1109 [[0 : i64, 1 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1115 = tensor.expand_shape %1114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1116 = tensor.empty() : tensor<1x14x8x256xf32>
    %1117 = linalg.transpose ins(%1115:tensor<1x8x256x14xf32>) outs(%1116:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1118 = arith.constant {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1119 = tensor.splat %1118 {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x2xf32>
    %1120 = tensor.collapse_shape %1101 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1121 = tensor.expand_shape %1120 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1122 = "tensor.insert_slice"(%1121, %1119) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1123 = tensor.collapse_shape %1117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1124 = tensor.expand_shape %1123 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1125 = "tensor.insert_slice"(%1124, %1122) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_4", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1126 = "tensor.extract_slice"(%1125) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1127 = tensor.collapse_shape %1126 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1129 = "tensor.extract_slice"(%1125) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1130 = tensor.collapse_shape %1129 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1131 = tensor.expand_shape %1130 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1132 = "tensor.extract_slice"(%23) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %1133 = tensor.collapse_shape %1132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %1134 = tensor.expand_shape %1133 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %1135 = "tensor.extract_slice"(%23) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %1136 = tensor.collapse_shape %1135 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %1137 = tensor.expand_shape %1136 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %1138 = tensor.empty() : tensor<1x14x8x256xf32>
    %1139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1128, %1134 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1138 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb80(%1140: f32, %1141: f32, %1142: f32):
      %1143 = arith.mulf %1140, %1141 : f32
      linalg.yield %1143 : f32
    } -> tensor<1x14x8x256xf32>
    %1144 = tensor.empty() : tensor<1x14x8x256xf32>
    %1145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1131, %1137 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1144 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb81(%1146: f32, %1147: f32, %1148: f32):
      %1149 = arith.mulf %1146, %1147 : f32
      linalg.yield %1149 : f32
    } -> tensor<1x14x8x256xf32>
    %1150 = tensor.empty() : tensor<1x14x8x256xf32>
    %1151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1128, %1137 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1150 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb82(%1152: f32, %1153: f32, %1154: f32):
      %1155 = arith.mulf %1152, %1153 : f32
      linalg.yield %1155 : f32
    } -> tensor<1x14x8x256xf32>
    %1156 = tensor.empty() : tensor<1x14x8x256xf32>
    %1157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1131, %1134 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1156 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb83(%1158: f32, %1159: f32, %1160: f32):
      %1161 = arith.mulf %1158, %1159 : f32
      linalg.yield %1161 : f32
    } -> tensor<1x14x8x256xf32>
    %1162 = tensor.empty() : tensor<1x14x8x256xf32>
    %1163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1139, %1145 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%1162 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb84(%1164: f32, %1165: f32, %1166: f32):
      %1167 = arith.subf %1164, %1165 : f32
      linalg.yield %1167 : f32
    } -> tensor<1x14x8x256xf32>
    %1168 = tensor.empty() : tensor<1x14x8x256xf32>
    %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1151, %1157 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%1168 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb85(%1170: f32, %1171: f32, %1172: f32):
      %1173 = arith.addf %1170, %1171 : f32
      linalg.yield %1173 : f32
    } -> tensor<1x14x8x256xf32>
    %1174 = arith.constant {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1175 = tensor.splat %1174 {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x2xf32>
    %1176 = tensor.collapse_shape %1163 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1177 = tensor.expand_shape %1176 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1178 = "tensor.insert_slice"(%1177, %1175) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1179 = tensor.collapse_shape %1169 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1180 = tensor.expand_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1181 = "tensor.insert_slice"(%1180, %1178) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_2", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1182 = "tensor.extract_slice"(%1181) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1183 = tensor.collapse_shape %1182 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1184 = tensor.expand_shape %1183 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1185 = "tensor.extract_slice"(%1181) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1186 = tensor.collapse_shape %1185 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1187 = tensor.expand_shape %1186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1188 = tensor.empty() : tensor<1x8x256x14xf32>
    %1189 = linalg.transpose ins(%1184:tensor<1x14x8x256xf32>) outs(%1188:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1190 = tensor.collapse_shape %1189 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1191 = tensor.expand_shape %1190 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1192 = tensor.empty() : tensor<1x8x256x14xf32>
    %1193 = linalg.transpose ins(%1187:tensor<1x14x8x256xf32>) outs(%1192:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1194 = tensor.collapse_shape %1193 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1195 = tensor.expand_shape %1194 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1196 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D"> : tensor<14x14xf32>
    %1197 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CCB5C21234CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBC00000000D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BDCB5CA1A3D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD00000000379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D300BF223379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD00000000379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDCB5C21A4379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBD00000000D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643DFEB34924D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD000000004CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC300B72A44CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC00000000CB5C2123CB5CA1A3300BF223CB5C21A4FEB34924300B72A432318D24CB5CA1A46488B524FEB3C9A40AC1C925300BF2A4DEA63AA6000000004CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3CCB5CA1A44CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C00000000D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD6488B524D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D00000000379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DFEB3C9A4379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3D00000000379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD7EA235A5379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D00000000D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D300BF2A4D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D000000004CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCD7D6D3254CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3C"> : tensor<14x14xf32>
    %1198 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1199 = tensor.splat %1198 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1200 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1191, %1196 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1199 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1201 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1202 = tensor.splat %1201 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1203 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1195, %1197 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1202 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1204 = tensor.empty() : tensor<2048x14xf32>
    %1205 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1200, %1203 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1204 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb86(%1206: f32, %1207: f32, %1208: f32):
      %1209 = arith.subf %1206, %1207 : f32
      linalg.yield %1209 : f32
    } -> tensor<2048x14xf32>
    %1210 = tensor.collapse_shape %1205 [[0 : i64, 1 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1211 = tensor.expand_shape %1210 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1212 = tensor.empty() : tensor<1x14x8x256xf32>
    %1213 = linalg.transpose ins(%1211:tensor<1x8x256x14xf32>) outs(%1212:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1214 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1215 = tensor.splat %1214 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1216 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1195, %1196 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1215 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1217 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1218 = tensor.splat %1217 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32>
    %1219 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1191, %1197 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1218 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1220 = tensor.empty() : tensor<2048x14xf32>
    %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1216, %1219 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1220 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb87(%1222: f32, %1223: f32, %1224: f32):
      %1225 = arith.addf %1222, %1223 : f32
      linalg.yield %1225 : f32
    } -> tensor<2048x14xf32>
    %1226 = tensor.collapse_shape %1221 [[0 : i64, 1 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1227 = tensor.expand_shape %1226 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1228 = tensor.empty() : tensor<1x14x8x256xf32>
    %1229 = linalg.transpose ins(%1227:tensor<1x8x256x14xf32>) outs(%1228:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1230 = tensor.empty() : tensor<1x14x256x8xf32>
    %1231 = linalg.transpose ins(%1213:tensor<1x14x8x256xf32>) outs(%1230:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %1232 = tensor.collapse_shape %1231 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %1233 = tensor.expand_shape %1232 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %1234 = tensor.empty() : tensor<1x14x256x8xf32>
    %1235 = linalg.transpose ins(%1229:tensor<1x14x8x256xf32>) outs(%1234:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %1236 = tensor.collapse_shape %1235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %1237 = tensor.expand_shape %1236 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %1238 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F00000040E5A5E63F079D9F3F87DCE33E87DCE3BE079D9FBFE5A5E6BF000000C0E5A5E6BF079D9FBF87DCE3BE87DCE33E079D9F3FE5A5E63F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F0000004087DCE33EE5A5E6BF079D9FBF079D9F3FE5A5E63F87DCE3BE000000C087DCE3BEE5A5E63F079D9F3F079D9FBFE5A5E6BF87DCE33E0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE00000040079D9FBF87DCE3BEE5A5E63FE5A5E6BF87DCE33E079D9F3F000000C0079D9F3F87DCE33EE5A5E6BFE5A5E63F87DCE3BE079D9FBF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF"> : tensor<8x14xf32>
    %1239 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002265E3F1C26C83FE094F93FE094F93F1C26C83F02265E3F32318D2502265EBF1C26C8BFE094F9BFE094F9BF1C26C8BF02265EBF000000001C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF32310DA61C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF00000000E094F93F02265E3F1C26C8BF1C26C8BF02265E3FE094F93FCAC95326E094F9BF02265EBF1C26C83F1C26C83F02265EBFE094F9BF00000000E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF32318DA6E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF000000001C26C83FE094F9BF02265E3F02265E3FE094F9BF1C26C83F7E7DB0261C26C8BFE094F93F02265EBF02265EBFE094F93F1C26C8BF0000000002265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBFCAC9D3A602265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBF0000000032310D2532318DA5CAC9D32532310DA67E7D3026CAC953A61716772632318DA658D79E267E7DB0A6E988B027CAC9D3A6025223A8"> : tensor<8x14xf32>
    %1240 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1241 = tensor.splat %1240 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x14xf32>
    %1242 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1233, %1238 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%1241 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %1243 = arith.constant {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} 0.000000e+00 : f32
    %1244 = tensor.splat %1243 {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x14xf32>
    %1245 = linalg.matmul {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} ins(%1237, %1239 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%1244 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %1246 = tensor.empty() : tensor<3584x14xf32>
    %1247 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1242, %1245 : tensor<3584x14xf32>, tensor<3584x14xf32>) outs(%1246 : tensor<3584x14xf32>) attrs =  {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} {
    ^bb88(%1248: f32, %1249: f32, %1250: f32):
      %1251 = arith.subf %1248, %1249 : f32
      linalg.yield %1251 : f32
    } -> tensor<3584x14xf32>
    %1252 = tensor.collapse_shape %1247 [[0 : i64, 1 : i64]] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<3584x14xf32> into tensor<50176xf32>
    %1253 = tensor.expand_shape %1252 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 14] {prov.region_id = "fft_5", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<50176xf32> into tensor<1x14x256x14xf32>
    %1254 = tensor.empty() : tensor<1x14x14x256xf32>
    %1255 = linalg.transpose ins(%1253:tensor<1x14x256x14xf32>) outs(%1254:tensor<1x14x14x256xf32>) permutation = [0, 1, 3, 2]
    %1256 = tensor.collapse_shape %1255 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<1x14x14x256xf32> into tensor<50176xf32>
    %1257 = tensor.expand_shape %1256 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.filter"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1258 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} 0.000000e+00 : f32
    %1259 = tensor.splat %1258 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32>
    %1260 = linalg.reduce ins(%1257:tensor<1x196x256xf32>) outs(%1259:tensor<1x196xf32>) dimensions = [2]
    (%1261: f32, %1262: f32) {
      %1263 = arith.addf %1261, %1262 : f32
      linalg.yield %1263 : f32
    }
    %1264 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} 2.560000e+02 : f32
    %1265 = tensor.splat %1264 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32>
    %1266 = tensor.empty() : tensor<1x196xf32>
    %1267 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1260, %1265 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1266 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb89(%1268: f32, %1269: f32, %1270: f32):
      %1271 = arith.divf %1268, %1269 : f32
      linalg.yield %1271 : f32
    } -> tensor<1x196xf32>
    %1272 = tensor.collapse_shape %1267 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %1273 = tensor.expand_shape %1272 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1274 = tensor.empty() : tensor<1x196x256xf32>
    %1275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1257, %1273 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1274 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb90(%1276: f32, %1277: f32, %1278: f32):
      %1279 = arith.subf %1276, %1277 : f32
      linalg.yield %1279 : f32
    } -> tensor<1x196x256xf32>
    %1280 = tensor.empty() : tensor<1x196x256xf32>
    %1281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %1275 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1280 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb91(%1282: f32, %1283: f32, %1284: f32):
      %1285 = arith.mulf %1282, %1283 : f32
      linalg.yield %1285 : f32
    } -> tensor<1x196x256xf32>
    %1286 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} 0.000000e+00 : f32
    %1287 = tensor.splat %1286 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32>
    %1288 = linalg.reduce ins(%1281:tensor<1x196x256xf32>) outs(%1287:tensor<1x196xf32>) dimensions = [2]
    (%1289: f32, %1290: f32) {
      %1291 = arith.addf %1289, %1290 : f32
      linalg.yield %1291 : f32
    }
    %1292 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} 2.560000e+02 : f32
    %1293 = tensor.splat %1292 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32>
    %1294 = tensor.empty() : tensor<1x196xf32>
    %1295 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1288, %1293 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1294 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb92(%1296: f32, %1297: f32, %1298: f32):
      %1299 = arith.divf %1296, %1297 : f32
      linalg.yield %1299 : f32
    } -> tensor<1x196xf32>
    %1300 = tensor.collapse_shape %1295 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %1301 = tensor.expand_shape %1300 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1302 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} 1.000000e-06 : f32
    %1303 = tensor.splat %1302 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} : tensor<1x196x1xf32>
    %1304 = tensor.empty() : tensor<1x196x1xf32>
    %1305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1301, %1303 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%1304 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb93(%1306: f32, %1307: f32, %1308: f32):
      %1309 = arith.addf %1306, %1307 : f32
      linalg.yield %1309 : f32
    } -> tensor<1x196x1xf32>
    %1310 = tensor.empty() : tensor<1x196x1xf32>
    %1311 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1305 : tensor<1x196x1xf32>) outs(%1310 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb94(%1312: f32, %1313: f32):
      %1314 = math.rsqrt %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x196x1xf32>
    %1315 = tensor.empty() : tensor<1x196x256xf32>
    %1316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %1311 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1315 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb95(%1317: f32, %1318: f32, %1319: f32):
      %1320 = arith.mulf %1317, %1318 : f32
      linalg.yield %1320 : f32
    } -> tensor<1x196x256xf32>
    %1321 = tensor.empty() : tensor<1x196x256xf32>
    %1322 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316, %24 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1321 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb96(%1323: f32, %1324: f32, %1325: f32):
      %1326 = arith.mulf %1323, %1324 : f32
      linalg.yield %1326 : f32
    } -> tensor<1x196x256xf32>
    %1327 = tensor.empty() : tensor<1x196x256xf32>
    %1328 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1322, %25 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1327 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.norm2"} {
    ^bb97(%1329: f32, %1330: f32, %1331: f32):
      %1332 = arith.addf %1329, %1330 : f32
      linalg.yield %1332 : f32
    } -> tensor<1x196x256xf32>
    %1333 = tensor.collapse_shape %1328 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1334 = tensor.expand_shape %1333 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %1335 = tensor.empty() : tensor<256x1024xf32>
    %1336 = linalg.transpose ins(%26:tensor<1024x256xf32>) outs(%1335:tensor<256x1024xf32>) permutation = [1, 0]
    %1337 = tensor.empty() : tensor<196x1024xf32>
    %1338 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1339 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1338 : f32) outs(%1337 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %1340 = linalg.matmul {prov.region_id = "matmul_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1", prov.transposed_b = "true"} ins(%1334, %1336 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%1339 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %1341 = tensor.empty() : tensor<196x1024xf32>
    %1342 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1340, %27 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%1341 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1"} {
    ^bb98(%1343: f32, %1344: f32, %1345: f32):
      %1346 = arith.addf %1343, %1344 : f32
      linalg.yield %1346 : f32
    } -> tensor<196x1024xf32>
    %1347 = tensor.collapse_shape %1342 [[0 : i64, 1 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %1348 = tensor.expand_shape %1347 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %1349 = tensor.empty() : tensor<1x196x1024xf32>
    %1350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1348 : tensor<1x196x1024xf32>) outs(%1349 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.act"} {
    ^bb99(%1351: f32, %1352: f32):
      %1353 = arith.constant 5.000000e-01 : f32
      %1354 = arith.constant 1.000000e+00 : f32
      %1355 = arith.constant 0.707106769 : f32
      %1356 = arith.mulf %1351, %1355 : f32
      %1357 = math.erf %1356 : f32
      %1358 = arith.addf %1354, %1357 : f32
      %1359 = arith.mulf %1353, %1351 : f32
      %1360 = arith.mulf %1359, %1358 : f32
      linalg.yield %1360 : f32
    } -> tensor<1x196x1024xf32>
    %1361 = tensor.collapse_shape %1350 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %1362 = tensor.expand_shape %1361 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %1363 = tensor.empty() : tensor<1024x256xf32>
    %1364 = linalg.transpose ins(%28:tensor<256x1024xf32>) outs(%1363:tensor<1024x256xf32>) permutation = [1, 0]
    %1365 = tensor.empty() : tensor<196x256xf32>
    %1366 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1367 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1366 : f32) outs(%1365 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1368 = linalg.matmul {prov.region_id = "matmul_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2", prov.transposed_b = "true"} ins(%1362, %1364 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%1367 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1369 = tensor.empty() : tensor<196x256xf32>
    %1370 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1368, %29 : tensor<196x256xf32>, tensor<256xf32>) outs(%1369 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2"} {
    ^bb100(%1371: f32, %1372: f32, %1373: f32):
      %1374 = arith.addf %1371, %1372 : f32
      linalg.yield %1374 : f32
    } -> tensor<196x256xf32>
    %1375 = tensor.collapse_shape %1370 [[0 : i64, 1 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %1376 = tensor.expand_shape %1375 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1377 = tensor.empty() : tensor<1x196x256xf32>
    %1378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %1376 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1377 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2"} {
    ^bb101(%1379: f32, %1380: f32, %1381: f32):
      %1382 = arith.addf %1379, %1380 : f32
      linalg.yield %1382 : f32
    } -> tensor<1x196x256xf32>
    %1383 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} 0.000000e+00 : f32
    %1384 = tensor.splat %1383 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32>
    %1385 = linalg.reduce ins(%1378:tensor<1x196x256xf32>) outs(%1384:tensor<1x196xf32>) dimensions = [2]
    (%1386: f32, %1387: f32) {
      %1388 = arith.addf %1386, %1387 : f32
      linalg.yield %1388 : f32
    }
    %1389 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} 2.560000e+02 : f32
    %1390 = tensor.splat %1389 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32>
    %1391 = tensor.empty() : tensor<1x196xf32>
    %1392 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1385, %1390 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1391 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb102(%1393: f32, %1394: f32, %1395: f32):
      %1396 = arith.divf %1393, %1394 : f32
      linalg.yield %1396 : f32
    } -> tensor<1x196xf32>
    %1397 = tensor.collapse_shape %1392 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %1398 = tensor.expand_shape %1397 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1399 = tensor.empty() : tensor<1x196x256xf32>
    %1400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1378, %1398 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1399 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb103(%1401: f32, %1402: f32, %1403: f32):
      %1404 = arith.subf %1401, %1402 : f32
      linalg.yield %1404 : f32
    } -> tensor<1x196x256xf32>
    %1405 = tensor.empty() : tensor<1x196x256xf32>
    %1406 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1400, %1400 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1405 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb104(%1407: f32, %1408: f32, %1409: f32):
      %1410 = arith.mulf %1407, %1408 : f32
      linalg.yield %1410 : f32
    } -> tensor<1x196x256xf32>
    %1411 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} 0.000000e+00 : f32
    %1412 = tensor.splat %1411 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32>
    %1413 = linalg.reduce ins(%1406:tensor<1x196x256xf32>) outs(%1412:tensor<1x196xf32>) dimensions = [2]
    (%1414: f32, %1415: f32) {
      %1416 = arith.addf %1414, %1415 : f32
      linalg.yield %1416 : f32
    }
    %1417 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} 2.560000e+02 : f32
    %1418 = tensor.splat %1417 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32>
    %1419 = tensor.empty() : tensor<1x196xf32>
    %1420 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1413, %1418 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1419 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb105(%1421: f32, %1422: f32, %1423: f32):
      %1424 = arith.divf %1421, %1422 : f32
      linalg.yield %1424 : f32
    } -> tensor<1x196xf32>
    %1425 = tensor.collapse_shape %1420 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %1426 = tensor.expand_shape %1425 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1427 = arith.constant {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} 1.000000e-06 : f32
    %1428 = tensor.splat %1427 {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} : tensor<1x196x1xf32>
    %1429 = tensor.empty() : tensor<1x196x1xf32>
    %1430 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1426, %1428 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%1429 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb106(%1431: f32, %1432: f32, %1433: f32):
      %1434 = arith.addf %1431, %1432 : f32
      linalg.yield %1434 : f32
    } -> tensor<1x196x1xf32>
    %1435 = tensor.empty() : tensor<1x196x1xf32>
    %1436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1430 : tensor<1x196x1xf32>) outs(%1435 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb107(%1437: f32, %1438: f32):
      %1439 = math.rsqrt %1437 : f32
      linalg.yield %1439 : f32
    } -> tensor<1x196x1xf32>
    %1440 = tensor.empty() : tensor<1x196x256xf32>
    %1441 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1400, %1436 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1440 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb108(%1442: f32, %1443: f32, %1444: f32):
      %1445 = arith.mulf %1442, %1443 : f32
      linalg.yield %1445 : f32
    } -> tensor<1x196x256xf32>
    %1446 = tensor.empty() : tensor<1x196x256xf32>
    %1447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1441, %30 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1446 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb109(%1448: f32, %1449: f32, %1450: f32):
      %1451 = arith.mulf %1448, %1449 : f32
      linalg.yield %1451 : f32
    } -> tensor<1x196x256xf32>
    %1452 = tensor.empty() : tensor<1x196x256xf32>
    %1453 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1447, %31 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1452 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_6", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm1"} {
    ^bb110(%1454: f32, %1455: f32, %1456: f32):
      %1457 = arith.addf %1454, %1455 : f32
      linalg.yield %1457 : f32
    } -> tensor<1x196x256xf32>
    %1458 = tensor.collapse_shape %1453 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1459 = tensor.expand_shape %1458 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 14, 256] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<50176xf32> into tensor<1x14x14x256xf32>
    %1460 = tensor.empty() : tensor<1x14x256x14xf32>
    %1461 = linalg.transpose ins(%1459:tensor<1x14x14x256xf32>) outs(%1460:tensor<1x14x256x14xf32>) permutation = [0, 1, 3, 2]
    %1462 = tensor.collapse_shape %1461 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x256x14xf32> into tensor<50176xf32>
    %1463 = tensor.expand_shape %1462 [[0 : i64, 1 : i64]] output_shape [3584, 14] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<50176xf32> into tensor<3584x14xf32>
    %1464 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BD2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD"> : tensor<14x8xf32>
    %1465 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000804CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCCB5C21A300000080D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643DCB5CA12300000080379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD300BF2A300000080379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DCB5C212400000080D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BDFEB349A4000000804CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C300B722400000080CB5C21A3CB5CA123300BF2A3CB5C2124FEB349A4300B722432318DA4000000804CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBCCB5CA12400000080D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D6488B5A400000080379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDFEB3C92400000080379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D7EA2352500000080D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD300BF224000000804CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CD7D6D3A5"> : tensor<14x8xf32>
    %1466 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1467 = tensor.splat %1466 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x8xf32>
    %1468 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1463, %1464 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%1467 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %1469 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1470 = tensor.splat %1469 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x8xf32>
    %1471 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1463, %1465 : tensor<3584x14xf32>, tensor<14x8xf32>) outs(%1470 : tensor<3584x8xf32>) -> tensor<3584x8xf32>
    %1472 = tensor.collapse_shape %1468 [[0 : i64, 1 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %1473 = tensor.expand_shape %1472 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %1474 = tensor.empty() : tensor<1x14x8x256xf32>
    %1475 = linalg.transpose ins(%1473:tensor<1x14x256x8xf32>) outs(%1474:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %1476 = tensor.collapse_shape %1471 [[0 : i64, 1 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x8xf32> into tensor<28672xf32>
    %1477 = tensor.expand_shape %1476 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 8] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x256x8xf32>
    %1478 = tensor.empty() : tensor<1x14x8x256xf32>
    %1479 = linalg.transpose ins(%1477:tensor<1x14x256x8xf32>) outs(%1478:tensor<1x14x8x256xf32>) permutation = [0, 1, 3, 2]
    %1480 = tensor.empty() : tensor<1x8x256x14xf32>
    %1481 = linalg.transpose ins(%1475:tensor<1x14x8x256xf32>) outs(%1480:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1482 = tensor.collapse_shape %1481 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1483 = tensor.expand_shape %1482 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1484 = tensor.empty() : tensor<1x8x256x14xf32>
    %1485 = linalg.transpose ins(%1479:tensor<1x14x8x256xf32>) outs(%1484:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1486 = tensor.collapse_shape %1485 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1487 = tensor.expand_shape %1486 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1488 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803FE5A566BF079D1F3F87DC63BE87DC63BE079D1F3FE5A566BF0000803F079D1FBF87DC63BEE5A5663FE5A566BF87DC633E079D1F3F000080BF079D1F3F87DC633EE5A566BFE5A5663F87DC63BE079D1FBF0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC63BEE5A566BF079D1F3F079D1F3FE5A566BF87DC63BE0000803F87DC633EE5A566BF079D1FBF079D1F3FE5A5663F87DC63BE000080BF87DC63BEE5A5663F079D1F3F079D1FBFE5A566BF87DC633E0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803F079D1F3F87DC63BEE5A566BFE5A566BF87DC63BE079D1F3F0000803FE5A5663F079D1F3F87DC633E87DC63BE079D1FBFE5A566BF000080BFE5A566BF079D1FBF87DC63BE87DC633E079D1F3FE5A5663F"> : tensor<14x14xf32>
    %1489 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x0000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800000008000000080000000800226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE32310DA50226DE3E1C26483FE094793FE094793F1C26483F0226DE3E000000801C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F32318D251C2648BFE09479BF0226DEBE0226DE3EE094793F1C26483F00000080E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BFCAC9D3A5E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F00000080E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F32310D26E09479BF0226DE3E1C26483F1C2648BF0226DEBEE094793F000000801C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF7E7D30A61C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F000000800226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3ECAC953260226DEBE1C26483FE09479BFE094793F1C2648BF0226DE3E0000008032310DA532318D25CAC9D3A532310D267E7D30A6CAC95326171677A632318D2658D79EA67E7DB026E988B0A7CAC9D32602522328000000800226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE32318D260226DE3E1C2648BFE094793FE09479BF1C26483F0226DEBE000000801C26483FE09479BF0226DE3E0226DE3EE09479BF1C26483F58D79EA61C2648BFE094793F0226DEBE0226DEBEE094793F1C2648BF00000080E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF7E7DB026E094793F0226DEBE1C2648BF1C26483F0226DE3EE09479BF00000080E094793F0226DE3E1C2648BF1C2648BF0226DE3EE094793F2EEE1E27E09479BF0226DEBE1C26483F1C26483F0226DEBEE09479BF000000801C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BFCAC9D3261C26483FE094793F0226DE3E0226DEBEE09479BF1C2648BF000000800226DE3E1C26483FE094793FE094793F1C26483F0226DE3EFC5BB9A70226DEBE1C2648BFE09479BFE09479BF1C2648BF0226DEBE"> : tensor<14x14xf32>
    %1490 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1491 = tensor.splat %1490 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1492 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1483, %1488 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1491 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1493 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1494 = tensor.splat %1493 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1495 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1487, %1489 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1494 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1496 = tensor.empty() : tensor<2048x14xf32>
    %1497 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1492, %1495 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1496 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb111(%1498: f32, %1499: f32, %1500: f32):
      %1501 = arith.subf %1498, %1499 : f32
      linalg.yield %1501 : f32
    } -> tensor<2048x14xf32>
    %1502 = tensor.collapse_shape %1497 [[0 : i64, 1 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1503 = tensor.expand_shape %1502 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1504 = tensor.empty() : tensor<1x14x8x256xf32>
    %1505 = linalg.transpose ins(%1503:tensor<1x8x256x14xf32>) outs(%1504:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1506 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1507 = tensor.splat %1506 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1508 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1487, %1488 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1507 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1509 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1510 = tensor.splat %1509 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1511 = linalg.matmul {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1483, %1489 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1510 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1512 = tensor.empty() : tensor<2048x14xf32>
    %1513 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1508, %1511 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1512 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb112(%1514: f32, %1515: f32, %1516: f32):
      %1517 = arith.addf %1514, %1515 : f32
      linalg.yield %1517 : f32
    } -> tensor<2048x14xf32>
    %1518 = tensor.collapse_shape %1513 [[0 : i64, 1 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1519 = tensor.expand_shape %1518 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1520 = tensor.empty() : tensor<1x14x8x256xf32>
    %1521 = linalg.transpose ins(%1519:tensor<1x8x256x14xf32>) outs(%1520:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1522 = arith.constant {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1523 = tensor.splat %1522 {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x2xf32>
    %1524 = tensor.collapse_shape %1505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1525 = tensor.expand_shape %1524 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1526 = "tensor.insert_slice"(%1525, %1523) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1527 = tensor.collapse_shape %1521 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1528 = tensor.expand_shape %1527 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1529 = "tensor.insert_slice"(%1528, %1526) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "fft_6", prov.family = "spectral", prov._pattern_hint = "fft_rfft2", prov.op = "fft_rfft2", prov.aten = "aten._fft_r2c.default", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1530 = "tensor.extract_slice"(%1529) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1531 = tensor.collapse_shape %1530 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1532 = tensor.expand_shape %1531 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1533 = "tensor.extract_slice"(%1529) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1534 = tensor.collapse_shape %1533 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1535 = tensor.expand_shape %1534 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1536 = "tensor.extract_slice"(%32) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %1537 = tensor.collapse_shape %1536 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %1538 = tensor.expand_shape %1537 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %1539 = "tensor.extract_slice"(%32) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<14x8x256x2xf32>) -> tensor<14x8x256x1xf32>
    %1540 = tensor.collapse_shape %1539 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<14x8x256x1xf32> into tensor<28672xf32>
    %1541 = tensor.expand_shape %1540 [[0 : i64, 1 : i64, 2 : i64]] output_shape [14, 8, 256] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<14x8x256xf32>
    %1542 = tensor.empty() : tensor<1x14x8x256xf32>
    %1543 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1532, %1538 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1542 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb113(%1544: f32, %1545: f32, %1546: f32):
      %1547 = arith.mulf %1544, %1545 : f32
      linalg.yield %1547 : f32
    } -> tensor<1x14x8x256xf32>
    %1548 = tensor.empty() : tensor<1x14x8x256xf32>
    %1549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1535, %1541 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1548 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb114(%1550: f32, %1551: f32, %1552: f32):
      %1553 = arith.mulf %1550, %1551 : f32
      linalg.yield %1553 : f32
    } -> tensor<1x14x8x256xf32>
    %1554 = tensor.empty() : tensor<1x14x8x256xf32>
    %1555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1532, %1541 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1554 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb115(%1556: f32, %1557: f32, %1558: f32):
      %1559 = arith.mulf %1556, %1557 : f32
      linalg.yield %1559 : f32
    } -> tensor<1x14x8x256xf32>
    %1560 = tensor.empty() : tensor<1x14x8x256xf32>
    %1561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1535, %1538 : tensor<1x14x8x256xf32>, tensor<14x8x256xf32>) outs(%1560 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb116(%1562: f32, %1563: f32, %1564: f32):
      %1565 = arith.mulf %1562, %1563 : f32
      linalg.yield %1565 : f32
    } -> tensor<1x14x8x256xf32>
    %1566 = tensor.empty() : tensor<1x14x8x256xf32>
    %1567 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1543, %1549 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%1566 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb117(%1568: f32, %1569: f32, %1570: f32):
      %1571 = arith.subf %1568, %1569 : f32
      linalg.yield %1571 : f32
    } -> tensor<1x14x8x256xf32>
    %1572 = tensor.empty() : tensor<1x14x8x256xf32>
    %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1555, %1561 : tensor<1x14x8x256xf32>, tensor<1x14x8x256xf32>) outs(%1572 : tensor<1x14x8x256xf32>) attrs =  {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb118(%1574: f32, %1575: f32, %1576: f32):
      %1577 = arith.addf %1574, %1575 : f32
      linalg.yield %1577 : f32
    } -> tensor<1x14x8x256xf32>
    %1578 = arith.constant {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1579 = tensor.splat %1578 {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x2xf32>
    %1580 = tensor.collapse_shape %1567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1581 = tensor.expand_shape %1580 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1582 = "tensor.insert_slice"(%1581, %1579) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1583 = tensor.collapse_shape %1573 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256xf32> into tensor<28672xf32>
    %1584 = tensor.expand_shape %1583 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 14, 8, 256, 1] {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256x1xf32>
    %1585 = "tensor.insert_slice"(%1584, %1582) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "complex_mul_3", prov.family = "elementwise", prov._pattern_hint = "mul", prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "complex64", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x1xf32>, tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x2xf32>
    %1586 = "tensor.extract_slice"(%1585) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1587 = tensor.collapse_shape %1586 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1588 = tensor.expand_shape %1587 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1589 = "tensor.extract_slice"(%1585) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 14, 8, 256, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : (tensor<1x14x8x256x2xf32>) -> tensor<1x14x8x256x1xf32>
    %1590 = tensor.collapse_shape %1589 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x8x256x1xf32> into tensor<28672xf32>
    %1591 = tensor.expand_shape %1590 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 8, 256] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x14x8x256xf32>
    %1592 = tensor.empty() : tensor<1x8x256x14xf32>
    %1593 = linalg.transpose ins(%1588:tensor<1x14x8x256xf32>) outs(%1592:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1594 = tensor.collapse_shape %1593 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1595 = tensor.expand_shape %1594 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1596 = tensor.empty() : tensor<1x8x256x14xf32>
    %1597 = linalg.transpose ins(%1591:tensor<1x14x8x256xf32>) outs(%1596:tensor<1x8x256x14xf32>) permutation = [0, 2, 3, 1]
    %1598 = tensor.collapse_shape %1597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x8x256x14xf32> into tensor<28672xf32>
    %1599 = tensor.expand_shape %1598 [[0 : i64, 1 : i64]] output_shape [2048, 14] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<2048x14xf32>
    %1600 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D254992BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D83CC83BD516A363DDF3482BCDF3482BC516A363D83CC83BD2549923D516A36BDDF3482BC83CC833D83CC83BDDF34823C516A363D254992BD516A363DDF34823C83CC83BD83CC833DDF3482BC516A36BD2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF3482BC83CC83BD516A363D516A363D83CC83BDDF3482BC2549923DDF34823C83CC83BD516A36BD516A363D83CC833DDF3482BC254992BDDF3482BC83CC833D516A363D516A36BD83CC83BDDF34823C2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D516A363DDF3482BC83CC83BD83CC83BDDF3482BC516A363D2549923D83CC833D516A363DDF34823CDF3482BC516A36BD83CC83BD254992BD83CC83BD516A36BDDF3482BCDF34823C516A363D83CC833D"> : tensor<14x14xf32>
    %1601 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3CCB5C21234CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBC00000000D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BDCB5CA1A3D6BD643D379E8E3D4CE2FD3C4CE2FDBC379E8EBDD6BD64BD00000000379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D300BF223379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD00000000379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBDCB5C21A4379E8E3D4CE2FDBCD6BD64BDD6BD643D4CE2FD3C379E8EBD00000000D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643DFEB34924D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD000000004CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC300B72A44CE2FD3CD6BD64BD379E8E3D379E8EBDD6BD643D4CE2FDBC00000000CB5C2123CB5CA1A3300BF223CB5C21A4FEB34924300B72A432318D24CB5CA1A46488B524FEB3C9A40AC1C925300BF2A4DEA63AA6000000004CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3CCB5CA1A44CE2FDBCD6BD643D379E8EBD379E8E3DD6BD64BD4CE2FD3C00000000D6BD64BD379E8E3D4CE2FDBC4CE2FDBC379E8E3DD6BD64BD6488B524D6BD643D379E8EBD4CE2FD3C4CE2FD3C379E8EBDD6BD643D00000000379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3DFEB3C9A4379E8EBD4CE2FD3CD6BD643DD6BD64BD4CE2FDBC379E8E3D00000000379E8EBD4CE2FDBCD6BD643DD6BD643D4CE2FDBC379E8EBD7EA235A5379E8E3D4CE2FD3CD6BD64BDD6BD64BD4CE2FD3C379E8E3D00000000D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D300BF2A4D6BD64BD379E8EBD4CE2FDBC4CE2FD3C379E8E3DD6BD643D000000004CE2FDBCD6BD64BD379E8EBD379E8EBDD6BD64BD4CE2FDBCD7D6D3254CE2FD3CD6BD643D379E8E3D379E8E3DD6BD643D4CE2FD3C"> : tensor<14x14xf32>
    %1602 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1603 = tensor.splat %1602 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1604 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1595, %1600 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1603 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1605 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1606 = tensor.splat %1605 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1607 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1599, %1601 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1606 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1608 = tensor.empty() : tensor<2048x14xf32>
    %1609 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1604, %1607 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1608 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb119(%1610: f32, %1611: f32, %1612: f32):
      %1613 = arith.subf %1610, %1611 : f32
      linalg.yield %1613 : f32
    } -> tensor<2048x14xf32>
    %1614 = tensor.collapse_shape %1609 [[0 : i64, 1 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1615 = tensor.expand_shape %1614 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1616 = tensor.empty() : tensor<1x14x8x256xf32>
    %1617 = linalg.transpose ins(%1615:tensor<1x8x256x14xf32>) outs(%1616:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1618 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1619 = tensor.splat %1618 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1620 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1599, %1600 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1619 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1621 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1622 = tensor.splat %1621 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32>
    %1623 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1595, %1601 : tensor<2048x14xf32>, tensor<14x14xf32>) outs(%1622 : tensor<2048x14xf32>) -> tensor<2048x14xf32>
    %1624 = tensor.empty() : tensor<2048x14xf32>
    %1625 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1620, %1623 : tensor<2048x14xf32>, tensor<2048x14xf32>) outs(%1624 : tensor<2048x14xf32>) attrs =  {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb120(%1626: f32, %1627: f32, %1628: f32):
      %1629 = arith.addf %1626, %1627 : f32
      linalg.yield %1629 : f32
    } -> tensor<2048x14xf32>
    %1630 = tensor.collapse_shape %1625 [[0 : i64, 1 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<2048x14xf32> into tensor<28672xf32>
    %1631 = tensor.expand_shape %1630 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 256, 14] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<1x8x256x14xf32>
    %1632 = tensor.empty() : tensor<1x14x8x256xf32>
    %1633 = linalg.transpose ins(%1631:tensor<1x8x256x14xf32>) outs(%1632:tensor<1x14x8x256xf32>) permutation = [0, 3, 1, 2]
    %1634 = tensor.empty() : tensor<1x14x256x8xf32>
    %1635 = linalg.transpose ins(%1617:tensor<1x14x8x256xf32>) outs(%1634:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %1636 = tensor.collapse_shape %1635 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %1637 = tensor.expand_shape %1636 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %1638 = tensor.empty() : tensor<1x14x256x8xf32>
    %1639 = linalg.transpose ins(%1633:tensor<1x14x8x256xf32>) outs(%1638:tensor<1x14x256x8xf32>) permutation = [0, 1, 3, 2]
    %1640 = tensor.collapse_shape %1639 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x256x8xf32> into tensor<28672xf32>
    %1641 = tensor.expand_shape %1640 [[0 : i64, 1 : i64]] output_shape [3584, 8] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<28672xf32> into tensor<3584x8xf32>
    %1642 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F00000040E5A5E63F079D9F3F87DCE33E87DCE3BE079D9FBFE5A5E6BF000000C0E5A5E6BF079D9FBF87DCE3BE87DCE33E079D9F3FE5A5E63F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F00000040079D9F3F87DCE3BEE5A5E6BFE5A5E6BF87DCE3BE079D9F3F0000004087DCE33EE5A5E6BF079D9FBF079D9F3FE5A5E63F87DCE3BE000000C087DCE3BEE5A5E63F079D9F3F079D9FBFE5A5E6BF87DCE33E0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE0000004087DCE3BEE5A5E6BF079D9F3F079D9F3FE5A5E6BF87DCE3BE00000040079D9FBF87DCE3BEE5A5E63FE5A5E6BF87DCE33E079D9F3F000000C0079D9F3F87DCE33EE5A5E6BFE5A5E63F87DCE3BE079D9FBF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF00000040E5A5E6BF079D9F3F87DCE3BE87DCE3BE079D9F3FE5A5E6BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF0000803F000080BF"> : tensor<8x14xf32>
    %1643 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002265E3F1C26C83FE094F93FE094F93F1C26C83F02265E3F32318D2502265EBF1C26C8BFE094F9BFE094F9BF1C26C8BF02265EBF000000001C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF32310DA61C26C83FE094F93F02265E3F02265EBFE094F9BF1C26C8BF00000000E094F93F02265E3F1C26C8BF1C26C8BF02265E3FE094F93FCAC95326E094F9BF02265EBF1C26C83F1C26C83F02265EBFE094F9BF00000000E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF32318DA6E094F93F02265EBF1C26C8BF1C26C83F02265E3FE094F9BF000000001C26C83FE094F9BF02265E3F02265E3FE094F9BF1C26C83F7E7DB0261C26C8BFE094F93F02265EBF02265EBFE094F93F1C26C8BF0000000002265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBFCAC9D3A602265E3F1C26C8BFE094F93FE094F9BF1C26C83F02265EBF0000000032310D2532318DA5CAC9D32532310DA67E7D3026CAC953A61716772632318DA658D79E267E7DB0A6E988B027CAC9D3A6025223A8"> : tensor<8x14xf32>
    %1644 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1645 = tensor.splat %1644 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x14xf32>
    %1646 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1637, %1642 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%1645 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %1647 = arith.constant {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} 0.000000e+00 : f32
    %1648 = tensor.splat %1647 {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x14xf32>
    %1649 = linalg.matmul {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} ins(%1641, %1643 : tensor<3584x8xf32>, tensor<8x14xf32>) outs(%1648 : tensor<3584x14xf32>) -> tensor<3584x14xf32>
    %1650 = tensor.empty() : tensor<3584x14xf32>
    %1651 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1646, %1649 : tensor<3584x14xf32>, tensor<3584x14xf32>) outs(%1650 : tensor<3584x14xf32>) attrs =  {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} {
    ^bb121(%1652: f32, %1653: f32, %1654: f32):
      %1655 = arith.subf %1652, %1653 : f32
      linalg.yield %1655 : f32
    } -> tensor<3584x14xf32>
    %1656 = tensor.collapse_shape %1651 [[0 : i64, 1 : i64]] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<3584x14xf32> into tensor<50176xf32>
    %1657 = tensor.expand_shape %1656 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 14, 256, 14] {prov.region_id = "fft_7", prov.family = "spectral", prov._pattern_hint = "fft_irfft2", prov.op = "fft_irfft2", prov.aten = "aten._fft_c2r.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<50176xf32> into tensor<1x14x256x14xf32>
    %1658 = tensor.empty() : tensor<1x14x14x256xf32>
    %1659 = linalg.transpose ins(%1657:tensor<1x14x256x14xf32>) outs(%1658:tensor<1x14x14x256xf32>) permutation = [0, 1, 3, 2]
    %1660 = tensor.collapse_shape %1659 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<1x14x14x256xf32> into tensor<50176xf32>
    %1661 = tensor.expand_shape %1660 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.filter"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1662 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} 0.000000e+00 : f32
    %1663 = tensor.splat %1662 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32>
    %1664 = linalg.reduce ins(%1661:tensor<1x196x256xf32>) outs(%1663:tensor<1x196xf32>) dimensions = [2]
    (%1665: f32, %1666: f32) {
      %1667 = arith.addf %1665, %1666 : f32
      linalg.yield %1667 : f32
    }
    %1668 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} 2.560000e+02 : f32
    %1669 = tensor.splat %1668 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32>
    %1670 = tensor.empty() : tensor<1x196xf32>
    %1671 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1664, %1669 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1670 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb122(%1672: f32, %1673: f32, %1674: f32):
      %1675 = arith.divf %1672, %1673 : f32
      linalg.yield %1675 : f32
    } -> tensor<1x196xf32>
    %1676 = tensor.collapse_shape %1671 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %1677 = tensor.expand_shape %1676 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1678 = tensor.empty() : tensor<1x196x256xf32>
    %1679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661, %1677 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1678 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb123(%1680: f32, %1681: f32, %1682: f32):
      %1683 = arith.subf %1680, %1681 : f32
      linalg.yield %1683 : f32
    } -> tensor<1x196x256xf32>
    %1684 = tensor.empty() : tensor<1x196x256xf32>
    %1685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1679, %1679 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1684 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb124(%1686: f32, %1687: f32, %1688: f32):
      %1689 = arith.mulf %1686, %1687 : f32
      linalg.yield %1689 : f32
    } -> tensor<1x196x256xf32>
    %1690 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} 0.000000e+00 : f32
    %1691 = tensor.splat %1690 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32>
    %1692 = linalg.reduce ins(%1685:tensor<1x196x256xf32>) outs(%1691:tensor<1x196xf32>) dimensions = [2]
    (%1693: f32, %1694: f32) {
      %1695 = arith.addf %1693, %1694 : f32
      linalg.yield %1695 : f32
    }
    %1696 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} 2.560000e+02 : f32
    %1697 = tensor.splat %1696 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32>
    %1698 = tensor.empty() : tensor<1x196xf32>
    %1699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1692, %1697 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1698 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb125(%1700: f32, %1701: f32, %1702: f32):
      %1703 = arith.divf %1700, %1701 : f32
      linalg.yield %1703 : f32
    } -> tensor<1x196xf32>
    %1704 = tensor.collapse_shape %1699 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1706 = arith.constant {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} 1.000000e-06 : f32
    %1707 = tensor.splat %1706 {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} : tensor<1x196x1xf32>
    %1708 = tensor.empty() : tensor<1x196x1xf32>
    %1709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1705, %1707 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%1708 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb126(%1710: f32, %1711: f32, %1712: f32):
      %1713 = arith.addf %1710, %1711 : f32
      linalg.yield %1713 : f32
    } -> tensor<1x196x1xf32>
    %1714 = tensor.empty() : tensor<1x196x1xf32>
    %1715 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1709 : tensor<1x196x1xf32>) outs(%1714 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb127(%1716: f32, %1717: f32):
      %1718 = math.rsqrt %1716 : f32
      linalg.yield %1718 : f32
    } -> tensor<1x196x1xf32>
    %1719 = tensor.empty() : tensor<1x196x256xf32>
    %1720 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1679, %1715 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1719 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb128(%1721: f32, %1722: f32, %1723: f32):
      %1724 = arith.mulf %1721, %1722 : f32
      linalg.yield %1724 : f32
    } -> tensor<1x196x256xf32>
    %1725 = tensor.empty() : tensor<1x196x256xf32>
    %1726 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1720, %33 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1725 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb129(%1727: f32, %1728: f32, %1729: f32):
      %1730 = arith.mulf %1727, %1728 : f32
      linalg.yield %1730 : f32
    } -> tensor<1x196x256xf32>
    %1731 = tensor.empty() : tensor<1x196x256xf32>
    %1732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1726, %34 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1731 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_7", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.norm2"} {
    ^bb130(%1733: f32, %1734: f32, %1735: f32):
      %1736 = arith.addf %1733, %1734 : f32
      linalg.yield %1736 : f32
    } -> tensor<1x196x256xf32>
    %1737 = tensor.collapse_shape %1732 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1738 = tensor.expand_shape %1737 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %1739 = tensor.empty() : tensor<256x1024xf32>
    %1740 = linalg.transpose ins(%35:tensor<1024x256xf32>) outs(%1739:tensor<256x1024xf32>) permutation = [1, 0]
    %1741 = tensor.empty() : tensor<196x1024xf32>
    %1742 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1743 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1742 : f32) outs(%1741 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %1744 = linalg.matmul {prov.region_id = "matmul_6", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1", prov.transposed_b = "true"} ins(%1738, %1740 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%1743 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %1745 = tensor.empty() : tensor<196x1024xf32>
    %1746 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1744, %36 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%1745 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1"} {
    ^bb131(%1747: f32, %1748: f32, %1749: f32):
      %1750 = arith.addf %1747, %1748 : f32
      linalg.yield %1750 : f32
    } -> tensor<196x1024xf32>
    %1751 = tensor.collapse_shape %1746 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %1752 = tensor.expand_shape %1751 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %1753 = tensor.empty() : tensor<1x196x1024xf32>
    %1754 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1752 : tensor<1x196x1024xf32>) outs(%1753 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.act"} {
    ^bb132(%1755: f32, %1756: f32):
      %1757 = arith.constant 5.000000e-01 : f32
      %1758 = arith.constant 1.000000e+00 : f32
      %1759 = arith.constant 0.707106769 : f32
      %1760 = arith.mulf %1755, %1759 : f32
      %1761 = math.erf %1760 : f32
      %1762 = arith.addf %1758, %1761 : f32
      %1763 = arith.mulf %1757, %1755 : f32
      %1764 = arith.mulf %1763, %1762 : f32
      linalg.yield %1764 : f32
    } -> tensor<1x196x1024xf32>
    %1765 = tensor.collapse_shape %1754 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %1766 = tensor.expand_shape %1765 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %1767 = tensor.empty() : tensor<1024x256xf32>
    %1768 = linalg.transpose ins(%37:tensor<256x1024xf32>) outs(%1767:tensor<1024x256xf32>) permutation = [1, 0]
    %1769 = tensor.empty() : tensor<196x256xf32>
    %1770 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1771 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1770 : f32) outs(%1769 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1772 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2", prov.transposed_b = "true"} ins(%1766, %1768 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%1771 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1773 = tensor.empty() : tensor<196x256xf32>
    %1774 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1772, %38 : tensor<196x256xf32>, tensor<256xf32>) outs(%1773 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2"} {
    ^bb133(%1775: f32, %1776: f32, %1777: f32):
      %1778 = arith.addf %1775, %1776 : f32
      linalg.yield %1778 : f32
    } -> tensor<196x256xf32>
    %1779 = tensor.collapse_shape %1774 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %1780 = tensor.expand_shape %1779 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1781 = tensor.empty() : tensor<1x196x256xf32>
    %1782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1378, %1780 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1781 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3"} {
    ^bb134(%1783: f32, %1784: f32, %1785: f32):
      %1786 = arith.addf %1783, %1784 : f32
      linalg.yield %1786 : f32
    } -> tensor<1x196x256xf32>
    %1787 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} 0.000000e+00 : f32
    %1788 = tensor.splat %1787 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32>
    %1789 = linalg.reduce ins(%1782:tensor<1x196x256xf32>) outs(%1788:tensor<1x196xf32>) dimensions = [2]
    (%1790: f32, %1791: f32) {
      %1792 = arith.addf %1790, %1791 : f32
      linalg.yield %1792 : f32
    }
    %1793 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} 2.560000e+02 : f32
    %1794 = tensor.splat %1793 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32>
    %1795 = tensor.empty() : tensor<1x196xf32>
    %1796 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1789, %1794 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1795 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb135(%1797: f32, %1798: f32, %1799: f32):
      %1800 = arith.divf %1797, %1798 : f32
      linalg.yield %1800 : f32
    } -> tensor<1x196xf32>
    %1801 = tensor.collapse_shape %1796 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %1802 = tensor.expand_shape %1801 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1803 = tensor.empty() : tensor<1x196x256xf32>
    %1804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1782, %1802 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1803 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb136(%1805: f32, %1806: f32, %1807: f32):
      %1808 = arith.subf %1805, %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<1x196x256xf32>
    %1809 = tensor.empty() : tensor<1x196x256xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804, %1804 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1809 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb137(%1811: f32, %1812: f32, %1813: f32):
      %1814 = arith.mulf %1811, %1812 : f32
      linalg.yield %1814 : f32
    } -> tensor<1x196x256xf32>
    %1815 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} 0.000000e+00 : f32
    %1816 = tensor.splat %1815 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32>
    %1817 = linalg.reduce ins(%1810:tensor<1x196x256xf32>) outs(%1816:tensor<1x196xf32>) dimensions = [2]
    (%1818: f32, %1819: f32) {
      %1820 = arith.addf %1818, %1819 : f32
      linalg.yield %1820 : f32
    }
    %1821 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} 2.560000e+02 : f32
    %1822 = tensor.splat %1821 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32>
    %1823 = tensor.empty() : tensor<1x196xf32>
    %1824 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1817, %1822 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%1823 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb138(%1825: f32, %1826: f32, %1827: f32):
      %1828 = arith.divf %1825, %1826 : f32
      linalg.yield %1828 : f32
    } -> tensor<1x196xf32>
    %1829 = tensor.collapse_shape %1824 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %1830 = tensor.expand_shape %1829 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %1831 = arith.constant {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} 1.000000e-06 : f32
    %1832 = tensor.splat %1831 {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} : tensor<1x196x1xf32>
    %1833 = tensor.empty() : tensor<1x196x1xf32>
    %1834 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1830, %1832 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%1833 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb139(%1835: f32, %1836: f32, %1837: f32):
      %1838 = arith.addf %1835, %1836 : f32
      linalg.yield %1838 : f32
    } -> tensor<1x196x1xf32>
    %1839 = tensor.empty() : tensor<1x196x1xf32>
    %1840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1834 : tensor<1x196x1xf32>) outs(%1839 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb140(%1841: f32, %1842: f32):
      %1843 = math.rsqrt %1841 : f32
      linalg.yield %1843 : f32
    } -> tensor<1x196x1xf32>
    %1844 = tensor.empty() : tensor<1x196x256xf32>
    %1845 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1804, %1840 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%1844 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb141(%1846: f32, %1847: f32, %1848: f32):
      %1849 = arith.mulf %1846, %1847 : f32
      linalg.yield %1849 : f32
    } -> tensor<1x196x256xf32>
    %1850 = tensor.empty() : tensor<1x196x256xf32>
    %1851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1845, %39 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1850 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb142(%1852: f32, %1853: f32, %1854: f32):
      %1855 = arith.mulf %1852, %1853 : f32
      linalg.yield %1855 : f32
    } -> tensor<1x196x256xf32>
    %1856 = tensor.empty() : tensor<1x196x256xf32>
    %1857 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1851, %40 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%1856 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_8", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm1"} {
    ^bb143(%1858: f32, %1859: f32, %1860: f32):
      %1861 = arith.addf %1858, %1859 : f32
      linalg.yield %1861 : f32
    } -> tensor<1x196x256xf32>
    %1862 = tensor.collapse_shape %1857 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1863 = tensor.expand_shape %1862 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %1864 = tensor.empty() : tensor<256x768xf32>
    %1865 = linalg.transpose ins(%47:tensor<768x256xf32>) outs(%1864:tensor<256x768xf32>) permutation = [1, 0]
    %1866 = tensor.empty() : tensor<196x768xf32>
    %1867 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1868 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1867 : f32) outs(%1866 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %1869 = linalg.matmul {prov.region_id = "matmul_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv", prov.transposed_b = "true"} ins(%1863, %1865 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%1868 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %1870 = tensor.empty() : tensor<196x768xf32>
    %1871 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1869, %48 : tensor<196x768xf32>, tensor<768xf32>) outs(%1870 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv"} {
    ^bb144(%1872: f32, %1873: f32, %1874: f32):
      %1875 = arith.addf %1872, %1873 : f32
      linalg.yield %1875 : f32
    } -> tensor<196x768xf32>
    %1876 = tensor.collapse_shape %1871 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %1877 = tensor.expand_shape %1876 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %1878 = tensor.collapse_shape %1877 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %1879 = tensor.expand_shape %1878 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %1880 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %1881 = linalg.transpose ins(%1879:tensor<1x196x3x4x64xf32>) outs(%1880:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %1882 = "tensor.extract_slice"(%1881) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %1883 = tensor.collapse_shape %1882 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %1884 = tensor.expand_shape %1883 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %1885 = "tensor.extract_slice"(%1881) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %1886 = tensor.collapse_shape %1885 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %1887 = tensor.expand_shape %1886 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %1888 = "tensor.extract_slice"(%1881) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %1889 = tensor.collapse_shape %1888 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %1890 = tensor.expand_shape %1889 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %1891 = tensor.empty() : tensor<1x4x64x196xf32>
    %1892 = linalg.transpose ins(%1887:tensor<1x4x196x64xf32>) outs(%1891:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %1893 = tensor.empty() : tensor<1x4x196x64xf32>
    %1894 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1884 : tensor<1x4x196x64xf32>) outs(%1893 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb145(%1895: f32, %1896: f32):
      linalg.yield %1895 : f32
    } -> tensor<1x4x196x64xf32>
    %1897 = tensor.collapse_shape %1894 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %1898 = tensor.expand_shape %1897 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %1899 = tensor.empty() : tensor<1x4x64x196xf32>
    %1900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1892 : tensor<1x4x64x196xf32>) outs(%1899 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb146(%1901: f32, %1902: f32):
      linalg.yield %1901 : f32
    } -> tensor<1x4x64x196xf32>
    %1903 = tensor.collapse_shape %1900 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %1904 = tensor.expand_shape %1903 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %1905 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} 0.000000e+00 : f32
    %1906 = tensor.splat %1905 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<4x196x196xf32>
    %1907 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1898, %1904 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%1906 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb147(%1908: f32, %1909: f32, %1910: f32):
      %1911 = arith.mulf %1908, %1909 : f32
      %1912 = arith.addf %1910, %1911 : f32
      linalg.yield %1912 : f32
    } -> tensor<4x196x196xf32>
    %1913 = tensor.collapse_shape %1907 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %1914 = tensor.expand_shape %1913 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %1915 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} 1.250000e-01 : f32
    %1916 = tensor.splat %1915 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196x196xf32>
    %1917 = tensor.empty() : tensor<1x4x196x196xf32>
    %1918 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1914, %1916 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%1917 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb148(%1919: f32, %1920: f32, %1921: f32):
      %1922 = arith.mulf %1919, %1920 : f32
      linalg.yield %1922 : f32
    } -> tensor<1x4x196x196xf32>
    %1923 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} 0xff800000 : f32
    %1924 = tensor.splat %1923 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196xf32>
    %1925 = linalg.reduce ins(%1918:tensor<1x4x196x196xf32>) outs(%1924:tensor<1x4x196xf32>) dimensions = [3]
    (%1926: f32, %1927: f32) {
      %1928 = arith.maximumf %1926, %1927 : f32
      linalg.yield %1928 : f32
    }
    %1929 = tensor.collapse_shape %1925 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %1930 = tensor.expand_shape %1929 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %1931 = tensor.empty() : tensor<1x4x196x196xf32>
    %1932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1918, %1930 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%1931 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb149(%1933: f32, %1934: f32, %1935: f32):
      %1936 = arith.subf %1933, %1934 : f32
      linalg.yield %1936 : f32
    } -> tensor<1x4x196x196xf32>
    %1937 = tensor.empty() : tensor<1x4x196x196xf32>
    %1938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1932 : tensor<1x4x196x196xf32>) outs(%1937 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb150(%1939: f32, %1940: f32):
      %1941 = math.exp %1939 : f32
      linalg.yield %1941 : f32
    } -> tensor<1x4x196x196xf32>
    %1942 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} 0.000000e+00 : f32
    %1943 = tensor.splat %1942 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196xf32>
    %1944 = linalg.reduce ins(%1938:tensor<1x4x196x196xf32>) outs(%1943:tensor<1x4x196xf32>) dimensions = [3]
    (%1945: f32, %1946: f32) {
      %1947 = arith.addf %1945, %1946 : f32
      linalg.yield %1947 : f32
    }
    %1948 = tensor.collapse_shape %1944 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %1949 = tensor.expand_shape %1948 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %1950 = tensor.empty() : tensor<1x4x196x196xf32>
    %1951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1938, %1949 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%1950 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb151(%1952: f32, %1953: f32, %1954: f32):
      %1955 = arith.divf %1952, %1953 : f32
      linalg.yield %1955 : f32
    } -> tensor<1x4x196x196xf32>
    %1956 = tensor.empty() : tensor<1x4x196x196xf32>
    %1957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1951 : tensor<1x4x196x196xf32>) outs(%1956 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb152(%1958: f32, %1959: f32):
      linalg.yield %1958 : f32
    } -> tensor<1x4x196x196xf32>
    %1960 = tensor.collapse_shape %1957 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %1961 = tensor.expand_shape %1960 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %1962 = tensor.empty() : tensor<1x4x196x64xf32>
    %1963 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1890 : tensor<1x4x196x64xf32>) outs(%1962 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb153(%1964: f32, %1965: f32):
      linalg.yield %1964 : f32
    } -> tensor<1x4x196x64xf32>
    %1966 = tensor.collapse_shape %1963 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %1967 = tensor.expand_shape %1966 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %1968 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} 0.000000e+00 : f32
    %1969 = tensor.splat %1968 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<4x196x64xf32>
    %1970 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1961, %1967 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%1969 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} {
    ^bb154(%1971: f32, %1972: f32, %1973: f32):
      %1974 = arith.mulf %1971, %1972 : f32
      %1975 = arith.addf %1973, %1974 : f32
      linalg.yield %1975 : f32
    } -> tensor<4x196x64xf32>
    %1976 = tensor.collapse_shape %1970 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %1977 = tensor.expand_shape %1976 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %1978 = tensor.empty() : tensor<1x196x4x64xf32>
    %1979 = linalg.transpose ins(%1977:tensor<1x4x196x64xf32>) outs(%1978:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %1980 = tensor.collapse_shape %1979 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %1981 = tensor.expand_shape %1980 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1982 = tensor.collapse_shape %1981 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %1983 = tensor.expand_shape %1982 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %1984 = tensor.empty() : tensor<256x256xf32>
    %1985 = linalg.transpose ins(%49:tensor<256x256xf32>) outs(%1984:tensor<256x256xf32>) permutation = [1, 0]
    %1986 = tensor.empty() : tensor<196x256xf32>
    %1987 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1988 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1987 : f32) outs(%1986 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1989 = linalg.matmul {prov.region_id = "matmul_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj", prov.transposed_b = "true"} ins(%1983, %1985 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%1988 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %1990 = tensor.empty() : tensor<196x256xf32>
    %1991 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1989, %50 : tensor<196x256xf32>, tensor<256xf32>) outs(%1990 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj"} {
    ^bb155(%1992: f32, %1993: f32, %1994: f32):
      %1995 = arith.addf %1992, %1993 : f32
      linalg.yield %1995 : f32
    } -> tensor<196x256xf32>
    %1996 = tensor.collapse_shape %1991 [[0 : i64, 1 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %1997 = tensor.expand_shape %1996 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %1998 = tensor.empty() : tensor<1x196x256xf32>
    %1999 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1782, %1997 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%1998 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4"} {
    ^bb156(%2000: f32, %2001: f32, %2002: f32):
      %2003 = arith.addf %2000, %2001 : f32
      linalg.yield %2003 : f32
    } -> tensor<1x196x256xf32>
    %2004 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} 0.000000e+00 : f32
    %2005 = tensor.splat %2004 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32>
    %2006 = linalg.reduce ins(%1999:tensor<1x196x256xf32>) outs(%2005:tensor<1x196xf32>) dimensions = [2]
    (%2007: f32, %2008: f32) {
      %2009 = arith.addf %2007, %2008 : f32
      linalg.yield %2009 : f32
    }
    %2010 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} 2.560000e+02 : f32
    %2011 = tensor.splat %2010 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32>
    %2012 = tensor.empty() : tensor<1x196xf32>
    %2013 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2006, %2011 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2012 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb157(%2014: f32, %2015: f32, %2016: f32):
      %2017 = arith.divf %2014, %2015 : f32
      linalg.yield %2017 : f32
    } -> tensor<1x196xf32>
    %2018 = tensor.collapse_shape %2013 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2019 = tensor.expand_shape %2018 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2020 = tensor.empty() : tensor<1x196x256xf32>
    %2021 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1999, %2019 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2020 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb158(%2022: f32, %2023: f32, %2024: f32):
      %2025 = arith.subf %2022, %2023 : f32
      linalg.yield %2025 : f32
    } -> tensor<1x196x256xf32>
    %2026 = tensor.empty() : tensor<1x196x256xf32>
    %2027 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2021, %2021 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2026 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb159(%2028: f32, %2029: f32, %2030: f32):
      %2031 = arith.mulf %2028, %2029 : f32
      linalg.yield %2031 : f32
    } -> tensor<1x196x256xf32>
    %2032 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} 0.000000e+00 : f32
    %2033 = tensor.splat %2032 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32>
    %2034 = linalg.reduce ins(%2027:tensor<1x196x256xf32>) outs(%2033:tensor<1x196xf32>) dimensions = [2]
    (%2035: f32, %2036: f32) {
      %2037 = arith.addf %2035, %2036 : f32
      linalg.yield %2037 : f32
    }
    %2038 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} 2.560000e+02 : f32
    %2039 = tensor.splat %2038 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32>
    %2040 = tensor.empty() : tensor<1x196xf32>
    %2041 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2034, %2039 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2040 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb160(%2042: f32, %2043: f32, %2044: f32):
      %2045 = arith.divf %2042, %2043 : f32
      linalg.yield %2045 : f32
    } -> tensor<1x196xf32>
    %2046 = tensor.collapse_shape %2041 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2047 = tensor.expand_shape %2046 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2048 = arith.constant {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} 1.000000e-06 : f32
    %2049 = tensor.splat %2048 {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} : tensor<1x196x1xf32>
    %2050 = tensor.empty() : tensor<1x196x1xf32>
    %2051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2047, %2049 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2050 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb161(%2052: f32, %2053: f32, %2054: f32):
      %2055 = arith.addf %2052, %2053 : f32
      linalg.yield %2055 : f32
    } -> tensor<1x196x1xf32>
    %2056 = tensor.empty() : tensor<1x196x1xf32>
    %2057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2051 : tensor<1x196x1xf32>) outs(%2056 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb162(%2058: f32, %2059: f32):
      %2060 = math.rsqrt %2058 : f32
      linalg.yield %2060 : f32
    } -> tensor<1x196x1xf32>
    %2061 = tensor.empty() : tensor<1x196x256xf32>
    %2062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2021, %2057 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2061 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb163(%2063: f32, %2064: f32, %2065: f32):
      %2066 = arith.mulf %2063, %2064 : f32
      linalg.yield %2066 : f32
    } -> tensor<1x196x256xf32>
    %2067 = tensor.empty() : tensor<1x196x256xf32>
    %2068 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2062, %41 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2067 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb164(%2069: f32, %2070: f32, %2071: f32):
      %2072 = arith.mulf %2069, %2070 : f32
      linalg.yield %2072 : f32
    } -> tensor<1x196x256xf32>
    %2073 = tensor.empty() : tensor<1x196x256xf32>
    %2074 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2068, %42 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2073 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_9", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.norm2"} {
    ^bb165(%2075: f32, %2076: f32, %2077: f32):
      %2078 = arith.addf %2075, %2076 : f32
      linalg.yield %2078 : f32
    } -> tensor<1x196x256xf32>
    %2079 = tensor.collapse_shape %2074 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2080 = tensor.expand_shape %2079 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %2081 = tensor.empty() : tensor<256x1024xf32>
    %2082 = linalg.transpose ins(%43:tensor<1024x256xf32>) outs(%2081:tensor<256x1024xf32>) permutation = [1, 0]
    %2083 = tensor.empty() : tensor<196x1024xf32>
    %2084 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2085 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2084 : f32) outs(%2083 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2086 = linalg.matmul {prov.region_id = "matmul_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1", prov.transposed_b = "true"} ins(%2080, %2082 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%2085 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2087 = tensor.empty() : tensor<196x1024xf32>
    %2088 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2086, %44 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%2087 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1"} {
    ^bb166(%2089: f32, %2090: f32, %2091: f32):
      %2092 = arith.addf %2089, %2090 : f32
      linalg.yield %2092 : f32
    } -> tensor<196x1024xf32>
    %2093 = tensor.collapse_shape %2088 [[0 : i64, 1 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %2094 = tensor.expand_shape %2093 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %2095 = tensor.empty() : tensor<1x196x1024xf32>
    %2096 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2094 : tensor<1x196x1024xf32>) outs(%2095 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_4", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.act"} {
    ^bb167(%2097: f32, %2098: f32):
      %2099 = arith.constant 5.000000e-01 : f32
      %2100 = arith.constant 1.000000e+00 : f32
      %2101 = arith.constant 0.707106769 : f32
      %2102 = arith.mulf %2097, %2101 : f32
      %2103 = math.erf %2102 : f32
      %2104 = arith.addf %2100, %2103 : f32
      %2105 = arith.mulf %2099, %2097 : f32
      %2106 = arith.mulf %2105, %2104 : f32
      linalg.yield %2106 : f32
    } -> tensor<1x196x1024xf32>
    %2107 = tensor.collapse_shape %2096 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %2108 = tensor.expand_shape %2107 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %2109 = tensor.empty() : tensor<1024x256xf32>
    %2110 = linalg.transpose ins(%45:tensor<256x1024xf32>) outs(%2109:tensor<1024x256xf32>) permutation = [1, 0]
    %2111 = tensor.empty() : tensor<196x256xf32>
    %2112 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2113 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2112 : f32) outs(%2111 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2114 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2", prov.transposed_b = "true"} ins(%2108, %2110 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%2113 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2115 = tensor.empty() : tensor<196x256xf32>
    %2116 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2114, %46 : tensor<196x256xf32>, tensor<256xf32>) outs(%2115 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2"} {
    ^bb168(%2117: f32, %2118: f32, %2119: f32):
      %2120 = arith.addf %2117, %2118 : f32
      linalg.yield %2120 : f32
    } -> tensor<196x256xf32>
    %2121 = tensor.collapse_shape %2116 [[0 : i64, 1 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %2122 = tensor.expand_shape %2121 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2123 = tensor.empty() : tensor<1x196x256xf32>
    %2124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1999, %2122 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2123 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.4"} {
    ^bb169(%2125: f32, %2126: f32, %2127: f32):
      %2128 = arith.addf %2125, %2126 : f32
      linalg.yield %2128 : f32
    } -> tensor<1x196x256xf32>
    %2129 = arith.constant {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} 0.000000e+00 : f32
    %2130 = tensor.splat %2129 {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32>
    %2131 = linalg.reduce ins(%2124:tensor<1x196x256xf32>) outs(%2130:tensor<1x196xf32>) dimensions = [2]
    (%2132: f32, %2133: f32) {
      %2134 = arith.addf %2132, %2133 : f32
      linalg.yield %2134 : f32
    }
    %2135 = arith.constant {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} 2.560000e+02 : f32
    %2136 = tensor.splat %2135 {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32>
    %2137 = tensor.empty() : tensor<1x196xf32>
    %2138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2131, %2136 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2137 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb170(%2139: f32, %2140: f32, %2141: f32):
      %2142 = arith.divf %2139, %2140 : f32
      linalg.yield %2142 : f32
    } -> tensor<1x196xf32>
    %2143 = tensor.collapse_shape %2138 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2144 = tensor.expand_shape %2143 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2145 = tensor.empty() : tensor<1x196x256xf32>
    %2146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2124, %2144 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2145 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb171(%2147: f32, %2148: f32, %2149: f32):
      %2150 = arith.subf %2147, %2148 : f32
      linalg.yield %2150 : f32
    } -> tensor<1x196x256xf32>
    %2151 = tensor.empty() : tensor<1x196x256xf32>
    %2152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2146, %2146 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2151 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb172(%2153: f32, %2154: f32, %2155: f32):
      %2156 = arith.mulf %2153, %2154 : f32
      linalg.yield %2156 : f32
    } -> tensor<1x196x256xf32>
    %2157 = arith.constant {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} 0.000000e+00 : f32
    %2158 = tensor.splat %2157 {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32>
    %2159 = linalg.reduce ins(%2152:tensor<1x196x256xf32>) outs(%2158:tensor<1x196xf32>) dimensions = [2]
    (%2160: f32, %2161: f32) {
      %2162 = arith.addf %2160, %2161 : f32
      linalg.yield %2162 : f32
    }
    %2163 = arith.constant {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} 2.560000e+02 : f32
    %2164 = tensor.splat %2163 {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32>
    %2165 = tensor.empty() : tensor<1x196xf32>
    %2166 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2159, %2164 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2165 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb173(%2167: f32, %2168: f32, %2169: f32):
      %2170 = arith.divf %2167, %2168 : f32
      linalg.yield %2170 : f32
    } -> tensor<1x196xf32>
    %2171 = tensor.collapse_shape %2166 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2172 = tensor.expand_shape %2171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2173 = arith.constant {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} 1.000000e-06 : f32
    %2174 = tensor.splat %2173 {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} : tensor<1x196x1xf32>
    %2175 = tensor.empty() : tensor<1x196x1xf32>
    %2176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2172, %2174 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2175 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb174(%2177: f32, %2178: f32, %2179: f32):
      %2180 = arith.addf %2177, %2178 : f32
      linalg.yield %2180 : f32
    } -> tensor<1x196x1xf32>
    %2181 = tensor.empty() : tensor<1x196x1xf32>
    %2182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2176 : tensor<1x196x1xf32>) outs(%2181 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb175(%2183: f32, %2184: f32):
      %2185 = math.rsqrt %2183 : f32
      linalg.yield %2185 : f32
    } -> tensor<1x196x1xf32>
    %2186 = tensor.empty() : tensor<1x196x256xf32>
    %2187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2146, %2182 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2186 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb176(%2188: f32, %2189: f32, %2190: f32):
      %2191 = arith.mulf %2188, %2189 : f32
      linalg.yield %2191 : f32
    } -> tensor<1x196x256xf32>
    %2192 = tensor.empty() : tensor<1x196x256xf32>
    %2193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2187, %51 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2192 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb177(%2194: f32, %2195: f32, %2196: f32):
      %2197 = arith.mulf %2194, %2195 : f32
      linalg.yield %2197 : f32
    } -> tensor<1x196x256xf32>
    %2198 = tensor.empty() : tensor<1x196x256xf32>
    %2199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2193, %52 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2198 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_10", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm1"} {
    ^bb178(%2200: f32, %2201: f32, %2202: f32):
      %2203 = arith.addf %2200, %2201 : f32
      linalg.yield %2203 : f32
    } -> tensor<1x196x256xf32>
    %2204 = tensor.collapse_shape %2199 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2205 = tensor.expand_shape %2204 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %2206 = tensor.empty() : tensor<256x768xf32>
    %2207 = linalg.transpose ins(%59:tensor<768x256xf32>) outs(%2206:tensor<256x768xf32>) permutation = [1, 0]
    %2208 = tensor.empty() : tensor<196x768xf32>
    %2209 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2210 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2209 : f32) outs(%2208 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2211 = linalg.matmul {prov.region_id = "matmul_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv", prov.transposed_b = "true"} ins(%2205, %2207 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%2210 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2212 = tensor.empty() : tensor<196x768xf32>
    %2213 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2211, %60 : tensor<196x768xf32>, tensor<768xf32>) outs(%2212 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv"} {
    ^bb179(%2214: f32, %2215: f32, %2216: f32):
      %2217 = arith.addf %2214, %2215 : f32
      linalg.yield %2217 : f32
    } -> tensor<196x768xf32>
    %2218 = tensor.collapse_shape %2213 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %2219 = tensor.expand_shape %2218 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %2220 = tensor.collapse_shape %2219 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %2221 = tensor.expand_shape %2220 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %2222 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %2223 = linalg.transpose ins(%2221:tensor<1x196x3x4x64xf32>) outs(%2222:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %2224 = "tensor.extract_slice"(%2223) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2225 = tensor.collapse_shape %2224 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2226 = tensor.expand_shape %2225 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2227 = "tensor.extract_slice"(%2223) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2228 = tensor.collapse_shape %2227 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2229 = tensor.expand_shape %2228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2230 = "tensor.extract_slice"(%2223) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2231 = tensor.collapse_shape %2230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2232 = tensor.expand_shape %2231 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2233 = tensor.empty() : tensor<1x4x64x196xf32>
    %2234 = linalg.transpose ins(%2229:tensor<1x4x196x64xf32>) outs(%2233:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %2235 = tensor.empty() : tensor<1x4x196x64xf32>
    %2236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2226 : tensor<1x4x196x64xf32>) outs(%2235 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb180(%2237: f32, %2238: f32):
      linalg.yield %2237 : f32
    } -> tensor<1x4x196x64xf32>
    %2239 = tensor.collapse_shape %2236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2240 = tensor.expand_shape %2239 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2241 = tensor.empty() : tensor<1x4x64x196xf32>
    %2242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2234 : tensor<1x4x64x196xf32>) outs(%2241 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb181(%2243: f32, %2244: f32):
      linalg.yield %2243 : f32
    } -> tensor<1x4x64x196xf32>
    %2245 = tensor.collapse_shape %2242 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %2246 = tensor.expand_shape %2245 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %2247 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} 0.000000e+00 : f32
    %2248 = tensor.splat %2247 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<4x196x196xf32>
    %2249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2240, %2246 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%2248 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb182(%2250: f32, %2251: f32, %2252: f32):
      %2253 = arith.mulf %2250, %2251 : f32
      %2254 = arith.addf %2252, %2253 : f32
      linalg.yield %2254 : f32
    } -> tensor<4x196x196xf32>
    %2255 = tensor.collapse_shape %2249 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %2256 = tensor.expand_shape %2255 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %2257 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} 1.250000e-01 : f32
    %2258 = tensor.splat %2257 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196x196xf32>
    %2259 = tensor.empty() : tensor<1x4x196x196xf32>
    %2260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2256, %2258 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%2259 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb183(%2261: f32, %2262: f32, %2263: f32):
      %2264 = arith.mulf %2261, %2262 : f32
      linalg.yield %2264 : f32
    } -> tensor<1x4x196x196xf32>
    %2265 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} 0xff800000 : f32
    %2266 = tensor.splat %2265 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196xf32>
    %2267 = linalg.reduce ins(%2260:tensor<1x4x196x196xf32>) outs(%2266:tensor<1x4x196xf32>) dimensions = [3]
    (%2268: f32, %2269: f32) {
      %2270 = arith.maximumf %2268, %2269 : f32
      linalg.yield %2270 : f32
    }
    %2271 = tensor.collapse_shape %2267 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2272 = tensor.expand_shape %2271 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2273 = tensor.empty() : tensor<1x4x196x196xf32>
    %2274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2260, %2272 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2273 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb184(%2275: f32, %2276: f32, %2277: f32):
      %2278 = arith.subf %2275, %2276 : f32
      linalg.yield %2278 : f32
    } -> tensor<1x4x196x196xf32>
    %2279 = tensor.empty() : tensor<1x4x196x196xf32>
    %2280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2274 : tensor<1x4x196x196xf32>) outs(%2279 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb185(%2281: f32, %2282: f32):
      %2283 = math.exp %2281 : f32
      linalg.yield %2283 : f32
    } -> tensor<1x4x196x196xf32>
    %2284 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} 0.000000e+00 : f32
    %2285 = tensor.splat %2284 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196xf32>
    %2286 = linalg.reduce ins(%2280:tensor<1x4x196x196xf32>) outs(%2285:tensor<1x4x196xf32>) dimensions = [3]
    (%2287: f32, %2288: f32) {
      %2289 = arith.addf %2287, %2288 : f32
      linalg.yield %2289 : f32
    }
    %2290 = tensor.collapse_shape %2286 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2291 = tensor.expand_shape %2290 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2292 = tensor.empty() : tensor<1x4x196x196xf32>
    %2293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2280, %2291 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2292 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb186(%2294: f32, %2295: f32, %2296: f32):
      %2297 = arith.divf %2294, %2295 : f32
      linalg.yield %2297 : f32
    } -> tensor<1x4x196x196xf32>
    %2298 = tensor.empty() : tensor<1x4x196x196xf32>
    %2299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2293 : tensor<1x4x196x196xf32>) outs(%2298 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb187(%2300: f32, %2301: f32):
      linalg.yield %2300 : f32
    } -> tensor<1x4x196x196xf32>
    %2302 = tensor.collapse_shape %2299 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %2303 = tensor.expand_shape %2302 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %2304 = tensor.empty() : tensor<1x4x196x64xf32>
    %2305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2232 : tensor<1x4x196x64xf32>) outs(%2304 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb188(%2306: f32, %2307: f32):
      linalg.yield %2306 : f32
    } -> tensor<1x4x196x64xf32>
    %2308 = tensor.collapse_shape %2305 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2309 = tensor.expand_shape %2308 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2310 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} 0.000000e+00 : f32
    %2311 = tensor.splat %2310 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<4x196x64xf32>
    %2312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2303, %2309 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%2311 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} {
    ^bb189(%2313: f32, %2314: f32, %2315: f32):
      %2316 = arith.mulf %2313, %2314 : f32
      %2317 = arith.addf %2315, %2316 : f32
      linalg.yield %2317 : f32
    } -> tensor<4x196x64xf32>
    %2318 = tensor.collapse_shape %2312 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %2319 = tensor.expand_shape %2318 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2320 = tensor.empty() : tensor<1x196x4x64xf32>
    %2321 = linalg.transpose ins(%2319:tensor<1x4x196x64xf32>) outs(%2320:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %2322 = tensor.collapse_shape %2321 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %2323 = tensor.expand_shape %2322 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2324 = tensor.collapse_shape %2323 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2325 = tensor.expand_shape %2324 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %2326 = tensor.empty() : tensor<256x256xf32>
    %2327 = linalg.transpose ins(%61:tensor<256x256xf32>) outs(%2326:tensor<256x256xf32>) permutation = [1, 0]
    %2328 = tensor.empty() : tensor<196x256xf32>
    %2329 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2330 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2329 : f32) outs(%2328 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2331 = linalg.matmul {prov.region_id = "matmul_17", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj", prov.transposed_b = "true"} ins(%2325, %2327 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%2330 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2332 = tensor.empty() : tensor<196x256xf32>
    %2333 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2331, %62 : tensor<196x256xf32>, tensor<256xf32>) outs(%2332 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj"} {
    ^bb190(%2334: f32, %2335: f32, %2336: f32):
      %2337 = arith.addf %2334, %2335 : f32
      linalg.yield %2337 : f32
    } -> tensor<196x256xf32>
    %2338 = tensor.collapse_shape %2333 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %2339 = tensor.expand_shape %2338 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2340 = tensor.empty() : tensor<1x196x256xf32>
    %2341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2124, %2339 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2340 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5"} {
    ^bb191(%2342: f32, %2343: f32, %2344: f32):
      %2345 = arith.addf %2342, %2343 : f32
      linalg.yield %2345 : f32
    } -> tensor<1x196x256xf32>
    %2346 = arith.constant {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} 0.000000e+00 : f32
    %2347 = tensor.splat %2346 {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32>
    %2348 = linalg.reduce ins(%2341:tensor<1x196x256xf32>) outs(%2347:tensor<1x196xf32>) dimensions = [2]
    (%2349: f32, %2350: f32) {
      %2351 = arith.addf %2349, %2350 : f32
      linalg.yield %2351 : f32
    }
    %2352 = arith.constant {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} 2.560000e+02 : f32
    %2353 = tensor.splat %2352 {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32>
    %2354 = tensor.empty() : tensor<1x196xf32>
    %2355 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2348, %2353 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2354 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb192(%2356: f32, %2357: f32, %2358: f32):
      %2359 = arith.divf %2356, %2357 : f32
      linalg.yield %2359 : f32
    } -> tensor<1x196xf32>
    %2360 = tensor.collapse_shape %2355 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2361 = tensor.expand_shape %2360 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2362 = tensor.empty() : tensor<1x196x256xf32>
    %2363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2341, %2361 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2362 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb193(%2364: f32, %2365: f32, %2366: f32):
      %2367 = arith.subf %2364, %2365 : f32
      linalg.yield %2367 : f32
    } -> tensor<1x196x256xf32>
    %2368 = tensor.empty() : tensor<1x196x256xf32>
    %2369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2363, %2363 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2368 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb194(%2370: f32, %2371: f32, %2372: f32):
      %2373 = arith.mulf %2370, %2371 : f32
      linalg.yield %2373 : f32
    } -> tensor<1x196x256xf32>
    %2374 = arith.constant {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} 0.000000e+00 : f32
    %2375 = tensor.splat %2374 {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32>
    %2376 = linalg.reduce ins(%2369:tensor<1x196x256xf32>) outs(%2375:tensor<1x196xf32>) dimensions = [2]
    (%2377: f32, %2378: f32) {
      %2379 = arith.addf %2377, %2378 : f32
      linalg.yield %2379 : f32
    }
    %2380 = arith.constant {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} 2.560000e+02 : f32
    %2381 = tensor.splat %2380 {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32>
    %2382 = tensor.empty() : tensor<1x196xf32>
    %2383 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2376, %2381 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2382 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb195(%2384: f32, %2385: f32, %2386: f32):
      %2387 = arith.divf %2384, %2385 : f32
      linalg.yield %2387 : f32
    } -> tensor<1x196xf32>
    %2388 = tensor.collapse_shape %2383 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2389 = tensor.expand_shape %2388 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2390 = arith.constant {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} 1.000000e-06 : f32
    %2391 = tensor.splat %2390 {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} : tensor<1x196x1xf32>
    %2392 = tensor.empty() : tensor<1x196x1xf32>
    %2393 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2389, %2391 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2392 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb196(%2394: f32, %2395: f32, %2396: f32):
      %2397 = arith.addf %2394, %2395 : f32
      linalg.yield %2397 : f32
    } -> tensor<1x196x1xf32>
    %2398 = tensor.empty() : tensor<1x196x1xf32>
    %2399 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2393 : tensor<1x196x1xf32>) outs(%2398 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb197(%2400: f32, %2401: f32):
      %2402 = math.rsqrt %2400 : f32
      linalg.yield %2402 : f32
    } -> tensor<1x196x1xf32>
    %2403 = tensor.empty() : tensor<1x196x256xf32>
    %2404 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2363, %2399 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2403 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb198(%2405: f32, %2406: f32, %2407: f32):
      %2408 = arith.mulf %2405, %2406 : f32
      linalg.yield %2408 : f32
    } -> tensor<1x196x256xf32>
    %2409 = tensor.empty() : tensor<1x196x256xf32>
    %2410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2404, %53 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2409 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb199(%2411: f32, %2412: f32, %2413: f32):
      %2414 = arith.mulf %2411, %2412 : f32
      linalg.yield %2414 : f32
    } -> tensor<1x196x256xf32>
    %2415 = tensor.empty() : tensor<1x196x256xf32>
    %2416 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2410, %54 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2415 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_11", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.norm2"} {
    ^bb200(%2417: f32, %2418: f32, %2419: f32):
      %2420 = arith.addf %2417, %2418 : f32
      linalg.yield %2420 : f32
    } -> tensor<1x196x256xf32>
    %2421 = tensor.collapse_shape %2416 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2422 = tensor.expand_shape %2421 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %2423 = tensor.empty() : tensor<256x1024xf32>
    %2424 = linalg.transpose ins(%55:tensor<1024x256xf32>) outs(%2423:tensor<256x1024xf32>) permutation = [1, 0]
    %2425 = tensor.empty() : tensor<196x1024xf32>
    %2426 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2427 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2426 : f32) outs(%2425 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2428 = linalg.matmul {prov.region_id = "matmul_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1", prov.transposed_b = "true"} ins(%2422, %2424 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%2427 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2429 = tensor.empty() : tensor<196x1024xf32>
    %2430 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2428, %56 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%2429 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1"} {
    ^bb201(%2431: f32, %2432: f32, %2433: f32):
      %2434 = arith.addf %2431, %2432 : f32
      linalg.yield %2434 : f32
    } -> tensor<196x1024xf32>
    %2435 = tensor.collapse_shape %2430 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %2436 = tensor.expand_shape %2435 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %2437 = tensor.empty() : tensor<1x196x1024xf32>
    %2438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2436 : tensor<1x196x1024xf32>) outs(%2437 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_5", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.act"} {
    ^bb202(%2439: f32, %2440: f32):
      %2441 = arith.constant 5.000000e-01 : f32
      %2442 = arith.constant 1.000000e+00 : f32
      %2443 = arith.constant 0.707106769 : f32
      %2444 = arith.mulf %2439, %2443 : f32
      %2445 = math.erf %2444 : f32
      %2446 = arith.addf %2442, %2445 : f32
      %2447 = arith.mulf %2441, %2439 : f32
      %2448 = arith.mulf %2447, %2446 : f32
      linalg.yield %2448 : f32
    } -> tensor<1x196x1024xf32>
    %2449 = tensor.collapse_shape %2438 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %2450 = tensor.expand_shape %2449 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %2451 = tensor.empty() : tensor<1024x256xf32>
    %2452 = linalg.transpose ins(%57:tensor<256x1024xf32>) outs(%2451:tensor<1024x256xf32>) permutation = [1, 0]
    %2453 = tensor.empty() : tensor<196x256xf32>
    %2454 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2455 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2454 : f32) outs(%2453 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2456 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2", prov.transposed_b = "true"} ins(%2450, %2452 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%2455 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2457 = tensor.empty() : tensor<196x256xf32>
    %2458 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2456, %58 : tensor<196x256xf32>, tensor<256xf32>) outs(%2457 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2"} {
    ^bb203(%2459: f32, %2460: f32, %2461: f32):
      %2462 = arith.addf %2459, %2460 : f32
      linalg.yield %2462 : f32
    } -> tensor<196x256xf32>
    %2463 = tensor.collapse_shape %2458 [[0 : i64, 1 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %2464 = tensor.expand_shape %2463 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2465 = tensor.empty() : tensor<1x196x256xf32>
    %2466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2341, %2464 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2465 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.5"} {
    ^bb204(%2467: f32, %2468: f32, %2469: f32):
      %2470 = arith.addf %2467, %2468 : f32
      linalg.yield %2470 : f32
    } -> tensor<1x196x256xf32>
    %2471 = arith.constant {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} 0.000000e+00 : f32
    %2472 = tensor.splat %2471 {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32>
    %2473 = linalg.reduce ins(%2466:tensor<1x196x256xf32>) outs(%2472:tensor<1x196xf32>) dimensions = [2]
    (%2474: f32, %2475: f32) {
      %2476 = arith.addf %2474, %2475 : f32
      linalg.yield %2476 : f32
    }
    %2477 = arith.constant {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} 2.560000e+02 : f32
    %2478 = tensor.splat %2477 {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32>
    %2479 = tensor.empty() : tensor<1x196xf32>
    %2480 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2473, %2478 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2479 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb205(%2481: f32, %2482: f32, %2483: f32):
      %2484 = arith.divf %2481, %2482 : f32
      linalg.yield %2484 : f32
    } -> tensor<1x196xf32>
    %2485 = tensor.collapse_shape %2480 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2486 = tensor.expand_shape %2485 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2487 = tensor.empty() : tensor<1x196x256xf32>
    %2488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2466, %2486 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2487 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb206(%2489: f32, %2490: f32, %2491: f32):
      %2492 = arith.subf %2489, %2490 : f32
      linalg.yield %2492 : f32
    } -> tensor<1x196x256xf32>
    %2493 = tensor.empty() : tensor<1x196x256xf32>
    %2494 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2488, %2488 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2493 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb207(%2495: f32, %2496: f32, %2497: f32):
      %2498 = arith.mulf %2495, %2496 : f32
      linalg.yield %2498 : f32
    } -> tensor<1x196x256xf32>
    %2499 = arith.constant {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} 0.000000e+00 : f32
    %2500 = tensor.splat %2499 {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32>
    %2501 = linalg.reduce ins(%2494:tensor<1x196x256xf32>) outs(%2500:tensor<1x196xf32>) dimensions = [2]
    (%2502: f32, %2503: f32) {
      %2504 = arith.addf %2502, %2503 : f32
      linalg.yield %2504 : f32
    }
    %2505 = arith.constant {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} 2.560000e+02 : f32
    %2506 = tensor.splat %2505 {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32>
    %2507 = tensor.empty() : tensor<1x196xf32>
    %2508 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2501, %2506 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2507 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb208(%2509: f32, %2510: f32, %2511: f32):
      %2512 = arith.divf %2509, %2510 : f32
      linalg.yield %2512 : f32
    } -> tensor<1x196xf32>
    %2513 = tensor.collapse_shape %2508 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2514 = tensor.expand_shape %2513 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2515 = arith.constant {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} 1.000000e-06 : f32
    %2516 = tensor.splat %2515 {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} : tensor<1x196x1xf32>
    %2517 = tensor.empty() : tensor<1x196x1xf32>
    %2518 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2514, %2516 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2517 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb209(%2519: f32, %2520: f32, %2521: f32):
      %2522 = arith.addf %2519, %2520 : f32
      linalg.yield %2522 : f32
    } -> tensor<1x196x1xf32>
    %2523 = tensor.empty() : tensor<1x196x1xf32>
    %2524 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2518 : tensor<1x196x1xf32>) outs(%2523 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb210(%2525: f32, %2526: f32):
      %2527 = math.rsqrt %2525 : f32
      linalg.yield %2527 : f32
    } -> tensor<1x196x1xf32>
    %2528 = tensor.empty() : tensor<1x196x256xf32>
    %2529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2488, %2524 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2528 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb211(%2530: f32, %2531: f32, %2532: f32):
      %2533 = arith.mulf %2530, %2531 : f32
      linalg.yield %2533 : f32
    } -> tensor<1x196x256xf32>
    %2534 = tensor.empty() : tensor<1x196x256xf32>
    %2535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2529, %63 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2534 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb212(%2536: f32, %2537: f32, %2538: f32):
      %2539 = arith.mulf %2536, %2537 : f32
      linalg.yield %2539 : f32
    } -> tensor<1x196x256xf32>
    %2540 = tensor.empty() : tensor<1x196x256xf32>
    %2541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2535, %64 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2540 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_12", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm1"} {
    ^bb213(%2542: f32, %2543: f32, %2544: f32):
      %2545 = arith.addf %2542, %2543 : f32
      linalg.yield %2545 : f32
    } -> tensor<1x196x256xf32>
    %2546 = tensor.collapse_shape %2541 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2547 = tensor.expand_shape %2546 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %2548 = tensor.empty() : tensor<256x768xf32>
    %2549 = linalg.transpose ins(%71:tensor<768x256xf32>) outs(%2548:tensor<256x768xf32>) permutation = [1, 0]
    %2550 = tensor.empty() : tensor<196x768xf32>
    %2551 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2552 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2551 : f32) outs(%2550 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2553 = linalg.matmul {prov.region_id = "matmul_20", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv", prov.transposed_b = "true"} ins(%2547, %2549 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%2552 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2554 = tensor.empty() : tensor<196x768xf32>
    %2555 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2553, %72 : tensor<196x768xf32>, tensor<768xf32>) outs(%2554 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv"} {
    ^bb214(%2556: f32, %2557: f32, %2558: f32):
      %2559 = arith.addf %2556, %2557 : f32
      linalg.yield %2559 : f32
    } -> tensor<196x768xf32>
    %2560 = tensor.collapse_shape %2555 [[0 : i64, 1 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %2561 = tensor.expand_shape %2560 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %2562 = tensor.collapse_shape %2561 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %2563 = tensor.expand_shape %2562 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %2564 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %2565 = linalg.transpose ins(%2563:tensor<1x196x3x4x64xf32>) outs(%2564:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %2566 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2567 = tensor.collapse_shape %2566 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2568 = tensor.expand_shape %2567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2569 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2570 = tensor.collapse_shape %2569 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2571 = tensor.expand_shape %2570 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2572 = "tensor.extract_slice"(%2565) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2573 = tensor.collapse_shape %2572 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2574 = tensor.expand_shape %2573 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2575 = tensor.empty() : tensor<1x4x64x196xf32>
    %2576 = linalg.transpose ins(%2571:tensor<1x4x196x64xf32>) outs(%2575:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %2577 = tensor.empty() : tensor<1x4x196x64xf32>
    %2578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2568 : tensor<1x4x196x64xf32>) outs(%2577 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb215(%2579: f32, %2580: f32):
      linalg.yield %2579 : f32
    } -> tensor<1x4x196x64xf32>
    %2581 = tensor.collapse_shape %2578 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2582 = tensor.expand_shape %2581 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2583 = tensor.empty() : tensor<1x4x64x196xf32>
    %2584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2576 : tensor<1x4x64x196xf32>) outs(%2583 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb216(%2585: f32, %2586: f32):
      linalg.yield %2585 : f32
    } -> tensor<1x4x64x196xf32>
    %2587 = tensor.collapse_shape %2584 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %2588 = tensor.expand_shape %2587 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %2589 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} 0.000000e+00 : f32
    %2590 = tensor.splat %2589 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<4x196x196xf32>
    %2591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2582, %2588 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%2590 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb217(%2592: f32, %2593: f32, %2594: f32):
      %2595 = arith.mulf %2592, %2593 : f32
      %2596 = arith.addf %2594, %2595 : f32
      linalg.yield %2596 : f32
    } -> tensor<4x196x196xf32>
    %2597 = tensor.collapse_shape %2591 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %2598 = tensor.expand_shape %2597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %2599 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} 1.250000e-01 : f32
    %2600 = tensor.splat %2599 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196x196xf32>
    %2601 = tensor.empty() : tensor<1x4x196x196xf32>
    %2602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2598, %2600 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%2601 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb218(%2603: f32, %2604: f32, %2605: f32):
      %2606 = arith.mulf %2603, %2604 : f32
      linalg.yield %2606 : f32
    } -> tensor<1x4x196x196xf32>
    %2607 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} 0xff800000 : f32
    %2608 = tensor.splat %2607 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196xf32>
    %2609 = linalg.reduce ins(%2602:tensor<1x4x196x196xf32>) outs(%2608:tensor<1x4x196xf32>) dimensions = [3]
    (%2610: f32, %2611: f32) {
      %2612 = arith.maximumf %2610, %2611 : f32
      linalg.yield %2612 : f32
    }
    %2613 = tensor.collapse_shape %2609 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2614 = tensor.expand_shape %2613 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2615 = tensor.empty() : tensor<1x4x196x196xf32>
    %2616 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2602, %2614 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2615 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb219(%2617: f32, %2618: f32, %2619: f32):
      %2620 = arith.subf %2617, %2618 : f32
      linalg.yield %2620 : f32
    } -> tensor<1x4x196x196xf32>
    %2621 = tensor.empty() : tensor<1x4x196x196xf32>
    %2622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2616 : tensor<1x4x196x196xf32>) outs(%2621 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb220(%2623: f32, %2624: f32):
      %2625 = math.exp %2623 : f32
      linalg.yield %2625 : f32
    } -> tensor<1x4x196x196xf32>
    %2626 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} 0.000000e+00 : f32
    %2627 = tensor.splat %2626 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196xf32>
    %2628 = linalg.reduce ins(%2622:tensor<1x4x196x196xf32>) outs(%2627:tensor<1x4x196xf32>) dimensions = [3]
    (%2629: f32, %2630: f32) {
      %2631 = arith.addf %2629, %2630 : f32
      linalg.yield %2631 : f32
    }
    %2632 = tensor.collapse_shape %2628 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2633 = tensor.expand_shape %2632 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2634 = tensor.empty() : tensor<1x4x196x196xf32>
    %2635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2622, %2633 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2634 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb221(%2636: f32, %2637: f32, %2638: f32):
      %2639 = arith.divf %2636, %2637 : f32
      linalg.yield %2639 : f32
    } -> tensor<1x4x196x196xf32>
    %2640 = tensor.empty() : tensor<1x4x196x196xf32>
    %2641 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2635 : tensor<1x4x196x196xf32>) outs(%2640 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb222(%2642: f32, %2643: f32):
      linalg.yield %2642 : f32
    } -> tensor<1x4x196x196xf32>
    %2644 = tensor.collapse_shape %2641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %2645 = tensor.expand_shape %2644 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %2646 = tensor.empty() : tensor<1x4x196x64xf32>
    %2647 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2574 : tensor<1x4x196x64xf32>) outs(%2646 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb223(%2648: f32, %2649: f32):
      linalg.yield %2648 : f32
    } -> tensor<1x4x196x64xf32>
    %2650 = tensor.collapse_shape %2647 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2651 = tensor.expand_shape %2650 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2652 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} 0.000000e+00 : f32
    %2653 = tensor.splat %2652 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<4x196x64xf32>
    %2654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2645, %2651 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%2653 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} {
    ^bb224(%2655: f32, %2656: f32, %2657: f32):
      %2658 = arith.mulf %2655, %2656 : f32
      %2659 = arith.addf %2657, %2658 : f32
      linalg.yield %2659 : f32
    } -> tensor<4x196x64xf32>
    %2660 = tensor.collapse_shape %2654 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %2661 = tensor.expand_shape %2660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2662 = tensor.empty() : tensor<1x196x4x64xf32>
    %2663 = linalg.transpose ins(%2661:tensor<1x4x196x64xf32>) outs(%2662:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %2664 = tensor.collapse_shape %2663 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %2665 = tensor.expand_shape %2664 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2666 = tensor.collapse_shape %2665 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2667 = tensor.expand_shape %2666 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %2668 = tensor.empty() : tensor<256x256xf32>
    %2669 = linalg.transpose ins(%73:tensor<256x256xf32>) outs(%2668:tensor<256x256xf32>) permutation = [1, 0]
    %2670 = tensor.empty() : tensor<196x256xf32>
    %2671 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2672 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2671 : f32) outs(%2670 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2673 = linalg.matmul {prov.region_id = "matmul_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj", prov.transposed_b = "true"} ins(%2667, %2669 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%2672 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2674 = tensor.empty() : tensor<196x256xf32>
    %2675 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2673, %74 : tensor<196x256xf32>, tensor<256xf32>) outs(%2674 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj"} {
    ^bb225(%2676: f32, %2677: f32, %2678: f32):
      %2679 = arith.addf %2676, %2677 : f32
      linalg.yield %2679 : f32
    } -> tensor<196x256xf32>
    %2680 = tensor.collapse_shape %2675 [[0 : i64, 1 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %2681 = tensor.expand_shape %2680 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2682 = tensor.empty() : tensor<1x196x256xf32>
    %2683 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2466, %2681 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2682 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6"} {
    ^bb226(%2684: f32, %2685: f32, %2686: f32):
      %2687 = arith.addf %2684, %2685 : f32
      linalg.yield %2687 : f32
    } -> tensor<1x196x256xf32>
    %2688 = arith.constant {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} 0.000000e+00 : f32
    %2689 = tensor.splat %2688 {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32>
    %2690 = linalg.reduce ins(%2683:tensor<1x196x256xf32>) outs(%2689:tensor<1x196xf32>) dimensions = [2]
    (%2691: f32, %2692: f32) {
      %2693 = arith.addf %2691, %2692 : f32
      linalg.yield %2693 : f32
    }
    %2694 = arith.constant {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} 2.560000e+02 : f32
    %2695 = tensor.splat %2694 {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32>
    %2696 = tensor.empty() : tensor<1x196xf32>
    %2697 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2690, %2695 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2696 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb227(%2698: f32, %2699: f32, %2700: f32):
      %2701 = arith.divf %2698, %2699 : f32
      linalg.yield %2701 : f32
    } -> tensor<1x196xf32>
    %2702 = tensor.collapse_shape %2697 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2703 = tensor.expand_shape %2702 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2704 = tensor.empty() : tensor<1x196x256xf32>
    %2705 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2683, %2703 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2704 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb228(%2706: f32, %2707: f32, %2708: f32):
      %2709 = arith.subf %2706, %2707 : f32
      linalg.yield %2709 : f32
    } -> tensor<1x196x256xf32>
    %2710 = tensor.empty() : tensor<1x196x256xf32>
    %2711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2705, %2705 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2710 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb229(%2712: f32, %2713: f32, %2714: f32):
      %2715 = arith.mulf %2712, %2713 : f32
      linalg.yield %2715 : f32
    } -> tensor<1x196x256xf32>
    %2716 = arith.constant {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} 0.000000e+00 : f32
    %2717 = tensor.splat %2716 {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32>
    %2718 = linalg.reduce ins(%2711:tensor<1x196x256xf32>) outs(%2717:tensor<1x196xf32>) dimensions = [2]
    (%2719: f32, %2720: f32) {
      %2721 = arith.addf %2719, %2720 : f32
      linalg.yield %2721 : f32
    }
    %2722 = arith.constant {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} 2.560000e+02 : f32
    %2723 = tensor.splat %2722 {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32>
    %2724 = tensor.empty() : tensor<1x196xf32>
    %2725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2718, %2723 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2724 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb230(%2726: f32, %2727: f32, %2728: f32):
      %2729 = arith.divf %2726, %2727 : f32
      linalg.yield %2729 : f32
    } -> tensor<1x196xf32>
    %2730 = tensor.collapse_shape %2725 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %2731 = tensor.expand_shape %2730 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2732 = arith.constant {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} 1.000000e-06 : f32
    %2733 = tensor.splat %2732 {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} : tensor<1x196x1xf32>
    %2734 = tensor.empty() : tensor<1x196x1xf32>
    %2735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2731, %2733 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2734 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb231(%2736: f32, %2737: f32, %2738: f32):
      %2739 = arith.addf %2736, %2737 : f32
      linalg.yield %2739 : f32
    } -> tensor<1x196x1xf32>
    %2740 = tensor.empty() : tensor<1x196x1xf32>
    %2741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2735 : tensor<1x196x1xf32>) outs(%2740 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb232(%2742: f32, %2743: f32):
      %2744 = math.rsqrt %2742 : f32
      linalg.yield %2744 : f32
    } -> tensor<1x196x1xf32>
    %2745 = tensor.empty() : tensor<1x196x256xf32>
    %2746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2705, %2741 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2745 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb233(%2747: f32, %2748: f32, %2749: f32):
      %2750 = arith.mulf %2747, %2748 : f32
      linalg.yield %2750 : f32
    } -> tensor<1x196x256xf32>
    %2751 = tensor.empty() : tensor<1x196x256xf32>
    %2752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2746, %65 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2751 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb234(%2753: f32, %2754: f32, %2755: f32):
      %2756 = arith.mulf %2753, %2754 : f32
      linalg.yield %2756 : f32
    } -> tensor<1x196x256xf32>
    %2757 = tensor.empty() : tensor<1x196x256xf32>
    %2758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2752, %66 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2757 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_13", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.norm2"} {
    ^bb235(%2759: f32, %2760: f32, %2761: f32):
      %2762 = arith.addf %2759, %2760 : f32
      linalg.yield %2762 : f32
    } -> tensor<1x196x256xf32>
    %2763 = tensor.collapse_shape %2758 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2764 = tensor.expand_shape %2763 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %2765 = tensor.empty() : tensor<256x1024xf32>
    %2766 = linalg.transpose ins(%67:tensor<1024x256xf32>) outs(%2765:tensor<256x1024xf32>) permutation = [1, 0]
    %2767 = tensor.empty() : tensor<196x1024xf32>
    %2768 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2769 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2768 : f32) outs(%2767 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2770 = linalg.matmul {prov.region_id = "matmul_24", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1", prov.transposed_b = "true"} ins(%2764, %2766 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%2769 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %2771 = tensor.empty() : tensor<196x1024xf32>
    %2772 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2770, %68 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%2771 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1"} {
    ^bb236(%2773: f32, %2774: f32, %2775: f32):
      %2776 = arith.addf %2773, %2774 : f32
      linalg.yield %2776 : f32
    } -> tensor<196x1024xf32>
    %2777 = tensor.collapse_shape %2772 [[0 : i64, 1 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %2778 = tensor.expand_shape %2777 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %2779 = tensor.empty() : tensor<1x196x1024xf32>
    %2780 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2778 : tensor<1x196x1024xf32>) outs(%2779 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_6", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.act"} {
    ^bb237(%2781: f32, %2782: f32):
      %2783 = arith.constant 5.000000e-01 : f32
      %2784 = arith.constant 1.000000e+00 : f32
      %2785 = arith.constant 0.707106769 : f32
      %2786 = arith.mulf %2781, %2785 : f32
      %2787 = math.erf %2786 : f32
      %2788 = arith.addf %2784, %2787 : f32
      %2789 = arith.mulf %2783, %2781 : f32
      %2790 = arith.mulf %2789, %2788 : f32
      linalg.yield %2790 : f32
    } -> tensor<1x196x1024xf32>
    %2791 = tensor.collapse_shape %2780 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %2792 = tensor.expand_shape %2791 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %2793 = tensor.empty() : tensor<1024x256xf32>
    %2794 = linalg.transpose ins(%69:tensor<256x1024xf32>) outs(%2793:tensor<1024x256xf32>) permutation = [1, 0]
    %2795 = tensor.empty() : tensor<196x256xf32>
    %2796 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2797 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2796 : f32) outs(%2795 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2798 = linalg.matmul {prov.region_id = "matmul_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2", prov.transposed_b = "true"} ins(%2792, %2794 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%2797 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %2799 = tensor.empty() : tensor<196x256xf32>
    %2800 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2798, %70 : tensor<196x256xf32>, tensor<256xf32>) outs(%2799 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2"} {
    ^bb238(%2801: f32, %2802: f32, %2803: f32):
      %2804 = arith.addf %2801, %2802 : f32
      linalg.yield %2804 : f32
    } -> tensor<196x256xf32>
    %2805 = tensor.collapse_shape %2800 [[0 : i64, 1 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %2806 = tensor.expand_shape %2805 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %2807 = tensor.empty() : tensor<1x196x256xf32>
    %2808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2683, %2806 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2807 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.6"} {
    ^bb239(%2809: f32, %2810: f32, %2811: f32):
      %2812 = arith.addf %2809, %2810 : f32
      linalg.yield %2812 : f32
    } -> tensor<1x196x256xf32>
    %2813 = arith.constant {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} 0.000000e+00 : f32
    %2814 = tensor.splat %2813 {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32>
    %2815 = linalg.reduce ins(%2808:tensor<1x196x256xf32>) outs(%2814:tensor<1x196xf32>) dimensions = [2]
    (%2816: f32, %2817: f32) {
      %2818 = arith.addf %2816, %2817 : f32
      linalg.yield %2818 : f32
    }
    %2819 = arith.constant {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} 2.560000e+02 : f32
    %2820 = tensor.splat %2819 {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32>
    %2821 = tensor.empty() : tensor<1x196xf32>
    %2822 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2815, %2820 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2821 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb240(%2823: f32, %2824: f32, %2825: f32):
      %2826 = arith.divf %2823, %2824 : f32
      linalg.yield %2826 : f32
    } -> tensor<1x196xf32>
    %2827 = tensor.collapse_shape %2822 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2828 = tensor.expand_shape %2827 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2829 = tensor.empty() : tensor<1x196x256xf32>
    %2830 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2808, %2828 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2829 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb241(%2831: f32, %2832: f32, %2833: f32):
      %2834 = arith.subf %2831, %2832 : f32
      linalg.yield %2834 : f32
    } -> tensor<1x196x256xf32>
    %2835 = tensor.empty() : tensor<1x196x256xf32>
    %2836 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2830, %2830 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%2835 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb242(%2837: f32, %2838: f32, %2839: f32):
      %2840 = arith.mulf %2837, %2838 : f32
      linalg.yield %2840 : f32
    } -> tensor<1x196x256xf32>
    %2841 = arith.constant {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} 0.000000e+00 : f32
    %2842 = tensor.splat %2841 {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32>
    %2843 = linalg.reduce ins(%2836:tensor<1x196x256xf32>) outs(%2842:tensor<1x196xf32>) dimensions = [2]
    (%2844: f32, %2845: f32) {
      %2846 = arith.addf %2844, %2845 : f32
      linalg.yield %2846 : f32
    }
    %2847 = arith.constant {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} 2.560000e+02 : f32
    %2848 = tensor.splat %2847 {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32>
    %2849 = tensor.empty() : tensor<1x196xf32>
    %2850 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2843, %2848 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%2849 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb243(%2851: f32, %2852: f32, %2853: f32):
      %2854 = arith.divf %2851, %2852 : f32
      linalg.yield %2854 : f32
    } -> tensor<1x196xf32>
    %2855 = tensor.collapse_shape %2850 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %2856 = tensor.expand_shape %2855 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %2857 = arith.constant {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} 1.000000e-06 : f32
    %2858 = tensor.splat %2857 {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} : tensor<1x196x1xf32>
    %2859 = tensor.empty() : tensor<1x196x1xf32>
    %2860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2856, %2858 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%2859 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb244(%2861: f32, %2862: f32, %2863: f32):
      %2864 = arith.addf %2861, %2862 : f32
      linalg.yield %2864 : f32
    } -> tensor<1x196x1xf32>
    %2865 = tensor.empty() : tensor<1x196x1xf32>
    %2866 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2860 : tensor<1x196x1xf32>) outs(%2865 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb245(%2867: f32, %2868: f32):
      %2869 = math.rsqrt %2867 : f32
      linalg.yield %2869 : f32
    } -> tensor<1x196x1xf32>
    %2870 = tensor.empty() : tensor<1x196x256xf32>
    %2871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2830, %2866 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%2870 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb246(%2872: f32, %2873: f32, %2874: f32):
      %2875 = arith.mulf %2872, %2873 : f32
      linalg.yield %2875 : f32
    } -> tensor<1x196x256xf32>
    %2876 = tensor.empty() : tensor<1x196x256xf32>
    %2877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2871, %75 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2876 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb247(%2878: f32, %2879: f32, %2880: f32):
      %2881 = arith.mulf %2878, %2879 : f32
      linalg.yield %2881 : f32
    } -> tensor<1x196x256xf32>
    %2882 = tensor.empty() : tensor<1x196x256xf32>
    %2883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2877, %76 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%2882 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_14", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm1"} {
    ^bb248(%2884: f32, %2885: f32, %2886: f32):
      %2887 = arith.addf %2884, %2885 : f32
      linalg.yield %2887 : f32
    } -> tensor<1x196x256xf32>
    %2888 = tensor.collapse_shape %2883 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %2889 = tensor.expand_shape %2888 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %2890 = tensor.empty() : tensor<256x768xf32>
    %2891 = linalg.transpose ins(%83:tensor<768x256xf32>) outs(%2890:tensor<256x768xf32>) permutation = [1, 0]
    %2892 = tensor.empty() : tensor<196x768xf32>
    %2893 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %2894 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2893 : f32) outs(%2892 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2895 = linalg.matmul {prov.region_id = "matmul_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv", prov.transposed_b = "true"} ins(%2889, %2891 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%2894 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %2896 = tensor.empty() : tensor<196x768xf32>
    %2897 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2895, %84 : tensor<196x768xf32>, tensor<768xf32>) outs(%2896 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv"} {
    ^bb249(%2898: f32, %2899: f32, %2900: f32):
      %2901 = arith.addf %2898, %2899 : f32
      linalg.yield %2901 : f32
    } -> tensor<196x768xf32>
    %2902 = tensor.collapse_shape %2897 [[0 : i64, 1 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %2903 = tensor.expand_shape %2902 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %2904 = tensor.collapse_shape %2903 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %2905 = tensor.expand_shape %2904 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %2906 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %2907 = linalg.transpose ins(%2905:tensor<1x196x3x4x64xf32>) outs(%2906:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %2908 = "tensor.extract_slice"(%2907) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2909 = tensor.collapse_shape %2908 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2910 = tensor.expand_shape %2909 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2911 = "tensor.extract_slice"(%2907) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2912 = tensor.collapse_shape %2911 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2913 = tensor.expand_shape %2912 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2914 = "tensor.extract_slice"(%2907) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %2915 = tensor.collapse_shape %2914 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %2916 = tensor.expand_shape %2915 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %2917 = tensor.empty() : tensor<1x4x64x196xf32>
    %2918 = linalg.transpose ins(%2913:tensor<1x4x196x64xf32>) outs(%2917:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %2919 = tensor.empty() : tensor<1x4x196x64xf32>
    %2920 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2910 : tensor<1x4x196x64xf32>) outs(%2919 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb250(%2921: f32, %2922: f32):
      linalg.yield %2921 : f32
    } -> tensor<1x4x196x64xf32>
    %2923 = tensor.collapse_shape %2920 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2924 = tensor.expand_shape %2923 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2925 = tensor.empty() : tensor<1x4x64x196xf32>
    %2926 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2918 : tensor<1x4x64x196xf32>) outs(%2925 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb251(%2927: f32, %2928: f32):
      linalg.yield %2927 : f32
    } -> tensor<1x4x64x196xf32>
    %2929 = tensor.collapse_shape %2926 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %2930 = tensor.expand_shape %2929 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %2931 = arith.constant {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} 0.000000e+00 : f32
    %2932 = tensor.splat %2931 {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<4x196x196xf32>
    %2933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2924, %2930 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%2932 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb252(%2934: f32, %2935: f32, %2936: f32):
      %2937 = arith.mulf %2934, %2935 : f32
      %2938 = arith.addf %2936, %2937 : f32
      linalg.yield %2938 : f32
    } -> tensor<4x196x196xf32>
    %2939 = tensor.collapse_shape %2933 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %2940 = tensor.expand_shape %2939 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %2941 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} 1.250000e-01 : f32
    %2942 = tensor.splat %2941 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196x196xf32>
    %2943 = tensor.empty() : tensor<1x4x196x196xf32>
    %2944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2940, %2942 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%2943 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb253(%2945: f32, %2946: f32, %2947: f32):
      %2948 = arith.mulf %2945, %2946 : f32
      linalg.yield %2948 : f32
    } -> tensor<1x4x196x196xf32>
    %2949 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} 0xff800000 : f32
    %2950 = tensor.splat %2949 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196xf32>
    %2951 = linalg.reduce ins(%2944:tensor<1x4x196x196xf32>) outs(%2950:tensor<1x4x196xf32>) dimensions = [3]
    (%2952: f32, %2953: f32) {
      %2954 = arith.maximumf %2952, %2953 : f32
      linalg.yield %2954 : f32
    }
    %2955 = tensor.collapse_shape %2951 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2956 = tensor.expand_shape %2955 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2957 = tensor.empty() : tensor<1x4x196x196xf32>
    %2958 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2944, %2956 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2957 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb254(%2959: f32, %2960: f32, %2961: f32):
      %2962 = arith.subf %2959, %2960 : f32
      linalg.yield %2962 : f32
    } -> tensor<1x4x196x196xf32>
    %2963 = tensor.empty() : tensor<1x4x196x196xf32>
    %2964 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2958 : tensor<1x4x196x196xf32>) outs(%2963 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb255(%2965: f32, %2966: f32):
      %2967 = math.exp %2965 : f32
      linalg.yield %2967 : f32
    } -> tensor<1x4x196x196xf32>
    %2968 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} 0.000000e+00 : f32
    %2969 = tensor.splat %2968 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196xf32>
    %2970 = linalg.reduce ins(%2964:tensor<1x4x196x196xf32>) outs(%2969:tensor<1x4x196xf32>) dimensions = [3]
    (%2971: f32, %2972: f32) {
      %2973 = arith.addf %2971, %2972 : f32
      linalg.yield %2973 : f32
    }
    %2974 = tensor.collapse_shape %2970 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %2975 = tensor.expand_shape %2974 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %2976 = tensor.empty() : tensor<1x4x196x196xf32>
    %2977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2964, %2975 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%2976 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb256(%2978: f32, %2979: f32, %2980: f32):
      %2981 = arith.divf %2978, %2979 : f32
      linalg.yield %2981 : f32
    } -> tensor<1x4x196x196xf32>
    %2982 = tensor.empty() : tensor<1x4x196x196xf32>
    %2983 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2977 : tensor<1x4x196x196xf32>) outs(%2982 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb257(%2984: f32, %2985: f32):
      linalg.yield %2984 : f32
    } -> tensor<1x4x196x196xf32>
    %2986 = tensor.collapse_shape %2983 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %2987 = tensor.expand_shape %2986 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %2988 = tensor.empty() : tensor<1x4x196x64xf32>
    %2989 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2916 : tensor<1x4x196x64xf32>) outs(%2988 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb258(%2990: f32, %2991: f32):
      linalg.yield %2990 : f32
    } -> tensor<1x4x196x64xf32>
    %2992 = tensor.collapse_shape %2989 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %2993 = tensor.expand_shape %2992 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %2994 = arith.constant {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} 0.000000e+00 : f32
    %2995 = tensor.splat %2994 {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<4x196x64xf32>
    %2996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2987, %2993 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%2995 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} {
    ^bb259(%2997: f32, %2998: f32, %2999: f32):
      %3000 = arith.mulf %2997, %2998 : f32
      %3001 = arith.addf %2999, %3000 : f32
      linalg.yield %3001 : f32
    } -> tensor<4x196x64xf32>
    %3002 = tensor.collapse_shape %2996 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %3003 = tensor.expand_shape %3002 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3004 = tensor.empty() : tensor<1x196x4x64xf32>
    %3005 = linalg.transpose ins(%3003:tensor<1x4x196x64xf32>) outs(%3004:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %3006 = tensor.collapse_shape %3005 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %3007 = tensor.expand_shape %3006 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3008 = tensor.collapse_shape %3007 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3009 = tensor.expand_shape %3008 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %3010 = tensor.empty() : tensor<256x256xf32>
    %3011 = linalg.transpose ins(%85:tensor<256x256xf32>) outs(%3010:tensor<256x256xf32>) permutation = [1, 0]
    %3012 = tensor.empty() : tensor<196x256xf32>
    %3013 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3014 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3013 : f32) outs(%3012 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3015 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj", prov.transposed_b = "true"} ins(%3009, %3011 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%3014 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3016 = tensor.empty() : tensor<196x256xf32>
    %3017 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3015, %86 : tensor<196x256xf32>, tensor<256xf32>) outs(%3016 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj"} {
    ^bb260(%3018: f32, %3019: f32, %3020: f32):
      %3021 = arith.addf %3018, %3019 : f32
      linalg.yield %3021 : f32
    } -> tensor<196x256xf32>
    %3022 = tensor.collapse_shape %3017 [[0 : i64, 1 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %3023 = tensor.expand_shape %3022 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3024 = tensor.empty() : tensor<1x196x256xf32>
    %3025 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2808, %3023 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3024 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7"} {
    ^bb261(%3026: f32, %3027: f32, %3028: f32):
      %3029 = arith.addf %3026, %3027 : f32
      linalg.yield %3029 : f32
    } -> tensor<1x196x256xf32>
    %3030 = arith.constant {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} 0.000000e+00 : f32
    %3031 = tensor.splat %3030 {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32>
    %3032 = linalg.reduce ins(%3025:tensor<1x196x256xf32>) outs(%3031:tensor<1x196xf32>) dimensions = [2]
    (%3033: f32, %3034: f32) {
      %3035 = arith.addf %3033, %3034 : f32
      linalg.yield %3035 : f32
    }
    %3036 = arith.constant {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} 2.560000e+02 : f32
    %3037 = tensor.splat %3036 {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32>
    %3038 = tensor.empty() : tensor<1x196xf32>
    %3039 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3032, %3037 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3038 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb262(%3040: f32, %3041: f32, %3042: f32):
      %3043 = arith.divf %3040, %3041 : f32
      linalg.yield %3043 : f32
    } -> tensor<1x196xf32>
    %3044 = tensor.collapse_shape %3039 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3045 = tensor.expand_shape %3044 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3046 = tensor.empty() : tensor<1x196x256xf32>
    %3047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3025, %3045 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3046 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb263(%3048: f32, %3049: f32, %3050: f32):
      %3051 = arith.subf %3048, %3049 : f32
      linalg.yield %3051 : f32
    } -> tensor<1x196x256xf32>
    %3052 = tensor.empty() : tensor<1x196x256xf32>
    %3053 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3047, %3047 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3052 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb264(%3054: f32, %3055: f32, %3056: f32):
      %3057 = arith.mulf %3054, %3055 : f32
      linalg.yield %3057 : f32
    } -> tensor<1x196x256xf32>
    %3058 = arith.constant {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} 0.000000e+00 : f32
    %3059 = tensor.splat %3058 {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32>
    %3060 = linalg.reduce ins(%3053:tensor<1x196x256xf32>) outs(%3059:tensor<1x196xf32>) dimensions = [2]
    (%3061: f32, %3062: f32) {
      %3063 = arith.addf %3061, %3062 : f32
      linalg.yield %3063 : f32
    }
    %3064 = arith.constant {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} 2.560000e+02 : f32
    %3065 = tensor.splat %3064 {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32>
    %3066 = tensor.empty() : tensor<1x196xf32>
    %3067 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3060, %3065 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3066 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb265(%3068: f32, %3069: f32, %3070: f32):
      %3071 = arith.divf %3068, %3069 : f32
      linalg.yield %3071 : f32
    } -> tensor<1x196xf32>
    %3072 = tensor.collapse_shape %3067 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3073 = tensor.expand_shape %3072 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3074 = arith.constant {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} 1.000000e-06 : f32
    %3075 = tensor.splat %3074 {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} : tensor<1x196x1xf32>
    %3076 = tensor.empty() : tensor<1x196x1xf32>
    %3077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3073, %3075 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3076 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb266(%3078: f32, %3079: f32, %3080: f32):
      %3081 = arith.addf %3078, %3079 : f32
      linalg.yield %3081 : f32
    } -> tensor<1x196x1xf32>
    %3082 = tensor.empty() : tensor<1x196x1xf32>
    %3083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3077 : tensor<1x196x1xf32>) outs(%3082 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb267(%3084: f32, %3085: f32):
      %3086 = math.rsqrt %3084 : f32
      linalg.yield %3086 : f32
    } -> tensor<1x196x1xf32>
    %3087 = tensor.empty() : tensor<1x196x256xf32>
    %3088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3047, %3083 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3087 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb268(%3089: f32, %3090: f32, %3091: f32):
      %3092 = arith.mulf %3089, %3090 : f32
      linalg.yield %3092 : f32
    } -> tensor<1x196x256xf32>
    %3093 = tensor.empty() : tensor<1x196x256xf32>
    %3094 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3088, %77 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3093 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb269(%3095: f32, %3096: f32, %3097: f32):
      %3098 = arith.mulf %3095, %3096 : f32
      linalg.yield %3098 : f32
    } -> tensor<1x196x256xf32>
    %3099 = tensor.empty() : tensor<1x196x256xf32>
    %3100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3094, %78 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3099 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_15", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.norm2"} {
    ^bb270(%3101: f32, %3102: f32, %3103: f32):
      %3104 = arith.addf %3101, %3102 : f32
      linalg.yield %3104 : f32
    } -> tensor<1x196x256xf32>
    %3105 = tensor.collapse_shape %3100 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3106 = tensor.expand_shape %3105 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %3107 = tensor.empty() : tensor<256x1024xf32>
    %3108 = linalg.transpose ins(%79:tensor<1024x256xf32>) outs(%3107:tensor<256x1024xf32>) permutation = [1, 0]
    %3109 = tensor.empty() : tensor<196x1024xf32>
    %3110 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3111 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3110 : f32) outs(%3109 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3112 = linalg.matmul {prov.region_id = "matmul_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1", prov.transposed_b = "true"} ins(%3106, %3108 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%3111 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3113 = tensor.empty() : tensor<196x1024xf32>
    %3114 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3112, %80 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%3113 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1"} {
    ^bb271(%3115: f32, %3116: f32, %3117: f32):
      %3118 = arith.addf %3115, %3116 : f32
      linalg.yield %3118 : f32
    } -> tensor<196x1024xf32>
    %3119 = tensor.collapse_shape %3114 [[0 : i64, 1 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %3120 = tensor.expand_shape %3119 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %3121 = tensor.empty() : tensor<1x196x1024xf32>
    %3122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3120 : tensor<1x196x1024xf32>) outs(%3121 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_7", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.act"} {
    ^bb272(%3123: f32, %3124: f32):
      %3125 = arith.constant 5.000000e-01 : f32
      %3126 = arith.constant 1.000000e+00 : f32
      %3127 = arith.constant 0.707106769 : f32
      %3128 = arith.mulf %3123, %3127 : f32
      %3129 = math.erf %3128 : f32
      %3130 = arith.addf %3126, %3129 : f32
      %3131 = arith.mulf %3125, %3123 : f32
      %3132 = arith.mulf %3131, %3130 : f32
      linalg.yield %3132 : f32
    } -> tensor<1x196x1024xf32>
    %3133 = tensor.collapse_shape %3122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %3134 = tensor.expand_shape %3133 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %3135 = tensor.empty() : tensor<1024x256xf32>
    %3136 = linalg.transpose ins(%81:tensor<256x1024xf32>) outs(%3135:tensor<1024x256xf32>) permutation = [1, 0]
    %3137 = tensor.empty() : tensor<196x256xf32>
    %3138 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3139 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3138 : f32) outs(%3137 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3140 = linalg.matmul {prov.region_id = "matmul_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2", prov.transposed_b = "true"} ins(%3134, %3136 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%3139 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3141 = tensor.empty() : tensor<196x256xf32>
    %3142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3140, %82 : tensor<196x256xf32>, tensor<256xf32>) outs(%3141 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2"} {
    ^bb273(%3143: f32, %3144: f32, %3145: f32):
      %3146 = arith.addf %3143, %3144 : f32
      linalg.yield %3146 : f32
    } -> tensor<196x256xf32>
    %3147 = tensor.collapse_shape %3142 [[0 : i64, 1 : i64]] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %3148 = tensor.expand_shape %3147 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3149 = tensor.empty() : tensor<1x196x256xf32>
    %3150 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3025, %3148 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3149 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.7"} {
    ^bb274(%3151: f32, %3152: f32, %3153: f32):
      %3154 = arith.addf %3151, %3152 : f32
      linalg.yield %3154 : f32
    } -> tensor<1x196x256xf32>
    %3155 = arith.constant {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} 0.000000e+00 : f32
    %3156 = tensor.splat %3155 {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32>
    %3157 = linalg.reduce ins(%3150:tensor<1x196x256xf32>) outs(%3156:tensor<1x196xf32>) dimensions = [2]
    (%3158: f32, %3159: f32) {
      %3160 = arith.addf %3158, %3159 : f32
      linalg.yield %3160 : f32
    }
    %3161 = arith.constant {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} 2.560000e+02 : f32
    %3162 = tensor.splat %3161 {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32>
    %3163 = tensor.empty() : tensor<1x196xf32>
    %3164 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3157, %3162 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3163 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb275(%3165: f32, %3166: f32, %3167: f32):
      %3168 = arith.divf %3165, %3166 : f32
      linalg.yield %3168 : f32
    } -> tensor<1x196xf32>
    %3169 = tensor.collapse_shape %3164 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3170 = tensor.expand_shape %3169 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3171 = tensor.empty() : tensor<1x196x256xf32>
    %3172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3150, %3170 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3171 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb276(%3173: f32, %3174: f32, %3175: f32):
      %3176 = arith.subf %3173, %3174 : f32
      linalg.yield %3176 : f32
    } -> tensor<1x196x256xf32>
    %3177 = tensor.empty() : tensor<1x196x256xf32>
    %3178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3172, %3172 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3177 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb277(%3179: f32, %3180: f32, %3181: f32):
      %3182 = arith.mulf %3179, %3180 : f32
      linalg.yield %3182 : f32
    } -> tensor<1x196x256xf32>
    %3183 = arith.constant {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} 0.000000e+00 : f32
    %3184 = tensor.splat %3183 {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32>
    %3185 = linalg.reduce ins(%3178:tensor<1x196x256xf32>) outs(%3184:tensor<1x196xf32>) dimensions = [2]
    (%3186: f32, %3187: f32) {
      %3188 = arith.addf %3186, %3187 : f32
      linalg.yield %3188 : f32
    }
    %3189 = arith.constant {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} 2.560000e+02 : f32
    %3190 = tensor.splat %3189 {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32>
    %3191 = tensor.empty() : tensor<1x196xf32>
    %3192 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3185, %3190 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3191 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb278(%3193: f32, %3194: f32, %3195: f32):
      %3196 = arith.divf %3193, %3194 : f32
      linalg.yield %3196 : f32
    } -> tensor<1x196xf32>
    %3197 = tensor.collapse_shape %3192 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3198 = tensor.expand_shape %3197 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3199 = arith.constant {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} 1.000000e-06 : f32
    %3200 = tensor.splat %3199 {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} : tensor<1x196x1xf32>
    %3201 = tensor.empty() : tensor<1x196x1xf32>
    %3202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3198, %3200 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3201 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb279(%3203: f32, %3204: f32, %3205: f32):
      %3206 = arith.addf %3203, %3204 : f32
      linalg.yield %3206 : f32
    } -> tensor<1x196x1xf32>
    %3207 = tensor.empty() : tensor<1x196x1xf32>
    %3208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3202 : tensor<1x196x1xf32>) outs(%3207 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb280(%3209: f32, %3210: f32):
      %3211 = math.rsqrt %3209 : f32
      linalg.yield %3211 : f32
    } -> tensor<1x196x1xf32>
    %3212 = tensor.empty() : tensor<1x196x256xf32>
    %3213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3172, %3208 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3212 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb281(%3214: f32, %3215: f32, %3216: f32):
      %3217 = arith.mulf %3214, %3215 : f32
      linalg.yield %3217 : f32
    } -> tensor<1x196x256xf32>
    %3218 = tensor.empty() : tensor<1x196x256xf32>
    %3219 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3213, %87 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3218 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb282(%3220: f32, %3221: f32, %3222: f32):
      %3223 = arith.mulf %3220, %3221 : f32
      linalg.yield %3223 : f32
    } -> tensor<1x196x256xf32>
    %3224 = tensor.empty() : tensor<1x196x256xf32>
    %3225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3219, %88 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3224 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_16", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm1"} {
    ^bb283(%3226: f32, %3227: f32, %3228: f32):
      %3229 = arith.addf %3226, %3227 : f32
      linalg.yield %3229 : f32
    } -> tensor<1x196x256xf32>
    %3230 = tensor.collapse_shape %3225 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3231 = tensor.expand_shape %3230 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %3232 = tensor.empty() : tensor<256x768xf32>
    %3233 = linalg.transpose ins(%95:tensor<768x256xf32>) outs(%3232:tensor<256x768xf32>) permutation = [1, 0]
    %3234 = tensor.empty() : tensor<196x768xf32>
    %3235 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3236 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3235 : f32) outs(%3234 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3237 = linalg.matmul {prov.region_id = "matmul_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv", prov.transposed_b = "true"} ins(%3231, %3233 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%3236 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3238 = tensor.empty() : tensor<196x768xf32>
    %3239 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3237, %96 : tensor<196x768xf32>, tensor<768xf32>) outs(%3238 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv"} {
    ^bb284(%3240: f32, %3241: f32, %3242: f32):
      %3243 = arith.addf %3240, %3241 : f32
      linalg.yield %3243 : f32
    } -> tensor<196x768xf32>
    %3244 = tensor.collapse_shape %3239 [[0 : i64, 1 : i64]] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %3245 = tensor.expand_shape %3244 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %3246 = tensor.collapse_shape %3245 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %3247 = tensor.expand_shape %3246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %3248 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %3249 = linalg.transpose ins(%3247:tensor<1x196x3x4x64xf32>) outs(%3248:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %3250 = "tensor.extract_slice"(%3249) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3251 = tensor.collapse_shape %3250 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3252 = tensor.expand_shape %3251 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3253 = "tensor.extract_slice"(%3249) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3254 = tensor.collapse_shape %3253 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3255 = tensor.expand_shape %3254 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3256 = "tensor.extract_slice"(%3249) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3257 = tensor.collapse_shape %3256 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3258 = tensor.expand_shape %3257 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3259 = tensor.empty() : tensor<1x4x64x196xf32>
    %3260 = linalg.transpose ins(%3255:tensor<1x4x196x64xf32>) outs(%3259:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %3261 = tensor.empty() : tensor<1x4x196x64xf32>
    %3262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3252 : tensor<1x4x196x64xf32>) outs(%3261 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb285(%3263: f32, %3264: f32):
      linalg.yield %3263 : f32
    } -> tensor<1x4x196x64xf32>
    %3265 = tensor.collapse_shape %3262 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %3266 = tensor.expand_shape %3265 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %3267 = tensor.empty() : tensor<1x4x64x196xf32>
    %3268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3260 : tensor<1x4x64x196xf32>) outs(%3267 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb286(%3269: f32, %3270: f32):
      linalg.yield %3269 : f32
    } -> tensor<1x4x64x196xf32>
    %3271 = tensor.collapse_shape %3268 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %3272 = tensor.expand_shape %3271 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_93", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %3273 = arith.constant {prov.region_id = "matmul_33", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} 0.000000e+00 : f32
    %3274 = tensor.splat %3273 {prov.region_id = "matmul_33", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<4x196x196xf32>
    %3275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3266, %3272 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%3274 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_33", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb287(%3276: f32, %3277: f32, %3278: f32):
      %3279 = arith.mulf %3276, %3277 : f32
      %3280 = arith.addf %3278, %3279 : f32
      linalg.yield %3280 : f32
    } -> tensor<4x196x196xf32>
    %3281 = tensor.collapse_shape %3275 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %3282 = tensor.expand_shape %3281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_94", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %3283 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} 1.250000e-01 : f32
    %3284 = tensor.splat %3283 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196x196xf32>
    %3285 = tensor.empty() : tensor<1x4x196x196xf32>
    %3286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3282, %3284 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%3285 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb288(%3287: f32, %3288: f32, %3289: f32):
      %3290 = arith.mulf %3287, %3288 : f32
      linalg.yield %3290 : f32
    } -> tensor<1x4x196x196xf32>
    %3291 = arith.constant {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} 0xff800000 : f32
    %3292 = tensor.splat %3291 {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196xf32>
    %3293 = linalg.reduce ins(%3286:tensor<1x4x196x196xf32>) outs(%3292:tensor<1x4x196xf32>) dimensions = [3]
    (%3294: f32, %3295: f32) {
      %3296 = arith.maximumf %3294, %3295 : f32
      linalg.yield %3296 : f32
    }
    %3297 = tensor.collapse_shape %3293 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %3298 = tensor.expand_shape %3297 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %3299 = tensor.empty() : tensor<1x4x196x196xf32>
    %3300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3286, %3298 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%3299 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb289(%3301: f32, %3302: f32, %3303: f32):
      %3304 = arith.subf %3301, %3302 : f32
      linalg.yield %3304 : f32
    } -> tensor<1x4x196x196xf32>
    %3305 = tensor.empty() : tensor<1x4x196x196xf32>
    %3306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3300 : tensor<1x4x196x196xf32>) outs(%3305 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb290(%3307: f32, %3308: f32):
      %3309 = math.exp %3307 : f32
      linalg.yield %3309 : f32
    } -> tensor<1x4x196x196xf32>
    %3310 = arith.constant {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} 0.000000e+00 : f32
    %3311 = tensor.splat %3310 {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196xf32>
    %3312 = linalg.reduce ins(%3306:tensor<1x4x196x196xf32>) outs(%3311:tensor<1x4x196xf32>) dimensions = [3]
    (%3313: f32, %3314: f32) {
      %3315 = arith.addf %3313, %3314 : f32
      linalg.yield %3315 : f32
    }
    %3316 = tensor.collapse_shape %3312 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %3317 = tensor.expand_shape %3316 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %3318 = tensor.empty() : tensor<1x4x196x196xf32>
    %3319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3306, %3317 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%3318 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb291(%3320: f32, %3321: f32, %3322: f32):
      %3323 = arith.divf %3320, %3321 : f32
      linalg.yield %3323 : f32
    } -> tensor<1x4x196x196xf32>
    %3324 = tensor.empty() : tensor<1x4x196x196xf32>
    %3325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3319 : tensor<1x4x196x196xf32>) outs(%3324 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb292(%3326: f32, %3327: f32):
      linalg.yield %3326 : f32
    } -> tensor<1x4x196x196xf32>
    %3328 = tensor.collapse_shape %3325 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %3329 = tensor.expand_shape %3328 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_95", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %3330 = tensor.empty() : tensor<1x4x196x64xf32>
    %3331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3258 : tensor<1x4x196x64xf32>) outs(%3330 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb293(%3332: f32, %3333: f32):
      linalg.yield %3332 : f32
    } -> tensor<1x4x196x64xf32>
    %3334 = tensor.collapse_shape %3331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %3335 = tensor.expand_shape %3334 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_96", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %3336 = arith.constant {prov.region_id = "matmul_34", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} 0.000000e+00 : f32
    %3337 = tensor.splat %3336 {prov.region_id = "matmul_34", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<4x196x64xf32>
    %3338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3329, %3335 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%3337 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_34", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} {
    ^bb294(%3339: f32, %3340: f32, %3341: f32):
      %3342 = arith.mulf %3339, %3340 : f32
      %3343 = arith.addf %3341, %3342 : f32
      linalg.yield %3343 : f32
    } -> tensor<4x196x64xf32>
    %3344 = tensor.collapse_shape %3338 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %3345 = tensor.expand_shape %3344 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_97", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3346 = tensor.empty() : tensor<1x196x4x64xf32>
    %3347 = linalg.transpose ins(%3345:tensor<1x4x196x64xf32>) outs(%3346:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %3348 = tensor.collapse_shape %3347 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %3349 = tensor.expand_shape %3348 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_98", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3350 = tensor.collapse_shape %3349 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3351 = tensor.expand_shape %3350 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_99", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %3352 = tensor.empty() : tensor<256x256xf32>
    %3353 = linalg.transpose ins(%97:tensor<256x256xf32>) outs(%3352:tensor<256x256xf32>) permutation = [1, 0]
    %3354 = tensor.empty() : tensor<196x256xf32>
    %3355 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3356 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3355 : f32) outs(%3354 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3357 = linalg.matmul {prov.region_id = "matmul_35", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj", prov.transposed_b = "true"} ins(%3351, %3353 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%3356 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3358 = tensor.empty() : tensor<196x256xf32>
    %3359 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3357, %98 : tensor<196x256xf32>, tensor<256xf32>) outs(%3358 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj"} {
    ^bb295(%3360: f32, %3361: f32, %3362: f32):
      %3363 = arith.addf %3360, %3361 : f32
      linalg.yield %3363 : f32
    } -> tensor<196x256xf32>
    %3364 = tensor.collapse_shape %3359 [[0 : i64, 1 : i64]] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %3365 = tensor.expand_shape %3364 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_100", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3366 = tensor.empty() : tensor<1x196x256xf32>
    %3367 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3150, %3365 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3366 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8"} {
    ^bb296(%3368: f32, %3369: f32, %3370: f32):
      %3371 = arith.addf %3368, %3369 : f32
      linalg.yield %3371 : f32
    } -> tensor<1x196x256xf32>
    %3372 = arith.constant {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} 0.000000e+00 : f32
    %3373 = tensor.splat %3372 {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32>
    %3374 = linalg.reduce ins(%3367:tensor<1x196x256xf32>) outs(%3373:tensor<1x196xf32>) dimensions = [2]
    (%3375: f32, %3376: f32) {
      %3377 = arith.addf %3375, %3376 : f32
      linalg.yield %3377 : f32
    }
    %3378 = arith.constant {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} 2.560000e+02 : f32
    %3379 = tensor.splat %3378 {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32>
    %3380 = tensor.empty() : tensor<1x196xf32>
    %3381 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3374, %3379 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3380 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb297(%3382: f32, %3383: f32, %3384: f32):
      %3385 = arith.divf %3382, %3383 : f32
      linalg.yield %3385 : f32
    } -> tensor<1x196xf32>
    %3386 = tensor.collapse_shape %3381 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3387 = tensor.expand_shape %3386 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3388 = tensor.empty() : tensor<1x196x256xf32>
    %3389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3367, %3387 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3388 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb298(%3390: f32, %3391: f32, %3392: f32):
      %3393 = arith.subf %3390, %3391 : f32
      linalg.yield %3393 : f32
    } -> tensor<1x196x256xf32>
    %3394 = tensor.empty() : tensor<1x196x256xf32>
    %3395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3389, %3389 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3394 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb299(%3396: f32, %3397: f32, %3398: f32):
      %3399 = arith.mulf %3396, %3397 : f32
      linalg.yield %3399 : f32
    } -> tensor<1x196x256xf32>
    %3400 = arith.constant {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} 0.000000e+00 : f32
    %3401 = tensor.splat %3400 {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32>
    %3402 = linalg.reduce ins(%3395:tensor<1x196x256xf32>) outs(%3401:tensor<1x196xf32>) dimensions = [2]
    (%3403: f32, %3404: f32) {
      %3405 = arith.addf %3403, %3404 : f32
      linalg.yield %3405 : f32
    }
    %3406 = arith.constant {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} 2.560000e+02 : f32
    %3407 = tensor.splat %3406 {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32>
    %3408 = tensor.empty() : tensor<1x196xf32>
    %3409 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3402, %3407 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3408 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb300(%3410: f32, %3411: f32, %3412: f32):
      %3413 = arith.divf %3410, %3411 : f32
      linalg.yield %3413 : f32
    } -> tensor<1x196xf32>
    %3414 = tensor.collapse_shape %3409 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3415 = tensor.expand_shape %3414 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3416 = arith.constant {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} 1.000000e-06 : f32
    %3417 = tensor.splat %3416 {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} : tensor<1x196x1xf32>
    %3418 = tensor.empty() : tensor<1x196x1xf32>
    %3419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3415, %3417 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3418 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb301(%3420: f32, %3421: f32, %3422: f32):
      %3423 = arith.addf %3420, %3421 : f32
      linalg.yield %3423 : f32
    } -> tensor<1x196x1xf32>
    %3424 = tensor.empty() : tensor<1x196x1xf32>
    %3425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3419 : tensor<1x196x1xf32>) outs(%3424 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb302(%3426: f32, %3427: f32):
      %3428 = math.rsqrt %3426 : f32
      linalg.yield %3428 : f32
    } -> tensor<1x196x1xf32>
    %3429 = tensor.empty() : tensor<1x196x256xf32>
    %3430 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3389, %3425 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3429 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb303(%3431: f32, %3432: f32, %3433: f32):
      %3434 = arith.mulf %3431, %3432 : f32
      linalg.yield %3434 : f32
    } -> tensor<1x196x256xf32>
    %3435 = tensor.empty() : tensor<1x196x256xf32>
    %3436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3430, %89 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3435 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb304(%3437: f32, %3438: f32, %3439: f32):
      %3440 = arith.mulf %3437, %3438 : f32
      linalg.yield %3440 : f32
    } -> tensor<1x196x256xf32>
    %3441 = tensor.empty() : tensor<1x196x256xf32>
    %3442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3436, %90 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3441 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_17", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.norm2"} {
    ^bb305(%3443: f32, %3444: f32, %3445: f32):
      %3446 = arith.addf %3443, %3444 : f32
      linalg.yield %3446 : f32
    } -> tensor<1x196x256xf32>
    %3447 = tensor.collapse_shape %3442 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3448 = tensor.expand_shape %3447 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_101", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %3449 = tensor.empty() : tensor<256x1024xf32>
    %3450 = linalg.transpose ins(%91:tensor<1024x256xf32>) outs(%3449:tensor<256x1024xf32>) permutation = [1, 0]
    %3451 = tensor.empty() : tensor<196x1024xf32>
    %3452 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3453 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3452 : f32) outs(%3451 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3454 = linalg.matmul {prov.region_id = "matmul_36", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1", prov.transposed_b = "true"} ins(%3448, %3450 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%3453 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3455 = tensor.empty() : tensor<196x1024xf32>
    %3456 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3454, %92 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%3455 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1"} {
    ^bb306(%3457: f32, %3458: f32, %3459: f32):
      %3460 = arith.addf %3457, %3458 : f32
      linalg.yield %3460 : f32
    } -> tensor<196x1024xf32>
    %3461 = tensor.collapse_shape %3456 [[0 : i64, 1 : i64]] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %3462 = tensor.expand_shape %3461 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_102", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %3463 = tensor.empty() : tensor<1x196x1024xf32>
    %3464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3462 : tensor<1x196x1024xf32>) outs(%3463 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_8", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.act"} {
    ^bb307(%3465: f32, %3466: f32):
      %3467 = arith.constant 5.000000e-01 : f32
      %3468 = arith.constant 1.000000e+00 : f32
      %3469 = arith.constant 0.707106769 : f32
      %3470 = arith.mulf %3465, %3469 : f32
      %3471 = math.erf %3470 : f32
      %3472 = arith.addf %3468, %3471 : f32
      %3473 = arith.mulf %3467, %3465 : f32
      %3474 = arith.mulf %3473, %3472 : f32
      linalg.yield %3474 : f32
    } -> tensor<1x196x1024xf32>
    %3475 = tensor.collapse_shape %3464 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %3476 = tensor.expand_shape %3475 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_103", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %3477 = tensor.empty() : tensor<1024x256xf32>
    %3478 = linalg.transpose ins(%93:tensor<256x1024xf32>) outs(%3477:tensor<1024x256xf32>) permutation = [1, 0]
    %3479 = tensor.empty() : tensor<196x256xf32>
    %3480 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3481 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3480 : f32) outs(%3479 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3482 = linalg.matmul {prov.region_id = "matmul_37", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2", prov.transposed_b = "true"} ins(%3476, %3478 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%3481 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3483 = tensor.empty() : tensor<196x256xf32>
    %3484 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3482, %94 : tensor<196x256xf32>, tensor<256xf32>) outs(%3483 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2"} {
    ^bb308(%3485: f32, %3486: f32, %3487: f32):
      %3488 = arith.addf %3485, %3486 : f32
      linalg.yield %3488 : f32
    } -> tensor<196x256xf32>
    %3489 = tensor.collapse_shape %3484 [[0 : i64, 1 : i64]] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %3490 = tensor.expand_shape %3489 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_104", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3491 = tensor.empty() : tensor<1x196x256xf32>
    %3492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3367, %3490 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3491 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_42", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.8"} {
    ^bb309(%3493: f32, %3494: f32, %3495: f32):
      %3496 = arith.addf %3493, %3494 : f32
      linalg.yield %3496 : f32
    } -> tensor<1x196x256xf32>
    %3497 = arith.constant {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} 0.000000e+00 : f32
    %3498 = tensor.splat %3497 {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32>
    %3499 = linalg.reduce ins(%3492:tensor<1x196x256xf32>) outs(%3498:tensor<1x196xf32>) dimensions = [2]
    (%3500: f32, %3501: f32) {
      %3502 = arith.addf %3500, %3501 : f32
      linalg.yield %3502 : f32
    }
    %3503 = arith.constant {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} 2.560000e+02 : f32
    %3504 = tensor.splat %3503 {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32>
    %3505 = tensor.empty() : tensor<1x196xf32>
    %3506 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3499, %3504 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3505 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb310(%3507: f32, %3508: f32, %3509: f32):
      %3510 = arith.divf %3507, %3508 : f32
      linalg.yield %3510 : f32
    } -> tensor<1x196xf32>
    %3511 = tensor.collapse_shape %3506 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3512 = tensor.expand_shape %3511 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3513 = tensor.empty() : tensor<1x196x256xf32>
    %3514 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3492, %3512 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3513 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb311(%3515: f32, %3516: f32, %3517: f32):
      %3518 = arith.subf %3515, %3516 : f32
      linalg.yield %3518 : f32
    } -> tensor<1x196x256xf32>
    %3519 = tensor.empty() : tensor<1x196x256xf32>
    %3520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3514, %3514 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3519 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb312(%3521: f32, %3522: f32, %3523: f32):
      %3524 = arith.mulf %3521, %3522 : f32
      linalg.yield %3524 : f32
    } -> tensor<1x196x256xf32>
    %3525 = arith.constant {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} 0.000000e+00 : f32
    %3526 = tensor.splat %3525 {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32>
    %3527 = linalg.reduce ins(%3520:tensor<1x196x256xf32>) outs(%3526:tensor<1x196xf32>) dimensions = [2]
    (%3528: f32, %3529: f32) {
      %3530 = arith.addf %3528, %3529 : f32
      linalg.yield %3530 : f32
    }
    %3531 = arith.constant {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} 2.560000e+02 : f32
    %3532 = tensor.splat %3531 {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32>
    %3533 = tensor.empty() : tensor<1x196xf32>
    %3534 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3527, %3532 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3533 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb313(%3535: f32, %3536: f32, %3537: f32):
      %3538 = arith.divf %3535, %3536 : f32
      linalg.yield %3538 : f32
    } -> tensor<1x196xf32>
    %3539 = tensor.collapse_shape %3534 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3540 = tensor.expand_shape %3539 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3541 = arith.constant {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} 1.000000e-06 : f32
    %3542 = tensor.splat %3541 {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} : tensor<1x196x1xf32>
    %3543 = tensor.empty() : tensor<1x196x1xf32>
    %3544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3540, %3542 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3543 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb314(%3545: f32, %3546: f32, %3547: f32):
      %3548 = arith.addf %3545, %3546 : f32
      linalg.yield %3548 : f32
    } -> tensor<1x196x1xf32>
    %3549 = tensor.empty() : tensor<1x196x1xf32>
    %3550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3544 : tensor<1x196x1xf32>) outs(%3549 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb315(%3551: f32, %3552: f32):
      %3553 = math.rsqrt %3551 : f32
      linalg.yield %3553 : f32
    } -> tensor<1x196x1xf32>
    %3554 = tensor.empty() : tensor<1x196x256xf32>
    %3555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3514, %3550 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3554 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb316(%3556: f32, %3557: f32, %3558: f32):
      %3559 = arith.mulf %3556, %3557 : f32
      linalg.yield %3559 : f32
    } -> tensor<1x196x256xf32>
    %3560 = tensor.empty() : tensor<1x196x256xf32>
    %3561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3555, %99 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3560 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb317(%3562: f32, %3563: f32, %3564: f32):
      %3565 = arith.mulf %3562, %3563 : f32
      linalg.yield %3565 : f32
    } -> tensor<1x196x256xf32>
    %3566 = tensor.empty() : tensor<1x196x256xf32>
    %3567 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3561, %100 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3566 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_18", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm1"} {
    ^bb318(%3568: f32, %3569: f32, %3570: f32):
      %3571 = arith.addf %3568, %3569 : f32
      linalg.yield %3571 : f32
    } -> tensor<1x196x256xf32>
    %3572 = tensor.collapse_shape %3567 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3573 = tensor.expand_shape %3572 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_105", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %3574 = tensor.empty() : tensor<256x768xf32>
    %3575 = linalg.transpose ins(%107:tensor<768x256xf32>) outs(%3574:tensor<256x768xf32>) permutation = [1, 0]
    %3576 = tensor.empty() : tensor<196x768xf32>
    %3577 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3578 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3577 : f32) outs(%3576 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3579 = linalg.matmul {prov.region_id = "matmul_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv", prov.transposed_b = "true"} ins(%3573, %3575 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%3578 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3580 = tensor.empty() : tensor<196x768xf32>
    %3581 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3579, %108 : tensor<196x768xf32>, tensor<768xf32>) outs(%3580 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_43", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv"} {
    ^bb319(%3582: f32, %3583: f32, %3584: f32):
      %3585 = arith.addf %3582, %3583 : f32
      linalg.yield %3585 : f32
    } -> tensor<196x768xf32>
    %3586 = tensor.collapse_shape %3581 [[0 : i64, 1 : i64]] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %3587 = tensor.expand_shape %3586 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_106", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %3588 = tensor.collapse_shape %3587 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %3589 = tensor.expand_shape %3588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_107", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %3590 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %3591 = linalg.transpose ins(%3589:tensor<1x196x3x4x64xf32>) outs(%3590:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %3592 = "tensor.extract_slice"(%3591) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3593 = tensor.collapse_shape %3592 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3594 = tensor.expand_shape %3593 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3595 = "tensor.extract_slice"(%3591) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3596 = tensor.collapse_shape %3595 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3597 = tensor.expand_shape %3596 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3598 = "tensor.extract_slice"(%3591) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3599 = tensor.collapse_shape %3598 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3600 = tensor.expand_shape %3599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3601 = tensor.empty() : tensor<1x4x64x196xf32>
    %3602 = linalg.transpose ins(%3597:tensor<1x4x196x64xf32>) outs(%3601:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %3603 = tensor.empty() : tensor<1x4x196x64xf32>
    %3604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3594 : tensor<1x4x196x64xf32>) outs(%3603 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb320(%3605: f32, %3606: f32):
      linalg.yield %3605 : f32
    } -> tensor<1x4x196x64xf32>
    %3607 = tensor.collapse_shape %3604 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %3608 = tensor.expand_shape %3607 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_108", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %3609 = tensor.empty() : tensor<1x4x64x196xf32>
    %3610 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3602 : tensor<1x4x64x196xf32>) outs(%3609 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb321(%3611: f32, %3612: f32):
      linalg.yield %3611 : f32
    } -> tensor<1x4x64x196xf32>
    %3613 = tensor.collapse_shape %3610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %3614 = tensor.expand_shape %3613 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_109", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %3615 = arith.constant {prov.region_id = "matmul_39", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} 0.000000e+00 : f32
    %3616 = tensor.splat %3615 {prov.region_id = "matmul_39", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<4x196x196xf32>
    %3617 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3608, %3614 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%3616 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_39", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb322(%3618: f32, %3619: f32, %3620: f32):
      %3621 = arith.mulf %3618, %3619 : f32
      %3622 = arith.addf %3620, %3621 : f32
      linalg.yield %3622 : f32
    } -> tensor<4x196x196xf32>
    %3623 = tensor.collapse_shape %3617 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %3624 = tensor.expand_shape %3623 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_110", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %3625 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} 1.250000e-01 : f32
    %3626 = tensor.splat %3625 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196x196xf32>
    %3627 = tensor.empty() : tensor<1x4x196x196xf32>
    %3628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3624, %3626 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%3627 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb323(%3629: f32, %3630: f32, %3631: f32):
      %3632 = arith.mulf %3629, %3630 : f32
      linalg.yield %3632 : f32
    } -> tensor<1x4x196x196xf32>
    %3633 = arith.constant {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} 0xff800000 : f32
    %3634 = tensor.splat %3633 {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196xf32>
    %3635 = linalg.reduce ins(%3628:tensor<1x4x196x196xf32>) outs(%3634:tensor<1x4x196xf32>) dimensions = [3]
    (%3636: f32, %3637: f32) {
      %3638 = arith.maximumf %3636, %3637 : f32
      linalg.yield %3638 : f32
    }
    %3639 = tensor.collapse_shape %3635 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %3640 = tensor.expand_shape %3639 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %3641 = tensor.empty() : tensor<1x4x196x196xf32>
    %3642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3628, %3640 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%3641 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb324(%3643: f32, %3644: f32, %3645: f32):
      %3646 = arith.subf %3643, %3644 : f32
      linalg.yield %3646 : f32
    } -> tensor<1x4x196x196xf32>
    %3647 = tensor.empty() : tensor<1x4x196x196xf32>
    %3648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3642 : tensor<1x4x196x196xf32>) outs(%3647 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb325(%3649: f32, %3650: f32):
      %3651 = math.exp %3649 : f32
      linalg.yield %3651 : f32
    } -> tensor<1x4x196x196xf32>
    %3652 = arith.constant {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} 0.000000e+00 : f32
    %3653 = tensor.splat %3652 {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196xf32>
    %3654 = linalg.reduce ins(%3648:tensor<1x4x196x196xf32>) outs(%3653:tensor<1x4x196xf32>) dimensions = [3]
    (%3655: f32, %3656: f32) {
      %3657 = arith.addf %3655, %3656 : f32
      linalg.yield %3657 : f32
    }
    %3658 = tensor.collapse_shape %3654 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %3659 = tensor.expand_shape %3658 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %3660 = tensor.empty() : tensor<1x4x196x196xf32>
    %3661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3648, %3659 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%3660 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb326(%3662: f32, %3663: f32, %3664: f32):
      %3665 = arith.divf %3662, %3663 : f32
      linalg.yield %3665 : f32
    } -> tensor<1x4x196x196xf32>
    %3666 = tensor.empty() : tensor<1x4x196x196xf32>
    %3667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3661 : tensor<1x4x196x196xf32>) outs(%3666 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_22", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb327(%3668: f32, %3669: f32):
      linalg.yield %3668 : f32
    } -> tensor<1x4x196x196xf32>
    %3670 = tensor.collapse_shape %3667 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_111", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %3671 = tensor.expand_shape %3670 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_111", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %3672 = tensor.empty() : tensor<1x4x196x64xf32>
    %3673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3600 : tensor<1x4x196x64xf32>) outs(%3672 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_23", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb328(%3674: f32, %3675: f32):
      linalg.yield %3674 : f32
    } -> tensor<1x4x196x64xf32>
    %3676 = tensor.collapse_shape %3673 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_112", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %3677 = tensor.expand_shape %3676 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_112", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %3678 = arith.constant {prov.region_id = "matmul_40", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} 0.000000e+00 : f32
    %3679 = tensor.splat %3678 {prov.region_id = "matmul_40", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<4x196x64xf32>
    %3680 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3671, %3677 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%3679 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_40", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} {
    ^bb329(%3681: f32, %3682: f32, %3683: f32):
      %3684 = arith.mulf %3681, %3682 : f32
      %3685 = arith.addf %3683, %3684 : f32
      linalg.yield %3685 : f32
    } -> tensor<4x196x64xf32>
    %3686 = tensor.collapse_shape %3680 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_113", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %3687 = tensor.expand_shape %3686 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_113", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3688 = tensor.empty() : tensor<1x196x4x64xf32>
    %3689 = linalg.transpose ins(%3687:tensor<1x4x196x64xf32>) outs(%3688:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %3690 = tensor.collapse_shape %3689 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_114", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %3691 = tensor.expand_shape %3690 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_114", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3692 = tensor.collapse_shape %3691 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_115", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3693 = tensor.expand_shape %3692 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_115", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %3694 = tensor.empty() : tensor<256x256xf32>
    %3695 = linalg.transpose ins(%109:tensor<256x256xf32>) outs(%3694:tensor<256x256xf32>) permutation = [1, 0]
    %3696 = tensor.empty() : tensor<196x256xf32>
    %3697 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3698 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3697 : f32) outs(%3696 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3699 = linalg.matmul {prov.region_id = "matmul_41", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj", prov.transposed_b = "true"} ins(%3693, %3695 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%3698 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3700 = tensor.empty() : tensor<196x256xf32>
    %3701 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3699, %110 : tensor<196x256xf32>, tensor<256xf32>) outs(%3700 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_44", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj"} {
    ^bb330(%3702: f32, %3703: f32, %3704: f32):
      %3705 = arith.addf %3702, %3703 : f32
      linalg.yield %3705 : f32
    } -> tensor<196x256xf32>
    %3706 = tensor.collapse_shape %3701 [[0 : i64, 1 : i64]] {prov.region_id = "view_116", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %3707 = tensor.expand_shape %3706 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_116", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3708 = tensor.empty() : tensor<1x196x256xf32>
    %3709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3492, %3707 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3708 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_45", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9"} {
    ^bb331(%3710: f32, %3711: f32, %3712: f32):
      %3713 = arith.addf %3710, %3711 : f32
      linalg.yield %3713 : f32
    } -> tensor<1x196x256xf32>
    %3714 = arith.constant {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} 0.000000e+00 : f32
    %3715 = tensor.splat %3714 {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32>
    %3716 = linalg.reduce ins(%3709:tensor<1x196x256xf32>) outs(%3715:tensor<1x196xf32>) dimensions = [2]
    (%3717: f32, %3718: f32) {
      %3719 = arith.addf %3717, %3718 : f32
      linalg.yield %3719 : f32
    }
    %3720 = arith.constant {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} 2.560000e+02 : f32
    %3721 = tensor.splat %3720 {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32>
    %3722 = tensor.empty() : tensor<1x196xf32>
    %3723 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3716, %3721 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3722 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb332(%3724: f32, %3725: f32, %3726: f32):
      %3727 = arith.divf %3724, %3725 : f32
      linalg.yield %3727 : f32
    } -> tensor<1x196xf32>
    %3728 = tensor.collapse_shape %3723 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3729 = tensor.expand_shape %3728 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3730 = tensor.empty() : tensor<1x196x256xf32>
    %3731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3709, %3729 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3730 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb333(%3732: f32, %3733: f32, %3734: f32):
      %3735 = arith.subf %3732, %3733 : f32
      linalg.yield %3735 : f32
    } -> tensor<1x196x256xf32>
    %3736 = tensor.empty() : tensor<1x196x256xf32>
    %3737 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3731, %3731 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3736 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb334(%3738: f32, %3739: f32, %3740: f32):
      %3741 = arith.mulf %3738, %3739 : f32
      linalg.yield %3741 : f32
    } -> tensor<1x196x256xf32>
    %3742 = arith.constant {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} 0.000000e+00 : f32
    %3743 = tensor.splat %3742 {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32>
    %3744 = linalg.reduce ins(%3737:tensor<1x196x256xf32>) outs(%3743:tensor<1x196xf32>) dimensions = [2]
    (%3745: f32, %3746: f32) {
      %3747 = arith.addf %3745, %3746 : f32
      linalg.yield %3747 : f32
    }
    %3748 = arith.constant {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} 2.560000e+02 : f32
    %3749 = tensor.splat %3748 {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32>
    %3750 = tensor.empty() : tensor<1x196xf32>
    %3751 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3744, %3749 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3750 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb335(%3752: f32, %3753: f32, %3754: f32):
      %3755 = arith.divf %3752, %3753 : f32
      linalg.yield %3755 : f32
    } -> tensor<1x196xf32>
    %3756 = tensor.collapse_shape %3751 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %3757 = tensor.expand_shape %3756 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3758 = arith.constant {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} 1.000000e-06 : f32
    %3759 = tensor.splat %3758 {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} : tensor<1x196x1xf32>
    %3760 = tensor.empty() : tensor<1x196x1xf32>
    %3761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3757, %3759 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3760 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb336(%3762: f32, %3763: f32, %3764: f32):
      %3765 = arith.addf %3762, %3763 : f32
      linalg.yield %3765 : f32
    } -> tensor<1x196x1xf32>
    %3766 = tensor.empty() : tensor<1x196x1xf32>
    %3767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3761 : tensor<1x196x1xf32>) outs(%3766 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb337(%3768: f32, %3769: f32):
      %3770 = math.rsqrt %3768 : f32
      linalg.yield %3770 : f32
    } -> tensor<1x196x1xf32>
    %3771 = tensor.empty() : tensor<1x196x256xf32>
    %3772 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3731, %3767 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3771 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb338(%3773: f32, %3774: f32, %3775: f32):
      %3776 = arith.mulf %3773, %3774 : f32
      linalg.yield %3776 : f32
    } -> tensor<1x196x256xf32>
    %3777 = tensor.empty() : tensor<1x196x256xf32>
    %3778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3772, %101 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3777 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb339(%3779: f32, %3780: f32, %3781: f32):
      %3782 = arith.mulf %3779, %3780 : f32
      linalg.yield %3782 : f32
    } -> tensor<1x196x256xf32>
    %3783 = tensor.empty() : tensor<1x196x256xf32>
    %3784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3778, %102 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3783 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_19", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.norm2"} {
    ^bb340(%3785: f32, %3786: f32, %3787: f32):
      %3788 = arith.addf %3785, %3786 : f32
      linalg.yield %3788 : f32
    } -> tensor<1x196x256xf32>
    %3789 = tensor.collapse_shape %3784 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_117", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3790 = tensor.expand_shape %3789 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_117", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %3791 = tensor.empty() : tensor<256x1024xf32>
    %3792 = linalg.transpose ins(%103:tensor<1024x256xf32>) outs(%3791:tensor<256x1024xf32>) permutation = [1, 0]
    %3793 = tensor.empty() : tensor<196x1024xf32>
    %3794 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3795 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3794 : f32) outs(%3793 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3796 = linalg.matmul {prov.region_id = "matmul_42", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1", prov.transposed_b = "true"} ins(%3790, %3792 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%3795 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %3797 = tensor.empty() : tensor<196x1024xf32>
    %3798 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3796, %104 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%3797 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_46", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1"} {
    ^bb341(%3799: f32, %3800: f32, %3801: f32):
      %3802 = arith.addf %3799, %3800 : f32
      linalg.yield %3802 : f32
    } -> tensor<196x1024xf32>
    %3803 = tensor.collapse_shape %3798 [[0 : i64, 1 : i64]] {prov.region_id = "view_118", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %3804 = tensor.expand_shape %3803 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_118", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %3805 = tensor.empty() : tensor<1x196x1024xf32>
    %3806 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3804 : tensor<1x196x1024xf32>) outs(%3805 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_9", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.act"} {
    ^bb342(%3807: f32, %3808: f32):
      %3809 = arith.constant 5.000000e-01 : f32
      %3810 = arith.constant 1.000000e+00 : f32
      %3811 = arith.constant 0.707106769 : f32
      %3812 = arith.mulf %3807, %3811 : f32
      %3813 = math.erf %3812 : f32
      %3814 = arith.addf %3810, %3813 : f32
      %3815 = arith.mulf %3809, %3807 : f32
      %3816 = arith.mulf %3815, %3814 : f32
      linalg.yield %3816 : f32
    } -> tensor<1x196x1024xf32>
    %3817 = tensor.collapse_shape %3806 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_119", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %3818 = tensor.expand_shape %3817 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_119", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %3819 = tensor.empty() : tensor<1024x256xf32>
    %3820 = linalg.transpose ins(%105:tensor<256x1024xf32>) outs(%3819:tensor<1024x256xf32>) permutation = [1, 0]
    %3821 = tensor.empty() : tensor<196x256xf32>
    %3822 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3823 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3822 : f32) outs(%3821 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3824 = linalg.matmul {prov.region_id = "matmul_43", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2", prov.transposed_b = "true"} ins(%3818, %3820 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%3823 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %3825 = tensor.empty() : tensor<196x256xf32>
    %3826 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3824, %106 : tensor<196x256xf32>, tensor<256xf32>) outs(%3825 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_47", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2"} {
    ^bb343(%3827: f32, %3828: f32, %3829: f32):
      %3830 = arith.addf %3827, %3828 : f32
      linalg.yield %3830 : f32
    } -> tensor<196x256xf32>
    %3831 = tensor.collapse_shape %3826 [[0 : i64, 1 : i64]] {prov.region_id = "view_120", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %3832 = tensor.expand_shape %3831 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_120", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %3833 = tensor.empty() : tensor<1x196x256xf32>
    %3834 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3709, %3832 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3833 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_48", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.9"} {
    ^bb344(%3835: f32, %3836: f32, %3837: f32):
      %3838 = arith.addf %3835, %3836 : f32
      linalg.yield %3838 : f32
    } -> tensor<1x196x256xf32>
    %3839 = arith.constant {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} 0.000000e+00 : f32
    %3840 = tensor.splat %3839 {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32>
    %3841 = linalg.reduce ins(%3834:tensor<1x196x256xf32>) outs(%3840:tensor<1x196xf32>) dimensions = [2]
    (%3842: f32, %3843: f32) {
      %3844 = arith.addf %3842, %3843 : f32
      linalg.yield %3844 : f32
    }
    %3845 = arith.constant {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} 2.560000e+02 : f32
    %3846 = tensor.splat %3845 {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32>
    %3847 = tensor.empty() : tensor<1x196xf32>
    %3848 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3841, %3846 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3847 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb345(%3849: f32, %3850: f32, %3851: f32):
      %3852 = arith.divf %3849, %3850 : f32
      linalg.yield %3852 : f32
    } -> tensor<1x196xf32>
    %3853 = tensor.collapse_shape %3848 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3854 = tensor.expand_shape %3853 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3855 = tensor.empty() : tensor<1x196x256xf32>
    %3856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3834, %3854 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3855 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb346(%3857: f32, %3858: f32, %3859: f32):
      %3860 = arith.subf %3857, %3858 : f32
      linalg.yield %3860 : f32
    } -> tensor<1x196x256xf32>
    %3861 = tensor.empty() : tensor<1x196x256xf32>
    %3862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3856, %3856 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%3861 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb347(%3863: f32, %3864: f32, %3865: f32):
      %3866 = arith.mulf %3863, %3864 : f32
      linalg.yield %3866 : f32
    } -> tensor<1x196x256xf32>
    %3867 = arith.constant {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} 0.000000e+00 : f32
    %3868 = tensor.splat %3867 {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32>
    %3869 = linalg.reduce ins(%3862:tensor<1x196x256xf32>) outs(%3868:tensor<1x196xf32>) dimensions = [2]
    (%3870: f32, %3871: f32) {
      %3872 = arith.addf %3870, %3871 : f32
      linalg.yield %3872 : f32
    }
    %3873 = arith.constant {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} 2.560000e+02 : f32
    %3874 = tensor.splat %3873 {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32>
    %3875 = tensor.empty() : tensor<1x196xf32>
    %3876 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3869, %3874 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%3875 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb348(%3877: f32, %3878: f32, %3879: f32):
      %3880 = arith.divf %3877, %3878 : f32
      linalg.yield %3880 : f32
    } -> tensor<1x196xf32>
    %3881 = tensor.collapse_shape %3876 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %3882 = tensor.expand_shape %3881 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %3883 = arith.constant {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} 1.000000e-06 : f32
    %3884 = tensor.splat %3883 {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} : tensor<1x196x1xf32>
    %3885 = tensor.empty() : tensor<1x196x1xf32>
    %3886 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3882, %3884 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%3885 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb349(%3887: f32, %3888: f32, %3889: f32):
      %3890 = arith.addf %3887, %3888 : f32
      linalg.yield %3890 : f32
    } -> tensor<1x196x1xf32>
    %3891 = tensor.empty() : tensor<1x196x1xf32>
    %3892 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3886 : tensor<1x196x1xf32>) outs(%3891 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb350(%3893: f32, %3894: f32):
      %3895 = math.rsqrt %3893 : f32
      linalg.yield %3895 : f32
    } -> tensor<1x196x1xf32>
    %3896 = tensor.empty() : tensor<1x196x256xf32>
    %3897 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3856, %3892 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%3896 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb351(%3898: f32, %3899: f32, %3900: f32):
      %3901 = arith.mulf %3898, %3899 : f32
      linalg.yield %3901 : f32
    } -> tensor<1x196x256xf32>
    %3902 = tensor.empty() : tensor<1x196x256xf32>
    %3903 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3897, %111 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3902 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb352(%3904: f32, %3905: f32, %3906: f32):
      %3907 = arith.mulf %3904, %3905 : f32
      linalg.yield %3907 : f32
    } -> tensor<1x196x256xf32>
    %3908 = tensor.empty() : tensor<1x196x256xf32>
    %3909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3903, %112 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%3908 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_20", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm1"} {
    ^bb353(%3910: f32, %3911: f32, %3912: f32):
      %3913 = arith.addf %3910, %3911 : f32
      linalg.yield %3913 : f32
    } -> tensor<1x196x256xf32>
    %3914 = tensor.collapse_shape %3909 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_121", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %3915 = tensor.expand_shape %3914 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_121", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %3916 = tensor.empty() : tensor<256x768xf32>
    %3917 = linalg.transpose ins(%119:tensor<768x256xf32>) outs(%3916:tensor<256x768xf32>) permutation = [1, 0]
    %3918 = tensor.empty() : tensor<196x768xf32>
    %3919 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %3920 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3919 : f32) outs(%3918 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3921 = linalg.matmul {prov.region_id = "matmul_44", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv", prov.transposed_b = "true"} ins(%3915, %3917 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%3920 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %3922 = tensor.empty() : tensor<196x768xf32>
    %3923 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3921, %120 : tensor<196x768xf32>, tensor<768xf32>) outs(%3922 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_49", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv"} {
    ^bb354(%3924: f32, %3925: f32, %3926: f32):
      %3927 = arith.addf %3924, %3925 : f32
      linalg.yield %3927 : f32
    } -> tensor<196x768xf32>
    %3928 = tensor.collapse_shape %3923 [[0 : i64, 1 : i64]] {prov.region_id = "view_122", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %3929 = tensor.expand_shape %3928 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_122", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %3930 = tensor.collapse_shape %3929 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_123", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %3931 = tensor.expand_shape %3930 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_123", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %3932 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %3933 = linalg.transpose ins(%3931:tensor<1x196x3x4x64xf32>) outs(%3932:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %3934 = "tensor.extract_slice"(%3933) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_18", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3935 = tensor.collapse_shape %3934 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_18", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3936 = tensor.expand_shape %3935 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_18", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3937 = "tensor.extract_slice"(%3933) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_19", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3938 = tensor.collapse_shape %3937 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_19", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3939 = tensor.expand_shape %3938 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_19", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3940 = "tensor.extract_slice"(%3933) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_20", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %3941 = tensor.collapse_shape %3940 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_20", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %3942 = tensor.expand_shape %3941 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_20", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %3943 = tensor.empty() : tensor<1x4x64x196xf32>
    %3944 = linalg.transpose ins(%3939:tensor<1x4x196x64xf32>) outs(%3943:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %3945 = tensor.empty() : tensor<1x4x196x64xf32>
    %3946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3936 : tensor<1x4x196x64xf32>) outs(%3945 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_24", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb355(%3947: f32, %3948: f32):
      linalg.yield %3947 : f32
    } -> tensor<1x4x196x64xf32>
    %3949 = tensor.collapse_shape %3946 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_124", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %3950 = tensor.expand_shape %3949 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_124", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %3951 = tensor.empty() : tensor<1x4x64x196xf32>
    %3952 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3944 : tensor<1x4x64x196xf32>) outs(%3951 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_25", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb356(%3953: f32, %3954: f32):
      linalg.yield %3953 : f32
    } -> tensor<1x4x64x196xf32>
    %3955 = tensor.collapse_shape %3952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_125", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %3956 = tensor.expand_shape %3955 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_125", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %3957 = arith.constant {prov.region_id = "matmul_45", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} 0.000000e+00 : f32
    %3958 = tensor.splat %3957 {prov.region_id = "matmul_45", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<4x196x196xf32>
    %3959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3950, %3956 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%3958 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_45", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb357(%3960: f32, %3961: f32, %3962: f32):
      %3963 = arith.mulf %3960, %3961 : f32
      %3964 = arith.addf %3962, %3963 : f32
      linalg.yield %3964 : f32
    } -> tensor<4x196x196xf32>
    %3965 = tensor.collapse_shape %3959 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_126", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %3966 = tensor.expand_shape %3965 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_126", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %3967 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} 1.250000e-01 : f32
    %3968 = tensor.splat %3967 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196x196xf32>
    %3969 = tensor.empty() : tensor<1x4x196x196xf32>
    %3970 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3966, %3968 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%3969 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb358(%3971: f32, %3972: f32, %3973: f32):
      %3974 = arith.mulf %3971, %3972 : f32
      linalg.yield %3974 : f32
    } -> tensor<1x4x196x196xf32>
    %3975 = arith.constant {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} 0xff800000 : f32
    %3976 = tensor.splat %3975 {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196xf32>
    %3977 = linalg.reduce ins(%3970:tensor<1x4x196x196xf32>) outs(%3976:tensor<1x4x196xf32>) dimensions = [3]
    (%3978: f32, %3979: f32) {
      %3980 = arith.maximumf %3978, %3979 : f32
      linalg.yield %3980 : f32
    }
    %3981 = tensor.collapse_shape %3977 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %3982 = tensor.expand_shape %3981 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %3983 = tensor.empty() : tensor<1x4x196x196xf32>
    %3984 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3970, %3982 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%3983 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb359(%3985: f32, %3986: f32, %3987: f32):
      %3988 = arith.subf %3985, %3986 : f32
      linalg.yield %3988 : f32
    } -> tensor<1x4x196x196xf32>
    %3989 = tensor.empty() : tensor<1x4x196x196xf32>
    %3990 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3984 : tensor<1x4x196x196xf32>) outs(%3989 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb360(%3991: f32, %3992: f32):
      %3993 = math.exp %3991 : f32
      linalg.yield %3993 : f32
    } -> tensor<1x4x196x196xf32>
    %3994 = arith.constant {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} 0.000000e+00 : f32
    %3995 = tensor.splat %3994 {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196xf32>
    %3996 = linalg.reduce ins(%3990:tensor<1x4x196x196xf32>) outs(%3995:tensor<1x4x196xf32>) dimensions = [3]
    (%3997: f32, %3998: f32) {
      %3999 = arith.addf %3997, %3998 : f32
      linalg.yield %3999 : f32
    }
    %4000 = tensor.collapse_shape %3996 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %4001 = tensor.expand_shape %4000 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %4002 = tensor.empty() : tensor<1x4x196x196xf32>
    %4003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3990, %4001 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%4002 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb361(%4004: f32, %4005: f32, %4006: f32):
      %4007 = arith.divf %4004, %4005 : f32
      linalg.yield %4007 : f32
    } -> tensor<1x4x196x196xf32>
    %4008 = tensor.empty() : tensor<1x4x196x196xf32>
    %4009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4003 : tensor<1x4x196x196xf32>) outs(%4008 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_26", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb362(%4010: f32, %4011: f32):
      linalg.yield %4010 : f32
    } -> tensor<1x4x196x196xf32>
    %4012 = tensor.collapse_shape %4009 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_127", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %4013 = tensor.expand_shape %4012 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_127", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %4014 = tensor.empty() : tensor<1x4x196x64xf32>
    %4015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3942 : tensor<1x4x196x64xf32>) outs(%4014 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_27", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb363(%4016: f32, %4017: f32):
      linalg.yield %4016 : f32
    } -> tensor<1x4x196x64xf32>
    %4018 = tensor.collapse_shape %4015 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_128", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %4019 = tensor.expand_shape %4018 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_128", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %4020 = arith.constant {prov.region_id = "matmul_46", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} 0.000000e+00 : f32
    %4021 = tensor.splat %4020 {prov.region_id = "matmul_46", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<4x196x64xf32>
    %4022 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4013, %4019 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%4021 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_46", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} {
    ^bb364(%4023: f32, %4024: f32, %4025: f32):
      %4026 = arith.mulf %4023, %4024 : f32
      %4027 = arith.addf %4025, %4026 : f32
      linalg.yield %4027 : f32
    } -> tensor<4x196x64xf32>
    %4028 = tensor.collapse_shape %4022 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_129", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %4029 = tensor.expand_shape %4028 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_129", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %4030 = tensor.empty() : tensor<1x196x4x64xf32>
    %4031 = linalg.transpose ins(%4029:tensor<1x4x196x64xf32>) outs(%4030:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %4032 = tensor.collapse_shape %4031 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_130", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %4033 = tensor.expand_shape %4032 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_130", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4034 = tensor.collapse_shape %4033 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_131", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %4035 = tensor.expand_shape %4034 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_131", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %4036 = tensor.empty() : tensor<256x256xf32>
    %4037 = linalg.transpose ins(%121:tensor<256x256xf32>) outs(%4036:tensor<256x256xf32>) permutation = [1, 0]
    %4038 = tensor.empty() : tensor<196x256xf32>
    %4039 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4040 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4039 : f32) outs(%4038 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4041 = linalg.matmul {prov.region_id = "matmul_47", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj", prov.transposed_b = "true"} ins(%4035, %4037 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%4040 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4042 = tensor.empty() : tensor<196x256xf32>
    %4043 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4041, %122 : tensor<196x256xf32>, tensor<256xf32>) outs(%4042 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_50", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj"} {
    ^bb365(%4044: f32, %4045: f32, %4046: f32):
      %4047 = arith.addf %4044, %4045 : f32
      linalg.yield %4047 : f32
    } -> tensor<196x256xf32>
    %4048 = tensor.collapse_shape %4043 [[0 : i64, 1 : i64]] {prov.region_id = "view_132", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %4049 = tensor.expand_shape %4048 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_132", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4050 = tensor.empty() : tensor<1x196x256xf32>
    %4051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3834, %4049 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4050 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_51", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10"} {
    ^bb366(%4052: f32, %4053: f32, %4054: f32):
      %4055 = arith.addf %4052, %4053 : f32
      linalg.yield %4055 : f32
    } -> tensor<1x196x256xf32>
    %4056 = arith.constant {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} 0.000000e+00 : f32
    %4057 = tensor.splat %4056 {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32>
    %4058 = linalg.reduce ins(%4051:tensor<1x196x256xf32>) outs(%4057:tensor<1x196xf32>) dimensions = [2]
    (%4059: f32, %4060: f32) {
      %4061 = arith.addf %4059, %4060 : f32
      linalg.yield %4061 : f32
    }
    %4062 = arith.constant {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} 2.560000e+02 : f32
    %4063 = tensor.splat %4062 {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32>
    %4064 = tensor.empty() : tensor<1x196xf32>
    %4065 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4058, %4063 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4064 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb367(%4066: f32, %4067: f32, %4068: f32):
      %4069 = arith.divf %4066, %4067 : f32
      linalg.yield %4069 : f32
    } -> tensor<1x196xf32>
    %4070 = tensor.collapse_shape %4065 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %4071 = tensor.expand_shape %4070 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4072 = tensor.empty() : tensor<1x196x256xf32>
    %4073 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4051, %4071 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4072 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb368(%4074: f32, %4075: f32, %4076: f32):
      %4077 = arith.subf %4074, %4075 : f32
      linalg.yield %4077 : f32
    } -> tensor<1x196x256xf32>
    %4078 = tensor.empty() : tensor<1x196x256xf32>
    %4079 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4073, %4073 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4078 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb369(%4080: f32, %4081: f32, %4082: f32):
      %4083 = arith.mulf %4080, %4081 : f32
      linalg.yield %4083 : f32
    } -> tensor<1x196x256xf32>
    %4084 = arith.constant {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} 0.000000e+00 : f32
    %4085 = tensor.splat %4084 {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32>
    %4086 = linalg.reduce ins(%4079:tensor<1x196x256xf32>) outs(%4085:tensor<1x196xf32>) dimensions = [2]
    (%4087: f32, %4088: f32) {
      %4089 = arith.addf %4087, %4088 : f32
      linalg.yield %4089 : f32
    }
    %4090 = arith.constant {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} 2.560000e+02 : f32
    %4091 = tensor.splat %4090 {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32>
    %4092 = tensor.empty() : tensor<1x196xf32>
    %4093 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4086, %4091 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4092 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb370(%4094: f32, %4095: f32, %4096: f32):
      %4097 = arith.divf %4094, %4095 : f32
      linalg.yield %4097 : f32
    } -> tensor<1x196xf32>
    %4098 = tensor.collapse_shape %4093 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %4099 = tensor.expand_shape %4098 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4100 = arith.constant {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} 1.000000e-06 : f32
    %4101 = tensor.splat %4100 {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} : tensor<1x196x1xf32>
    %4102 = tensor.empty() : tensor<1x196x1xf32>
    %4103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4099, %4101 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%4102 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb371(%4104: f32, %4105: f32, %4106: f32):
      %4107 = arith.addf %4104, %4105 : f32
      linalg.yield %4107 : f32
    } -> tensor<1x196x1xf32>
    %4108 = tensor.empty() : tensor<1x196x1xf32>
    %4109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4103 : tensor<1x196x1xf32>) outs(%4108 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb372(%4110: f32, %4111: f32):
      %4112 = math.rsqrt %4110 : f32
      linalg.yield %4112 : f32
    } -> tensor<1x196x1xf32>
    %4113 = tensor.empty() : tensor<1x196x256xf32>
    %4114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4073, %4109 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4113 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb373(%4115: f32, %4116: f32, %4117: f32):
      %4118 = arith.mulf %4115, %4116 : f32
      linalg.yield %4118 : f32
    } -> tensor<1x196x256xf32>
    %4119 = tensor.empty() : tensor<1x196x256xf32>
    %4120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4114, %113 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4119 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb374(%4121: f32, %4122: f32, %4123: f32):
      %4124 = arith.mulf %4121, %4122 : f32
      linalg.yield %4124 : f32
    } -> tensor<1x196x256xf32>
    %4125 = tensor.empty() : tensor<1x196x256xf32>
    %4126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4120, %114 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4125 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_21", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.norm2"} {
    ^bb375(%4127: f32, %4128: f32, %4129: f32):
      %4130 = arith.addf %4127, %4128 : f32
      linalg.yield %4130 : f32
    } -> tensor<1x196x256xf32>
    %4131 = tensor.collapse_shape %4126 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_133", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %4132 = tensor.expand_shape %4131 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_133", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %4133 = tensor.empty() : tensor<256x1024xf32>
    %4134 = linalg.transpose ins(%115:tensor<1024x256xf32>) outs(%4133:tensor<256x1024xf32>) permutation = [1, 0]
    %4135 = tensor.empty() : tensor<196x1024xf32>
    %4136 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4137 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4136 : f32) outs(%4135 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %4138 = linalg.matmul {prov.region_id = "matmul_48", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1", prov.transposed_b = "true"} ins(%4132, %4134 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%4137 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %4139 = tensor.empty() : tensor<196x1024xf32>
    %4140 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4138, %116 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%4139 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_52", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1"} {
    ^bb376(%4141: f32, %4142: f32, %4143: f32):
      %4144 = arith.addf %4141, %4142 : f32
      linalg.yield %4144 : f32
    } -> tensor<196x1024xf32>
    %4145 = tensor.collapse_shape %4140 [[0 : i64, 1 : i64]] {prov.region_id = "view_134", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %4146 = tensor.expand_shape %4145 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_134", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %4147 = tensor.empty() : tensor<1x196x1024xf32>
    %4148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4146 : tensor<1x196x1024xf32>) outs(%4147 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_10", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.act"} {
    ^bb377(%4149: f32, %4150: f32):
      %4151 = arith.constant 5.000000e-01 : f32
      %4152 = arith.constant 1.000000e+00 : f32
      %4153 = arith.constant 0.707106769 : f32
      %4154 = arith.mulf %4149, %4153 : f32
      %4155 = math.erf %4154 : f32
      %4156 = arith.addf %4152, %4155 : f32
      %4157 = arith.mulf %4151, %4149 : f32
      %4158 = arith.mulf %4157, %4156 : f32
      linalg.yield %4158 : f32
    } -> tensor<1x196x1024xf32>
    %4159 = tensor.collapse_shape %4148 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_135", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %4160 = tensor.expand_shape %4159 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_135", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %4161 = tensor.empty() : tensor<1024x256xf32>
    %4162 = linalg.transpose ins(%117:tensor<256x1024xf32>) outs(%4161:tensor<1024x256xf32>) permutation = [1, 0]
    %4163 = tensor.empty() : tensor<196x256xf32>
    %4164 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4165 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4164 : f32) outs(%4163 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4166 = linalg.matmul {prov.region_id = "matmul_49", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2", prov.transposed_b = "true"} ins(%4160, %4162 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%4165 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4167 = tensor.empty() : tensor<196x256xf32>
    %4168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4166, %118 : tensor<196x256xf32>, tensor<256xf32>) outs(%4167 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_53", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2"} {
    ^bb378(%4169: f32, %4170: f32, %4171: f32):
      %4172 = arith.addf %4169, %4170 : f32
      linalg.yield %4172 : f32
    } -> tensor<196x256xf32>
    %4173 = tensor.collapse_shape %4168 [[0 : i64, 1 : i64]] {prov.region_id = "view_136", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %4174 = tensor.expand_shape %4173 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_136", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4175 = tensor.empty() : tensor<1x196x256xf32>
    %4176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4051, %4174 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4175 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_54", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.10"} {
    ^bb379(%4177: f32, %4178: f32, %4179: f32):
      %4180 = arith.addf %4177, %4178 : f32
      linalg.yield %4180 : f32
    } -> tensor<1x196x256xf32>
    %4181 = arith.constant {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} 0.000000e+00 : f32
    %4182 = tensor.splat %4181 {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32>
    %4183 = linalg.reduce ins(%4176:tensor<1x196x256xf32>) outs(%4182:tensor<1x196xf32>) dimensions = [2]
    (%4184: f32, %4185: f32) {
      %4186 = arith.addf %4184, %4185 : f32
      linalg.yield %4186 : f32
    }
    %4187 = arith.constant {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} 2.560000e+02 : f32
    %4188 = tensor.splat %4187 {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32>
    %4189 = tensor.empty() : tensor<1x196xf32>
    %4190 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4183, %4188 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4189 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb380(%4191: f32, %4192: f32, %4193: f32):
      %4194 = arith.divf %4191, %4192 : f32
      linalg.yield %4194 : f32
    } -> tensor<1x196xf32>
    %4195 = tensor.collapse_shape %4190 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %4196 = tensor.expand_shape %4195 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4197 = tensor.empty() : tensor<1x196x256xf32>
    %4198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4176, %4196 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4197 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb381(%4199: f32, %4200: f32, %4201: f32):
      %4202 = arith.subf %4199, %4200 : f32
      linalg.yield %4202 : f32
    } -> tensor<1x196x256xf32>
    %4203 = tensor.empty() : tensor<1x196x256xf32>
    %4204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4198, %4198 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4203 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb382(%4205: f32, %4206: f32, %4207: f32):
      %4208 = arith.mulf %4205, %4206 : f32
      linalg.yield %4208 : f32
    } -> tensor<1x196x256xf32>
    %4209 = arith.constant {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} 0.000000e+00 : f32
    %4210 = tensor.splat %4209 {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32>
    %4211 = linalg.reduce ins(%4204:tensor<1x196x256xf32>) outs(%4210:tensor<1x196xf32>) dimensions = [2]
    (%4212: f32, %4213: f32) {
      %4214 = arith.addf %4212, %4213 : f32
      linalg.yield %4214 : f32
    }
    %4215 = arith.constant {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} 2.560000e+02 : f32
    %4216 = tensor.splat %4215 {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32>
    %4217 = tensor.empty() : tensor<1x196xf32>
    %4218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4211, %4216 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4217 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb383(%4219: f32, %4220: f32, %4221: f32):
      %4222 = arith.divf %4219, %4220 : f32
      linalg.yield %4222 : f32
    } -> tensor<1x196xf32>
    %4223 = tensor.collapse_shape %4218 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196xf32> into tensor<196xf32>
    %4224 = tensor.expand_shape %4223 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4225 = arith.constant {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} 1.000000e-06 : f32
    %4226 = tensor.splat %4225 {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} : tensor<1x196x1xf32>
    %4227 = tensor.empty() : tensor<1x196x1xf32>
    %4228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4224, %4226 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%4227 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb384(%4229: f32, %4230: f32, %4231: f32):
      %4232 = arith.addf %4229, %4230 : f32
      linalg.yield %4232 : f32
    } -> tensor<1x196x1xf32>
    %4233 = tensor.empty() : tensor<1x196x1xf32>
    %4234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4228 : tensor<1x196x1xf32>) outs(%4233 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb385(%4235: f32, %4236: f32):
      %4237 = math.rsqrt %4235 : f32
      linalg.yield %4237 : f32
    } -> tensor<1x196x1xf32>
    %4238 = tensor.empty() : tensor<1x196x256xf32>
    %4239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4198, %4234 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4238 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb386(%4240: f32, %4241: f32, %4242: f32):
      %4243 = arith.mulf %4240, %4241 : f32
      linalg.yield %4243 : f32
    } -> tensor<1x196x256xf32>
    %4244 = tensor.empty() : tensor<1x196x256xf32>
    %4245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4239, %123 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4244 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb387(%4246: f32, %4247: f32, %4248: f32):
      %4249 = arith.mulf %4246, %4247 : f32
      linalg.yield %4249 : f32
    } -> tensor<1x196x256xf32>
    %4250 = tensor.empty() : tensor<1x196x256xf32>
    %4251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4245, %124 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4250 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_22", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm1"} {
    ^bb388(%4252: f32, %4253: f32, %4254: f32):
      %4255 = arith.addf %4252, %4253 : f32
      linalg.yield %4255 : f32
    } -> tensor<1x196x256xf32>
    %4256 = tensor.collapse_shape %4251 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_137", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %4257 = tensor.expand_shape %4256 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_137", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv"} : tensor<50176xf32> into tensor<196x256xf32>
    %4258 = tensor.empty() : tensor<256x768xf32>
    %4259 = linalg.transpose ins(%131:tensor<768x256xf32>) outs(%4258:tensor<256x768xf32>) permutation = [1, 0]
    %4260 = tensor.empty() : tensor<196x768xf32>
    %4261 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4262 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4261 : f32) outs(%4260 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %4263 = linalg.matmul {prov.region_id = "matmul_50", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv", prov.transposed_b = "true"} ins(%4257, %4259 : tensor<196x256xf32>, tensor<256x768xf32>) outs(%4262 : tensor<196x768xf32>) -> tensor<196x768xf32>
    %4264 = tensor.empty() : tensor<196x768xf32>
    %4265 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4263, %132 : tensor<196x768xf32>, tensor<768xf32>) outs(%4264 : tensor<196x768xf32>) attrs =  {prov.region_id = "add_55", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv"} {
    ^bb389(%4266: f32, %4267: f32, %4268: f32):
      %4269 = arith.addf %4266, %4267 : f32
      linalg.yield %4269 : f32
    } -> tensor<196x768xf32>
    %4270 = tensor.collapse_shape %4265 [[0 : i64, 1 : i64]] {prov.region_id = "view_138", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv"} : tensor<196x768xf32> into tensor<150528xf32>
    %4271 = tensor.expand_shape %4270 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 768] {prov.region_id = "view_138", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.qkv"} : tensor<150528xf32> into tensor<1x196x768xf32>
    %4272 = tensor.collapse_shape %4271 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_139", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x196x768xf32> into tensor<150528xf32>
    %4273 = tensor.expand_shape %4272 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 196, 3, 4, 64] {prov.region_id = "view_139", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<150528xf32> into tensor<1x196x3x4x64xf32>
    %4274 = tensor.empty() : tensor<3x1x4x196x64xf32>
    %4275 = linalg.transpose ins(%4273:tensor<1x196x3x4x64xf32>) outs(%4274:tensor<3x1x4x196x64xf32>) permutation = [2, 0, 3, 1, 4]
    %4276 = "tensor.extract_slice"(%4275) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_21", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %4277 = tensor.collapse_shape %4276 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_21", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %4278 = tensor.expand_shape %4277 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_21", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %4279 = "tensor.extract_slice"(%4275) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_22", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %4280 = tensor.collapse_shape %4279 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_22", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %4281 = tensor.expand_shape %4280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_22", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %4282 = "tensor.extract_slice"(%4275) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 4, 196, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_23", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : (tensor<3x1x4x196x64xf32>) -> tensor<1x1x4x196x64xf32>
    %4283 = tensor.collapse_shape %4282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "select_23", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x1x4x196x64xf32> into tensor<50176xf32>
    %4284 = tensor.expand_shape %4283 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "select_23", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %4285 = tensor.empty() : tensor<1x4x64x196xf32>
    %4286 = linalg.transpose ins(%4281:tensor<1x4x196x64xf32>) outs(%4285:tensor<1x4x64x196xf32>) permutation = [0, 1, 3, 2]
    %4287 = tensor.empty() : tensor<1x4x196x64xf32>
    %4288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4278 : tensor<1x4x196x64xf32>) outs(%4287 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_28", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb390(%4289: f32, %4290: f32):
      linalg.yield %4289 : f32
    } -> tensor<1x4x196x64xf32>
    %4291 = tensor.collapse_shape %4288 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_140", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %4292 = tensor.expand_shape %4291 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_140", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %4293 = tensor.empty() : tensor<1x4x64x196xf32>
    %4294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4286 : tensor<1x4x64x196xf32>) outs(%4293 : tensor<1x4x64x196xf32>) attrs =  {prov.region_id = "expand_29", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb391(%4295: f32, %4296: f32):
      linalg.yield %4295 : f32
    } -> tensor<1x4x64x196xf32>
    %4297 = tensor.collapse_shape %4294 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_141", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x64x196xf32> into tensor<50176xf32>
    %4298 = tensor.expand_shape %4297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 64, 196] {prov.region_id = "view_141", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<4x64x196xf32>
    %4299 = arith.constant {prov.region_id = "matmul_51", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} 0.000000e+00 : f32
    %4300 = tensor.splat %4299 {prov.region_id = "matmul_51", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<4x196x196xf32>
    %4301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4292, %4298 : tensor<4x196x64xf32>, tensor<4x64x196xf32>) outs(%4300 : tensor<4x196x196xf32>) attrs =  {prov.region_id = "matmul_51", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb392(%4302: f32, %4303: f32, %4304: f32):
      %4305 = arith.mulf %4302, %4303 : f32
      %4306 = arith.addf %4304, %4305 : f32
      linalg.yield %4306 : f32
    } -> tensor<4x196x196xf32>
    %4307 = tensor.collapse_shape %4301 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_142", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<4x196x196xf32> into tensor<153664xf32>
    %4308 = tensor.expand_shape %4307 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 196] {prov.region_id = "view_142", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<153664xf32> into tensor<1x4x196x196xf32>
    %4309 = arith.constant {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} 1.250000e-01 : f32
    %4310 = tensor.splat %4309 {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196x196xf32>
    %4311 = tensor.empty() : tensor<1x4x196x196xf32>
    %4312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4308, %4310 : tensor<1x4x196x196xf32>, tensor<1x4x196x196xf32>) outs(%4311 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb393(%4313: f32, %4314: f32, %4315: f32):
      %4316 = arith.mulf %4313, %4314 : f32
      linalg.yield %4316 : f32
    } -> tensor<1x4x196x196xf32>
    %4317 = arith.constant {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} 0xff800000 : f32
    %4318 = tensor.splat %4317 {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196xf32>
    %4319 = linalg.reduce ins(%4312:tensor<1x4x196x196xf32>) outs(%4318:tensor<1x4x196xf32>) dimensions = [3]
    (%4320: f32, %4321: f32) {
      %4322 = arith.maximumf %4320, %4321 : f32
      linalg.yield %4322 : f32
    }
    %4323 = tensor.collapse_shape %4319 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %4324 = tensor.expand_shape %4323 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %4325 = tensor.empty() : tensor<1x4x196x196xf32>
    %4326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4312, %4324 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%4325 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb394(%4327: f32, %4328: f32, %4329: f32):
      %4330 = arith.subf %4327, %4328 : f32
      linalg.yield %4330 : f32
    } -> tensor<1x4x196x196xf32>
    %4331 = tensor.empty() : tensor<1x4x196x196xf32>
    %4332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4326 : tensor<1x4x196x196xf32>) outs(%4331 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb395(%4333: f32, %4334: f32):
      %4335 = math.exp %4333 : f32
      linalg.yield %4335 : f32
    } -> tensor<1x4x196x196xf32>
    %4336 = arith.constant {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} 0.000000e+00 : f32
    %4337 = tensor.splat %4336 {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196xf32>
    %4338 = linalg.reduce ins(%4332:tensor<1x4x196x196xf32>) outs(%4337:tensor<1x4x196xf32>) dimensions = [3]
    (%4339: f32, %4340: f32) {
      %4341 = arith.addf %4339, %4340 : f32
      linalg.yield %4341 : f32
    }
    %4342 = tensor.collapse_shape %4338 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196xf32> into tensor<784xf32>
    %4343 = tensor.expand_shape %4342 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 1] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<784xf32> into tensor<1x4x196x1xf32>
    %4344 = tensor.empty() : tensor<1x4x196x196xf32>
    %4345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4332, %4343 : tensor<1x4x196x196xf32>, tensor<1x4x196x1xf32>) outs(%4344 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb396(%4346: f32, %4347: f32, %4348: f32):
      %4349 = arith.divf %4346, %4347 : f32
      linalg.yield %4349 : f32
    } -> tensor<1x4x196x196xf32>
    %4350 = tensor.empty() : tensor<1x4x196x196xf32>
    %4351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4345 : tensor<1x4x196x196xf32>) outs(%4350 : tensor<1x4x196x196xf32>) attrs =  {prov.region_id = "expand_30", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb397(%4352: f32, %4353: f32):
      linalg.yield %4352 : f32
    } -> tensor<1x4x196x196xf32>
    %4354 = tensor.collapse_shape %4351 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_143", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196x196xf32> into tensor<153664xf32>
    %4355 = tensor.expand_shape %4354 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 196] {prov.region_id = "view_143", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<153664xf32> into tensor<4x196x196xf32>
    %4356 = tensor.empty() : tensor<1x4x196x64xf32>
    %4357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4284 : tensor<1x4x196x64xf32>) outs(%4356 : tensor<1x4x196x64xf32>) attrs =  {prov.region_id = "expand_31", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb398(%4358: f32, %4359: f32):
      linalg.yield %4358 : f32
    } -> tensor<1x4x196x64xf32>
    %4360 = tensor.collapse_shape %4357 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_144", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x4x196x64xf32> into tensor<50176xf32>
    %4361 = tensor.expand_shape %4360 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 196, 64] {prov.region_id = "view_144", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<4x196x64xf32>
    %4362 = arith.constant {prov.region_id = "matmul_52", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} 0.000000e+00 : f32
    %4363 = tensor.splat %4362 {prov.region_id = "matmul_52", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<4x196x64xf32>
    %4364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4355, %4361 : tensor<4x196x196xf32>, tensor<4x196x64xf32>) outs(%4363 : tensor<4x196x64xf32>) attrs =  {prov.region_id = "matmul_52", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} {
    ^bb399(%4365: f32, %4366: f32, %4367: f32):
      %4368 = arith.mulf %4365, %4366 : f32
      %4369 = arith.addf %4367, %4368 : f32
      linalg.yield %4369 : f32
    } -> tensor<4x196x64xf32>
    %4370 = tensor.collapse_shape %4364 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_145", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<4x196x64xf32> into tensor<50176xf32>
    %4371 = tensor.expand_shape %4370 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 196, 64] {prov.region_id = "view_145", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<1x4x196x64xf32>
    %4372 = tensor.empty() : tensor<1x196x4x64xf32>
    %4373 = linalg.transpose ins(%4371:tensor<1x4x196x64xf32>) outs(%4372:tensor<1x196x4x64xf32>) permutation = [0, 2, 1, 3]
    %4374 = tensor.collapse_shape %4373 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_146", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<1x196x4x64xf32> into tensor<50176xf32>
    %4375 = tensor.expand_shape %4374 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_146", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4376 = tensor.collapse_shape %4375 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_147", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %4377 = tensor.expand_shape %4376 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_147", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj"} : tensor<50176xf32> into tensor<196x256xf32>
    %4378 = tensor.empty() : tensor<256x256xf32>
    %4379 = linalg.transpose ins(%133:tensor<256x256xf32>) outs(%4378:tensor<256x256xf32>) permutation = [1, 0]
    %4380 = tensor.empty() : tensor<196x256xf32>
    %4381 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4382 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4381 : f32) outs(%4380 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4383 = linalg.matmul {prov.region_id = "matmul_53", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj", prov.transposed_b = "true"} ins(%4377, %4379 : tensor<196x256xf32>, tensor<256x256xf32>) outs(%4382 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4384 = tensor.empty() : tensor<196x256xf32>
    %4385 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4383, %134 : tensor<196x256xf32>, tensor<256xf32>) outs(%4384 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_56", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj"} {
    ^bb400(%4386: f32, %4387: f32, %4388: f32):
      %4389 = arith.addf %4386, %4387 : f32
      linalg.yield %4389 : f32
    } -> tensor<196x256xf32>
    %4390 = tensor.collapse_shape %4385 [[0 : i64, 1 : i64]] {prov.region_id = "view_148", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj"} : tensor<196x256xf32> into tensor<50176xf32>
    %4391 = tensor.expand_shape %4390 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_148", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.attn.proj"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4392 = tensor.empty() : tensor<1x196x256xf32>
    %4393 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4176, %4391 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4392 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_57", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11"} {
    ^bb401(%4394: f32, %4395: f32, %4396: f32):
      %4397 = arith.addf %4394, %4395 : f32
      linalg.yield %4397 : f32
    } -> tensor<1x196x256xf32>
    %4398 = arith.constant {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} 0.000000e+00 : f32
    %4399 = tensor.splat %4398 {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32>
    %4400 = linalg.reduce ins(%4393:tensor<1x196x256xf32>) outs(%4399:tensor<1x196xf32>) dimensions = [2]
    (%4401: f32, %4402: f32) {
      %4403 = arith.addf %4401, %4402 : f32
      linalg.yield %4403 : f32
    }
    %4404 = arith.constant {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} 2.560000e+02 : f32
    %4405 = tensor.splat %4404 {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32>
    %4406 = tensor.empty() : tensor<1x196xf32>
    %4407 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4400, %4405 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4406 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb402(%4408: f32, %4409: f32, %4410: f32):
      %4411 = arith.divf %4408, %4409 : f32
      linalg.yield %4411 : f32
    } -> tensor<1x196xf32>
    %4412 = tensor.collapse_shape %4407 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %4413 = tensor.expand_shape %4412 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4414 = tensor.empty() : tensor<1x196x256xf32>
    %4415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4393, %4413 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4414 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb403(%4416: f32, %4417: f32, %4418: f32):
      %4419 = arith.subf %4416, %4417 : f32
      linalg.yield %4419 : f32
    } -> tensor<1x196x256xf32>
    %4420 = tensor.empty() : tensor<1x196x256xf32>
    %4421 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4415, %4415 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4420 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb404(%4422: f32, %4423: f32, %4424: f32):
      %4425 = arith.mulf %4422, %4423 : f32
      linalg.yield %4425 : f32
    } -> tensor<1x196x256xf32>
    %4426 = arith.constant {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} 0.000000e+00 : f32
    %4427 = tensor.splat %4426 {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32>
    %4428 = linalg.reduce ins(%4421:tensor<1x196x256xf32>) outs(%4427:tensor<1x196xf32>) dimensions = [2]
    (%4429: f32, %4430: f32) {
      %4431 = arith.addf %4429, %4430 : f32
      linalg.yield %4431 : f32
    }
    %4432 = arith.constant {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} 2.560000e+02 : f32
    %4433 = tensor.splat %4432 {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32>
    %4434 = tensor.empty() : tensor<1x196xf32>
    %4435 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4428, %4433 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4434 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb405(%4436: f32, %4437: f32, %4438: f32):
      %4439 = arith.divf %4436, %4437 : f32
      linalg.yield %4439 : f32
    } -> tensor<1x196xf32>
    %4440 = tensor.collapse_shape %4435 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196xf32> into tensor<196xf32>
    %4441 = tensor.expand_shape %4440 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4442 = arith.constant {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} 1.000000e-06 : f32
    %4443 = tensor.splat %4442 {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} : tensor<1x196x1xf32>
    %4444 = tensor.empty() : tensor<1x196x1xf32>
    %4445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4441, %4443 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%4444 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb406(%4446: f32, %4447: f32, %4448: f32):
      %4449 = arith.addf %4446, %4447 : f32
      linalg.yield %4449 : f32
    } -> tensor<1x196x1xf32>
    %4450 = tensor.empty() : tensor<1x196x1xf32>
    %4451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4445 : tensor<1x196x1xf32>) outs(%4450 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb407(%4452: f32, %4453: f32):
      %4454 = math.rsqrt %4452 : f32
      linalg.yield %4454 : f32
    } -> tensor<1x196x1xf32>
    %4455 = tensor.empty() : tensor<1x196x256xf32>
    %4456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4415, %4451 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4455 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb408(%4457: f32, %4458: f32, %4459: f32):
      %4460 = arith.mulf %4457, %4458 : f32
      linalg.yield %4460 : f32
    } -> tensor<1x196x256xf32>
    %4461 = tensor.empty() : tensor<1x196x256xf32>
    %4462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4456, %125 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4461 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb409(%4463: f32, %4464: f32, %4465: f32):
      %4466 = arith.mulf %4463, %4464 : f32
      linalg.yield %4466 : f32
    } -> tensor<1x196x256xf32>
    %4467 = tensor.empty() : tensor<1x196x256xf32>
    %4468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4462, %126 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4467 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_23", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.norm2"} {
    ^bb410(%4469: f32, %4470: f32, %4471: f32):
      %4472 = arith.addf %4469, %4470 : f32
      linalg.yield %4472 : f32
    } -> tensor<1x196x256xf32>
    %4473 = tensor.collapse_shape %4468 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_149", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1"} : tensor<1x196x256xf32> into tensor<50176xf32>
    %4474 = tensor.expand_shape %4473 [[0 : i64, 1 : i64]] output_shape [196, 256] {prov.region_id = "view_149", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1"} : tensor<50176xf32> into tensor<196x256xf32>
    %4475 = tensor.empty() : tensor<256x1024xf32>
    %4476 = linalg.transpose ins(%127:tensor<1024x256xf32>) outs(%4475:tensor<256x1024xf32>) permutation = [1, 0]
    %4477 = tensor.empty() : tensor<196x1024xf32>
    %4478 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4479 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4478 : f32) outs(%4477 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %4480 = linalg.matmul {prov.region_id = "matmul_54", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1", prov.transposed_b = "true"} ins(%4474, %4476 : tensor<196x256xf32>, tensor<256x1024xf32>) outs(%4479 : tensor<196x1024xf32>) -> tensor<196x1024xf32>
    %4481 = tensor.empty() : tensor<196x1024xf32>
    %4482 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4480, %128 : tensor<196x1024xf32>, tensor<1024xf32>) outs(%4481 : tensor<196x1024xf32>) attrs =  {prov.region_id = "add_58", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1"} {
    ^bb411(%4483: f32, %4484: f32, %4485: f32):
      %4486 = arith.addf %4483, %4484 : f32
      linalg.yield %4486 : f32
    } -> tensor<196x1024xf32>
    %4487 = tensor.collapse_shape %4482 [[0 : i64, 1 : i64]] {prov.region_id = "view_150", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1"} : tensor<196x1024xf32> into tensor<200704xf32>
    %4488 = tensor.expand_shape %4487 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1024] {prov.region_id = "view_150", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc1"} : tensor<200704xf32> into tensor<1x196x1024xf32>
    %4489 = tensor.empty() : tensor<1x196x1024xf32>
    %4490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4488 : tensor<1x196x1024xf32>) outs(%4489 : tensor<1x196x1024xf32>) attrs =  {prov.region_id = "gelu_11", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.act"} {
    ^bb412(%4491: f32, %4492: f32):
      %4493 = arith.constant 5.000000e-01 : f32
      %4494 = arith.constant 1.000000e+00 : f32
      %4495 = arith.constant 0.707106769 : f32
      %4496 = arith.mulf %4491, %4495 : f32
      %4497 = math.erf %4496 : f32
      %4498 = arith.addf %4494, %4497 : f32
      %4499 = arith.mulf %4493, %4491 : f32
      %4500 = arith.mulf %4499, %4498 : f32
      linalg.yield %4500 : f32
    } -> tensor<1x196x1024xf32>
    %4501 = tensor.collapse_shape %4490 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_151", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2"} : tensor<1x196x1024xf32> into tensor<200704xf32>
    %4502 = tensor.expand_shape %4501 [[0 : i64, 1 : i64]] output_shape [196, 1024] {prov.region_id = "view_151", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2"} : tensor<200704xf32> into tensor<196x1024xf32>
    %4503 = tensor.empty() : tensor<1024x256xf32>
    %4504 = linalg.transpose ins(%129:tensor<256x1024xf32>) outs(%4503:tensor<1024x256xf32>) permutation = [1, 0]
    %4505 = tensor.empty() : tensor<196x256xf32>
    %4506 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %4507 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%4506 : f32) outs(%4505 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4508 = linalg.matmul {prov.region_id = "matmul_55", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2", prov.transposed_b = "true"} ins(%4502, %4504 : tensor<196x1024xf32>, tensor<1024x256xf32>) outs(%4507 : tensor<196x256xf32>) -> tensor<196x256xf32>
    %4509 = tensor.empty() : tensor<196x256xf32>
    %4510 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4508, %130 : tensor<196x256xf32>, tensor<256xf32>) outs(%4509 : tensor<196x256xf32>) attrs =  {prov.region_id = "add_59", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2"} {
    ^bb413(%4511: f32, %4512: f32, %4513: f32):
      %4514 = arith.addf %4511, %4512 : f32
      linalg.yield %4514 : f32
    } -> tensor<196x256xf32>
    %4515 = tensor.collapse_shape %4510 [[0 : i64, 1 : i64]] {prov.region_id = "view_152", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2"} : tensor<196x256xf32> into tensor<50176xf32>
    %4516 = tensor.expand_shape %4515 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 256] {prov.region_id = "view_152", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11.mlp.fc2"} : tensor<50176xf32> into tensor<1x196x256xf32>
    %4517 = tensor.empty() : tensor<1x196x256xf32>
    %4518 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4393, %4516 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4517 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "add_60", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.11"} {
    ^bb414(%4519: f32, %4520: f32, %4521: f32):
      %4522 = arith.addf %4519, %4520 : f32
      linalg.yield %4522 : f32
    } -> tensor<1x196x256xf32>
    %4523 = arith.constant {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %4524 = tensor.splat %4523 {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32>
    %4525 = linalg.reduce ins(%4518:tensor<1x196x256xf32>) outs(%4524:tensor<1x196xf32>) dimensions = [2]
    (%4526: f32, %4527: f32) {
      %4528 = arith.addf %4526, %4527 : f32
      linalg.yield %4528 : f32
    }
    %4529 = arith.constant {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 2.560000e+02 : f32
    %4530 = tensor.splat %4529 {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32>
    %4531 = tensor.empty() : tensor<1x196xf32>
    %4532 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4525, %4530 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4531 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb415(%4533: f32, %4534: f32, %4535: f32):
      %4536 = arith.divf %4533, %4534 : f32
      linalg.yield %4536 : f32
    } -> tensor<1x196xf32>
    %4537 = tensor.collapse_shape %4532 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32> into tensor<196xf32>
    %4538 = tensor.expand_shape %4537 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4539 = tensor.empty() : tensor<1x196x256xf32>
    %4540 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4518, %4538 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4539 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb416(%4541: f32, %4542: f32, %4543: f32):
      %4544 = arith.subf %4541, %4542 : f32
      linalg.yield %4544 : f32
    } -> tensor<1x196x256xf32>
    %4545 = tensor.empty() : tensor<1x196x256xf32>
    %4546 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4540, %4540 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%4545 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb417(%4547: f32, %4548: f32, %4549: f32):
      %4550 = arith.mulf %4547, %4548 : f32
      linalg.yield %4550 : f32
    } -> tensor<1x196x256xf32>
    %4551 = arith.constant {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %4552 = tensor.splat %4551 {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32>
    %4553 = linalg.reduce ins(%4546:tensor<1x196x256xf32>) outs(%4552:tensor<1x196xf32>) dimensions = [2]
    (%4554: f32, %4555: f32) {
      %4556 = arith.addf %4554, %4555 : f32
      linalg.yield %4556 : f32
    }
    %4557 = arith.constant {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 2.560000e+02 : f32
    %4558 = tensor.splat %4557 {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32>
    %4559 = tensor.empty() : tensor<1x196xf32>
    %4560 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4553, %4558 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%4559 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb418(%4561: f32, %4562: f32, %4563: f32):
      %4564 = arith.divf %4561, %4562 : f32
      linalg.yield %4564 : f32
    } -> tensor<1x196xf32>
    %4565 = tensor.collapse_shape %4560 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196xf32> into tensor<196xf32>
    %4566 = tensor.expand_shape %4565 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<196xf32> into tensor<1x196x1xf32>
    %4567 = arith.constant {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.000000e-06 : f32
    %4568 = tensor.splat %4567 {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x196x1xf32>
    %4569 = tensor.empty() : tensor<1x196x1xf32>
    %4570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4566, %4568 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%4569 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb419(%4571: f32, %4572: f32, %4573: f32):
      %4574 = arith.addf %4571, %4572 : f32
      linalg.yield %4574 : f32
    } -> tensor<1x196x1xf32>
    %4575 = tensor.empty() : tensor<1x196x1xf32>
    %4576 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4570 : tensor<1x196x1xf32>) outs(%4575 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb420(%4577: f32, %4578: f32):
      %4579 = math.rsqrt %4577 : f32
      linalg.yield %4579 : f32
    } -> tensor<1x196x1xf32>
    %4580 = tensor.empty() : tensor<1x196x256xf32>
    %4581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4540, %4576 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%4580 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb421(%4582: f32, %4583: f32, %4584: f32):
      %4585 = arith.mulf %4582, %4583 : f32
      linalg.yield %4585 : f32
    } -> tensor<1x196x256xf32>
    %4586 = tensor.empty() : tensor<1x196x256xf32>
    %4587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4581, %135 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4586 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb422(%4588: f32, %4589: f32, %4590: f32):
      %4591 = arith.mulf %4588, %4589 : f32
      linalg.yield %4591 : f32
    } -> tensor<1x196x256xf32>
    %4592 = tensor.empty() : tensor<1x196x256xf32>
    %4593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4587, %136 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%4592 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_24", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb423(%4594: f32, %4595: f32, %4596: f32):
      %4597 = arith.addf %4594, %4595 : f32
      linalg.yield %4597 : f32
    } -> tensor<1x196x256xf32>
    %4598 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %4599 = tensor.splat %4598 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x256xf32>
    %4600 = linalg.reduce ins(%4593:tensor<1x196x256xf32>) outs(%4599:tensor<1x256xf32>) dimensions = [1]
    (%4601: f32, %4602: f32) {
      %4603 = arith.addf %4601, %4602 : f32
      linalg.yield %4603 : f32
    }
    %4604 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.960000e+02 : f32
    %4605 = tensor.splat %4604 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x256xf32>
    %4606 = tensor.empty() : tensor<1x256xf32>
    %4607 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4600, %4605 : tensor<1x256xf32>, tensor<1x256xf32>) outs(%4606 : tensor<1x256xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb424(%4608: f32, %4609: f32, %4610: f32):
      %4611 = arith.divf %4608, %4609 : f32
      linalg.yield %4611 : f32
    } -> tensor<1x256xf32>
    %4612 = tensor.empty() : tensor<256x1000xf32>
    %4613 = linalg.transpose ins(%137:tensor<1000x256xf32>) outs(%4612:tensor<256x1000xf32>) permutation = [1, 0]
    %4614 = tensor.empty() : tensor<1x1000xf32>
    %4615 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %4616 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%4615 : f32) outs(%4614 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %4617 = linalg.matmul {prov.region_id = "matmul_56", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head", prov.transposed_b = "true"} ins(%4607, %4613 : tensor<1x256xf32>, tensor<256x1000xf32>) outs(%4616 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %4618 = tensor.empty() : tensor<1x1000xf32>
    %4619 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4617, %138 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%4618 : tensor<1x1000xf32>) attrs =  {prov.region_id = "add_61", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head"} {
    ^bb425(%4620: f32, %4621: f32, %4622: f32):
      %4623 = arith.addf %4620, %4621 : f32
      linalg.yield %4623 : f32
    } -> tensor<1x1000xf32>
    func.return %4619 : tensor<1x1000xf32>
  }
}
