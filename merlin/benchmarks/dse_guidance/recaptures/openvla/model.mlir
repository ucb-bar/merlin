builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x1x192xf32>, %1: tensor<1x17x192xf32>, %2: tensor<192x3x16x16xf32>, %3: tensor<192xf32>, %4: tensor<192xf32>, %5: tensor<192xf32>, %6: tensor<576x192xf32>, %7: tensor<576xf32>, %8: tensor<192x192xf32>, %9: tensor<192xf32>, %10: tensor<192xf32>, %11: tensor<192xf32>, %12: tensor<768x192xf32>, %13: tensor<768xf32>, %14: tensor<192x768xf32>, %15: tensor<192xf32>, %16: tensor<192xf32>, %17: tensor<192xf32>, %18: tensor<576x192xf32>, %19: tensor<576xf32>, %20: tensor<192x192xf32>, %21: tensor<192xf32>, %22: tensor<192xf32>, %23: tensor<192xf32>, %24: tensor<768x192xf32>, %25: tensor<768xf32>, %26: tensor<192x768xf32>, %27: tensor<192xf32>, %28: tensor<192xf32>, %29: tensor<192xf32>, %30: tensor<1x1x384xf32>, %31: tensor<1x17x384xf32>, %32: tensor<384x3x16x16xf32>, %33: tensor<384xf32>, %34: tensor<384xf32>, %35: tensor<384xf32>, %36: tensor<1152x384xf32>, %37: tensor<1152xf32>, %38: tensor<384x384xf32>, %39: tensor<384xf32>, %40: tensor<384xf32>, %41: tensor<384xf32>, %42: tensor<1536x384xf32>, %43: tensor<1536xf32>, %44: tensor<384x1536xf32>, %45: tensor<384xf32>, %46: tensor<384xf32>, %47: tensor<384xf32>, %48: tensor<1152x384xf32>, %49: tensor<1152xf32>, %50: tensor<384x384xf32>, %51: tensor<384xf32>, %52: tensor<384xf32>, %53: tensor<384xf32>, %54: tensor<1536x384xf32>, %55: tensor<1536xf32>, %56: tensor<384x1536xf32>, %57: tensor<384xf32>, %58: tensor<384xf32>, %59: tensor<384xf32>, %60: tensor<2304x576xf32>, %61: tensor<2304xf32>, %62: tensor<128x2304xf32>, %63: tensor<128xf32>, %64: tensor<128x128xf32>, %65: tensor<128xf32>, %66: tensor<512x128xf32>, %67: tensor<512x128xf32>, %68: tensor<512x128xf32>, %69: tensor<512x128xf32>, %70: tensor<128x512xf32>, %71: tensor<256x128xf32>, %72: tensor<256x128xf32>, %73: tensor<128x256xf32>, %74: tensor<128xf32>, %75: tensor<128xf32>, %76: tensor<512x128xf32>, %77: tensor<512x128xf32>, %78: tensor<512x128xf32>, %79: tensor<128x512xf32>, %80: tensor<256x128xf32>, %81: tensor<256x128xf32>, %82: tensor<128x256xf32>, %83: tensor<128xf32>, %84: tensor<128xf32>, %85: tensor<128xf32>, %86: tensor<512x128xf32>, %87: tensor<f32>, %88: tensor<64xf32>, %89: tensor<64xf32>, %90: tensor<1x4xi64>, %91: tensor<1x6x64x64xf32>) -> tensor<1x20x512xf32> {
    %92 = "tensor.extract_slice"(%91) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 3, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone"} : (tensor<1x6x64x64xf32>) -> tensor<1x3x64x64xf32>
    %93 = "tensor.extract_slice"(%91) <{static_offsets = array<i64: 0, 3, 0, 0>, static_sizes = array<i64: 1, 3, 64, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone"} : (tensor<1x6x64x64xf32>) -> tensor<1x3x64x64xf32>
    %94 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed.proj"} 0.000000e+00 : f32
    %95 = tensor.splat %94 {prov.region_id = "conv_0", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed.proj"} : tensor<1x192x4x4xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, ((d2 * 16) + d5), ((d3 * 16) + d6))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%92, %2 : tensor<1x3x64x64xf32>, tensor<192x3x16x16xf32>) outs(%95 : tensor<1x192x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed.proj"} {
    ^bb0(%97: f32, %98: f32, %99: f32):
      %100 = arith.mulf %97, %98 : f32
      %101 = arith.addf %99, %100 : f32
      linalg.yield %101 : f32
    } -> tensor<1x192x4x4xf32>
    %102 = tensor.empty() : tensor<1x192x4x4xf32>
    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96, %3 : tensor<1x192x4x4xf32>, tensor<192xf32>) outs(%102 : tensor<1x192x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed.proj"} {
    ^bb1(%104: f32, %105: f32, %106: f32):
      %107 = arith.addf %104, %105 : f32
      linalg.yield %107 : f32
    } -> tensor<1x192x4x4xf32>
    %108 = tensor.collapse_shape %103 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed"} : tensor<1x192x4x4xf32> into tensor<3072xf32>
    %109 = tensor.expand_shape %108 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 192, 16] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.patch_embed"} : tensor<3072xf32> into tensor<1x192x16xf32>
    %110 = tensor.empty() : tensor<1x16x192xf32>
    %111 = linalg.transpose ins(%109:tensor<1x192x16xf32>) outs(%110:tensor<1x16x192xf32>) permutation = [0, 2, 1]
    %112 = tensor.empty() : tensor<1x1x192xf32>
    %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x192xf32>) outs(%112 : tensor<1x1x192xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer"} {
    ^bb2(%114: f32, %115: f32):
      linalg.yield %114 : f32
    } -> tensor<1x1x192xf32>
    %116 = tensor.concat dim(1) %113, %111 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer"} : (tensor<1x1x192xf32>, tensor<1x16x192xf32>) -> tensor<1x17x192xf32>
    %117 = tensor.empty() : tensor<1x17x192xf32>
    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %1 : tensor<1x17x192xf32>, tensor<1x17x192xf32>) outs(%117 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer"} {
    ^bb3(%119: f32, %120: f32, %121: f32):
      %122 = arith.addf %119, %120 : f32
      linalg.yield %122 : f32
    } -> tensor<1x17x192xf32>
    %123 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} 0.000000e+00 : f32
    %124 = tensor.splat %123 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %125 = linalg.reduce ins(%118:tensor<1x17x192xf32>) outs(%124:tensor<1x17xf32>) dimensions = [2]
    (%126: f32, %127: f32) {
      %128 = arith.addf %126, %127 : f32
      linalg.yield %128 : f32
    }
    %129 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} 1.920000e+02 : f32
    %130 = tensor.splat %129 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %131 = tensor.empty() : tensor<1x17xf32>
    %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %130 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%131 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb4(%133: f32, %134: f32, %135: f32):
      %136 = arith.divf %133, %134 : f32
      linalg.yield %136 : f32
    } -> tensor<1x17xf32>
    %137 = tensor.collapse_shape %132 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32> into tensor<17xf32>
    %138 = tensor.expand_shape %137 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<17xf32> into tensor<1x17x1xf32>
    %139 = tensor.empty() : tensor<1x17x192xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118, %138 : tensor<1x17x192xf32>, tensor<1x17x1xf32>) outs(%139 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb5(%141: f32, %142: f32, %143: f32):
      %144 = arith.subf %141, %142 : f32
      linalg.yield %144 : f32
    } -> tensor<1x17x192xf32>
    %145 = tensor.empty() : tensor<1x17x192xf32>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %140 : tensor<1x17x192xf32>, tensor<1x17x192xf32>) outs(%145 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb6(%147: f32, %148: f32, %149: f32):
      %150 = arith.mulf %147, %148 : f32
      linalg.yield %150 : f32
    } -> tensor<1x17x192xf32>
    %151 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} 0.000000e+00 : f32
    %152 = tensor.splat %151 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %153 = linalg.reduce ins(%146:tensor<1x17x192xf32>) outs(%152:tensor<1x17xf32>) dimensions = [2]
    (%154: f32, %155: f32) {
      %156 = arith.addf %154, %155 : f32
      linalg.yield %156 : f32
    }
    %157 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} 1.920000e+02 : f32
    %158 = tensor.splat %157 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %159 = tensor.empty() : tensor<1x17xf32>
    %160 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%153, %158 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%159 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb7(%161: f32, %162: f32, %163: f32):
      %164 = arith.divf %161, %162 : f32
      linalg.yield %164 : f32
    } -> tensor<1x17xf32>
    %165 = tensor.collapse_shape %160 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17xf32> into tensor<17xf32>
    %166 = tensor.expand_shape %165 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<17xf32> into tensor<1x17x1xf32>
    %167 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} 1.000000e-06 : f32
    %168 = tensor.splat %167 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} : tensor<1x17x1xf32>
    %169 = tensor.empty() : tensor<1x17x1xf32>
    %170 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%166, %168 : tensor<1x17x1xf32>, tensor<1x17x1xf32>) outs(%169 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb8(%171: f32, %172: f32, %173: f32):
      %174 = arith.addf %171, %172 : f32
      linalg.yield %174 : f32
    } -> tensor<1x17x1xf32>
    %175 = tensor.empty() : tensor<1x17x1xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%170 : tensor<1x17x1xf32>) outs(%175 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb9(%177: f32, %178: f32):
      %179 = math.rsqrt %177 : f32
      linalg.yield %179 : f32
    } -> tensor<1x17x1xf32>
    %180 = tensor.empty() : tensor<1x17x192xf32>
    %181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %176 : tensor<1x17x192xf32>, tensor<1x17x1xf32>) outs(%180 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb10(%182: f32, %183: f32, %184: f32):
      %185 = arith.mulf %182, %183 : f32
      linalg.yield %185 : f32
    } -> tensor<1x17x192xf32>
    %186 = tensor.empty() : tensor<1x17x192xf32>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181, %4 : tensor<1x17x192xf32>, tensor<192xf32>) outs(%186 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb11(%188: f32, %189: f32, %190: f32):
      %191 = arith.mulf %188, %189 : f32
      linalg.yield %191 : f32
    } -> tensor<1x17x192xf32>
    %192 = tensor.empty() : tensor<1x17x192xf32>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187, %5 : tensor<1x17x192xf32>, tensor<192xf32>) outs(%192 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm1"} {
    ^bb12(%194: f32, %195: f32, %196: f32):
      %197 = arith.addf %194, %195 : f32
      linalg.yield %197 : f32
    } -> tensor<1x17x192xf32>
    %198 = tensor.collapse_shape %193 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv"} : tensor<1x17x192xf32> into tensor<3264xf32>
    %199 = tensor.expand_shape %198 [[0 : i64, 1 : i64]] output_shape [17, 192] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv"} : tensor<3264xf32> into tensor<17x192xf32>
    %200 = tensor.empty() : tensor<192x576xf32>
    %201 = linalg.transpose ins(%6:tensor<576x192xf32>) outs(%200:tensor<192x576xf32>) permutation = [1, 0]
    %202 = tensor.empty() : tensor<17x576xf32>
    %203 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %204 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%203 : f32) outs(%202 : tensor<17x576xf32>) -> tensor<17x576xf32>
    %205 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv", prov.transposed_b = "true"} ins(%199, %201 : tensor<17x192xf32>, tensor<192x576xf32>) outs(%204 : tensor<17x576xf32>) -> tensor<17x576xf32>
    %206 = tensor.empty() : tensor<17x576xf32>
    %207 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%205, %7 : tensor<17x576xf32>, tensor<576xf32>) outs(%206 : tensor<17x576xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv"} {
    ^bb13(%208: f32, %209: f32, %210: f32):
      %211 = arith.addf %208, %209 : f32
      linalg.yield %211 : f32
    } -> tensor<17x576xf32>
    %212 = tensor.collapse_shape %207 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv"} : tensor<17x576xf32> into tensor<9792xf32>
    %213 = tensor.expand_shape %212 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 576] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.qkv"} : tensor<9792xf32> into tensor<1x17x576xf32>
    %214 = tensor.collapse_shape %213 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x17x576xf32> into tensor<9792xf32>
    %215 = tensor.expand_shape %214 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 17, 3, 3, 64] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<9792xf32> into tensor<1x17x3x3x64xf32>
    %216 = tensor.empty() : tensor<3x1x3x17x64xf32>
    %217 = linalg.transpose ins(%215:tensor<1x17x3x3x64xf32>) outs(%216:tensor<3x1x3x17x64xf32>) permutation = [2, 0, 3, 1, 4]
    %218 = "tensor.extract_slice"(%217) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 3, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : (tensor<3x1x3x17x64xf32>) -> tensor<1x1x3x17x64xf32>
    %219 = "tensor.extract_slice"(%217) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 3, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : (tensor<3x1x3x17x64xf32>) -> tensor<1x1x3x17x64xf32>
    %220 = "tensor.extract_slice"(%217) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 3, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : (tensor<3x1x3x17x64xf32>) -> tensor<1x1x3x17x64xf32>
    %221 = tensor.collapse_shape %218 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x1x3x17x64xf32> into tensor<3264xf32>
    %222 = tensor.expand_shape %221 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 64] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<1x3x17x64xf32>
    %223 = tensor.collapse_shape %219 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x1x3x17x64xf32> into tensor<3264xf32>
    %224 = tensor.expand_shape %223 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 64] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<1x3x17x64xf32>
    %225 = tensor.collapse_shape %220 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x1x3x17x64xf32> into tensor<3264xf32>
    %226 = tensor.expand_shape %225 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 64] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<1x3x17x64xf32>
    %227 = tensor.empty() : tensor<1x3x64x17xf32>
    %228 = linalg.transpose ins(%224:tensor<1x3x17x64xf32>) outs(%227:tensor<1x3x64x17xf32>) permutation = [0, 1, 3, 2]
    %229 = tensor.empty() : tensor<1x3x17x64xf32>
    %230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%222 : tensor<1x3x17x64xf32>) outs(%229 : tensor<1x3x17x64xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb14(%231: f32, %232: f32):
      linalg.yield %231 : f32
    } -> tensor<1x3x17x64xf32>
    %233 = tensor.collapse_shape %230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17x64xf32> into tensor<3264xf32>
    %234 = tensor.expand_shape %233 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 17, 64] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<3x17x64xf32>
    %235 = tensor.empty() : tensor<1x3x64x17xf32>
    %236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%228 : tensor<1x3x64x17xf32>) outs(%235 : tensor<1x3x64x17xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb15(%237: f32, %238: f32):
      linalg.yield %237 : f32
    } -> tensor<1x3x64x17xf32>
    %239 = tensor.collapse_shape %236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x64x17xf32> into tensor<3264xf32>
    %240 = tensor.expand_shape %239 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 64, 17] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<3x64x17xf32>
    %241 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %242 = tensor.splat %241 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3x17x17xf32>
    %243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%234, %240 : tensor<3x17x64xf32>, tensor<3x64x17xf32>) outs(%242 : tensor<3x17x17xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb16(%244: f32, %245: f32, %246: f32):
      %247 = arith.mulf %244, %245 : f32
      %248 = arith.addf %246, %247 : f32
      linalg.yield %248 : f32
    } -> tensor<3x17x17xf32>
    %249 = tensor.collapse_shape %243 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3x17x17xf32> into tensor<867xf32>
    %250 = tensor.expand_shape %249 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 17] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<867xf32> into tensor<1x3x17x17xf32>
    %251 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} 1.250000e-01 : f32
    %252 = tensor.splat %251 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17x17xf32>
    %253 = tensor.empty() : tensor<1x3x17x17xf32>
    %254 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%250, %252 : tensor<1x3x17x17xf32>, tensor<1x3x17x17xf32>) outs(%253 : tensor<1x3x17x17xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb17(%255: f32, %256: f32, %257: f32):
      %258 = arith.mulf %255, %256 : f32
      linalg.yield %258 : f32
    } -> tensor<1x3x17x17xf32>
    %259 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} 0xff800000 : f32
    %260 = tensor.splat %259 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17xf32>
    %261 = linalg.reduce ins(%254:tensor<1x3x17x17xf32>) outs(%260:tensor<1x3x17xf32>) dimensions = [3]
    (%262: f32, %263: f32) {
      %264 = arith.maximumf %262, %263 : f32
      linalg.yield %264 : f32
    }
    %265 = tensor.collapse_shape %261 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17xf32> into tensor<51xf32>
    %266 = tensor.expand_shape %265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<51xf32> into tensor<1x3x17x1xf32>
    %267 = tensor.empty() : tensor<1x3x17x17xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254, %266 : tensor<1x3x17x17xf32>, tensor<1x3x17x1xf32>) outs(%267 : tensor<1x3x17x17xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb18(%269: f32, %270: f32, %271: f32):
      %272 = arith.subf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<1x3x17x17xf32>
    %273 = tensor.empty() : tensor<1x3x17x17xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268 : tensor<1x3x17x17xf32>) outs(%273 : tensor<1x3x17x17xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb19(%275: f32, %276: f32):
      %277 = math.exp %275 : f32
      linalg.yield %277 : f32
    } -> tensor<1x3x17x17xf32>
    %278 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %279 = tensor.splat %278 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17xf32>
    %280 = linalg.reduce ins(%274:tensor<1x3x17x17xf32>) outs(%279:tensor<1x3x17xf32>) dimensions = [3]
    (%281: f32, %282: f32) {
      %283 = arith.addf %281, %282 : f32
      linalg.yield %283 : f32
    }
    %284 = tensor.collapse_shape %280 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17xf32> into tensor<51xf32>
    %285 = tensor.expand_shape %284 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<51xf32> into tensor<1x3x17x1xf32>
    %286 = tensor.empty() : tensor<1x3x17x17xf32>
    %287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %285 : tensor<1x3x17x17xf32>, tensor<1x3x17x1xf32>) outs(%286 : tensor<1x3x17x17xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb20(%288: f32, %289: f32, %290: f32):
      %291 = arith.divf %288, %289 : f32
      linalg.yield %291 : f32
    } -> tensor<1x3x17x17xf32>
    %292 = tensor.empty() : tensor<1x3x17x17xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287 : tensor<1x3x17x17xf32>) outs(%292 : tensor<1x3x17x17xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb21(%294: f32, %295: f32):
      linalg.yield %294 : f32
    } -> tensor<1x3x17x17xf32>
    %296 = tensor.collapse_shape %293 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17x17xf32> into tensor<867xf32>
    %297 = tensor.expand_shape %296 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 17, 17] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<867xf32> into tensor<3x17x17xf32>
    %298 = tensor.empty() : tensor<1x3x17x64xf32>
    %299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226 : tensor<1x3x17x64xf32>) outs(%298 : tensor<1x3x17x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb22(%300: f32, %301: f32):
      linalg.yield %300 : f32
    } -> tensor<1x3x17x64xf32>
    %302 = tensor.collapse_shape %299 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x3x17x64xf32> into tensor<3264xf32>
    %303 = tensor.expand_shape %302 [[0 : i64, 1 : i64, 2 : i64]] output_shape [3, 17, 64] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<3x17x64xf32>
    %304 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %305 = tensor.splat %304 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3x17x64xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%297, %303 : tensor<3x17x17xf32>, tensor<3x17x64xf32>) outs(%305 : tensor<3x17x64xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} {
    ^bb23(%307: f32, %308: f32, %309: f32):
      %310 = arith.mulf %307, %308 : f32
      %311 = arith.addf %309, %310 : f32
      linalg.yield %311 : f32
    } -> tensor<3x17x64xf32>
    %312 = tensor.collapse_shape %306 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3x17x64xf32> into tensor<3264xf32>
    %313 = tensor.expand_shape %312 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 3, 17, 64] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<1x3x17x64xf32>
    %314 = tensor.empty() : tensor<1x17x3x64xf32>
    %315 = linalg.transpose ins(%313:tensor<1x3x17x64xf32>) outs(%314:tensor<1x17x3x64xf32>) permutation = [0, 2, 1, 3]
    %316 = tensor.collapse_shape %315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<1x17x3x64xf32> into tensor<3264xf32>
    %317 = tensor.expand_shape %316 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 192] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn"} : tensor<3264xf32> into tensor<1x17x192xf32>
    %318 = tensor.collapse_shape %317 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj"} : tensor<1x17x192xf32> into tensor<3264xf32>
    %319 = tensor.expand_shape %318 [[0 : i64, 1 : i64]] output_shape [17, 192] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj"} : tensor<3264xf32> into tensor<17x192xf32>
    %320 = tensor.empty() : tensor<192x192xf32>
    %321 = linalg.transpose ins(%8:tensor<192x192xf32>) outs(%320:tensor<192x192xf32>) permutation = [1, 0]
    %322 = tensor.empty() : tensor<17x192xf32>
    %323 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %324 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%323 : f32) outs(%322 : tensor<17x192xf32>) -> tensor<17x192xf32>
    %325 = linalg.matmul {prov.region_id = "matmul_3", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj", prov.transposed_b = "true"} ins(%319, %321 : tensor<17x192xf32>, tensor<192x192xf32>) outs(%324 : tensor<17x192xf32>) -> tensor<17x192xf32>
    %326 = tensor.empty() : tensor<17x192xf32>
    %327 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325, %9 : tensor<17x192xf32>, tensor<192xf32>) outs(%326 : tensor<17x192xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj"} {
    ^bb24(%328: f32, %329: f32, %330: f32):
      %331 = arith.addf %328, %329 : f32
      linalg.yield %331 : f32
    } -> tensor<17x192xf32>
    %332 = tensor.collapse_shape %327 [[0 : i64, 1 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj"} : tensor<17x192xf32> into tensor<3264xf32>
    %333 = tensor.expand_shape %332 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 192] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.attn.proj"} : tensor<3264xf32> into tensor<1x17x192xf32>
    %334 = tensor.empty() : tensor<1x17x192xf32>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118, %333 : tensor<1x17x192xf32>, tensor<1x17x192xf32>) outs(%334 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0"} {
    ^bb25(%336: f32, %337: f32, %338: f32):
      %339 = arith.addf %336, %337 : f32
      linalg.yield %339 : f32
    } -> tensor<1x17x192xf32>
    %340 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} 0.000000e+00 : f32
    %341 = tensor.splat %340 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %342 = linalg.reduce ins(%335:tensor<1x17x192xf32>) outs(%341:tensor<1x17xf32>) dimensions = [2]
    (%343: f32, %344: f32) {
      %345 = arith.addf %343, %344 : f32
      linalg.yield %345 : f32
    }
    %346 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} 1.920000e+02 : f32
    %347 = tensor.splat %346 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %348 = tensor.empty() : tensor<1x17xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%342, %347 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%348 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb26(%350: f32, %351: f32, %352: f32):
      %353 = arith.divf %350, %351 : f32
      linalg.yield %353 : f32
    } -> tensor<1x17xf32>
    %354 = tensor.collapse_shape %349 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32> into tensor<17xf32>
    %355 = tensor.expand_shape %354 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<17xf32> into tensor<1x17x1xf32>
    %356 = tensor.empty() : tensor<1x17x192xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%335, %355 : tensor<1x17x192xf32>, tensor<1x17x1xf32>) outs(%356 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb27(%358: f32, %359: f32, %360: f32):
      %361 = arith.subf %358, %359 : f32
      linalg.yield %361 : f32
    } -> tensor<1x17x192xf32>
    %362 = tensor.empty() : tensor<1x17x192xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%357, %357 : tensor<1x17x192xf32>, tensor<1x17x192xf32>) outs(%362 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb28(%364: f32, %365: f32, %366: f32):
      %367 = arith.mulf %364, %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x17x192xf32>
    %368 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} 0.000000e+00 : f32
    %369 = tensor.splat %368 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %370 = linalg.reduce ins(%363:tensor<1x17x192xf32>) outs(%369:tensor<1x17xf32>) dimensions = [2]
    (%371: f32, %372: f32) {
      %373 = arith.addf %371, %372 : f32
      linalg.yield %373 : f32
    }
    %374 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} 1.920000e+02 : f32
    %375 = tensor.splat %374 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %376 = tensor.empty() : tensor<1x17xf32>
    %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%370, %375 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%376 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb29(%378: f32, %379: f32, %380: f32):
      %381 = arith.divf %378, %379 : f32
      linalg.yield %381 : f32
    } -> tensor<1x17xf32>
    %382 = tensor.collapse_shape %377 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17xf32> into tensor<17xf32>
    %383 = tensor.expand_shape %382 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<17xf32> into tensor<1x17x1xf32>
    %384 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} 1.000000e-06 : f32
    %385 = tensor.splat %384 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} : tensor<1x17x1xf32>
    %386 = tensor.empty() : tensor<1x17x1xf32>
    %387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%383, %385 : tensor<1x17x1xf32>, tensor<1x17x1xf32>) outs(%386 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb30(%388: f32, %389: f32, %390: f32):
      %391 = arith.addf %388, %389 : f32
      linalg.yield %391 : f32
    } -> tensor<1x17x1xf32>
    %392 = tensor.empty() : tensor<1x17x1xf32>
    %393 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%387 : tensor<1x17x1xf32>) outs(%392 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb31(%394: f32, %395: f32):
      %396 = math.rsqrt %394 : f32
      linalg.yield %396 : f32
    } -> tensor<1x17x1xf32>
    %397 = tensor.empty() : tensor<1x17x192xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%357, %393 : tensor<1x17x192xf32>, tensor<1x17x1xf32>) outs(%397 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb32(%399: f32, %400: f32, %401: f32):
      %402 = arith.mulf %399, %400 : f32
      linalg.yield %402 : f32
    } -> tensor<1x17x192xf32>
    %403 = tensor.empty() : tensor<1x17x192xf32>
    %404 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%398, %10 : tensor<1x17x192xf32>, tensor<192xf32>) outs(%403 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb33(%405: f32, %406: f32, %407: f32):
      %408 = arith.mulf %405, %406 : f32
      linalg.yield %408 : f32
    } -> tensor<1x17x192xf32>
    %409 = tensor.empty() : tensor<1x17x192xf32>
    %410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%404, %11 : tensor<1x17x192xf32>, tensor<192xf32>) outs(%409 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.norm2"} {
    ^bb34(%411: f32, %412: f32, %413: f32):
      %414 = arith.addf %411, %412 : f32
      linalg.yield %414 : f32
    } -> tensor<1x17x192xf32>
    %415 = tensor.collapse_shape %410 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1"} : tensor<1x17x192xf32> into tensor<3264xf32>
    %416 = tensor.expand_shape %415 [[0 : i64, 1 : i64]] output_shape [17, 192] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1"} : tensor<3264xf32> into tensor<17x192xf32>
    %417 = tensor.empty() : tensor<192x768xf32>
    %418 = linalg.transpose ins(%12:tensor<768x192xf32>) outs(%417:tensor<192x768xf32>) permutation = [1, 0]
    %419 = tensor.empty() : tensor<17x768xf32>
    %420 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %421 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%420 : f32) outs(%419 : tensor<17x768xf32>) -> tensor<17x768xf32>
    %422 = linalg.matmul {prov.region_id = "matmul_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1", prov.transposed_b = "true"} ins(%416, %418 : tensor<17x192xf32>, tensor<192x768xf32>) outs(%421 : tensor<17x768xf32>) -> tensor<17x768xf32>
    %423 = tensor.empty() : tensor<17x768xf32>
    %424 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%422, %13 : tensor<17x768xf32>, tensor<768xf32>) outs(%423 : tensor<17x768xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1"} {
    ^bb35(%425: f32, %426: f32, %427: f32):
      %428 = arith.addf %425, %426 : f32
      linalg.yield %428 : f32
    } -> tensor<17x768xf32>
    %429 = tensor.collapse_shape %424 [[0 : i64, 1 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1"} : tensor<17x768xf32> into tensor<13056xf32>
    %430 = tensor.expand_shape %429 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 768] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc1"} : tensor<13056xf32> into tensor<1x17x768xf32>
    %431 = tensor.empty() : tensor<1x17x768xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%430 : tensor<1x17x768xf32>) outs(%431 : tensor<1x17x768xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.act"} {
    ^bb36(%433: f32, %434: f32):
      %435 = arith.constant 5.000000e-01 : f32
      %436 = arith.constant 1.000000e+00 : f32
      %437 = arith.constant 0.707106769 : f32
      %438 = arith.mulf %433, %437 : f32
      %439 = math.erf %438 : f32
      %440 = arith.addf %436, %439 : f32
      %441 = arith.mulf %435, %433 : f32
      %442 = arith.mulf %441, %440 : f32
      linalg.yield %442 : f32
    } -> tensor<1x17x768xf32>
    %443 = tensor.collapse_shape %432 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2"} : tensor<1x17x768xf32> into tensor<13056xf32>
    %444 = tensor.expand_shape %443 [[0 : i64, 1 : i64]] output_shape [17, 768] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2"} : tensor<13056xf32> into tensor<17x768xf32>
    %445 = tensor.empty() : tensor<768x192xf32>
    %446 = linalg.transpose ins(%14:tensor<192x768xf32>) outs(%445:tensor<768x192xf32>) permutation = [1, 0]
    %447 = tensor.empty() : tensor<17x192xf32>
    %448 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %449 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%448 : f32) outs(%447 : tensor<17x192xf32>) -> tensor<17x192xf32>
    %450 = linalg.matmul {prov.region_id = "matmul_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2", prov.transposed_b = "true"} ins(%444, %446 : tensor<17x768xf32>, tensor<768x192xf32>) outs(%449 : tensor<17x192xf32>) -> tensor<17x192xf32>
    %451 = tensor.empty() : tensor<17x192xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%450, %15 : tensor<17x192xf32>, tensor<192xf32>) outs(%451 : tensor<17x192xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2"} {
    ^bb37(%453: f32, %454: f32, %455: f32):
      %456 = arith.addf %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<17x192xf32>
    %457 = tensor.collapse_shape %452 [[0 : i64, 1 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2"} : tensor<17x192xf32> into tensor<3264xf32>
    %458 = tensor.expand_shape %457 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 192] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0.mlp.fc2"} : tensor<3264xf32> into tensor<1x17x192xf32>
    %459 = tensor.empty() : tensor<1x17x192xf32>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%335, %458 : tensor<1x17x192xf32>, tensor<1x17x192xf32>) outs(%459 : tensor<1x17x192xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer.blocks.0"} {
    ^bb38(%461: f32, %462: f32, %463: f32):
      %464 = arith.addf %461, %462 : f32
      linalg.yield %464 : f32
    } -> tensor<1x17x192xf32>
    %465 = "tensor.extract_slice"(%460) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 16, 192>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.featurizer"} : (tensor<1x17x192xf32>) -> tensor<1x16x192xf32>
    %466 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed.proj"} 0.000000e+00 : f32
    %467 = tensor.splat %466 {prov.region_id = "conv_1", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed.proj"} : tensor<1x384x4x4xf32>
    %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, ((d2 * 16) + d5), ((d3 * 16) + d6))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%93, %32 : tensor<1x3x64x64xf32>, tensor<384x3x16x16xf32>) outs(%467 : tensor<1x384x4x4xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed.proj"} {
    ^bb39(%469: f32, %470: f32, %471: f32):
      %472 = arith.mulf %469, %470 : f32
      %473 = arith.addf %471, %472 : f32
      linalg.yield %473 : f32
    } -> tensor<1x384x4x4xf32>
    %474 = tensor.empty() : tensor<1x384x4x4xf32>
    %475 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %33 : tensor<1x384x4x4xf32>, tensor<384xf32>) outs(%474 : tensor<1x384x4x4xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov._pattern_hint = "conv2d", prov.op = "conv2d", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed.proj"} {
    ^bb40(%476: f32, %477: f32, %478: f32):
      %479 = arith.addf %476, %477 : f32
      linalg.yield %479 : f32
    } -> tensor<1x384x4x4xf32>
    %480 = tensor.collapse_shape %475 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed"} : tensor<1x384x4x4xf32> into tensor<6144xf32>
    %481 = tensor.expand_shape %480 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 384, 16] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.patch_embed"} : tensor<6144xf32> into tensor<1x384x16xf32>
    %482 = tensor.empty() : tensor<1x16x384xf32>
    %483 = linalg.transpose ins(%481:tensor<1x384x16xf32>) outs(%482:tensor<1x16x384xf32>) permutation = [0, 2, 1]
    %484 = tensor.empty() : tensor<1x1x384xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<1x1x384xf32>) outs(%484 : tensor<1x1x384xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer"} {
    ^bb41(%486: f32, %487: f32):
      linalg.yield %486 : f32
    } -> tensor<1x1x384xf32>
    %488 = tensor.concat dim(1) %485, %483 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer"} : (tensor<1x1x384xf32>, tensor<1x16x384xf32>) -> tensor<1x17x384xf32>
    %489 = tensor.empty() : tensor<1x17x384xf32>
    %490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%488, %31 : tensor<1x17x384xf32>, tensor<1x17x384xf32>) outs(%489 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer"} {
    ^bb42(%491: f32, %492: f32, %493: f32):
      %494 = arith.addf %491, %492 : f32
      linalg.yield %494 : f32
    } -> tensor<1x17x384xf32>
    %495 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} 0.000000e+00 : f32
    %496 = tensor.splat %495 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %497 = linalg.reduce ins(%490:tensor<1x17x384xf32>) outs(%496:tensor<1x17xf32>) dimensions = [2]
    (%498: f32, %499: f32) {
      %500 = arith.addf %498, %499 : f32
      linalg.yield %500 : f32
    }
    %501 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} 3.840000e+02 : f32
    %502 = tensor.splat %501 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %503 = tensor.empty() : tensor<1x17xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%497, %502 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%503 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb43(%505: f32, %506: f32, %507: f32):
      %508 = arith.divf %505, %506 : f32
      linalg.yield %508 : f32
    } -> tensor<1x17xf32>
    %509 = tensor.collapse_shape %504 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32> into tensor<17xf32>
    %510 = tensor.expand_shape %509 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<17xf32> into tensor<1x17x1xf32>
    %511 = tensor.empty() : tensor<1x17x384xf32>
    %512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%490, %510 : tensor<1x17x384xf32>, tensor<1x17x1xf32>) outs(%511 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb44(%513: f32, %514: f32, %515: f32):
      %516 = arith.subf %513, %514 : f32
      linalg.yield %516 : f32
    } -> tensor<1x17x384xf32>
    %517 = tensor.empty() : tensor<1x17x384xf32>
    %518 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%512, %512 : tensor<1x17x384xf32>, tensor<1x17x384xf32>) outs(%517 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb45(%519: f32, %520: f32, %521: f32):
      %522 = arith.mulf %519, %520 : f32
      linalg.yield %522 : f32
    } -> tensor<1x17x384xf32>
    %523 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} 0.000000e+00 : f32
    %524 = tensor.splat %523 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %525 = linalg.reduce ins(%518:tensor<1x17x384xf32>) outs(%524:tensor<1x17xf32>) dimensions = [2]
    (%526: f32, %527: f32) {
      %528 = arith.addf %526, %527 : f32
      linalg.yield %528 : f32
    }
    %529 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} 3.840000e+02 : f32
    %530 = tensor.splat %529 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32>
    %531 = tensor.empty() : tensor<1x17xf32>
    %532 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%525, %530 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%531 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb46(%533: f32, %534: f32, %535: f32):
      %536 = arith.divf %533, %534 : f32
      linalg.yield %536 : f32
    } -> tensor<1x17xf32>
    %537 = tensor.collapse_shape %532 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17xf32> into tensor<17xf32>
    %538 = tensor.expand_shape %537 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<17xf32> into tensor<1x17x1xf32>
    %539 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} 1.000000e-06 : f32
    %540 = tensor.splat %539 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} : tensor<1x17x1xf32>
    %541 = tensor.empty() : tensor<1x17x1xf32>
    %542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%538, %540 : tensor<1x17x1xf32>, tensor<1x17x1xf32>) outs(%541 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb47(%543: f32, %544: f32, %545: f32):
      %546 = arith.addf %543, %544 : f32
      linalg.yield %546 : f32
    } -> tensor<1x17x1xf32>
    %547 = tensor.empty() : tensor<1x17x1xf32>
    %548 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542 : tensor<1x17x1xf32>) outs(%547 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb48(%549: f32, %550: f32):
      %551 = math.rsqrt %549 : f32
      linalg.yield %551 : f32
    } -> tensor<1x17x1xf32>
    %552 = tensor.empty() : tensor<1x17x384xf32>
    %553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%512, %548 : tensor<1x17x384xf32>, tensor<1x17x1xf32>) outs(%552 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb49(%554: f32, %555: f32, %556: f32):
      %557 = arith.mulf %554, %555 : f32
      linalg.yield %557 : f32
    } -> tensor<1x17x384xf32>
    %558 = tensor.empty() : tensor<1x17x384xf32>
    %559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553, %34 : tensor<1x17x384xf32>, tensor<384xf32>) outs(%558 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb50(%560: f32, %561: f32, %562: f32):
      %563 = arith.mulf %560, %561 : f32
      linalg.yield %563 : f32
    } -> tensor<1x17x384xf32>
    %564 = tensor.empty() : tensor<1x17x384xf32>
    %565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%559, %35 : tensor<1x17x384xf32>, tensor<384xf32>) outs(%564 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm1"} {
    ^bb51(%566: f32, %567: f32, %568: f32):
      %569 = arith.addf %566, %567 : f32
      linalg.yield %569 : f32
    } -> tensor<1x17x384xf32>
    %570 = tensor.collapse_shape %565 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv"} : tensor<1x17x384xf32> into tensor<6528xf32>
    %571 = tensor.expand_shape %570 [[0 : i64, 1 : i64]] output_shape [17, 384] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv"} : tensor<6528xf32> into tensor<17x384xf32>
    %572 = tensor.empty() : tensor<384x1152xf32>
    %573 = linalg.transpose ins(%36:tensor<1152x384xf32>) outs(%572:tensor<384x1152xf32>) permutation = [1, 0]
    %574 = tensor.empty() : tensor<17x1152xf32>
    %575 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %576 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%575 : f32) outs(%574 : tensor<17x1152xf32>) -> tensor<17x1152xf32>
    %577 = linalg.matmul {prov.region_id = "matmul_6", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv", prov.transposed_b = "true"} ins(%571, %573 : tensor<17x384xf32>, tensor<384x1152xf32>) outs(%576 : tensor<17x1152xf32>) -> tensor<17x1152xf32>
    %578 = tensor.empty() : tensor<17x1152xf32>
    %579 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%577, %37 : tensor<17x1152xf32>, tensor<1152xf32>) outs(%578 : tensor<17x1152xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv"} {
    ^bb52(%580: f32, %581: f32, %582: f32):
      %583 = arith.addf %580, %581 : f32
      linalg.yield %583 : f32
    } -> tensor<17x1152xf32>
    %584 = tensor.collapse_shape %579 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv"} : tensor<17x1152xf32> into tensor<19584xf32>
    %585 = tensor.expand_shape %584 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1152] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.qkv"} : tensor<19584xf32> into tensor<1x17x1152xf32>
    %586 = tensor.collapse_shape %585 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x17x1152xf32> into tensor<19584xf32>
    %587 = tensor.expand_shape %586 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 17, 3, 6, 64] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<19584xf32> into tensor<1x17x3x6x64xf32>
    %588 = tensor.empty() : tensor<3x1x6x17x64xf32>
    %589 = linalg.transpose ins(%587:tensor<1x17x3x6x64xf32>) outs(%588:tensor<3x1x6x17x64xf32>) permutation = [2, 0, 3, 1, 4]
    %590 = "tensor.extract_slice"(%589) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 6, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : (tensor<3x1x6x17x64xf32>) -> tensor<1x1x6x17x64xf32>
    %591 = "tensor.extract_slice"(%589) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 6, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : (tensor<3x1x6x17x64xf32>) -> tensor<1x1x6x17x64xf32>
    %592 = "tensor.extract_slice"(%589) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 6, 17, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : (tensor<3x1x6x17x64xf32>) -> tensor<1x1x6x17x64xf32>
    %593 = tensor.collapse_shape %590 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x1x6x17x64xf32> into tensor<6528xf32>
    %594 = tensor.expand_shape %593 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 64] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<1x6x17x64xf32>
    %595 = tensor.collapse_shape %591 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x1x6x17x64xf32> into tensor<6528xf32>
    %596 = tensor.expand_shape %595 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 64] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<1x6x17x64xf32>
    %597 = tensor.collapse_shape %592 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x1x6x17x64xf32> into tensor<6528xf32>
    %598 = tensor.expand_shape %597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 64] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<1x6x17x64xf32>
    %599 = tensor.empty() : tensor<1x6x64x17xf32>
    %600 = linalg.transpose ins(%596:tensor<1x6x17x64xf32>) outs(%599:tensor<1x6x64x17xf32>) permutation = [0, 1, 3, 2]
    %601 = tensor.empty() : tensor<1x6x17x64xf32>
    %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%594 : tensor<1x6x17x64xf32>) outs(%601 : tensor<1x6x17x64xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb53(%603: f32, %604: f32):
      linalg.yield %603 : f32
    } -> tensor<1x6x17x64xf32>
    %605 = tensor.collapse_shape %602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17x64xf32> into tensor<6528xf32>
    %606 = tensor.expand_shape %605 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 17, 64] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<6x17x64xf32>
    %607 = tensor.empty() : tensor<1x6x64x17xf32>
    %608 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%600 : tensor<1x6x64x17xf32>) outs(%607 : tensor<1x6x64x17xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb54(%609: f32, %610: f32):
      linalg.yield %609 : f32
    } -> tensor<1x6x64x17xf32>
    %611 = tensor.collapse_shape %608 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x64x17xf32> into tensor<6528xf32>
    %612 = tensor.expand_shape %611 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 64, 17] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<6x64x17xf32>
    %613 = arith.constant {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %614 = tensor.splat %613 {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6x17x17xf32>
    %615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%606, %612 : tensor<6x17x64xf32>, tensor<6x64x17xf32>) outs(%614 : tensor<6x17x17xf32>) attrs =  {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb55(%616: f32, %617: f32, %618: f32):
      %619 = arith.mulf %616, %617 : f32
      %620 = arith.addf %618, %619 : f32
      linalg.yield %620 : f32
    } -> tensor<6x17x17xf32>
    %621 = tensor.collapse_shape %615 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6x17x17xf32> into tensor<1734xf32>
    %622 = tensor.expand_shape %621 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 17] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1734xf32> into tensor<1x6x17x17xf32>
    %623 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} 1.250000e-01 : f32
    %624 = tensor.splat %623 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17x17xf32>
    %625 = tensor.empty() : tensor<1x6x17x17xf32>
    %626 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%622, %624 : tensor<1x6x17x17xf32>, tensor<1x6x17x17xf32>) outs(%625 : tensor<1x6x17x17xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb56(%627: f32, %628: f32, %629: f32):
      %630 = arith.mulf %627, %628 : f32
      linalg.yield %630 : f32
    } -> tensor<1x6x17x17xf32>
    %631 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} 0xff800000 : f32
    %632 = tensor.splat %631 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17xf32>
    %633 = linalg.reduce ins(%626:tensor<1x6x17x17xf32>) outs(%632:tensor<1x6x17xf32>) dimensions = [3]
    (%634: f32, %635: f32) {
      %636 = arith.maximumf %634, %635 : f32
      linalg.yield %636 : f32
    }
    %637 = tensor.collapse_shape %633 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17xf32> into tensor<102xf32>
    %638 = tensor.expand_shape %637 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<102xf32> into tensor<1x6x17x1xf32>
    %639 = tensor.empty() : tensor<1x6x17x17xf32>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%626, %638 : tensor<1x6x17x17xf32>, tensor<1x6x17x1xf32>) outs(%639 : tensor<1x6x17x17xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb57(%641: f32, %642: f32, %643: f32):
      %644 = arith.subf %641, %642 : f32
      linalg.yield %644 : f32
    } -> tensor<1x6x17x17xf32>
    %645 = tensor.empty() : tensor<1x6x17x17xf32>
    %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%640 : tensor<1x6x17x17xf32>) outs(%645 : tensor<1x6x17x17xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb58(%647: f32, %648: f32):
      %649 = math.exp %647 : f32
      linalg.yield %649 : f32
    } -> tensor<1x6x17x17xf32>
    %650 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %651 = tensor.splat %650 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17xf32>
    %652 = linalg.reduce ins(%646:tensor<1x6x17x17xf32>) outs(%651:tensor<1x6x17xf32>) dimensions = [3]
    (%653: f32, %654: f32) {
      %655 = arith.addf %653, %654 : f32
      linalg.yield %655 : f32
    }
    %656 = tensor.collapse_shape %652 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17xf32> into tensor<102xf32>
    %657 = tensor.expand_shape %656 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<102xf32> into tensor<1x6x17x1xf32>
    %658 = tensor.empty() : tensor<1x6x17x17xf32>
    %659 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%646, %657 : tensor<1x6x17x17xf32>, tensor<1x6x17x1xf32>) outs(%658 : tensor<1x6x17x17xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb59(%660: f32, %661: f32, %662: f32):
      %663 = arith.divf %660, %661 : f32
      linalg.yield %663 : f32
    } -> tensor<1x6x17x17xf32>
    %664 = tensor.empty() : tensor<1x6x17x17xf32>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%659 : tensor<1x6x17x17xf32>) outs(%664 : tensor<1x6x17x17xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb60(%666: f32, %667: f32):
      linalg.yield %666 : f32
    } -> tensor<1x6x17x17xf32>
    %668 = tensor.collapse_shape %665 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17x17xf32> into tensor<1734xf32>
    %669 = tensor.expand_shape %668 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 17, 17] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1734xf32> into tensor<6x17x17xf32>
    %670 = tensor.empty() : tensor<1x6x17x64xf32>
    %671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%598 : tensor<1x6x17x64xf32>) outs(%670 : tensor<1x6x17x64xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb61(%672: f32, %673: f32):
      linalg.yield %672 : f32
    } -> tensor<1x6x17x64xf32>
    %674 = tensor.collapse_shape %671 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x6x17x64xf32> into tensor<6528xf32>
    %675 = tensor.expand_shape %674 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 17, 64] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<6x17x64xf32>
    %676 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} 0.000000e+00 : f32
    %677 = tensor.splat %676 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6x17x64xf32>
    %678 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%669, %675 : tensor<6x17x17xf32>, tensor<6x17x64xf32>) outs(%677 : tensor<6x17x64xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} {
    ^bb62(%679: f32, %680: f32, %681: f32):
      %682 = arith.mulf %679, %680 : f32
      %683 = arith.addf %681, %682 : f32
      linalg.yield %683 : f32
    } -> tensor<6x17x64xf32>
    %684 = tensor.collapse_shape %678 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6x17x64xf32> into tensor<6528xf32>
    %685 = tensor.expand_shape %684 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 6, 17, 64] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<1x6x17x64xf32>
    %686 = tensor.empty() : tensor<1x17x6x64xf32>
    %687 = linalg.transpose ins(%685:tensor<1x6x17x64xf32>) outs(%686:tensor<1x17x6x64xf32>) permutation = [0, 2, 1, 3]
    %688 = tensor.collapse_shape %687 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<1x17x6x64xf32> into tensor<6528xf32>
    %689 = tensor.expand_shape %688 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 384] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn"} : tensor<6528xf32> into tensor<1x17x384xf32>
    %690 = tensor.collapse_shape %689 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj"} : tensor<1x17x384xf32> into tensor<6528xf32>
    %691 = tensor.expand_shape %690 [[0 : i64, 1 : i64]] output_shape [17, 384] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj"} : tensor<6528xf32> into tensor<17x384xf32>
    %692 = tensor.empty() : tensor<384x384xf32>
    %693 = linalg.transpose ins(%38:tensor<384x384xf32>) outs(%692:tensor<384x384xf32>) permutation = [1, 0]
    %694 = tensor.empty() : tensor<17x384xf32>
    %695 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %696 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%695 : f32) outs(%694 : tensor<17x384xf32>) -> tensor<17x384xf32>
    %697 = linalg.matmul {prov.region_id = "matmul_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj", prov.transposed_b = "true"} ins(%691, %693 : tensor<17x384xf32>, tensor<384x384xf32>) outs(%696 : tensor<17x384xf32>) -> tensor<17x384xf32>
    %698 = tensor.empty() : tensor<17x384xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%697, %39 : tensor<17x384xf32>, tensor<384xf32>) outs(%698 : tensor<17x384xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj"} {
    ^bb63(%700: f32, %701: f32, %702: f32):
      %703 = arith.addf %700, %701 : f32
      linalg.yield %703 : f32
    } -> tensor<17x384xf32>
    %704 = tensor.collapse_shape %699 [[0 : i64, 1 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj"} : tensor<17x384xf32> into tensor<6528xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 384] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.attn.proj"} : tensor<6528xf32> into tensor<1x17x384xf32>
    %706 = tensor.empty() : tensor<1x17x384xf32>
    %707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%490, %705 : tensor<1x17x384xf32>, tensor<1x17x384xf32>) outs(%706 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0"} {
    ^bb64(%708: f32, %709: f32, %710: f32):
      %711 = arith.addf %708, %709 : f32
      linalg.yield %711 : f32
    } -> tensor<1x17x384xf32>
    %712 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} 0.000000e+00 : f32
    %713 = tensor.splat %712 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %714 = linalg.reduce ins(%707:tensor<1x17x384xf32>) outs(%713:tensor<1x17xf32>) dimensions = [2]
    (%715: f32, %716: f32) {
      %717 = arith.addf %715, %716 : f32
      linalg.yield %717 : f32
    }
    %718 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} 3.840000e+02 : f32
    %719 = tensor.splat %718 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %720 = tensor.empty() : tensor<1x17xf32>
    %721 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%714, %719 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%720 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb65(%722: f32, %723: f32, %724: f32):
      %725 = arith.divf %722, %723 : f32
      linalg.yield %725 : f32
    } -> tensor<1x17xf32>
    %726 = tensor.collapse_shape %721 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32> into tensor<17xf32>
    %727 = tensor.expand_shape %726 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<17xf32> into tensor<1x17x1xf32>
    %728 = tensor.empty() : tensor<1x17x384xf32>
    %729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %727 : tensor<1x17x384xf32>, tensor<1x17x1xf32>) outs(%728 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb66(%730: f32, %731: f32, %732: f32):
      %733 = arith.subf %730, %731 : f32
      linalg.yield %733 : f32
    } -> tensor<1x17x384xf32>
    %734 = tensor.empty() : tensor<1x17x384xf32>
    %735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%729, %729 : tensor<1x17x384xf32>, tensor<1x17x384xf32>) outs(%734 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb67(%736: f32, %737: f32, %738: f32):
      %739 = arith.mulf %736, %737 : f32
      linalg.yield %739 : f32
    } -> tensor<1x17x384xf32>
    %740 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} 0.000000e+00 : f32
    %741 = tensor.splat %740 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %742 = linalg.reduce ins(%735:tensor<1x17x384xf32>) outs(%741:tensor<1x17xf32>) dimensions = [2]
    (%743: f32, %744: f32) {
      %745 = arith.addf %743, %744 : f32
      linalg.yield %745 : f32
    }
    %746 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} 3.840000e+02 : f32
    %747 = tensor.splat %746 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32>
    %748 = tensor.empty() : tensor<1x17xf32>
    %749 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%742, %747 : tensor<1x17xf32>, tensor<1x17xf32>) outs(%748 : tensor<1x17xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb68(%750: f32, %751: f32, %752: f32):
      %753 = arith.divf %750, %751 : f32
      linalg.yield %753 : f32
    } -> tensor<1x17xf32>
    %754 = tensor.collapse_shape %749 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17xf32> into tensor<17xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<17xf32> into tensor<1x17x1xf32>
    %756 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} 1.000000e-06 : f32
    %757 = tensor.splat %756 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} : tensor<1x17x1xf32>
    %758 = tensor.empty() : tensor<1x17x1xf32>
    %759 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%755, %757 : tensor<1x17x1xf32>, tensor<1x17x1xf32>) outs(%758 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb69(%760: f32, %761: f32, %762: f32):
      %763 = arith.addf %760, %761 : f32
      linalg.yield %763 : f32
    } -> tensor<1x17x1xf32>
    %764 = tensor.empty() : tensor<1x17x1xf32>
    %765 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%759 : tensor<1x17x1xf32>) outs(%764 : tensor<1x17x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb70(%766: f32, %767: f32):
      %768 = math.rsqrt %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x17x1xf32>
    %769 = tensor.empty() : tensor<1x17x384xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%729, %765 : tensor<1x17x384xf32>, tensor<1x17x1xf32>) outs(%769 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb71(%771: f32, %772: f32, %773: f32):
      %774 = arith.mulf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x17x384xf32>
    %775 = tensor.empty() : tensor<1x17x384xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%770, %40 : tensor<1x17x384xf32>, tensor<384xf32>) outs(%775 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb72(%777: f32, %778: f32, %779: f32):
      %780 = arith.mulf %777, %778 : f32
      linalg.yield %780 : f32
    } -> tensor<1x17x384xf32>
    %781 = tensor.empty() : tensor<1x17x384xf32>
    %782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%776, %41 : tensor<1x17x384xf32>, tensor<384xf32>) outs(%781 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.norm2"} {
    ^bb73(%783: f32, %784: f32, %785: f32):
      %786 = arith.addf %783, %784 : f32
      linalg.yield %786 : f32
    } -> tensor<1x17x384xf32>
    %787 = tensor.collapse_shape %782 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"} : tensor<1x17x384xf32> into tensor<6528xf32>
    %788 = tensor.expand_shape %787 [[0 : i64, 1 : i64]] output_shape [17, 384] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"} : tensor<6528xf32> into tensor<17x384xf32>
    %789 = tensor.empty() : tensor<384x1536xf32>
    %790 = linalg.transpose ins(%42:tensor<1536x384xf32>) outs(%789:tensor<384x1536xf32>) permutation = [1, 0]
    %791 = tensor.empty() : tensor<17x1536xf32>
    %792 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %793 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%792 : f32) outs(%791 : tensor<17x1536xf32>) -> tensor<17x1536xf32>
    %794 = linalg.matmul {prov.region_id = "matmul_10", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1", prov.transposed_b = "true"} ins(%788, %790 : tensor<17x384xf32>, tensor<384x1536xf32>) outs(%793 : tensor<17x1536xf32>) -> tensor<17x1536xf32>
    %795 = tensor.empty() : tensor<17x1536xf32>
    %796 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%794, %43 : tensor<17x1536xf32>, tensor<1536xf32>) outs(%795 : tensor<17x1536xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"} {
    ^bb74(%797: f32, %798: f32, %799: f32):
      %800 = arith.addf %797, %798 : f32
      linalg.yield %800 : f32
    } -> tensor<17x1536xf32>
    %801 = tensor.collapse_shape %796 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"} : tensor<17x1536xf32> into tensor<26112xf32>
    %802 = tensor.expand_shape %801 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 1536] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc1"} : tensor<26112xf32> into tensor<1x17x1536xf32>
    %803 = tensor.empty() : tensor<1x17x1536xf32>
    %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%802 : tensor<1x17x1536xf32>) outs(%803 : tensor<1x17x1536xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.act"} {
    ^bb75(%805: f32, %806: f32):
      %807 = arith.constant 5.000000e-01 : f32
      %808 = arith.constant 1.000000e+00 : f32
      %809 = arith.constant 0.707106769 : f32
      %810 = arith.mulf %805, %809 : f32
      %811 = math.erf %810 : f32
      %812 = arith.addf %808, %811 : f32
      %813 = arith.mulf %807, %805 : f32
      %814 = arith.mulf %813, %812 : f32
      linalg.yield %814 : f32
    } -> tensor<1x17x1536xf32>
    %815 = tensor.collapse_shape %804 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"} : tensor<1x17x1536xf32> into tensor<26112xf32>
    %816 = tensor.expand_shape %815 [[0 : i64, 1 : i64]] output_shape [17, 1536] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"} : tensor<26112xf32> into tensor<17x1536xf32>
    %817 = tensor.empty() : tensor<1536x384xf32>
    %818 = linalg.transpose ins(%44:tensor<384x1536xf32>) outs(%817:tensor<1536x384xf32>) permutation = [1, 0]
    %819 = tensor.empty() : tensor<17x384xf32>
    %820 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %821 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%820 : f32) outs(%819 : tensor<17x384xf32>) -> tensor<17x384xf32>
    %822 = linalg.matmul {prov.region_id = "matmul_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2", prov.transposed_b = "true"} ins(%816, %818 : tensor<17x1536xf32>, tensor<1536x384xf32>) outs(%821 : tensor<17x384xf32>) -> tensor<17x384xf32>
    %823 = tensor.empty() : tensor<17x384xf32>
    %824 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%822, %45 : tensor<17x384xf32>, tensor<384xf32>) outs(%823 : tensor<17x384xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"} {
    ^bb76(%825: f32, %826: f32, %827: f32):
      %828 = arith.addf %825, %826 : f32
      linalg.yield %828 : f32
    } -> tensor<17x384xf32>
    %829 = tensor.collapse_shape %824 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"} : tensor<17x384xf32> into tensor<6528xf32>
    %830 = tensor.expand_shape %829 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 17, 384] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0.mlp.fc2"} : tensor<6528xf32> into tensor<1x17x384xf32>
    %831 = tensor.empty() : tensor<1x17x384xf32>
    %832 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707, %830 : tensor<1x17x384xf32>, tensor<1x17x384xf32>) outs(%831 : tensor<1x17x384xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer.blocks.0"} {
    ^bb77(%833: f32, %834: f32, %835: f32):
      %836 = arith.addf %833, %834 : f32
      linalg.yield %836 : f32
    } -> tensor<1x17x384xf32>
    %837 = "tensor.extract_slice"(%832) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 16, 384>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone.fused_featurizer"} : (tensor<1x17x384xf32>) -> tensor<1x16x384xf32>
    %838 = tensor.concat dim(2) %465, %837 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.vision_backbone"} : (tensor<1x16x192xf32>, tensor<1x16x384xf32>) -> tensor<1x16x576xf32>
    %839 = tensor.collapse_shape %838 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1"} : tensor<1x16x576xf32> into tensor<9216xf32>
    %840 = tensor.expand_shape %839 [[0 : i64, 1 : i64]] output_shape [16, 576] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1"} : tensor<9216xf32> into tensor<16x576xf32>
    %841 = tensor.empty() : tensor<576x2304xf32>
    %842 = linalg.transpose ins(%60:tensor<2304x576xf32>) outs(%841:tensor<576x2304xf32>) permutation = [1, 0]
    %843 = tensor.empty() : tensor<16x2304xf32>
    %844 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %845 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%844 : f32) outs(%843 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %846 = linalg.matmul {prov.region_id = "matmul_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1", prov.transposed_b = "true"} ins(%840, %842 : tensor<16x576xf32>, tensor<576x2304xf32>) outs(%845 : tensor<16x2304xf32>) -> tensor<16x2304xf32>
    %847 = tensor.empty() : tensor<16x2304xf32>
    %848 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%846, %61 : tensor<16x2304xf32>, tensor<2304xf32>) outs(%847 : tensor<16x2304xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1"} {
    ^bb78(%849: f32, %850: f32, %851: f32):
      %852 = arith.addf %849, %850 : f32
      linalg.yield %852 : f32
    } -> tensor<16x2304xf32>
    %853 = tensor.collapse_shape %848 [[0 : i64, 1 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1"} : tensor<16x2304xf32> into tensor<36864xf32>
    %854 = tensor.expand_shape %853 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 2304] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc1"} : tensor<36864xf32> into tensor<1x16x2304xf32>
    %855 = tensor.empty() : tensor<1x16x2304xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%854 : tensor<1x16x2304xf32>) outs(%855 : tensor<1x16x2304xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.act_fn1"} {
    ^bb79(%857: f32, %858: f32):
      %859 = arith.constant 5.000000e-01 : f32
      %860 = arith.constant 1.000000e+00 : f32
      %861 = arith.constant 0.707106769 : f32
      %862 = arith.mulf %857, %861 : f32
      %863 = math.erf %862 : f32
      %864 = arith.addf %860, %863 : f32
      %865 = arith.mulf %859, %857 : f32
      %866 = arith.mulf %865, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x16x2304xf32>
    %867 = tensor.collapse_shape %856 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2"} : tensor<1x16x2304xf32> into tensor<36864xf32>
    %868 = tensor.expand_shape %867 [[0 : i64, 1 : i64]] output_shape [16, 2304] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2"} : tensor<36864xf32> into tensor<16x2304xf32>
    %869 = tensor.empty() : tensor<2304x128xf32>
    %870 = linalg.transpose ins(%62:tensor<128x2304xf32>) outs(%869:tensor<2304x128xf32>) permutation = [1, 0]
    %871 = tensor.empty() : tensor<16x128xf32>
    %872 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %873 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%872 : f32) outs(%871 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %874 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2", prov.transposed_b = "true"} ins(%868, %870 : tensor<16x2304xf32>, tensor<2304x128xf32>) outs(%873 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %875 = tensor.empty() : tensor<16x128xf32>
    %876 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%874, %63 : tensor<16x128xf32>, tensor<128xf32>) outs(%875 : tensor<16x128xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2"} {
    ^bb80(%877: f32, %878: f32, %879: f32):
      %880 = arith.addf %877, %878 : f32
      linalg.yield %880 : f32
    } -> tensor<16x128xf32>
    %881 = tensor.collapse_shape %876 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2"} : tensor<16x128xf32> into tensor<2048xf32>
    %882 = tensor.expand_shape %881 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 128] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc2"} : tensor<2048xf32> into tensor<1x16x128xf32>
    %883 = tensor.empty() : tensor<1x16x128xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%882 : tensor<1x16x128xf32>) outs(%883 : tensor<1x16x128xf32>) attrs =  {prov.region_id = "gelu_3", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.act_fn2"} {
    ^bb81(%885: f32, %886: f32):
      %887 = arith.constant 5.000000e-01 : f32
      %888 = arith.constant 1.000000e+00 : f32
      %889 = arith.constant 0.707106769 : f32
      %890 = arith.mulf %885, %889 : f32
      %891 = math.erf %890 : f32
      %892 = arith.addf %888, %891 : f32
      %893 = arith.mulf %887, %885 : f32
      %894 = arith.mulf %893, %892 : f32
      linalg.yield %894 : f32
    } -> tensor<1x16x128xf32>
    %895 = tensor.collapse_shape %884 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3"} : tensor<1x16x128xf32> into tensor<2048xf32>
    %896 = tensor.expand_shape %895 [[0 : i64, 1 : i64]] output_shape [16, 128] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3"} : tensor<2048xf32> into tensor<16x128xf32>
    %897 = tensor.empty() : tensor<128x128xf32>
    %898 = linalg.transpose ins(%64:tensor<128x128xf32>) outs(%897:tensor<128x128xf32>) permutation = [1, 0]
    %899 = tensor.empty() : tensor<16x128xf32>
    %900 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %901 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%900 : f32) outs(%899 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %902 = linalg.matmul {prov.region_id = "matmul_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3", prov.transposed_b = "true"} ins(%896, %898 : tensor<16x128xf32>, tensor<128x128xf32>) outs(%901 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %903 = tensor.empty() : tensor<16x128xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%902, %65 : tensor<16x128xf32>, tensor<128xf32>) outs(%903 : tensor<16x128xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3"} {
    ^bb82(%905: f32, %906: f32, %907: f32):
      %908 = arith.addf %905, %906 : f32
      linalg.yield %908 : f32
    } -> tensor<16x128xf32>
    %909 = tensor.collapse_shape %904 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3"} : tensor<16x128xf32> into tensor<2048xf32>
    %910 = tensor.expand_shape %909 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 16, 128] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.projector.fc3"} : tensor<2048xf32> into tensor<1x16x128xf32>
    %911 = tensor.empty() : tensor<1x4x128xf32>
    %912 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90 : tensor<1x4xi64>) outs(%911 : tensor<1x4x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.embed_tokens"} {
    ^bb83(%913: i64, %914: f32):
      %915 = arith.index_cast %913 : i64 to index
      %916 = linalg.index 2 : index
      %917 = tensor.extract %66[%915, %916] : tensor<512x128xf32>
      linalg.yield %917 : f32
    } -> tensor<1x4x128xf32>
    %918 = "tensor.extract_slice"(%912) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla"} : (tensor<1x4x128xf32>) -> tensor<1x1x128xf32>
    %919 = "tensor.extract_slice"(%912) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 3, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla"} : (tensor<1x4x128xf32>) -> tensor<1x3x128xf32>
    %920 = tensor.concat dim(1) %918, %910, %919 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla"} : (tensor<1x1x128xf32>, tensor<1x16x128xf32>, tensor<1x3x128xf32>) -> tensor<1x20x128xf32>
    %921 = tensor.empty() : tensor<20xi64>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%921 : tensor<20xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb84(%923: i64):
      %924 = linalg.index 0 : index
      %925 = arith.index_cast %924 : index to i64
      %926 = arith.constant 1 : i64
      %927 = arith.muli %925, %926 : i64
      %928 = arith.constant 0 : i64
      %929 = arith.addi %928, %927 : i64
      linalg.yield %929 : i64
    } -> tensor<20xi64>
    %930 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 0 : i64
    %931 = tensor.splat %930 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64>
    %932 = tensor.empty() : tensor<20xi64>
    %933 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%922, %931 : tensor<20xi64>, tensor<20xi64>) outs(%932 : tensor<20xi64>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb85(%934: i64, %935: i64, %936: i64):
      %937 = arith.addi %934, %935 : i64
      linalg.yield %937 : i64
    } -> tensor<20xi64>
    %938 = tensor.expand_shape %933 [[0 : i64, 1 : i64]] output_shape [1, 20] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x20xi64>
    %939 = "tensor.extract_slice"(%938) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x20xi64>) -> tensor<1x1xi64>
    %940 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 1 : i64
    %941 = tensor.splat %940 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1xi64>
    %942 = tensor.empty() : tensor<1x1xi64>
    %943 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%939, %941 : tensor<1x1xi64>, tensor<1x1xi64>) outs(%942 : tensor<1x1xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb86(%944: i64, %945: i64, %946: i64):
      %947 = arith.subi %944, %945 : i64
      linalg.yield %947 : i64
    } -> tensor<1x1xi64>
    %948 = tensor.concat dim(1) %943, %938 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1xi64>, tensor<1x20xi64>) -> tensor<1x21xi64>
    %949 = "tensor.extract_slice"(%948) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 20>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x21xi64>) -> tensor<1x20xi64>
    %950 = "tensor.extract_slice"(%948) <{static_offsets = array<i64: 0, 1>, static_sizes = array<i64: 1, 20>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x21xi64>) -> tensor<1x20xi64>
    %951 = tensor.empty() : tensor<1x20xi64>
    %952 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%950, %949 : tensor<1x20xi64>, tensor<1x20xi64>) outs(%951 : tensor<1x20xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb87(%953: i64, %954: i64, %955: i64):
      %956 = arith.subi %953, %954 : i64
      linalg.yield %956 : i64
    } -> tensor<1x20xi64>
    %957 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} 1 : i64
    %958 = tensor.splat %957 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x20xi64>
    %959 = tensor.empty() : tensor<1x20xi1>
    %960 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%952, %958 : tensor<1x20xi64>, tensor<1x20xi64>) outs(%959 : tensor<1x20xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb88(%961: i64, %962: i64, %963: i1):
      %964 = arith.cmpi ne, %961, %962 : i64
      linalg.yield %964 : i1
    } -> tensor<1x20xi1>
    %965 = arith.constant {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 0 : i64
    %966 = tensor.splat %965 {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x20xi64>
    %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%960 : tensor<1x20xi1>) outs(%966 : tensor<1x20xi64>) attrs =  {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb89(%968: i1, %969: i64):
      %970 = linalg.index 1 : index
      %971 = linalg.index 2 : index
      %972 = arith.cmpi ule, %971, %970 : index
      %973 = arith.extui %968 : i1 to i64
      %974 = arith.select %972, %973, %965 : i64
      %975 = arith.addi %969, %974 : i64
      linalg.yield %975 : i64
    } -> tensor<1x20xi64>
    %976 = tensor.empty() : tensor<1xi64>
    %977 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%976 : tensor<1xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb90(%978: i64):
      %979 = linalg.index 0 : index
      %980 = arith.index_cast %979 : index to i64
      %981 = arith.constant 1 : i64
      %982 = arith.muli %980, %981 : i64
      %983 = arith.constant 0 : i64
      %984 = arith.addi %983, %982 : i64
      linalg.yield %984 : i64
    } -> tensor<1xi64>
    %985 = tensor.empty() : tensor<20xi64>
    %986 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%985 : tensor<20xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb91(%987: i64):
      %988 = linalg.index 0 : index
      %989 = arith.index_cast %988 : index to i64
      %990 = arith.constant 1 : i64
      %991 = arith.muli %989, %990 : i64
      %992 = arith.constant 0 : i64
      %993 = arith.addi %992, %991 : i64
      linalg.yield %993 : i64
    } -> tensor<20xi64>
    %994 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 0 : i64
    %995 = tensor.splat %994 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64>
    %996 = tensor.empty() : tensor<20xi64>
    %997 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%986, %995 : tensor<20xi64>, tensor<20xi64>) outs(%996 : tensor<20xi64>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb92(%998: i64, %999: i64, %1000: i64):
      %1001 = arith.addi %998, %999 : i64
      linalg.yield %1001 : i64
    } -> tensor<20xi64>
    %1002 = tensor.empty() : tensor<20xi64>
    %1003 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1002 : tensor<20xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb93(%1004: i64):
      %1005 = linalg.index 0 : index
      %1006 = arith.index_cast %1005 : index to i64
      %1007 = arith.constant 1 : i64
      %1008 = arith.muli %1006, %1007 : i64
      %1009 = arith.constant 0 : i64
      %1010 = arith.addi %1009, %1008 : i64
      linalg.yield %1010 : i64
    } -> tensor<20xi64>
    %1011 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 0 : i64
    %1012 = tensor.splat %1011 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64>
    %1013 = tensor.empty() : tensor<20xi64>
    %1014 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1003, %1012 : tensor<20xi64>, tensor<20xi64>) outs(%1013 : tensor<20xi64>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb94(%1015: i64, %1016: i64, %1017: i64):
      %1018 = arith.addi %1015, %1016 : i64
      linalg.yield %1018 : i64
    } -> tensor<20xi64>
    %1019 = tensor.expand_shape %977 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1xi64> into tensor<1x1xi64>
    %1020 = tensor.collapse_shape %1019 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1xi64> into tensor<1xi64>
    %1021 = tensor.expand_shape %1020 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1xi64> into tensor<1x1x1xi64>
    %1022 = tensor.collapse_shape %1021 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x1xi64> into tensor<1xi64>
    %1023 = tensor.expand_shape %1022 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1xi64> into tensor<1x1x1x1xi64>
    %1024 = tensor.expand_shape %997 [[0 : i64, 1 : i64]] output_shape [1, 20] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x20xi64>
    %1025 = tensor.collapse_shape %1024 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x20xi64> into tensor<20xi64>
    %1026 = tensor.expand_shape %1025 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 20] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x1x20xi64>
    %1027 = tensor.collapse_shape %1026 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x20xi64> into tensor<20xi64>
    %1028 = tensor.expand_shape %1027 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x1x20x1xi64>
    %1029 = tensor.expand_shape %1014 [[0 : i64, 1 : i64]] output_shape [1, 20] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x20xi64>
    %1030 = tensor.collapse_shape %1029 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x20xi64> into tensor<20xi64>
    %1031 = tensor.expand_shape %1030 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 20] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x1x20xi64>
    %1032 = tensor.collapse_shape %1031 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x20xi64> into tensor<20xi64>
    %1033 = tensor.expand_shape %1032 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 20] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<20xi64> into tensor<1x1x1x20xi64>
    %1034 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} true
    %1035 = tensor.splat %1034 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<i1>
    %1036 = tensor.empty() : tensor<1x1x20x20xi1>
    %1037 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1033, %1028 : tensor<1x1x1x20xi64>, tensor<1x1x20x1xi64>) outs(%1036 : tensor<1x1x20x20xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb95(%1038: i64, %1039: i64, %1040: i1):
      %1041 = arith.cmpi sle, %1038, %1039 : i64
      linalg.yield %1041 : i1
    } -> tensor<1x1x20x20xi1>
    %1042 = tensor.empty() : tensor<1x1x20x20xi1>
    %1043 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1035, %1037 : tensor<i1>, tensor<1x1x20x20xi1>) outs(%1042 : tensor<1x1x20x20xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_and.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb96(%1044: i1, %1045: i1, %1046: i1):
      %1047 = arith.andi %1044, %1045 : i1
      linalg.yield %1047 : i1
    } -> tensor<1x1x20x20xi1>
    %1048 = tensor.empty() : tensor<1x1x20x1xi64>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1023, %1028 : tensor<1x1x1x1xi64>, tensor<1x1x20x1xi64>) outs(%1048 : tensor<1x1x20x1xi64>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb97(%1050: i64, %1051: i64, %1052: i64):
      %1053 = arith.index_cast %1050 : i64 to index
      %1054 = arith.index_cast %1051 : i64 to index
      %1055 = tensor.extract %967[%1053, %1054] : tensor<1x20xi64>
      linalg.yield %1055 : i64
    } -> tensor<1x1x20x1xi64>
    %1056 = tensor.empty() : tensor<1x1x1x20xi64>
    %1057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1023, %1033 : tensor<1x1x1x1xi64>, tensor<1x1x1x20xi64>) outs(%1056 : tensor<1x1x1x20xi64>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb98(%1058: i64, %1059: i64, %1060: i64):
      %1061 = arith.index_cast %1058 : i64 to index
      %1062 = arith.index_cast %1059 : i64 to index
      %1063 = tensor.extract %967[%1061, %1062] : tensor<1x20xi64>
      linalg.yield %1063 : i64
    } -> tensor<1x1x1x20xi64>
    %1064 = tensor.empty() : tensor<1x1x20x20xi1>
    %1065 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1049, %1057 : tensor<1x1x20x1xi64>, tensor<1x1x1x20xi64>) outs(%1064 : tensor<1x1x20x20xi1>) attrs =  {prov.region_id = "compare_2", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.eq.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb99(%1066: i64, %1067: i64, %1068: i1):
      %1069 = arith.cmpi eq, %1066, %1067 : i64
      linalg.yield %1069 : i1
    } -> tensor<1x1x20x20xi1>
    %1070 = tensor.empty() : tensor<1x1x20x20xi1>
    %1071 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1043, %1065 : tensor<1x1x20x20xi1>, tensor<1x1x20x20xi1>) outs(%1070 : tensor<1x1x20x20xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_and.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb100(%1072: i1, %1073: i1, %1074: i1):
      %1075 = arith.andi %1072, %1073 : i1
      linalg.yield %1075 : i1
    } -> tensor<1x1x20x20xi1>
    %1076 = tensor.empty() : tensor<1x1x20x20xi1>
    %1077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1071 : tensor<1x1x20x20xi1>) outs(%1076 : tensor<1x1x20x20xi1>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb101(%1078: i1, %1079: i1):
      linalg.yield %1078 : i1
    } -> tensor<1x1x20x20xi1>
    %1080 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} -3.40282347e+38 : f32
    %1081 = tensor.splat %1080 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<f32>
    %1082 = tensor.empty() : tensor<1x1x20x20xf32>
    %1083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1077, %87, %1081 : tensor<1x1x20x20xi1>, tensor<f32>, tensor<f32>) outs(%1082 : tensor<1x1x20x20xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb102(%1084: i1, %1085: f32, %1086: f32, %1087: f32):
      %1088 = arith.select %1084, %1085, %1086 : f32
      linalg.yield %1088 : f32
    } -> tensor<1x1x20x20xf32>
    %1089 = tensor.expand_shape %88 [[0 : i64, 1 : i64]] output_shape [1, 64] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<64xf32> into tensor<1x64xf32>
    %1090 = tensor.collapse_shape %1089 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<1x64xf32> into tensor<64xf32>
    %1091 = tensor.expand_shape %1090 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<64xf32> into tensor<1x64x1xf32>
    %1092 = tensor.empty() : tensor<1x64x1xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1091 : tensor<1x64x1xf32>) outs(%1092 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb103(%1094: f32, %1095: f32):
      linalg.yield %1094 : f32
    } -> tensor<1x64x1xf32>
    %1096 = tensor.collapse_shape %938 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<1x20xi64> into tensor<20xi64>
    %1097 = tensor.expand_shape %1096 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 20] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<20xi64> into tensor<1x1x20xi64>
    %1098 = tensor.empty() : tensor<1x1x20xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1097 : tensor<1x1x20xi64>) outs(%1098 : tensor<1x1x20xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb104(%1100: i64, %1101: f32):
      %1102 = arith.sitofp %1100 : i64 to f32
      linalg.yield %1102 : f32
    } -> tensor<1x1x20xf32>
    %1103 = tensor.empty() : tensor<1x64x1xf32>
    %1104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1093 : tensor<1x64x1xf32>) outs(%1103 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb105(%1105: f32, %1106: f32):
      linalg.yield %1105 : f32
    } -> tensor<1x64x1xf32>
    %1107 = tensor.empty() : tensor<1x1x20xf32>
    %1108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099 : tensor<1x1x20xf32>) outs(%1107 : tensor<1x1x20xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb106(%1109: f32, %1110: f32):
      linalg.yield %1109 : f32
    } -> tensor<1x1x20xf32>
    %1111 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} 0.000000e+00 : f32
    %1112 = tensor.splat %1111 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<1x64x20xf32>
    %1113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1104, %1108 : tensor<1x64x1xf32>, tensor<1x1x20xf32>) outs(%1112 : tensor<1x64x20xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb107(%1114: f32, %1115: f32, %1116: f32):
      %1117 = arith.mulf %1114, %1115 : f32
      %1118 = arith.addf %1116, %1117 : f32
      linalg.yield %1118 : f32
    } -> tensor<1x64x20xf32>
    %1119 = tensor.empty() : tensor<1x20x64xf32>
    %1120 = linalg.transpose ins(%1113:tensor<1x64x20xf32>) outs(%1119:tensor<1x20x64xf32>) permutation = [0, 2, 1]
    %1121 = tensor.concat dim(2) %1120, %1120 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : (tensor<1x20x64xf32>, tensor<1x20x64xf32>) -> tensor<1x20x128xf32>
    %1122 = tensor.empty() : tensor<1x20x128xf32>
    %1123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1121 : tensor<1x20x128xf32>) outs(%1122 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb108(%1124: f32, %1125: f32):
      %1126 = math.cos %1124 : f32
      linalg.yield %1126 : f32
    } -> tensor<1x20x128xf32>
    %1127 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} 1.000000e+00 : f32
    %1128 = tensor.splat %1127 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<1x20x128xf32>
    %1129 = tensor.empty() : tensor<1x20x128xf32>
    %1130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1123, %1128 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1129 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb109(%1131: f32, %1132: f32, %1133: f32):
      %1134 = arith.mulf %1131, %1132 : f32
      linalg.yield %1134 : f32
    } -> tensor<1x20x128xf32>
    %1135 = tensor.empty() : tensor<1x20x128xf32>
    %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1121 : tensor<1x20x128xf32>) outs(%1135 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb110(%1137: f32, %1138: f32):
      %1139 = math.sin %1137 : f32
      linalg.yield %1139 : f32
    } -> tensor<1x20x128xf32>
    %1140 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} 1.000000e+00 : f32
    %1141 = tensor.splat %1140 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} : tensor<1x20x128xf32>
    %1142 = tensor.empty() : tensor<1x20x128xf32>
    %1143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1136, %1141 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1142 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.rotary_emb"} {
    ^bb111(%1144: f32, %1145: f32, %1146: f32):
      %1147 = arith.mulf %1144, %1145 : f32
      linalg.yield %1147 : f32
    } -> tensor<1x20x128xf32>
    %1148 = tensor.empty() : tensor<1x20x128xf32>
    %1149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%920 : tensor<1x20x128xf32>) outs(%1148 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb112(%1150: f32, %1151: f32):
      %1152 = arith.constant 2.000000e+00 : f32
      %1153 = math.powf %1150, %1152 : f32
      linalg.yield %1153 : f32
    } -> tensor<1x20x128xf32>
    %1154 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 0.000000e+00 : f32
    %1155 = tensor.splat %1154 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x20xf32>
    %1156 = linalg.reduce ins(%1149:tensor<1x20x128xf32>) outs(%1155:tensor<1x20xf32>) dimensions = [2]
    (%1157: f32, %1158: f32) {
      %1159 = arith.addf %1157, %1158 : f32
      linalg.yield %1159 : f32
    }
    %1160 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 1.280000e+02 : f32
    %1161 = tensor.splat %1160 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x20xf32>
    %1162 = tensor.empty() : tensor<1x20xf32>
    %1163 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1156, %1161 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%1162 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb113(%1164: f32, %1165: f32, %1166: f32):
      %1167 = arith.divf %1164, %1165 : f32
      linalg.yield %1167 : f32
    } -> tensor<1x20xf32>
    %1168 = tensor.collapse_shape %1163 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x20xf32> into tensor<20xf32>
    %1169 = tensor.expand_shape %1168 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<20xf32> into tensor<1x20x1xf32>
    %1170 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 1.000000e-06 : f32
    %1171 = tensor.splat %1170 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x20x1xf32>
    %1172 = tensor.empty() : tensor<1x20x1xf32>
    %1173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1169, %1171 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%1172 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb114(%1174: f32, %1175: f32, %1176: f32):
      %1177 = arith.addf %1174, %1175 : f32
      linalg.yield %1177 : f32
    } -> tensor<1x20x1xf32>
    %1178 = tensor.empty() : tensor<1x20x1xf32>
    %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1173 : tensor<1x20x1xf32>) outs(%1178 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb115(%1180: f32, %1181: f32):
      %1182 = math.rsqrt %1180 : f32
      linalg.yield %1182 : f32
    } -> tensor<1x20x1xf32>
    %1183 = tensor.empty() : tensor<1x20x128xf32>
    %1184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%920, %1179 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%1183 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb116(%1185: f32, %1186: f32, %1187: f32):
      %1188 = arith.mulf %1185, %1186 : f32
      linalg.yield %1188 : f32
    } -> tensor<1x20x128xf32>
    %1189 = tensor.empty() : tensor<1x20x128xf32>
    %1190 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %1184 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%1189 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb117(%1191: f32, %1192: f32, %1193: f32):
      %1194 = arith.mulf %1191, %1192 : f32
      linalg.yield %1194 : f32
    } -> tensor<1x20x128xf32>
    %1195 = tensor.empty() : tensor<128x512xf32>
    %1196 = linalg.transpose ins(%67:tensor<512x128xf32>) outs(%1195:tensor<128x512xf32>) permutation = [1, 0]
    %1197 = tensor.collapse_shape %1190 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1198 = tensor.expand_shape %1197 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1199 = tensor.empty() : tensor<20x512xf32>
    %1200 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1201 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1200 : f32) outs(%1199 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1202 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj", prov.transposed_b = "true"} ins(%1198, %1196 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1201 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1203 = tensor.collapse_shape %1202 [[0 : i64, 1 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1204 = tensor.expand_shape %1203 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1205 = tensor.collapse_shape %1204 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1206 = tensor.expand_shape %1205 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1207 = tensor.empty() : tensor<1x4x20x128xf32>
    %1208 = linalg.transpose ins(%1206:tensor<1x20x4x128xf32>) outs(%1207:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1209 = tensor.empty() : tensor<128x512xf32>
    %1210 = linalg.transpose ins(%68:tensor<512x128xf32>) outs(%1209:tensor<128x512xf32>) permutation = [1, 0]
    %1211 = tensor.collapse_shape %1190 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1212 = tensor.expand_shape %1211 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1213 = tensor.empty() : tensor<20x512xf32>
    %1214 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1215 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1214 : f32) outs(%1213 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1216 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj", prov.transposed_b = "true"} ins(%1212, %1210 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1215 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1217 = tensor.collapse_shape %1216 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1218 = tensor.expand_shape %1217 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1219 = tensor.collapse_shape %1218 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1220 = tensor.expand_shape %1219 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1221 = tensor.empty() : tensor<1x4x20x128xf32>
    %1222 = linalg.transpose ins(%1220:tensor<1x20x4x128xf32>) outs(%1221:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1223 = tensor.empty() : tensor<128x512xf32>
    %1224 = linalg.transpose ins(%69:tensor<512x128xf32>) outs(%1223:tensor<128x512xf32>) permutation = [1, 0]
    %1225 = tensor.collapse_shape %1190 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1226 = tensor.expand_shape %1225 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1227 = tensor.empty() : tensor<20x512xf32>
    %1228 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1229 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1228 : f32) outs(%1227 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1230 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj", prov.transposed_b = "true"} ins(%1226, %1224 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1229 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1231 = tensor.collapse_shape %1230 [[0 : i64, 1 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1232 = tensor.expand_shape %1231 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1233 = tensor.collapse_shape %1232 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1235 = tensor.empty() : tensor<1x4x20x128xf32>
    %1236 = linalg.transpose ins(%1234:tensor<1x20x4x128xf32>) outs(%1235:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1237 = tensor.collapse_shape %1130 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1238 = tensor.expand_shape %1237 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %1239 = tensor.collapse_shape %1143 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1240 = tensor.expand_shape %1239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %1241 = tensor.empty() : tensor<1x4x20x128xf32>
    %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1208, %1238 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1241 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb118(%1243: f32, %1244: f32, %1245: f32):
      %1246 = arith.mulf %1243, %1244 : f32
      linalg.yield %1246 : f32
    } -> tensor<1x4x20x128xf32>
    %1247 = "tensor.extract_slice"(%1208) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1248 = "tensor.extract_slice"(%1208) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1249 = tensor.empty() : tensor<1x4x20x64xf32>
    %1250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1248 : tensor<1x4x20x64xf32>) outs(%1249 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb119(%1251: f32, %1252: f32):
      %1253 = arith.negf %1251 : f32
      linalg.yield %1253 : f32
    } -> tensor<1x4x20x64xf32>
    %1254 = tensor.concat dim(3) %1250, %1247 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %1255 = tensor.empty() : tensor<1x4x20x128xf32>
    %1256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1254, %1240 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1255 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb120(%1257: f32, %1258: f32, %1259: f32):
      %1260 = arith.mulf %1257, %1258 : f32
      linalg.yield %1260 : f32
    } -> tensor<1x4x20x128xf32>
    %1261 = tensor.empty() : tensor<1x4x20x128xf32>
    %1262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1242, %1256 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%1261 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb121(%1263: f32, %1264: f32, %1265: f32):
      %1266 = arith.addf %1263, %1264 : f32
      linalg.yield %1266 : f32
    } -> tensor<1x4x20x128xf32>
    %1267 = tensor.empty() : tensor<1x4x20x128xf32>
    %1268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1222, %1238 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1267 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb122(%1269: f32, %1270: f32, %1271: f32):
      %1272 = arith.mulf %1269, %1270 : f32
      linalg.yield %1272 : f32
    } -> tensor<1x4x20x128xf32>
    %1273 = "tensor.extract_slice"(%1222) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1274 = "tensor.extract_slice"(%1222) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1275 = tensor.empty() : tensor<1x4x20x64xf32>
    %1276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1274 : tensor<1x4x20x64xf32>) outs(%1275 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb123(%1277: f32, %1278: f32):
      %1279 = arith.negf %1277 : f32
      linalg.yield %1279 : f32
    } -> tensor<1x4x20x64xf32>
    %1280 = tensor.concat dim(3) %1276, %1273 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %1281 = tensor.empty() : tensor<1x4x20x128xf32>
    %1282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1280, %1240 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1281 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb124(%1283: f32, %1284: f32, %1285: f32):
      %1286 = arith.mulf %1283, %1284 : f32
      linalg.yield %1286 : f32
    } -> tensor<1x4x20x128xf32>
    %1287 = tensor.empty() : tensor<1x4x20x128xf32>
    %1288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1268, %1282 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%1287 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb125(%1289: f32, %1290: f32, %1291: f32):
      %1292 = arith.addf %1289, %1290 : f32
      linalg.yield %1292 : f32
    } -> tensor<1x4x20x128xf32>
    %1293 = tensor.empty() : tensor<1x4x128x20xf32>
    %1294 = linalg.transpose ins(%1288:tensor<1x4x20x128xf32>) outs(%1293:tensor<1x4x128x20xf32>) permutation = [0, 1, 3, 2]
    %1295 = tensor.empty() : tensor<1x4x20x128xf32>
    %1296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1262 : tensor<1x4x20x128xf32>) outs(%1295 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb126(%1297: f32, %1298: f32):
      linalg.yield %1297 : f32
    } -> tensor<1x4x20x128xf32>
    %1299 = tensor.collapse_shape %1296 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20x128xf32> into tensor<10240xf32>
    %1300 = tensor.expand_shape %1299 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 128] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<4x20x128xf32>
    %1301 = tensor.empty() : tensor<1x4x128x20xf32>
    %1302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1294 : tensor<1x4x128x20xf32>) outs(%1301 : tensor<1x4x128x20xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb127(%1303: f32, %1304: f32):
      linalg.yield %1303 : f32
    } -> tensor<1x4x128x20xf32>
    %1305 = tensor.collapse_shape %1302 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x128x20xf32> into tensor<10240xf32>
    %1306 = tensor.expand_shape %1305 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 128, 20] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<4x128x20xf32>
    %1307 = arith.constant {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %1308 = tensor.splat %1307 {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4x20x20xf32>
    %1309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1300, %1306 : tensor<4x20x128xf32>, tensor<4x128x20xf32>) outs(%1308 : tensor<4x20x20xf32>) attrs =  {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb128(%1310: f32, %1311: f32, %1312: f32):
      %1313 = arith.mulf %1310, %1311 : f32
      %1314 = arith.addf %1312, %1313 : f32
      linalg.yield %1314 : f32
    } -> tensor<4x20x20xf32>
    %1315 = tensor.collapse_shape %1309 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4x20x20xf32> into tensor<1600xf32>
    %1316 = tensor.expand_shape %1315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 20] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1600xf32> into tensor<1x4x20x20xf32>
    %1317 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.0883883461 : f32
    %1318 = tensor.splat %1317 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20x20xf32>
    %1319 = tensor.empty() : tensor<1x4x20x20xf32>
    %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1316, %1318 : tensor<1x4x20x20xf32>, tensor<1x4x20x20xf32>) outs(%1319 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb129(%1321: f32, %1322: f32, %1323: f32):
      %1324 = arith.mulf %1321, %1322 : f32
      linalg.yield %1324 : f32
    } -> tensor<1x4x20x20xf32>
    %1325 = tensor.empty() : tensor<1x4x20x20xf32>
    %1326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1320, %1083 : tensor<1x4x20x20xf32>, tensor<1x1x20x20xf32>) outs(%1325 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb130(%1327: f32, %1328: f32, %1329: f32):
      %1330 = arith.addf %1327, %1328 : f32
      linalg.yield %1330 : f32
    } -> tensor<1x4x20x20xf32>
    %1331 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0xff800000 : f32
    %1332 = tensor.splat %1331 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20xf32>
    %1333 = linalg.reduce ins(%1326:tensor<1x4x20x20xf32>) outs(%1332:tensor<1x4x20xf32>) dimensions = [3]
    (%1334: f32, %1335: f32) {
      %1336 = arith.maximumf %1334, %1335 : f32
      linalg.yield %1336 : f32
    }
    %1337 = tensor.collapse_shape %1333 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20xf32> into tensor<80xf32>
    %1338 = tensor.expand_shape %1337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %1339 = tensor.empty() : tensor<1x4x20x20xf32>
    %1340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1326, %1338 : tensor<1x4x20x20xf32>, tensor<1x4x20x1xf32>) outs(%1339 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb131(%1341: f32, %1342: f32, %1343: f32):
      %1344 = arith.subf %1341, %1342 : f32
      linalg.yield %1344 : f32
    } -> tensor<1x4x20x20xf32>
    %1345 = tensor.empty() : tensor<1x4x20x20xf32>
    %1346 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1340 : tensor<1x4x20x20xf32>) outs(%1345 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb132(%1347: f32, %1348: f32):
      %1349 = math.exp %1347 : f32
      linalg.yield %1349 : f32
    } -> tensor<1x4x20x20xf32>
    %1350 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %1351 = tensor.splat %1350 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20xf32>
    %1352 = linalg.reduce ins(%1346:tensor<1x4x20x20xf32>) outs(%1351:tensor<1x4x20xf32>) dimensions = [3]
    (%1353: f32, %1354: f32) {
      %1355 = arith.addf %1353, %1354 : f32
      linalg.yield %1355 : f32
    }
    %1356 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20xf32> into tensor<80xf32>
    %1357 = tensor.expand_shape %1356 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %1358 = tensor.empty() : tensor<1x4x20x20xf32>
    %1359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1346, %1357 : tensor<1x4x20x20xf32>, tensor<1x4x20x1xf32>) outs(%1358 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb133(%1360: f32, %1361: f32, %1362: f32):
      %1363 = arith.divf %1360, %1361 : f32
      linalg.yield %1363 : f32
    } -> tensor<1x4x20x20xf32>
    %1364 = tensor.empty() : tensor<1x4x20x20xf32>
    %1365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1359 : tensor<1x4x20x20xf32>) outs(%1364 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb134(%1366: f32, %1367: f32):
      linalg.yield %1366 : f32
    } -> tensor<1x4x20x20xf32>
    %1368 = tensor.collapse_shape %1365 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20x20xf32> into tensor<1600xf32>
    %1369 = tensor.expand_shape %1368 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 20] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1600xf32> into tensor<4x20x20xf32>
    %1370 = tensor.empty() : tensor<1x4x20x128xf32>
    %1371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1236 : tensor<1x4x20x128xf32>) outs(%1370 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb135(%1372: f32, %1373: f32):
      linalg.yield %1372 : f32
    } -> tensor<1x4x20x128xf32>
    %1374 = tensor.collapse_shape %1371 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x20x128xf32> into tensor<10240xf32>
    %1375 = tensor.expand_shape %1374 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 128] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<4x20x128xf32>
    %1376 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %1377 = tensor.splat %1376 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4x20x128xf32>
    %1378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1369, %1375 : tensor<4x20x20xf32>, tensor<4x20x128xf32>) outs(%1377 : tensor<4x20x128xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb136(%1379: f32, %1380: f32, %1381: f32):
      %1382 = arith.mulf %1379, %1380 : f32
      %1383 = arith.addf %1381, %1382 : f32
      linalg.yield %1383 : f32
    } -> tensor<4x20x128xf32>
    %1384 = tensor.collapse_shape %1378 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4x20x128xf32> into tensor<10240xf32>
    %1385 = tensor.expand_shape %1384 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 128] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<1x4x20x128xf32>
    %1386 = tensor.empty() : tensor<1x20x4x128xf32>
    %1387 = linalg.transpose ins(%1385:tensor<1x4x20x128xf32>) outs(%1386:tensor<1x20x4x128xf32>) permutation = [0, 2, 1, 3]
    %1388 = tensor.collapse_shape %1387 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x20x4x128xf32> into tensor<10240xf32>
    %1389 = tensor.expand_shape %1388 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1390 = tensor.empty() : tensor<512x128xf32>
    %1391 = linalg.transpose ins(%70:tensor<128x512xf32>) outs(%1390:tensor<512x128xf32>) permutation = [1, 0]
    %1392 = tensor.collapse_shape %1389 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1393 = tensor.expand_shape %1392 [[0 : i64, 1 : i64]] output_shape [20, 512] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<10240xf32> into tensor<20x512xf32>
    %1394 = tensor.empty() : tensor<20x128xf32>
    %1395 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1396 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1395 : f32) outs(%1394 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1397 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj", prov.transposed_b = "true"} ins(%1393, %1391 : tensor<20x512xf32>, tensor<512x128xf32>) outs(%1396 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1398 = tensor.collapse_shape %1397 [[0 : i64, 1 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<20x128xf32> into tensor<2560xf32>
    %1399 = tensor.expand_shape %1398 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 128] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<2560xf32> into tensor<1x20x128xf32>
    %1400 = tensor.empty() : tensor<1x20x128xf32>
    %1401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%920, %1399 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1400 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb137(%1402: f32, %1403: f32, %1404: f32):
      %1405 = arith.addf %1402, %1403 : f32
      linalg.yield %1405 : f32
    } -> tensor<1x20x128xf32>
    %1406 = tensor.empty() : tensor<1x20x128xf32>
    %1407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1401 : tensor<1x20x128xf32>) outs(%1406 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb138(%1408: f32, %1409: f32):
      %1410 = arith.constant 2.000000e+00 : f32
      %1411 = math.powf %1408, %1410 : f32
      linalg.yield %1411 : f32
    } -> tensor<1x20x128xf32>
    %1412 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %1413 = tensor.splat %1412 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x20xf32>
    %1414 = linalg.reduce ins(%1407:tensor<1x20x128xf32>) outs(%1413:tensor<1x20xf32>) dimensions = [2]
    (%1415: f32, %1416: f32) {
      %1417 = arith.addf %1415, %1416 : f32
      linalg.yield %1417 : f32
    }
    %1418 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 1.280000e+02 : f32
    %1419 = tensor.splat %1418 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x20xf32>
    %1420 = tensor.empty() : tensor<1x20xf32>
    %1421 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1414, %1419 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%1420 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb139(%1422: f32, %1423: f32, %1424: f32):
      %1425 = arith.divf %1422, %1423 : f32
      linalg.yield %1425 : f32
    } -> tensor<1x20xf32>
    %1426 = tensor.collapse_shape %1421 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x20xf32> into tensor<20xf32>
    %1427 = tensor.expand_shape %1426 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<20xf32> into tensor<1x20x1xf32>
    %1428 = arith.constant {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 1.000000e-06 : f32
    %1429 = tensor.splat %1428 {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x20x1xf32>
    %1430 = tensor.empty() : tensor<1x20x1xf32>
    %1431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1427, %1429 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%1430 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb140(%1432: f32, %1433: f32, %1434: f32):
      %1435 = arith.addf %1432, %1433 : f32
      linalg.yield %1435 : f32
    } -> tensor<1x20x1xf32>
    %1436 = tensor.empty() : tensor<1x20x1xf32>
    %1437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1431 : tensor<1x20x1xf32>) outs(%1436 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb141(%1438: f32, %1439: f32):
      %1440 = math.rsqrt %1438 : f32
      linalg.yield %1440 : f32
    } -> tensor<1x20x1xf32>
    %1441 = tensor.empty() : tensor<1x20x128xf32>
    %1442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1401, %1437 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%1441 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb142(%1443: f32, %1444: f32, %1445: f32):
      %1446 = arith.mulf %1443, %1444 : f32
      linalg.yield %1446 : f32
    } -> tensor<1x20x128xf32>
    %1447 = tensor.empty() : tensor<1x20x128xf32>
    %1448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %1442 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%1447 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb143(%1449: f32, %1450: f32, %1451: f32):
      %1452 = arith.mulf %1449, %1450 : f32
      linalg.yield %1452 : f32
    } -> tensor<1x20x128xf32>
    %1453 = tensor.empty() : tensor<128x256xf32>
    %1454 = linalg.transpose ins(%71:tensor<256x128xf32>) outs(%1453:tensor<128x256xf32>) permutation = [1, 0]
    %1455 = tensor.collapse_shape %1448 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1456 = tensor.expand_shape %1455 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1457 = tensor.empty() : tensor<20x256xf32>
    %1458 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1459 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1458 : f32) outs(%1457 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1460 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj", prov.transposed_b = "true"} ins(%1456, %1454 : tensor<20x128xf32>, tensor<128x256xf32>) outs(%1459 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1461 = tensor.collapse_shape %1460 [[0 : i64, 1 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<20x256xf32> into tensor<5120xf32>
    %1462 = tensor.expand_shape %1461 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 256] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<5120xf32> into tensor<1x20x256xf32>
    %1463 = tensor.empty() : tensor<1x20x256xf32>
    %1464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1462 : tensor<1x20x256xf32>) outs(%1463 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.act_fn"} {
    ^bb144(%1465: f32, %1466: f32):
      %1467 = arith.constant 1.000000e+00 : f32
      %1468 = arith.negf %1465 : f32
      %1469 = math.exp %1468 : f32
      %1470 = arith.addf %1467, %1469 : f32
      %1471 = arith.divf %1467, %1470 : f32
      linalg.yield %1471 : f32
    } -> tensor<1x20x256xf32>
    %1472 = tensor.empty() : tensor<1x20x256xf32>
    %1473 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1462, %1464 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%1472 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.act_fn"} {
    ^bb145(%1474: f32, %1475: f32, %1476: f32):
      %1477 = arith.mulf %1474, %1475 : f32
      linalg.yield %1477 : f32
    } -> tensor<1x20x256xf32>
    %1478 = tensor.empty() : tensor<128x256xf32>
    %1479 = linalg.transpose ins(%72:tensor<256x128xf32>) outs(%1478:tensor<128x256xf32>) permutation = [1, 0]
    %1480 = tensor.collapse_shape %1448 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1481 = tensor.expand_shape %1480 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1482 = tensor.empty() : tensor<20x256xf32>
    %1483 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1484 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1483 : f32) outs(%1482 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1485 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj", prov.transposed_b = "true"} ins(%1481, %1479 : tensor<20x128xf32>, tensor<128x256xf32>) outs(%1484 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1486 = tensor.collapse_shape %1485 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<20x256xf32> into tensor<5120xf32>
    %1487 = tensor.expand_shape %1486 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 256] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<5120xf32> into tensor<1x20x256xf32>
    %1488 = tensor.empty() : tensor<1x20x256xf32>
    %1489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1473, %1487 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%1488 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb146(%1490: f32, %1491: f32, %1492: f32):
      %1493 = arith.mulf %1490, %1491 : f32
      linalg.yield %1493 : f32
    } -> tensor<1x20x256xf32>
    %1494 = tensor.empty() : tensor<256x128xf32>
    %1495 = linalg.transpose ins(%73:tensor<128x256xf32>) outs(%1494:tensor<256x128xf32>) permutation = [1, 0]
    %1496 = tensor.collapse_shape %1489 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x20x256xf32> into tensor<5120xf32>
    %1497 = tensor.expand_shape %1496 [[0 : i64, 1 : i64]] output_shape [20, 256] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<5120xf32> into tensor<20x256xf32>
    %1498 = tensor.empty() : tensor<20x128xf32>
    %1499 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1500 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1499 : f32) outs(%1498 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1501 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj", prov.transposed_b = "true"} ins(%1497, %1495 : tensor<20x256xf32>, tensor<256x128xf32>) outs(%1500 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1502 = tensor.collapse_shape %1501 [[0 : i64, 1 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<20x128xf32> into tensor<2560xf32>
    %1503 = tensor.expand_shape %1502 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 128] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<2560xf32> into tensor<1x20x128xf32>
    %1504 = tensor.empty() : tensor<1x20x128xf32>
    %1505 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1401, %1503 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1504 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb147(%1506: f32, %1507: f32, %1508: f32):
      %1509 = arith.addf %1506, %1507 : f32
      linalg.yield %1509 : f32
    } -> tensor<1x20x128xf32>
    %1510 = tensor.empty() : tensor<1x20x128xf32>
    %1511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1505 : tensor<1x20x128xf32>) outs(%1510 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb148(%1512: f32, %1513: f32):
      %1514 = arith.constant 2.000000e+00 : f32
      %1515 = math.powf %1512, %1514 : f32
      linalg.yield %1515 : f32
    } -> tensor<1x20x128xf32>
    %1516 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 0.000000e+00 : f32
    %1517 = tensor.splat %1516 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x20xf32>
    %1518 = linalg.reduce ins(%1511:tensor<1x20x128xf32>) outs(%1517:tensor<1x20xf32>) dimensions = [2]
    (%1519: f32, %1520: f32) {
      %1521 = arith.addf %1519, %1520 : f32
      linalg.yield %1521 : f32
    }
    %1522 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 1.280000e+02 : f32
    %1523 = tensor.splat %1522 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x20xf32>
    %1524 = tensor.empty() : tensor<1x20xf32>
    %1525 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1518, %1523 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%1524 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb149(%1526: f32, %1527: f32, %1528: f32):
      %1529 = arith.divf %1526, %1527 : f32
      linalg.yield %1529 : f32
    } -> tensor<1x20xf32>
    %1530 = tensor.collapse_shape %1525 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x20xf32> into tensor<20xf32>
    %1531 = tensor.expand_shape %1530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<20xf32> into tensor<1x20x1xf32>
    %1532 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 1.000000e-06 : f32
    %1533 = tensor.splat %1532 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x20x1xf32>
    %1534 = tensor.empty() : tensor<1x20x1xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531, %1533 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%1534 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb150(%1536: f32, %1537: f32, %1538: f32):
      %1539 = arith.addf %1536, %1537 : f32
      linalg.yield %1539 : f32
    } -> tensor<1x20x1xf32>
    %1540 = tensor.empty() : tensor<1x20x1xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1535 : tensor<1x20x1xf32>) outs(%1540 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb151(%1542: f32, %1543: f32):
      %1544 = math.rsqrt %1542 : f32
      linalg.yield %1544 : f32
    } -> tensor<1x20x1xf32>
    %1545 = tensor.empty() : tensor<1x20x128xf32>
    %1546 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1505, %1541 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%1545 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb152(%1547: f32, %1548: f32, %1549: f32):
      %1550 = arith.mulf %1547, %1548 : f32
      linalg.yield %1550 : f32
    } -> tensor<1x20x128xf32>
    %1551 = tensor.empty() : tensor<1x20x128xf32>
    %1552 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %1546 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%1551 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb153(%1553: f32, %1554: f32, %1555: f32):
      %1556 = arith.mulf %1553, %1554 : f32
      linalg.yield %1556 : f32
    } -> tensor<1x20x128xf32>
    %1557 = tensor.empty() : tensor<128x512xf32>
    %1558 = linalg.transpose ins(%76:tensor<512x128xf32>) outs(%1557:tensor<128x512xf32>) permutation = [1, 0]
    %1559 = tensor.collapse_shape %1552 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1560 = tensor.expand_shape %1559 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1561 = tensor.empty() : tensor<20x512xf32>
    %1562 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1563 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1562 : f32) outs(%1561 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1564 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj", prov.transposed_b = "true"} ins(%1560, %1558 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1563 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1565 = tensor.collapse_shape %1564 [[0 : i64, 1 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1566 = tensor.expand_shape %1565 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1567 = tensor.collapse_shape %1566 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1568 = tensor.expand_shape %1567 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1569 = tensor.empty() : tensor<1x4x20x128xf32>
    %1570 = linalg.transpose ins(%1568:tensor<1x20x4x128xf32>) outs(%1569:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1571 = tensor.empty() : tensor<128x512xf32>
    %1572 = linalg.transpose ins(%77:tensor<512x128xf32>) outs(%1571:tensor<128x512xf32>) permutation = [1, 0]
    %1573 = tensor.collapse_shape %1552 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1574 = tensor.expand_shape %1573 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1575 = tensor.empty() : tensor<20x512xf32>
    %1576 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1577 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1576 : f32) outs(%1575 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1578 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj", prov.transposed_b = "true"} ins(%1574, %1572 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1577 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1579 = tensor.collapse_shape %1578 [[0 : i64, 1 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1580 = tensor.expand_shape %1579 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1581 = tensor.collapse_shape %1580 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1582 = tensor.expand_shape %1581 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1583 = tensor.empty() : tensor<1x4x20x128xf32>
    %1584 = linalg.transpose ins(%1582:tensor<1x20x4x128xf32>) outs(%1583:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1585 = tensor.empty() : tensor<128x512xf32>
    %1586 = linalg.transpose ins(%78:tensor<512x128xf32>) outs(%1585:tensor<128x512xf32>) permutation = [1, 0]
    %1587 = tensor.collapse_shape %1552 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1588 = tensor.expand_shape %1587 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1589 = tensor.empty() : tensor<20x512xf32>
    %1590 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1591 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1590 : f32) outs(%1589 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1592 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj", prov.transposed_b = "true"} ins(%1588, %1586 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1591 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1593 = tensor.collapse_shape %1592 [[0 : i64, 1 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<20x512xf32> into tensor<10240xf32>
    %1594 = tensor.expand_shape %1593 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1595 = tensor.collapse_shape %1594 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1596 = tensor.expand_shape %1595 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %1597 = tensor.empty() : tensor<1x4x20x128xf32>
    %1598 = linalg.transpose ins(%1596:tensor<1x20x4x128xf32>) outs(%1597:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %1599 = tensor.collapse_shape %1130 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1600 = tensor.expand_shape %1599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %1601 = tensor.collapse_shape %1143 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1602 = tensor.expand_shape %1601 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %1603 = tensor.empty() : tensor<1x4x20x128xf32>
    %1604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1570, %1600 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1603 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb154(%1605: f32, %1606: f32, %1607: f32):
      %1608 = arith.mulf %1605, %1606 : f32
      linalg.yield %1608 : f32
    } -> tensor<1x4x20x128xf32>
    %1609 = "tensor.extract_slice"(%1570) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1610 = "tensor.extract_slice"(%1570) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1611 = tensor.empty() : tensor<1x4x20x64xf32>
    %1612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1610 : tensor<1x4x20x64xf32>) outs(%1611 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb155(%1613: f32, %1614: f32):
      %1615 = arith.negf %1613 : f32
      linalg.yield %1615 : f32
    } -> tensor<1x4x20x64xf32>
    %1616 = tensor.concat dim(3) %1612, %1609 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %1617 = tensor.empty() : tensor<1x4x20x128xf32>
    %1618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1616, %1602 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1617 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb156(%1619: f32, %1620: f32, %1621: f32):
      %1622 = arith.mulf %1619, %1620 : f32
      linalg.yield %1622 : f32
    } -> tensor<1x4x20x128xf32>
    %1623 = tensor.empty() : tensor<1x4x20x128xf32>
    %1624 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1604, %1618 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%1623 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb157(%1625: f32, %1626: f32, %1627: f32):
      %1628 = arith.addf %1625, %1626 : f32
      linalg.yield %1628 : f32
    } -> tensor<1x4x20x128xf32>
    %1629 = tensor.empty() : tensor<1x4x20x128xf32>
    %1630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1584, %1600 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1629 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb158(%1631: f32, %1632: f32, %1633: f32):
      %1634 = arith.mulf %1631, %1632 : f32
      linalg.yield %1634 : f32
    } -> tensor<1x4x20x128xf32>
    %1635 = "tensor.extract_slice"(%1584) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1636 = "tensor.extract_slice"(%1584) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %1637 = tensor.empty() : tensor<1x4x20x64xf32>
    %1638 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1636 : tensor<1x4x20x64xf32>) outs(%1637 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb159(%1639: f32, %1640: f32):
      %1641 = arith.negf %1639 : f32
      linalg.yield %1641 : f32
    } -> tensor<1x4x20x64xf32>
    %1642 = tensor.concat dim(3) %1638, %1635 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %1643 = tensor.empty() : tensor<1x4x20x128xf32>
    %1644 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1642, %1602 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%1643 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb160(%1645: f32, %1646: f32, %1647: f32):
      %1648 = arith.mulf %1645, %1646 : f32
      linalg.yield %1648 : f32
    } -> tensor<1x4x20x128xf32>
    %1649 = tensor.empty() : tensor<1x4x20x128xf32>
    %1650 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1630, %1644 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%1649 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb161(%1651: f32, %1652: f32, %1653: f32):
      %1654 = arith.addf %1651, %1652 : f32
      linalg.yield %1654 : f32
    } -> tensor<1x4x20x128xf32>
    %1655 = tensor.empty() : tensor<1x4x128x20xf32>
    %1656 = linalg.transpose ins(%1650:tensor<1x4x20x128xf32>) outs(%1655:tensor<1x4x128x20xf32>) permutation = [0, 1, 3, 2]
    %1657 = tensor.empty() : tensor<1x4x20x128xf32>
    %1658 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1624 : tensor<1x4x20x128xf32>) outs(%1657 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb162(%1659: f32, %1660: f32):
      linalg.yield %1659 : f32
    } -> tensor<1x4x20x128xf32>
    %1661 = tensor.collapse_shape %1658 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20x128xf32> into tensor<10240xf32>
    %1662 = tensor.expand_shape %1661 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 128] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<4x20x128xf32>
    %1663 = tensor.empty() : tensor<1x4x128x20xf32>
    %1664 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1656 : tensor<1x4x128x20xf32>) outs(%1663 : tensor<1x4x128x20xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb163(%1665: f32, %1666: f32):
      linalg.yield %1665 : f32
    } -> tensor<1x4x128x20xf32>
    %1667 = tensor.collapse_shape %1664 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x128x20xf32> into tensor<10240xf32>
    %1668 = tensor.expand_shape %1667 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 128, 20] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<4x128x20xf32>
    %1669 = arith.constant {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1670 = tensor.splat %1669 {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4x20x20xf32>
    %1671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1662, %1668 : tensor<4x20x128xf32>, tensor<4x128x20xf32>) outs(%1670 : tensor<4x20x20xf32>) attrs =  {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb164(%1672: f32, %1673: f32, %1674: f32):
      %1675 = arith.mulf %1672, %1673 : f32
      %1676 = arith.addf %1674, %1675 : f32
      linalg.yield %1676 : f32
    } -> tensor<4x20x20xf32>
    %1677 = tensor.collapse_shape %1671 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4x20x20xf32> into tensor<1600xf32>
    %1678 = tensor.expand_shape %1677 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 20] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1600xf32> into tensor<1x4x20x20xf32>
    %1679 = arith.constant {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.0883883461 : f32
    %1680 = tensor.splat %1679 {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20x20xf32>
    %1681 = tensor.empty() : tensor<1x4x20x20xf32>
    %1682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1678, %1680 : tensor<1x4x20x20xf32>, tensor<1x4x20x20xf32>) outs(%1681 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb165(%1683: f32, %1684: f32, %1685: f32):
      %1686 = arith.mulf %1683, %1684 : f32
      linalg.yield %1686 : f32
    } -> tensor<1x4x20x20xf32>
    %1687 = tensor.empty() : tensor<1x4x20x20xf32>
    %1688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1682, %1083 : tensor<1x4x20x20xf32>, tensor<1x1x20x20xf32>) outs(%1687 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb166(%1689: f32, %1690: f32, %1691: f32):
      %1692 = arith.addf %1689, %1690 : f32
      linalg.yield %1692 : f32
    } -> tensor<1x4x20x20xf32>
    %1693 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0xff800000 : f32
    %1694 = tensor.splat %1693 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20xf32>
    %1695 = linalg.reduce ins(%1688:tensor<1x4x20x20xf32>) outs(%1694:tensor<1x4x20xf32>) dimensions = [3]
    (%1696: f32, %1697: f32) {
      %1698 = arith.maximumf %1696, %1697 : f32
      linalg.yield %1698 : f32
    }
    %1699 = tensor.collapse_shape %1695 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20xf32> into tensor<80xf32>
    %1700 = tensor.expand_shape %1699 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %1701 = tensor.empty() : tensor<1x4x20x20xf32>
    %1702 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1688, %1700 : tensor<1x4x20x20xf32>, tensor<1x4x20x1xf32>) outs(%1701 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb167(%1703: f32, %1704: f32, %1705: f32):
      %1706 = arith.subf %1703, %1704 : f32
      linalg.yield %1706 : f32
    } -> tensor<1x4x20x20xf32>
    %1707 = tensor.empty() : tensor<1x4x20x20xf32>
    %1708 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1702 : tensor<1x4x20x20xf32>) outs(%1707 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb168(%1709: f32, %1710: f32):
      %1711 = math.exp %1709 : f32
      linalg.yield %1711 : f32
    } -> tensor<1x4x20x20xf32>
    %1712 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1713 = tensor.splat %1712 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20xf32>
    %1714 = linalg.reduce ins(%1708:tensor<1x4x20x20xf32>) outs(%1713:tensor<1x4x20xf32>) dimensions = [3]
    (%1715: f32, %1716: f32) {
      %1717 = arith.addf %1715, %1716 : f32
      linalg.yield %1717 : f32
    }
    %1718 = tensor.collapse_shape %1714 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20xf32> into tensor<80xf32>
    %1719 = tensor.expand_shape %1718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %1720 = tensor.empty() : tensor<1x4x20x20xf32>
    %1721 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1708, %1719 : tensor<1x4x20x20xf32>, tensor<1x4x20x1xf32>) outs(%1720 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb169(%1722: f32, %1723: f32, %1724: f32):
      %1725 = arith.divf %1722, %1723 : f32
      linalg.yield %1725 : f32
    } -> tensor<1x4x20x20xf32>
    %1726 = tensor.empty() : tensor<1x4x20x20xf32>
    %1727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1721 : tensor<1x4x20x20xf32>) outs(%1726 : tensor<1x4x20x20xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb170(%1728: f32, %1729: f32):
      linalg.yield %1728 : f32
    } -> tensor<1x4x20x20xf32>
    %1730 = tensor.collapse_shape %1727 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20x20xf32> into tensor<1600xf32>
    %1731 = tensor.expand_shape %1730 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 20] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1600xf32> into tensor<4x20x20xf32>
    %1732 = tensor.empty() : tensor<1x4x20x128xf32>
    %1733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1598 : tensor<1x4x20x128xf32>) outs(%1732 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb171(%1734: f32, %1735: f32):
      linalg.yield %1734 : f32
    } -> tensor<1x4x20x128xf32>
    %1736 = tensor.collapse_shape %1733 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x20x128xf32> into tensor<10240xf32>
    %1737 = tensor.expand_shape %1736 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 20, 128] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<4x20x128xf32>
    %1738 = arith.constant {prov.region_id = "matmul_29", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1739 = tensor.splat %1738 {prov.region_id = "matmul_29", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4x20x128xf32>
    %1740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1731, %1737 : tensor<4x20x20xf32>, tensor<4x20x128xf32>) outs(%1739 : tensor<4x20x128xf32>) attrs =  {prov.region_id = "matmul_29", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb172(%1741: f32, %1742: f32, %1743: f32):
      %1744 = arith.mulf %1741, %1742 : f32
      %1745 = arith.addf %1743, %1744 : f32
      linalg.yield %1745 : f32
    } -> tensor<4x20x128xf32>
    %1746 = tensor.collapse_shape %1740 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4x20x128xf32> into tensor<10240xf32>
    %1747 = tensor.expand_shape %1746 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 128] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<1x4x20x128xf32>
    %1748 = tensor.empty() : tensor<1x20x4x128xf32>
    %1749 = linalg.transpose ins(%1747:tensor<1x4x20x128xf32>) outs(%1748:tensor<1x20x4x128xf32>) permutation = [0, 2, 1, 3]
    %1750 = tensor.collapse_shape %1749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x20x4x128xf32> into tensor<10240xf32>
    %1751 = tensor.expand_shape %1750 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %1752 = tensor.empty() : tensor<512x128xf32>
    %1753 = linalg.transpose ins(%79:tensor<128x512xf32>) outs(%1752:tensor<512x128xf32>) permutation = [1, 0]
    %1754 = tensor.collapse_shape %1751 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %1755 = tensor.expand_shape %1754 [[0 : i64, 1 : i64]] output_shape [20, 512] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<10240xf32> into tensor<20x512xf32>
    %1756 = tensor.empty() : tensor<20x128xf32>
    %1757 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1758 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1757 : f32) outs(%1756 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1759 = linalg.matmul {prov.region_id = "matmul_30", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj", prov.transposed_b = "true"} ins(%1755, %1753 : tensor<20x512xf32>, tensor<512x128xf32>) outs(%1758 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1760 = tensor.collapse_shape %1759 [[0 : i64, 1 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<20x128xf32> into tensor<2560xf32>
    %1761 = tensor.expand_shape %1760 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 128] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<2560xf32> into tensor<1x20x128xf32>
    %1762 = tensor.empty() : tensor<1x20x128xf32>
    %1763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1505, %1761 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1762 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb173(%1764: f32, %1765: f32, %1766: f32):
      %1767 = arith.addf %1764, %1765 : f32
      linalg.yield %1767 : f32
    } -> tensor<1x20x128xf32>
    %1768 = tensor.empty() : tensor<1x20x128xf32>
    %1769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763 : tensor<1x20x128xf32>) outs(%1768 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb174(%1770: f32, %1771: f32):
      %1772 = arith.constant 2.000000e+00 : f32
      %1773 = math.powf %1770, %1772 : f32
      linalg.yield %1773 : f32
    } -> tensor<1x20x128xf32>
    %1774 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %1775 = tensor.splat %1774 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x20xf32>
    %1776 = linalg.reduce ins(%1769:tensor<1x20x128xf32>) outs(%1775:tensor<1x20xf32>) dimensions = [2]
    (%1777: f32, %1778: f32) {
      %1779 = arith.addf %1777, %1778 : f32
      linalg.yield %1779 : f32
    }
    %1780 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 1.280000e+02 : f32
    %1781 = tensor.splat %1780 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x20xf32>
    %1782 = tensor.empty() : tensor<1x20xf32>
    %1783 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1776, %1781 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%1782 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb175(%1784: f32, %1785: f32, %1786: f32):
      %1787 = arith.divf %1784, %1785 : f32
      linalg.yield %1787 : f32
    } -> tensor<1x20xf32>
    %1788 = tensor.collapse_shape %1783 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x20xf32> into tensor<20xf32>
    %1789 = tensor.expand_shape %1788 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<20xf32> into tensor<1x20x1xf32>
    %1790 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 1.000000e-06 : f32
    %1791 = tensor.splat %1790 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x20x1xf32>
    %1792 = tensor.empty() : tensor<1x20x1xf32>
    %1793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1789, %1791 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%1792 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb176(%1794: f32, %1795: f32, %1796: f32):
      %1797 = arith.addf %1794, %1795 : f32
      linalg.yield %1797 : f32
    } -> tensor<1x20x1xf32>
    %1798 = tensor.empty() : tensor<1x20x1xf32>
    %1799 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1793 : tensor<1x20x1xf32>) outs(%1798 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb177(%1800: f32, %1801: f32):
      %1802 = math.rsqrt %1800 : f32
      linalg.yield %1802 : f32
    } -> tensor<1x20x1xf32>
    %1803 = tensor.empty() : tensor<1x20x128xf32>
    %1804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763, %1799 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%1803 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb178(%1805: f32, %1806: f32, %1807: f32):
      %1808 = arith.mulf %1805, %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<1x20x128xf32>
    %1809 = tensor.empty() : tensor<1x20x128xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %1804 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%1809 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb179(%1811: f32, %1812: f32, %1813: f32):
      %1814 = arith.mulf %1811, %1812 : f32
      linalg.yield %1814 : f32
    } -> tensor<1x20x128xf32>
    %1815 = tensor.empty() : tensor<128x256xf32>
    %1816 = linalg.transpose ins(%80:tensor<256x128xf32>) outs(%1815:tensor<128x256xf32>) permutation = [1, 0]
    %1817 = tensor.collapse_shape %1810 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1818 = tensor.expand_shape %1817 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1819 = tensor.empty() : tensor<20x256xf32>
    %1820 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1821 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1820 : f32) outs(%1819 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1822 = linalg.matmul {prov.region_id = "matmul_31", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj", prov.transposed_b = "true"} ins(%1818, %1816 : tensor<20x128xf32>, tensor<128x256xf32>) outs(%1821 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1823 = tensor.collapse_shape %1822 [[0 : i64, 1 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<20x256xf32> into tensor<5120xf32>
    %1824 = tensor.expand_shape %1823 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 256] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<5120xf32> into tensor<1x20x256xf32>
    %1825 = tensor.empty() : tensor<1x20x256xf32>
    %1826 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1824 : tensor<1x20x256xf32>) outs(%1825 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.act_fn"} {
    ^bb180(%1827: f32, %1828: f32):
      %1829 = arith.constant 1.000000e+00 : f32
      %1830 = arith.negf %1827 : f32
      %1831 = math.exp %1830 : f32
      %1832 = arith.addf %1829, %1831 : f32
      %1833 = arith.divf %1829, %1832 : f32
      linalg.yield %1833 : f32
    } -> tensor<1x20x256xf32>
    %1834 = tensor.empty() : tensor<1x20x256xf32>
    %1835 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1824, %1826 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%1834 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.act_fn"} {
    ^bb181(%1836: f32, %1837: f32, %1838: f32):
      %1839 = arith.mulf %1836, %1837 : f32
      linalg.yield %1839 : f32
    } -> tensor<1x20x256xf32>
    %1840 = tensor.empty() : tensor<128x256xf32>
    %1841 = linalg.transpose ins(%81:tensor<256x128xf32>) outs(%1840:tensor<128x256xf32>) permutation = [1, 0]
    %1842 = tensor.collapse_shape %1810 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1843 = tensor.expand_shape %1842 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<2560xf32> into tensor<20x128xf32>
    %1844 = tensor.empty() : tensor<20x256xf32>
    %1845 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1846 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1845 : f32) outs(%1844 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1847 = linalg.matmul {prov.region_id = "matmul_32", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj", prov.transposed_b = "true"} ins(%1843, %1841 : tensor<20x128xf32>, tensor<128x256xf32>) outs(%1846 : tensor<20x256xf32>) -> tensor<20x256xf32>
    %1848 = tensor.collapse_shape %1847 [[0 : i64, 1 : i64]] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<20x256xf32> into tensor<5120xf32>
    %1849 = tensor.expand_shape %1848 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 256] {prov.region_id = "view_88", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<5120xf32> into tensor<1x20x256xf32>
    %1850 = tensor.empty() : tensor<1x20x256xf32>
    %1851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1835, %1849 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%1850 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb182(%1852: f32, %1853: f32, %1854: f32):
      %1855 = arith.mulf %1852, %1853 : f32
      linalg.yield %1855 : f32
    } -> tensor<1x20x256xf32>
    %1856 = tensor.empty() : tensor<256x128xf32>
    %1857 = linalg.transpose ins(%82:tensor<128x256xf32>) outs(%1856:tensor<256x128xf32>) permutation = [1, 0]
    %1858 = tensor.collapse_shape %1851 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x20x256xf32> into tensor<5120xf32>
    %1859 = tensor.expand_shape %1858 [[0 : i64, 1 : i64]] output_shape [20, 256] {prov.region_id = "view_89", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<5120xf32> into tensor<20x256xf32>
    %1860 = tensor.empty() : tensor<20x128xf32>
    %1861 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1862 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1861 : f32) outs(%1860 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1863 = linalg.matmul {prov.region_id = "matmul_33", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj", prov.transposed_b = "true"} ins(%1859, %1857 : tensor<20x256xf32>, tensor<256x128xf32>) outs(%1862 : tensor<20x128xf32>) -> tensor<20x128xf32>
    %1864 = tensor.collapse_shape %1863 [[0 : i64, 1 : i64]] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<20x128xf32> into tensor<2560xf32>
    %1865 = tensor.expand_shape %1864 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 128] {prov.region_id = "view_90", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<2560xf32> into tensor<1x20x128xf32>
    %1866 = tensor.empty() : tensor<1x20x128xf32>
    %1867 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1763, %1865 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%1866 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb183(%1868: f32, %1869: f32, %1870: f32):
      %1871 = arith.addf %1868, %1869 : f32
      linalg.yield %1871 : f32
    } -> tensor<1x20x128xf32>
    %1872 = tensor.empty() : tensor<1x20x128xf32>
    %1873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1867 : tensor<1x20x128xf32>) outs(%1872 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb184(%1874: f32, %1875: f32):
      %1876 = arith.constant 2.000000e+00 : f32
      %1877 = math.powf %1874, %1876 : f32
      linalg.yield %1877 : f32
    } -> tensor<1x20x128xf32>
    %1878 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 0.000000e+00 : f32
    %1879 = tensor.splat %1878 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x20xf32>
    %1880 = linalg.reduce ins(%1873:tensor<1x20x128xf32>) outs(%1879:tensor<1x20xf32>) dimensions = [2]
    (%1881: f32, %1882: f32) {
      %1883 = arith.addf %1881, %1882 : f32
      linalg.yield %1883 : f32
    }
    %1884 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 1.280000e+02 : f32
    %1885 = tensor.splat %1884 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x20xf32>
    %1886 = tensor.empty() : tensor<1x20xf32>
    %1887 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1880, %1885 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%1886 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb185(%1888: f32, %1889: f32, %1890: f32):
      %1891 = arith.divf %1888, %1889 : f32
      linalg.yield %1891 : f32
    } -> tensor<1x20xf32>
    %1892 = tensor.collapse_shape %1887 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x20xf32> into tensor<20xf32>
    %1893 = tensor.expand_shape %1892 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<20xf32> into tensor<1x20x1xf32>
    %1894 = arith.constant {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 1.000000e-06 : f32
    %1895 = tensor.splat %1894 {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x20x1xf32>
    %1896 = tensor.empty() : tensor<1x20x1xf32>
    %1897 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1893, %1895 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%1896 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb186(%1898: f32, %1899: f32, %1900: f32):
      %1901 = arith.addf %1898, %1899 : f32
      linalg.yield %1901 : f32
    } -> tensor<1x20x1xf32>
    %1902 = tensor.empty() : tensor<1x20x1xf32>
    %1903 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1897 : tensor<1x20x1xf32>) outs(%1902 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb187(%1904: f32, %1905: f32):
      %1906 = math.rsqrt %1904 : f32
      linalg.yield %1906 : f32
    } -> tensor<1x20x1xf32>
    %1907 = tensor.empty() : tensor<1x20x128xf32>
    %1908 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1867, %1903 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%1907 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb188(%1909: f32, %1910: f32, %1911: f32):
      %1912 = arith.mulf %1909, %1910 : f32
      linalg.yield %1912 : f32
    } -> tensor<1x20x128xf32>
    %1913 = tensor.empty() : tensor<1x20x128xf32>
    %1914 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %1908 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%1913 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb189(%1915: f32, %1916: f32, %1917: f32):
      %1918 = arith.mulf %1915, %1916 : f32
      linalg.yield %1918 : f32
    } -> tensor<1x20x128xf32>
    %1919 = tensor.empty() : tensor<128x512xf32>
    %1920 = linalg.transpose ins(%86:tensor<512x128xf32>) outs(%1919:tensor<128x512xf32>) permutation = [1, 0]
    %1921 = tensor.collapse_shape %1914 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %1922 = tensor.expand_shape %1921 [[0 : i64, 1 : i64]] output_shape [20, 128] {prov.region_id = "view_91", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<2560xf32> into tensor<20x128xf32>
    %1923 = tensor.empty() : tensor<20x512xf32>
    %1924 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1925 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1924 : f32) outs(%1923 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1926 = linalg.matmul {prov.region_id = "matmul_34", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head", prov.transposed_b = "true"} ins(%1922, %1920 : tensor<20x128xf32>, tensor<128x512xf32>) outs(%1925 : tensor<20x512xf32>) -> tensor<20x512xf32>
    %1927 = tensor.collapse_shape %1926 [[0 : i64, 1 : i64]] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<20x512xf32> into tensor<10240xf32>
    %1928 = tensor.expand_shape %1927 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_92", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<10240xf32> into tensor<1x20x512xf32>
    func.return %1928 : tensor<1x20x512xf32>
  }
}
