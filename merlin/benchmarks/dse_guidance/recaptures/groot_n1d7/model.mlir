builtin.module attributes {prov.weights_file = "/path/to/model2MLIR/workloads/groot_n1d7/groot_n1d7.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x132x1024xf32>, %1: tensor<32x1024xf32>, %2: tensor<32x1024x1536xf32>, %3: tensor<32x1536xf32>, %4: tensor<32x132x1536xf32>, %5: tensor<32x1536xf32>, %6: tensor<32x3072x1536xf32>, %7: tensor<32x1536xf32>, %8: tensor<32x1536x1536xf32>, %9: tensor<32x1536xf32>, %10: tensor<32x1024x1024xf32>, %11: tensor<32x1024xf32>, %12: tensor<32x1024x132xf32>, %13: tensor<32x132xf32>, %14: tensor<2048xf32>, %15: tensor<2048xf32>, %16: tensor<1024x1536xf32>, %17: tensor<1536x256xf32>, %18: tensor<1536xf32>, %19: tensor<1536x1536xf32>, %20: tensor<1536xf32>, %21: tensor<3072x1536xf32>, %22: tensor<3072xf32>, %23: tensor<1536x1536xf32>, %24: tensor<1536xf32>, %25: tensor<1536x2048xf32>, %26: tensor<1536xf32>, %27: tensor<1536x2048xf32>, %28: tensor<1536xf32>, %29: tensor<1536x1536xf32>, %30: tensor<1536xf32>, %31: tensor<6144x1536xf32>, %32: tensor<6144xf32>, %33: tensor<1536x6144xf32>, %34: tensor<1536xf32>, %35: tensor<3072x1536xf32>, %36: tensor<3072xf32>, %37: tensor<1536x1536xf32>, %38: tensor<1536xf32>, %39: tensor<1536x1536xf32>, %40: tensor<1536xf32>, %41: tensor<1536x1536xf32>, %42: tensor<1536xf32>, %43: tensor<1536x1536xf32>, %44: tensor<1536xf32>, %45: tensor<6144x1536xf32>, %46: tensor<6144xf32>, %47: tensor<1536x6144xf32>, %48: tensor<1536xf32>, %49: tensor<3072x1536xf32>, %50: tensor<3072xf32>, %51: tensor<1024x1536xf32>, %52: tensor<1024xf32>, %53: tensor<f32>, %54: tensor<1x64x2048xf32>, %55: tensor<1x1x132xf32>, %56: tensor<1x40x132xf32>, %57: tensor<1xi64>, %58: tensor<1x64xi1>, %59: tensor<1x64xi1>, %60: tensor<1xi64>) -> tensor<1x40x132xf32> {
    %61 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} 0.000000e+00 : f32
    %62 = tensor.splat %61 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32>
    %63 = linalg.reduce ins(%54:tensor<1x64x2048xf32>) outs(%62:tensor<1x64xf32>) dimensions = [2]
    (%64: f32, %65: f32) {
      %66 = arith.addf %64, %65 : f32
      linalg.yield %66 : f32
    }
    %67 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} 2.048000e+03 : f32
    %68 = tensor.splat %67 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32>
    %69 = tensor.empty() : tensor<1x64xf32>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %68 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%69 : tensor<1x64xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb0(%71: f32, %72: f32, %73: f32):
      %74 = arith.divf %71, %72 : f32
      linalg.yield %74 : f32
    } -> tensor<1x64xf32>
    %75 = tensor.collapse_shape %70 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32> into tensor<64xf32>
    %76 = tensor.expand_shape %75 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<64xf32> into tensor<1x64x1xf32>
    %77 = tensor.empty() : tensor<1x64x2048xf32>
    %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54, %76 : tensor<1x64x2048xf32>, tensor<1x64x1xf32>) outs(%77 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb1(%79: f32, %80: f32, %81: f32):
      %82 = arith.subf %79, %80 : f32
      linalg.yield %82 : f32
    } -> tensor<1x64x2048xf32>
    %83 = tensor.empty() : tensor<1x64x2048xf32>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78, %78 : tensor<1x64x2048xf32>, tensor<1x64x2048xf32>) outs(%83 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb2(%85: f32, %86: f32, %87: f32):
      %88 = arith.mulf %85, %86 : f32
      linalg.yield %88 : f32
    } -> tensor<1x64x2048xf32>
    %89 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} 0.000000e+00 : f32
    %90 = tensor.splat %89 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32>
    %91 = linalg.reduce ins(%84:tensor<1x64x2048xf32>) outs(%90:tensor<1x64xf32>) dimensions = [2]
    (%92: f32, %93: f32) {
      %94 = arith.addf %92, %93 : f32
      linalg.yield %94 : f32
    }
    %95 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} 2.048000e+03 : f32
    %96 = tensor.splat %95 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32>
    %97 = tensor.empty() : tensor<1x64xf32>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %96 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%97 : tensor<1x64xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb3(%99: f32, %100: f32, %101: f32):
      %102 = arith.divf %99, %100 : f32
      linalg.yield %102 : f32
    } -> tensor<1x64xf32>
    %103 = tensor.collapse_shape %98 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64xf32> into tensor<64xf32>
    %104 = tensor.expand_shape %103 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<64xf32> into tensor<1x64x1xf32>
    %105 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} 1.000000e-05 : f32
    %106 = tensor.splat %105 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} : tensor<1x64x1xf32>
    %107 = tensor.empty() : tensor<1x64x1xf32>
    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %106 : tensor<1x64x1xf32>, tensor<1x64x1xf32>) outs(%107 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb4(%109: f32, %110: f32, %111: f32):
      %112 = arith.addf %109, %110 : f32
      linalg.yield %112 : f32
    } -> tensor<1x64x1xf32>
    %113 = tensor.empty() : tensor<1x64x1xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108 : tensor<1x64x1xf32>) outs(%113 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb5(%115: f32, %116: f32):
      %117 = math.rsqrt %115 : f32
      linalg.yield %117 : f32
    } -> tensor<1x64x1xf32>
    %118 = tensor.empty() : tensor<1x64x2048xf32>
    %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78, %114 : tensor<1x64x2048xf32>, tensor<1x64x1xf32>) outs(%118 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb6(%120: f32, %121: f32, %122: f32):
      %123 = arith.mulf %120, %121 : f32
      linalg.yield %123 : f32
    } -> tensor<1x64x2048xf32>
    %124 = tensor.empty() : tensor<1x64x2048xf32>
    %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %14 : tensor<1x64x2048xf32>, tensor<2048xf32>) outs(%124 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb7(%126: f32, %127: f32, %128: f32):
      %129 = arith.mulf %126, %127 : f32
      linalg.yield %129 : f32
    } -> tensor<1x64x2048xf32>
    %130 = tensor.empty() : tensor<1x64x2048xf32>
    %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %15 : tensor<1x64x2048xf32>, tensor<2048xf32>) outs(%130 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.vlln"} {
    ^bb8(%132: f32, %133: f32, %134: f32):
      %135 = arith.addf %132, %133 : f32
      linalg.yield %135 : f32
    } -> tensor<1x64x2048xf32>
    %136 = tensor.empty() : tensor<1x132x1024xf32>
    %137 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%136 : tensor<1x132x1024xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} {
    ^bb9(%138: i64, %139: f32):
      %140 = arith.index_cast %138 : i64 to index
      %141 = linalg.index 1 : index
      %142 = linalg.index 2 : index
      %143 = tensor.extract %0[%140, %141, %142] : tensor<32x132x1024xf32>
      linalg.yield %143 : f32
    } -> tensor<1x132x1024xf32>
    %144 = tensor.empty() : tensor<1x1024xf32>
    %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%144 : tensor<1x1024xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} {
    ^bb10(%146: i64, %147: f32):
      %148 = arith.index_cast %146 : i64 to index
      %149 = linalg.index 1 : index
      %150 = tensor.extract %1[%148, %149] : tensor<32x1024xf32>
      linalg.yield %150 : f32
    } -> tensor<1x1024xf32>
    %151 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} 0.000000e+00 : f32
    %152 = tensor.splat %151 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} : tensor<1x1x1024xf32>
    %153 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%55, %137 : tensor<1x1x132xf32>, tensor<1x132x1024xf32>) outs(%152 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} {
    ^bb11(%154: f32, %155: f32, %156: f32):
      %157 = arith.mulf %154, %155 : f32
      %158 = arith.addf %156, %157 : f32
      linalg.yield %158 : f32
    } -> tensor<1x1x1024xf32>
    %159 = tensor.collapse_shape %145 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %160 = tensor.expand_shape %159 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %161 = tensor.empty() : tensor<1x1x1024xf32>
    %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%153, %160 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%161 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer1"} {
    ^bb12(%163: f32, %164: f32, %165: f32):
      %166 = arith.addf %163, %164 : f32
      linalg.yield %166 : f32
    } -> tensor<1x1x1024xf32>
    %167 = tensor.empty() : tensor<1x1x1024xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x1x1024xf32>) outs(%167 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder"} {
    ^bb13(%169: f32, %170: f32):
      %171 = arith.constant 0.000000e+00 : f32
      %172 = arith.maximumf %169, %171 : f32
      linalg.yield %172 : f32
    } -> tensor<1x1x1024xf32>
    %173 = tensor.empty() : tensor<1x1024x1536xf32>
    %174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%173 : tensor<1x1024x1536xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} {
    ^bb14(%175: i64, %176: f32):
      %177 = arith.index_cast %175 : i64 to index
      %178 = linalg.index 1 : index
      %179 = linalg.index 2 : index
      %180 = tensor.extract %2[%177, %178, %179] : tensor<32x1024x1536xf32>
      linalg.yield %180 : f32
    } -> tensor<1x1024x1536xf32>
    %181 = tensor.empty() : tensor<1x1536xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%181 : tensor<1x1536xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} {
    ^bb15(%183: i64, %184: f32):
      %185 = arith.index_cast %183 : i64 to index
      %186 = linalg.index 1 : index
      %187 = tensor.extract %3[%185, %186] : tensor<32x1536xf32>
      linalg.yield %187 : f32
    } -> tensor<1x1536xf32>
    %188 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} 0.000000e+00 : f32
    %189 = tensor.splat %188 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} : tensor<1x1x1536xf32>
    %190 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%168, %174 : tensor<1x1x1024xf32>, tensor<1x1024x1536xf32>) outs(%189 : tensor<1x1x1536xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} {
    ^bb16(%191: f32, %192: f32, %193: f32):
      %194 = arith.mulf %191, %192 : f32
      %195 = arith.addf %193, %194 : f32
      linalg.yield %195 : f32
    } -> tensor<1x1x1536xf32>
    %196 = tensor.collapse_shape %182 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} : tensor<1x1536xf32> into tensor<1536xf32>
    %197 = tensor.expand_shape %196 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %198 = tensor.empty() : tensor<1x1x1536xf32>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%190, %197 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%198 : tensor<1x1x1536xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.state_encoder.layer2"} {
    ^bb17(%200: f32, %201: f32, %202: f32):
      %203 = arith.addf %200, %201 : f32
      linalg.yield %203 : f32
    } -> tensor<1x1x1536xf32>
    %204 = tensor.expand_shape %60 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "head", prov.fqn = "head.action_encoder"} : tensor<1xi64> into tensor<1x1xi64>
    %205 = tensor.empty() : tensor<1x40xi64>
    %206 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%204 : tensor<1x1xi64>) outs(%205 : tensor<1x40xi64>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "int64", prov.module = "head", prov.fqn = "head.action_encoder"} {
    ^bb18(%207: i64, %208: i64):
      linalg.yield %207 : i64
    } -> tensor<1x40xi64>
    %209 = tensor.empty() : tensor<1x132x1536xf32>
    %210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%209 : tensor<1x132x1536xf32>) attrs =  {prov.region_id = "gather_4", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} {
    ^bb19(%211: i64, %212: f32):
      %213 = arith.index_cast %211 : i64 to index
      %214 = linalg.index 1 : index
      %215 = linalg.index 2 : index
      %216 = tensor.extract %4[%213, %214, %215] : tensor<32x132x1536xf32>
      linalg.yield %216 : f32
    } -> tensor<1x132x1536xf32>
    %217 = tensor.empty() : tensor<1x1536xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%217 : tensor<1x1536xf32>) attrs =  {prov.region_id = "gather_5", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} {
    ^bb20(%219: i64, %220: f32):
      %221 = arith.index_cast %219 : i64 to index
      %222 = linalg.index 1 : index
      %223 = tensor.extract %5[%221, %222] : tensor<32x1536xf32>
      linalg.yield %223 : f32
    } -> tensor<1x1536xf32>
    %224 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} 0.000000e+00 : f32
    %225 = tensor.splat %224 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} : tensor<1x40x1536xf32>
    %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%56, %210 : tensor<1x40x132xf32>, tensor<1x132x1536xf32>) outs(%225 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} {
    ^bb21(%227: f32, %228: f32, %229: f32):
      %230 = arith.mulf %227, %228 : f32
      %231 = arith.addf %229, %230 : f32
      linalg.yield %231 : f32
    } -> tensor<1x40x1536xf32>
    %232 = tensor.collapse_shape %218 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} : tensor<1x1536xf32> into tensor<1536xf32>
    %233 = tensor.expand_shape %232 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %234 = tensor.empty() : tensor<1x40x1536xf32>
    %235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%226, %233 : tensor<1x40x1536xf32>, tensor<1x1x1536xf32>) outs(%234 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W1"} {
    ^bb22(%236: f32, %237: f32, %238: f32):
      %239 = arith.addf %236, %237 : f32
      linalg.yield %239 : f32
    } -> tensor<1x40x1536xf32>
    %240 = tensor.empty() : tensor<1x40xf32>
    %241 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%206 : tensor<1x40xi64>) outs(%240 : tensor<1x40xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb23(%242: i64, %243: f32):
      %244 = arith.sitofp %242 : i64 to f32
      linalg.yield %244 : f32
    } -> tensor<1x40xf32>
    %245 = tensor.empty() : tensor<768xf32>
    %246 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%245 : tensor<768xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb24(%247: f32):
      %248 = linalg.index 0 : index
      %249 = arith.index_cast %248 : index to i64
      %250 = arith.sitofp %249 : i64 to f32
      %251 = arith.constant 1.000000e+00 : f32
      %252 = arith.mulf %250, %251 : f32
      %253 = arith.constant 0.000000e+00 : f32
      %254 = arith.addf %253, %252 : f32
      linalg.yield %254 : f32
    } -> tensor<768xf32>
    %255 = tensor.empty() : tensor<768xf32>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%246 : tensor<768xf32>) outs(%255 : tensor<768xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb25(%257: f32, %258: f32):
      %259 = arith.negf %257 : f32
      linalg.yield %259 : f32
    } -> tensor<768xf32>
    %260 = tensor.empty() : tensor<f32>
    %261 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%53 : tensor<f32>) outs(%260 : tensor<f32>) attrs =  {prov.region_id = "log_0", prov._pattern_hint = "log", prov.op = "log", prov.family = "elementwise", prov.aten = "aten.log.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb26(%262: f32, %263: f32):
      %264 = math.log %262 : f32
      linalg.yield %264 : f32
    } -> tensor<f32>
    %265 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} 7.680000e+02 : f32
    %266 = tensor.splat %265 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} : tensor<f32>
    %267 = tensor.empty() : tensor<f32>
    %268 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%261, %266 : tensor<f32>, tensor<f32>) outs(%267 : tensor<f32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb27(%269: f32, %270: f32, %271: f32):
      %272 = arith.divf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<f32>
    %273 = tensor.empty() : tensor<768xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%256, %268 : tensor<768xf32>, tensor<f32>) outs(%273 : tensor<768xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb28(%275: f32, %276: f32, %277: f32):
      %278 = arith.mulf %275, %276 : f32
      linalg.yield %278 : f32
    } -> tensor<768xf32>
    %279 = tensor.collapse_shape %241 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} : tensor<1x40xf32> into tensor<40xf32>
    %280 = tensor.expand_shape %279 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 40, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} : tensor<40xf32> into tensor<1x40x1xf32>
    %281 = tensor.empty() : tensor<768xf32>
    %282 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%274 : tensor<768xf32>) outs(%281 : tensor<768xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb29(%283: f32, %284: f32):
      %285 = math.exp %283 : f32
      linalg.yield %285 : f32
    } -> tensor<768xf32>
    %286 = tensor.empty() : tensor<1x40x768xf32>
    %287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%280, %282 : tensor<1x40x1xf32>, tensor<768xf32>) outs(%286 : tensor<1x40x768xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb30(%288: f32, %289: f32, %290: f32):
      %291 = arith.mulf %288, %289 : f32
      linalg.yield %291 : f32
    } -> tensor<1x40x768xf32>
    %292 = tensor.empty() : tensor<1x40x768xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287 : tensor<1x40x768xf32>) outs(%292 : tensor<1x40x768xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb31(%294: f32, %295: f32):
      %296 = math.sin %294 : f32
      linalg.yield %296 : f32
    } -> tensor<1x40x768xf32>
    %297 = tensor.empty() : tensor<1x40x768xf32>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287 : tensor<1x40x768xf32>) outs(%297 : tensor<1x40x768xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} {
    ^bb32(%299: f32, %300: f32):
      %301 = math.cos %299 : f32
      linalg.yield %301 : f32
    } -> tensor<1x40x768xf32>
    %302 = tensor.concat dim(2) %293, %298 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.pos_encoding"} : (tensor<1x40x768xf32>, tensor<1x40x768xf32>) -> tensor<1x40x1536xf32>
    %303 = tensor.concat dim(2) %235, %302 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder"} : (tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x40x3072xf32>
    %304 = tensor.empty() : tensor<1x3072x1536xf32>
    %305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%304 : tensor<1x3072x1536xf32>) attrs =  {prov.region_id = "gather_6", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} {
    ^bb33(%306: i64, %307: f32):
      %308 = arith.index_cast %306 : i64 to index
      %309 = linalg.index 1 : index
      %310 = linalg.index 2 : index
      %311 = tensor.extract %6[%308, %309, %310] : tensor<32x3072x1536xf32>
      linalg.yield %311 : f32
    } -> tensor<1x3072x1536xf32>
    %312 = tensor.empty() : tensor<1x1536xf32>
    %313 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%312 : tensor<1x1536xf32>) attrs =  {prov.region_id = "gather_7", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} {
    ^bb34(%314: i64, %315: f32):
      %316 = arith.index_cast %314 : i64 to index
      %317 = linalg.index 1 : index
      %318 = tensor.extract %7[%316, %317] : tensor<32x1536xf32>
      linalg.yield %318 : f32
    } -> tensor<1x1536xf32>
    %319 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} 0.000000e+00 : f32
    %320 = tensor.splat %319 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} : tensor<1x40x1536xf32>
    %321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%303, %305 : tensor<1x40x3072xf32>, tensor<1x3072x1536xf32>) outs(%320 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} {
    ^bb35(%322: f32, %323: f32, %324: f32):
      %325 = arith.mulf %322, %323 : f32
      %326 = arith.addf %324, %325 : f32
      linalg.yield %326 : f32
    } -> tensor<1x40x1536xf32>
    %327 = tensor.collapse_shape %313 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} : tensor<1x1536xf32> into tensor<1536xf32>
    %328 = tensor.expand_shape %327 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %329 = tensor.empty() : tensor<1x40x1536xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%321, %328 : tensor<1x40x1536xf32>, tensor<1x1x1536xf32>) outs(%329 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W2"} {
    ^bb36(%331: f32, %332: f32, %333: f32):
      %334 = arith.addf %331, %332 : f32
      linalg.yield %334 : f32
    } -> tensor<1x40x1536xf32>
    %335 = tensor.empty() : tensor<1x40x1536xf32>
    %336 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%330 : tensor<1x40x1536xf32>) outs(%335 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder"} {
    ^bb37(%337: f32, %338: f32):
      %339 = arith.constant 1.000000e+00 : f32
      %340 = arith.negf %337 : f32
      %341 = math.exp %340 : f32
      %342 = arith.addf %339, %341 : f32
      %343 = arith.divf %339, %342 : f32
      linalg.yield %343 : f32
    } -> tensor<1x40x1536xf32>
    %344 = tensor.empty() : tensor<1x40x1536xf32>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%330, %336 : tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) outs(%344 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder"} {
    ^bb38(%346: f32, %347: f32, %348: f32):
      %349 = arith.mulf %346, %347 : f32
      linalg.yield %349 : f32
    } -> tensor<1x40x1536xf32>
    %350 = tensor.empty() : tensor<1x1536x1536xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%350 : tensor<1x1536x1536xf32>) attrs =  {prov.region_id = "gather_8", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} {
    ^bb39(%352: i64, %353: f32):
      %354 = arith.index_cast %352 : i64 to index
      %355 = linalg.index 1 : index
      %356 = linalg.index 2 : index
      %357 = tensor.extract %8[%354, %355, %356] : tensor<32x1536x1536xf32>
      linalg.yield %357 : f32
    } -> tensor<1x1536x1536xf32>
    %358 = tensor.empty() : tensor<1x1536xf32>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%358 : tensor<1x1536xf32>) attrs =  {prov.region_id = "gather_9", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} {
    ^bb40(%360: i64, %361: f32):
      %362 = arith.index_cast %360 : i64 to index
      %363 = linalg.index 1 : index
      %364 = tensor.extract %9[%362, %363] : tensor<32x1536xf32>
      linalg.yield %364 : f32
    } -> tensor<1x1536xf32>
    %365 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} 0.000000e+00 : f32
    %366 = tensor.splat %365 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} : tensor<1x40x1536xf32>
    %367 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%345, %351 : tensor<1x40x1536xf32>, tensor<1x1536x1536xf32>) outs(%366 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} {
    ^bb41(%368: f32, %369: f32, %370: f32):
      %371 = arith.mulf %368, %369 : f32
      %372 = arith.addf %370, %371 : f32
      linalg.yield %372 : f32
    } -> tensor<1x40x1536xf32>
    %373 = tensor.collapse_shape %359 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} : tensor<1x1536xf32> into tensor<1536xf32>
    %374 = tensor.expand_shape %373 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %375 = tensor.empty() : tensor<1x40x1536xf32>
    %376 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%367, %374 : tensor<1x40x1536xf32>, tensor<1x1x1536xf32>) outs(%375 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_encoder.W3"} {
    ^bb42(%377: f32, %378: f32, %379: f32):
      %380 = arith.addf %377, %378 : f32
      linalg.yield %380 : f32
    } -> tensor<1x40x1536xf32>
    %381 = tensor.empty() : tensor<40xi64>
    %382 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%381 : tensor<40xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb43(%383: i64):
      %384 = linalg.index 0 : index
      %385 = arith.index_cast %384 : index to i64
      %386 = arith.constant 1 : i64
      %387 = arith.muli %385, %386 : i64
      %388 = arith.constant 0 : i64
      %389 = arith.addi %388, %387 : i64
      linalg.yield %389 : i64
    } -> tensor<40xi64>
    %390 = tensor.empty() : tensor<40x1536xf32>
    %391 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%382 : tensor<40xi64>) outs(%390 : tensor<40x1536xf32>) attrs =  {prov.region_id = "gather_10", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.position_embedding"} {
    ^bb44(%392: i64, %393: f32):
      %394 = arith.index_cast %392 : i64 to index
      %395 = linalg.index 1 : index
      %396 = tensor.extract %16[%394, %395] : tensor<1024x1536xf32>
      linalg.yield %396 : f32
    } -> tensor<40x1536xf32>
    %397 = tensor.collapse_shape %391 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<40x1536xf32> into tensor<61440xf32>
    %398 = tensor.expand_shape %397 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 40, 1536] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<61440xf32> into tensor<1x40x1536xf32>
    %399 = tensor.empty() : tensor<1x40x1536xf32>
    %400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%376, %398 : tensor<1x40x1536xf32>, tensor<1x40x1536xf32>) outs(%399 : tensor<1x40x1536xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%401: f32, %402: f32, %403: f32):
      %404 = arith.addf %401, %402 : f32
      linalg.yield %404 : f32
    } -> tensor<1x40x1536xf32>
    %405 = tensor.concat dim(1) %199, %400 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x1536xf32>, tensor<1x40x1536xf32>) -> tensor<1x41x1536xf32>
    %406 = tensor.empty() : tensor<128xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%406 : tensor<128xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb46(%408: f32):
      %409 = linalg.index 0 : index
      %410 = arith.index_cast %409 : index to i64
      %411 = arith.sitofp %410 : i64 to f32
      %412 = arith.constant 1.000000e+00 : f32
      %413 = arith.mulf %411, %412 : f32
      %414 = arith.constant 0.000000e+00 : f32
      %415 = arith.addf %414, %413 : f32
      linalg.yield %415 : f32
    } -> tensor<128xf32>
    %416 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} -9.2103405 : f32
    %417 = tensor.splat %416 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : tensor<128xf32>
    %418 = tensor.empty() : tensor<128xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%407, %417 : tensor<128xf32>, tensor<128xf32>) outs(%418 : tensor<128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb47(%420: f32, %421: f32, %422: f32):
      %423 = arith.mulf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<128xf32>
    %424 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} 1.270000e+02 : f32
    %425 = tensor.splat %424 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : tensor<128xf32>
    %426 = tensor.empty() : tensor<128xf32>
    %427 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%419, %425 : tensor<128xf32>, tensor<128xf32>) outs(%426 : tensor<128xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb48(%428: f32, %429: f32, %430: f32):
      %431 = arith.divf %428, %429 : f32
      linalg.yield %431 : f32
    } -> tensor<128xf32>
    %432 = tensor.empty() : tensor<128xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%427 : tensor<128xf32>) outs(%432 : tensor<128xf32>) attrs =  {prov.region_id = "exp_1", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb49(%434: f32, %435: f32):
      %436 = math.exp %434 : f32
      linalg.yield %436 : f32
    } -> tensor<128xf32>
    %437 = "tensor.extract_slice"(%60) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1xi64>) -> tensor<1xi64>
    %438 = tensor.expand_shape %437 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : tensor<1xi64> into tensor<1x1xi64>
    %439 = tensor.empty() : tensor<1x1xf32>
    %440 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%438 : tensor<1x1xi64>) outs(%439 : tensor<1x1xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb50(%441: i64, %442: f32):
      %443 = arith.sitofp %441 : i64 to f32
      linalg.yield %443 : f32
    } -> tensor<1x1xf32>
    %444 = tensor.expand_shape %433 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : tensor<128xf32> into tensor<1x128xf32>
    %445 = "tensor.extract_slice"(%444) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %446 = tensor.empty() : tensor<1x128xf32>
    %447 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%440, %445 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%446 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb51(%448: f32, %449: f32, %450: f32):
      %451 = arith.mulf %448, %449 : f32
      linalg.yield %451 : f32
    } -> tensor<1x128xf32>
    %452 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} 1.000000e+00 : f32
    %453 = tensor.splat %452 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : tensor<1x128xf32>
    %454 = tensor.empty() : tensor<1x128xf32>
    %455 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%447, %453 : tensor<1x128xf32>, tensor<1x128xf32>) outs(%454 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb52(%456: f32, %457: f32, %458: f32):
      %459 = arith.mulf %456, %457 : f32
      linalg.yield %459 : f32
    } -> tensor<1x128xf32>
    %460 = tensor.empty() : tensor<1x128xf32>
    %461 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%455 : tensor<1x128xf32>) outs(%460 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb53(%462: f32, %463: f32):
      %464 = math.sin %462 : f32
      linalg.yield %464 : f32
    } -> tensor<1x128xf32>
    %465 = tensor.empty() : tensor<1x128xf32>
    %466 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%455 : tensor<1x128xf32>) outs(%465 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} {
    ^bb54(%467: f32, %468: f32):
      %469 = math.cos %467 : f32
      linalg.yield %469 : f32
    } -> tensor<1x128xf32>
    %470 = tensor.concat dim(1) %461, %466 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %471 = "tensor.extract_slice"(%470) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 256>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %472 = "tensor.extract_slice"(%471) <{static_offsets = array<i64: 0, 128>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x256xf32>) -> tensor<1x128xf32>
    %473 = "tensor.extract_slice"(%470) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 256>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x256xf32>) -> tensor<1x256xf32>
    %474 = "tensor.extract_slice"(%473) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 128>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x256xf32>) -> tensor<1x128xf32>
    %475 = tensor.concat dim(1) %472, %474 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.time_proj"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %476 = tensor.empty() : tensor<256x1536xf32>
    %477 = linalg.transpose ins(%17:tensor<1536x256xf32>) outs(%476:tensor<256x1536xf32>) permutation = [1, 0]
    %478 = tensor.empty() : tensor<1x1536xf32>
    %479 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %480 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%479 : f32) outs(%478 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %481 = linalg.matmul {prov.region_id = "matmul_5", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.linear_1", prov.transposed_b = "true"} ins(%475, %477 : tensor<1x256xf32>, tensor<256x1536xf32>) outs(%480 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %482 = tensor.empty() : tensor<1x1536xf32>
    %483 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%481, %18 : tensor<1x1536xf32>, tensor<1536xf32>) outs(%482 : tensor<1x1536xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.linear_1"} {
    ^bb55(%484: f32, %485: f32, %486: f32):
      %487 = arith.addf %484, %485 : f32
      linalg.yield %487 : f32
    } -> tensor<1x1536xf32>
    %488 = tensor.empty() : tensor<1x1536xf32>
    %489 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%483 : tensor<1x1536xf32>) outs(%488 : tensor<1x1536xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.act"} {
    ^bb56(%490: f32, %491: f32):
      %492 = arith.constant 1.000000e+00 : f32
      %493 = arith.negf %490 : f32
      %494 = math.exp %493 : f32
      %495 = arith.addf %492, %494 : f32
      %496 = arith.divf %492, %495 : f32
      linalg.yield %496 : f32
    } -> tensor<1x1536xf32>
    %497 = tensor.empty() : tensor<1x1536xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%483, %489 : tensor<1x1536xf32>, tensor<1x1536xf32>) outs(%497 : tensor<1x1536xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.act"} {
    ^bb57(%499: f32, %500: f32, %501: f32):
      %502 = arith.mulf %499, %500 : f32
      linalg.yield %502 : f32
    } -> tensor<1x1536xf32>
    %503 = tensor.empty() : tensor<1536x1536xf32>
    %504 = linalg.transpose ins(%19:tensor<1536x1536xf32>) outs(%503:tensor<1536x1536xf32>) permutation = [1, 0]
    %505 = tensor.empty() : tensor<1x1536xf32>
    %506 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %507 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%506 : f32) outs(%505 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %508 = linalg.matmul {prov.region_id = "matmul_6", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.linear_2", prov.transposed_b = "true"} ins(%498, %504 : tensor<1x1536xf32>, tensor<1536x1536xf32>) outs(%507 : tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %509 = tensor.empty() : tensor<1x1536xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%508, %20 : tensor<1x1536xf32>, tensor<1536xf32>) outs(%509 : tensor<1x1536xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.timestep_encoder.timestep_embedder.linear_2"} {
    ^bb58(%511: f32, %512: f32, %513: f32):
      %514 = arith.addf %511, %512 : f32
      linalg.yield %514 : f32
    } -> tensor<1x1536xf32>
    %515 = tensor.empty() : tensor<1x64xi1>
    %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<1x64xi1>) outs(%515 : tensor<1x64xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model"} {
    ^bb59(%517: i1, %518: i1):
      %519 = arith.constant true
      %520 = arith.xori %517, %519 : i1
      linalg.yield %520 : i1
    } -> tensor<1x64xi1>
    %521 = tensor.empty() : tensor<1x64xi1>
    %522 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%516, %58 : tensor<1x64xi1>, tensor<1x64xi1>) outs(%521 : tensor<1x64xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_and.Tensor", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model"} {
    ^bb60(%523: i1, %524: i1, %525: i1):
      %526 = arith.andi %523, %524 : i1
      linalg.yield %526 : i1
    } -> tensor<1x64xi1>
    %527 = tensor.empty() : tensor<1x1536xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510 : tensor<1x1536xf32>) outs(%527 : tensor<1x1536xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.silu"} {
    ^bb61(%529: f32, %530: f32):
      %531 = arith.constant 1.000000e+00 : f32
      %532 = arith.negf %529 : f32
      %533 = math.exp %532 : f32
      %534 = arith.addf %531, %533 : f32
      %535 = arith.divf %531, %534 : f32
      linalg.yield %535 : f32
    } -> tensor<1x1536xf32>
    %536 = tensor.empty() : tensor<1x1536xf32>
    %537 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510, %528 : tensor<1x1536xf32>, tensor<1x1536xf32>) outs(%536 : tensor<1x1536xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.silu"} {
    ^bb62(%538: f32, %539: f32, %540: f32):
      %541 = arith.mulf %538, %539 : f32
      linalg.yield %541 : f32
    } -> tensor<1x1536xf32>
    %542 = tensor.empty() : tensor<1536x3072xf32>
    %543 = linalg.transpose ins(%21:tensor<3072x1536xf32>) outs(%542:tensor<1536x3072xf32>) permutation = [1, 0]
    %544 = tensor.empty() : tensor<1x3072xf32>
    %545 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %546 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%545 : f32) outs(%544 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %547 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.linear", prov.transposed_b = "true"} ins(%537, %543 : tensor<1x1536xf32>, tensor<1536x3072xf32>) outs(%546 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %548 = tensor.empty() : tensor<1x3072xf32>
    %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%547, %22 : tensor<1x3072xf32>, tensor<3072xf32>) outs(%548 : tensor<1x3072xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.linear"} {
    ^bb63(%550: f32, %551: f32, %552: f32):
      %553 = arith.addf %550, %551 : f32
      linalg.yield %553 : f32
    } -> tensor<1x3072xf32>
    %554 = "tensor.extract_slice"(%549) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %555 = "tensor.extract_slice"(%549) <{static_offsets = array<i64: 0, 1536>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %556 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} 0.000000e+00 : f32
    %557 = tensor.splat %556 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32>
    %558 = linalg.reduce ins(%405:tensor<1x41x1536xf32>) outs(%557:tensor<1x41xf32>) dimensions = [2]
    (%559: f32, %560: f32) {
      %561 = arith.addf %559, %560 : f32
      linalg.yield %561 : f32
    }
    %562 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} 1.536000e+03 : f32
    %563 = tensor.splat %562 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32>
    %564 = tensor.empty() : tensor<1x41xf32>
    %565 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%558, %563 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%564 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb64(%566: f32, %567: f32, %568: f32):
      %569 = arith.divf %566, %567 : f32
      linalg.yield %569 : f32
    } -> tensor<1x41xf32>
    %570 = tensor.collapse_shape %565 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32> into tensor<41xf32>
    %571 = tensor.expand_shape %570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<41xf32> into tensor<1x41x1xf32>
    %572 = tensor.empty() : tensor<1x41x1536xf32>
    %573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%405, %571 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%572 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb65(%574: f32, %575: f32, %576: f32):
      %577 = arith.subf %574, %575 : f32
      linalg.yield %577 : f32
    } -> tensor<1x41x1536xf32>
    %578 = tensor.empty() : tensor<1x41x1536xf32>
    %579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573, %573 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%578 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb66(%580: f32, %581: f32, %582: f32):
      %583 = arith.mulf %580, %581 : f32
      linalg.yield %583 : f32
    } -> tensor<1x41x1536xf32>
    %584 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} 0.000000e+00 : f32
    %585 = tensor.splat %584 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32>
    %586 = linalg.reduce ins(%579:tensor<1x41x1536xf32>) outs(%585:tensor<1x41xf32>) dimensions = [2]
    (%587: f32, %588: f32) {
      %589 = arith.addf %587, %588 : f32
      linalg.yield %589 : f32
    }
    %590 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} 1.536000e+03 : f32
    %591 = tensor.splat %590 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32>
    %592 = tensor.empty() : tensor<1x41xf32>
    %593 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%586, %591 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%592 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb67(%594: f32, %595: f32, %596: f32):
      %597 = arith.divf %594, %595 : f32
      linalg.yield %597 : f32
    } -> tensor<1x41xf32>
    %598 = tensor.collapse_shape %593 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41xf32> into tensor<41xf32>
    %599 = tensor.expand_shape %598 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<41xf32> into tensor<1x41x1xf32>
    %600 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} 1.000000e-05 : f32
    %601 = tensor.splat %600 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} : tensor<1x41x1xf32>
    %602 = tensor.empty() : tensor<1x41x1xf32>
    %603 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599, %601 : tensor<1x41x1xf32>, tensor<1x41x1xf32>) outs(%602 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb68(%604: f32, %605: f32, %606: f32):
      %607 = arith.addf %604, %605 : f32
      linalg.yield %607 : f32
    } -> tensor<1x41x1xf32>
    %608 = tensor.empty() : tensor<1x41x1xf32>
    %609 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%603 : tensor<1x41x1xf32>) outs(%608 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb69(%610: f32, %611: f32):
      %612 = math.rsqrt %610 : f32
      linalg.yield %612 : f32
    } -> tensor<1x41x1xf32>
    %613 = tensor.empty() : tensor<1x41x1536xf32>
    %614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573, %609 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%613 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1.norm"} {
    ^bb70(%615: f32, %616: f32, %617: f32):
      %618 = arith.mulf %615, %616 : f32
      linalg.yield %618 : f32
    } -> tensor<1x41x1536xf32>
    %619 = "tensor.extract_slice"(%554) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %620 = tensor.collapse_shape %619 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : tensor<1x1536xf32> into tensor<1536xf32>
    %621 = tensor.expand_shape %620 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %622 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} 1.000000e+00 : f32
    %623 = tensor.splat %622 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : tensor<1x1x1536xf32>
    %624 = tensor.empty() : tensor<1x1x1536xf32>
    %625 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %623 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%624 : tensor<1x1x1536xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} {
    ^bb71(%626: f32, %627: f32, %628: f32):
      %629 = arith.addf %626, %627 : f32
      linalg.yield %629 : f32
    } -> tensor<1x1x1536xf32>
    %630 = tensor.empty() : tensor<1x41x1536xf32>
    %631 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614, %625 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%630 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} {
    ^bb72(%632: f32, %633: f32, %634: f32):
      %635 = arith.mulf %632, %633 : f32
      linalg.yield %635 : f32
    } -> tensor<1x41x1536xf32>
    %636 = "tensor.extract_slice"(%555) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %637 = tensor.collapse_shape %636 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : tensor<1x1536xf32> into tensor<1536xf32>
    %638 = tensor.expand_shape %637 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %639 = tensor.empty() : tensor<1x41x1536xf32>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631, %638 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%639 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm1"} {
    ^bb73(%641: f32, %642: f32, %643: f32):
      %644 = arith.addf %641, %642 : f32
      linalg.yield %644 : f32
    } -> tensor<1x41x1536xf32>
    %645 = tensor.collapse_shape %522 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x64xi1> into tensor<64xi1>
    %646 = tensor.expand_shape %645 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 64] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<64xi1> into tensor<1x1x64xi1>
    %647 = tensor.empty() : tensor<1x32x64xi1>
    %648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646 : tensor<1x1x64xi1>) outs(%647 : tensor<1x32x64xi1>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb74(%649: i1, %650: i1):
      linalg.yield %649 : i1
    } -> tensor<1x32x64xi1>
    %651 = tensor.collapse_shape %648 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x64xi1> into tensor<2048xi1>
    %652 = tensor.expand_shape %651 [[0 : i64, 1 : i64]] output_shape [32, 64] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<2048xi1> into tensor<32x64xi1>
    %653 = tensor.collapse_shape %652 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<32x64xi1> into tensor<2048xi1>
    %654 = tensor.expand_shape %653 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 64] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<2048xi1> into tensor<1x32x1x64xi1>
    %655 = tensor.collapse_shape %640 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %656 = tensor.expand_shape %655 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q"} : tensor<62976xf32> into tensor<41x1536xf32>
    %657 = tensor.empty() : tensor<1536x1536xf32>
    %658 = linalg.transpose ins(%23:tensor<1536x1536xf32>) outs(%657:tensor<1536x1536xf32>) permutation = [1, 0]
    %659 = tensor.empty() : tensor<41x1536xf32>
    %660 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %661 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%660 : f32) outs(%659 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %662 = linalg.matmul {prov.region_id = "matmul_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q", prov.transposed_b = "true"} ins(%656, %658 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%661 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %663 = tensor.empty() : tensor<41x1536xf32>
    %664 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%662, %24 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%663 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q"} {
    ^bb75(%665: f32, %666: f32, %667: f32):
      %668 = arith.addf %665, %666 : f32
      linalg.yield %668 : f32
    } -> tensor<41x1536xf32>
    %669 = tensor.collapse_shape %664 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q"} : tensor<41x1536xf32> into tensor<62976xf32>
    %670 = tensor.expand_shape %669 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_q"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %671 = tensor.collapse_shape %131 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k"} : tensor<1x64x2048xf32> into tensor<131072xf32>
    %672 = tensor.expand_shape %671 [[0 : i64, 1 : i64]] output_shape [64, 2048] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k"} : tensor<131072xf32> into tensor<64x2048xf32>
    %673 = tensor.empty() : tensor<2048x1536xf32>
    %674 = linalg.transpose ins(%25:tensor<1536x2048xf32>) outs(%673:tensor<2048x1536xf32>) permutation = [1, 0]
    %675 = tensor.empty() : tensor<64x1536xf32>
    %676 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %677 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%676 : f32) outs(%675 : tensor<64x1536xf32>) -> tensor<64x1536xf32>
    %678 = linalg.matmul {prov.region_id = "matmul_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k", prov.transposed_b = "true"} ins(%672, %674 : tensor<64x2048xf32>, tensor<2048x1536xf32>) outs(%677 : tensor<64x1536xf32>) -> tensor<64x1536xf32>
    %679 = tensor.empty() : tensor<64x1536xf32>
    %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%678, %26 : tensor<64x1536xf32>, tensor<1536xf32>) outs(%679 : tensor<64x1536xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k"} {
    ^bb76(%681: f32, %682: f32, %683: f32):
      %684 = arith.addf %681, %682 : f32
      linalg.yield %684 : f32
    } -> tensor<64x1536xf32>
    %685 = tensor.collapse_shape %680 [[0 : i64, 1 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k"} : tensor<64x1536xf32> into tensor<98304xf32>
    %686 = tensor.expand_shape %685 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1536] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_k"} : tensor<98304xf32> into tensor<1x64x1536xf32>
    %687 = tensor.collapse_shape %131 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v"} : tensor<1x64x2048xf32> into tensor<131072xf32>
    %688 = tensor.expand_shape %687 [[0 : i64, 1 : i64]] output_shape [64, 2048] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v"} : tensor<131072xf32> into tensor<64x2048xf32>
    %689 = tensor.empty() : tensor<2048x1536xf32>
    %690 = linalg.transpose ins(%27:tensor<1536x2048xf32>) outs(%689:tensor<2048x1536xf32>) permutation = [1, 0]
    %691 = tensor.empty() : tensor<64x1536xf32>
    %692 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %693 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%692 : f32) outs(%691 : tensor<64x1536xf32>) -> tensor<64x1536xf32>
    %694 = linalg.matmul {prov.region_id = "matmul_10", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v", prov.transposed_b = "true"} ins(%688, %690 : tensor<64x2048xf32>, tensor<2048x1536xf32>) outs(%693 : tensor<64x1536xf32>) -> tensor<64x1536xf32>
    %695 = tensor.empty() : tensor<64x1536xf32>
    %696 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%694, %28 : tensor<64x1536xf32>, tensor<1536xf32>) outs(%695 : tensor<64x1536xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v"} {
    ^bb77(%697: f32, %698: f32, %699: f32):
      %700 = arith.addf %697, %698 : f32
      linalg.yield %700 : f32
    } -> tensor<64x1536xf32>
    %701 = tensor.collapse_shape %696 [[0 : i64, 1 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v"} : tensor<64x1536xf32> into tensor<98304xf32>
    %702 = tensor.expand_shape %701 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1536] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_v"} : tensor<98304xf32> into tensor<1x64x1536xf32>
    %703 = tensor.collapse_shape %670 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %704 = tensor.expand_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 41, 32, 48] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<62976xf32> into tensor<1x41x32x48xf32>
    %705 = tensor.empty() : tensor<1x32x41x48xf32>
    %706 = linalg.transpose ins(%704:tensor<1x41x32x48xf32>) outs(%705:tensor<1x32x41x48xf32>) permutation = [0, 2, 1, 3]
    %707 = tensor.collapse_shape %686 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x64x1536xf32> into tensor<98304xf32>
    %708 = tensor.expand_shape %707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 32, 48] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<98304xf32> into tensor<1x64x32x48xf32>
    %709 = tensor.empty() : tensor<1x32x64x48xf32>
    %710 = linalg.transpose ins(%708:tensor<1x64x32x48xf32>) outs(%709:tensor<1x32x64x48xf32>) permutation = [0, 2, 1, 3]
    %711 = tensor.collapse_shape %702 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x64x1536xf32> into tensor<98304xf32>
    %712 = tensor.expand_shape %711 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 32, 48] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<98304xf32> into tensor<1x64x32x48xf32>
    %713 = tensor.empty() : tensor<1x32x64x48xf32>
    %714 = linalg.transpose ins(%712:tensor<1x64x32x48xf32>) outs(%713:tensor<1x32x64x48xf32>) permutation = [0, 2, 1, 3]
    %715 = tensor.empty() : tensor<1x32x48x64xf32>
    %716 = linalg.transpose ins(%710:tensor<1x32x64x48xf32>) outs(%715:tensor<1x32x48x64xf32>) permutation = [0, 1, 3, 2]
    %717 = tensor.empty() : tensor<1x32x41x48xf32>
    %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706 : tensor<1x32x41x48xf32>) outs(%717 : tensor<1x32x41x48xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb78(%719: f32, %720: f32):
      linalg.yield %719 : f32
    } -> tensor<1x32x41x48xf32>
    %721 = tensor.collapse_shape %718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41x48xf32> into tensor<62976xf32>
    %722 = tensor.expand_shape %721 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 41, 48] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<62976xf32> into tensor<32x41x48xf32>
    %723 = tensor.empty() : tensor<1x32x48x64xf32>
    %724 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%716 : tensor<1x32x48x64xf32>) outs(%723 : tensor<1x32x48x64xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb79(%725: f32, %726: f32):
      linalg.yield %725 : f32
    } -> tensor<1x32x48x64xf32>
    %727 = tensor.collapse_shape %724 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x48x64xf32> into tensor<98304xf32>
    %728 = tensor.expand_shape %727 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 48, 64] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<98304xf32> into tensor<32x48x64xf32>
    %729 = arith.constant {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0.000000e+00 : f32
    %730 = tensor.splat %729 {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<32x41x64xf32>
    %731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%722, %728 : tensor<32x41x48xf32>, tensor<32x48x64xf32>) outs(%730 : tensor<32x41x64xf32>) attrs =  {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb80(%732: f32, %733: f32, %734: f32):
      %735 = arith.mulf %732, %733 : f32
      %736 = arith.addf %734, %735 : f32
      linalg.yield %736 : f32
    } -> tensor<32x41x64xf32>
    %737 = tensor.collapse_shape %731 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<32x41x64xf32> into tensor<83968xf32>
    %738 = tensor.expand_shape %737 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 64] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<83968xf32> into tensor<1x32x41x64xf32>
    %739 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0.144337565 : f32
    %740 = tensor.splat %739 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41x64xf32>
    %741 = tensor.empty() : tensor<1x32x41x64xf32>
    %742 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%738, %740 : tensor<1x32x41x64xf32>, tensor<1x32x41x64xf32>) outs(%741 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb81(%743: f32, %744: f32, %745: f32):
      %746 = arith.mulf %743, %744 : f32
      linalg.yield %746 : f32
    } -> tensor<1x32x41x64xf32>
    %747 = tensor.empty() : tensor<1x32x1x64xi1>
    %748 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654 : tensor<1x32x1x64xi1>) outs(%747 : tensor<1x32x1x64xi1>) attrs =  {prov.region_id = "bitwise_2", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb82(%749: i1, %750: i1):
      %751 = arith.constant true
      %752 = arith.xori %749, %751 : i1
      linalg.yield %752 : i1
    } -> tensor<1x32x1x64xi1>
    %753 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0xff800000 : f32
    %754 = tensor.splat %753 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<f32>
    %755 = tensor.empty() : tensor<1x32x41x64xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%748, %754, %742 : tensor<1x32x1x64xi1>, tensor<f32>, tensor<1x32x41x64xf32>) outs(%755 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb83(%757: i1, %758: f32, %759: f32, %760: f32):
      %761 = arith.select %757, %758, %759 : f32
      linalg.yield %761 : f32
    } -> tensor<1x32x41x64xf32>
    %762 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0xff800000 : f32
    %763 = tensor.splat %762 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41xf32>
    %764 = linalg.reduce ins(%756:tensor<1x32x41x64xf32>) outs(%763:tensor<1x32x41xf32>) dimensions = [3]
    (%765: f32, %766: f32) {
      %767 = arith.maximumf %765, %766 : f32
      linalg.yield %767 : f32
    }
    %768 = tensor.collapse_shape %764 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41xf32> into tensor<1312xf32>
    %769 = tensor.expand_shape %768 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1312xf32> into tensor<1x32x41x1xf32>
    %770 = tensor.empty() : tensor<1x32x41x64xf32>
    %771 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756, %769 : tensor<1x32x41x64xf32>, tensor<1x32x41x1xf32>) outs(%770 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb84(%772: f32, %773: f32, %774: f32):
      %775 = arith.subf %772, %773 : f32
      linalg.yield %775 : f32
    } -> tensor<1x32x41x64xf32>
    %776 = tensor.empty() : tensor<1x32x41x64xf32>
    %777 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771 : tensor<1x32x41x64xf32>) outs(%776 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb85(%778: f32, %779: f32):
      %780 = math.exp %778 : f32
      linalg.yield %780 : f32
    } -> tensor<1x32x41x64xf32>
    %781 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0.000000e+00 : f32
    %782 = tensor.splat %781 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41xf32>
    %783 = linalg.reduce ins(%777:tensor<1x32x41x64xf32>) outs(%782:tensor<1x32x41xf32>) dimensions = [3]
    (%784: f32, %785: f32) {
      %786 = arith.addf %784, %785 : f32
      linalg.yield %786 : f32
    }
    %787 = tensor.collapse_shape %783 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41xf32> into tensor<1312xf32>
    %788 = tensor.expand_shape %787 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1312xf32> into tensor<1x32x41x1xf32>
    %789 = tensor.empty() : tensor<1x32x41x64xf32>
    %790 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%777, %788 : tensor<1x32x41x64xf32>, tensor<1x32x41x1xf32>) outs(%789 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb86(%791: f32, %792: f32, %793: f32):
      %794 = arith.divf %791, %792 : f32
      linalg.yield %794 : f32
    } -> tensor<1x32x41x64xf32>
    %795 = tensor.empty() : tensor<1x32x41x64xf32>
    %796 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%790 : tensor<1x32x41x64xf32>) outs(%795 : tensor<1x32x41x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb87(%797: f32, %798: f32):
      linalg.yield %797 : f32
    } -> tensor<1x32x41x64xf32>
    %799 = tensor.collapse_shape %796 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x41x64xf32> into tensor<83968xf32>
    %800 = tensor.expand_shape %799 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 41, 64] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<83968xf32> into tensor<32x41x64xf32>
    %801 = tensor.empty() : tensor<1x32x64x48xf32>
    %802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%714 : tensor<1x32x64x48xf32>) outs(%801 : tensor<1x32x64x48xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb88(%803: f32, %804: f32):
      linalg.yield %803 : f32
    } -> tensor<1x32x64x48xf32>
    %805 = tensor.collapse_shape %802 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x32x64x48xf32> into tensor<98304xf32>
    %806 = tensor.expand_shape %805 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 48] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<98304xf32> into tensor<32x64x48xf32>
    %807 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 0.000000e+00 : f32
    %808 = tensor.splat %807 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<32x41x48xf32>
    %809 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%800, %806 : tensor<32x41x64xf32>, tensor<32x64x48xf32>) outs(%808 : tensor<32x41x48xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb89(%810: f32, %811: f32, %812: f32):
      %813 = arith.mulf %810, %811 : f32
      %814 = arith.addf %812, %813 : f32
      linalg.yield %814 : f32
    } -> tensor<32x41x48xf32>
    %815 = tensor.collapse_shape %809 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<32x41x48xf32> into tensor<62976xf32>
    %816 = tensor.expand_shape %815 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 48] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<62976xf32> into tensor<1x32x41x48xf32>
    %817 = tensor.empty() : tensor<1x41x32x48xf32>
    %818 = linalg.transpose ins(%816:tensor<1x32x41x48xf32>) outs(%817:tensor<1x41x32x48xf32>) permutation = [0, 2, 1, 3]
    %819 = tensor.collapse_shape %818 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x41x32x48xf32> into tensor<62976xf32>
    %820 = tensor.expand_shape %819 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %821 = tensor.collapse_shape %820 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %822 = tensor.expand_shape %821 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0"} : tensor<62976xf32> into tensor<41x1536xf32>
    %823 = tensor.empty() : tensor<1536x1536xf32>
    %824 = linalg.transpose ins(%29:tensor<1536x1536xf32>) outs(%823:tensor<1536x1536xf32>) permutation = [1, 0]
    %825 = tensor.empty() : tensor<41x1536xf32>
    %826 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %827 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%826 : f32) outs(%825 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %828 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0", prov.transposed_b = "true"} ins(%822, %824 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%827 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %829 = tensor.empty() : tensor<41x1536xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%828, %30 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%829 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0"} {
    ^bb90(%831: f32, %832: f32, %833: f32):
      %834 = arith.addf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<41x1536xf32>
    %835 = tensor.collapse_shape %830 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0"} : tensor<41x1536xf32> into tensor<62976xf32>
    %836 = tensor.expand_shape %835 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1.to_out.0"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %837 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} 1.000000e+00 : f32
    %838 = tensor.splat %837 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} : tensor<1x41x1536xf32>
    %839 = tensor.empty() : tensor<1x41x1536xf32>
    %840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%836, %838 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%839 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.attn1"} {
    ^bb91(%841: f32, %842: f32, %843: f32):
      %844 = arith.divf %841, %842 : f32
      linalg.yield %844 : f32
    } -> tensor<1x41x1536xf32>
    %845 = tensor.empty() : tensor<1x41x1536xf32>
    %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%840, %405 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%845 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0"} {
    ^bb92(%847: f32, %848: f32, %849: f32):
      %850 = arith.addf %847, %848 : f32
      linalg.yield %850 : f32
    } -> tensor<1x41x1536xf32>
    %851 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} 0.000000e+00 : f32
    %852 = tensor.splat %851 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32>
    %853 = linalg.reduce ins(%846:tensor<1x41x1536xf32>) outs(%852:tensor<1x41xf32>) dimensions = [2]
    (%854: f32, %855: f32) {
      %856 = arith.addf %854, %855 : f32
      linalg.yield %856 : f32
    }
    %857 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} 1.536000e+03 : f32
    %858 = tensor.splat %857 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32>
    %859 = tensor.empty() : tensor<1x41xf32>
    %860 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%853, %858 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%859 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb93(%861: f32, %862: f32, %863: f32):
      %864 = arith.divf %861, %862 : f32
      linalg.yield %864 : f32
    } -> tensor<1x41xf32>
    %865 = tensor.collapse_shape %860 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32> into tensor<41xf32>
    %866 = tensor.expand_shape %865 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<41xf32> into tensor<1x41x1xf32>
    %867 = tensor.empty() : tensor<1x41x1536xf32>
    %868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%846, %866 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%867 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb94(%869: f32, %870: f32, %871: f32):
      %872 = arith.subf %869, %870 : f32
      linalg.yield %872 : f32
    } -> tensor<1x41x1536xf32>
    %873 = tensor.empty() : tensor<1x41x1536xf32>
    %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%868, %868 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%873 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb95(%875: f32, %876: f32, %877: f32):
      %878 = arith.mulf %875, %876 : f32
      linalg.yield %878 : f32
    } -> tensor<1x41x1536xf32>
    %879 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} 0.000000e+00 : f32
    %880 = tensor.splat %879 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32>
    %881 = linalg.reduce ins(%874:tensor<1x41x1536xf32>) outs(%880:tensor<1x41xf32>) dimensions = [2]
    (%882: f32, %883: f32) {
      %884 = arith.addf %882, %883 : f32
      linalg.yield %884 : f32
    }
    %885 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} 1.536000e+03 : f32
    %886 = tensor.splat %885 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32>
    %887 = tensor.empty() : tensor<1x41xf32>
    %888 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%881, %886 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%887 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb96(%889: f32, %890: f32, %891: f32):
      %892 = arith.divf %889, %890 : f32
      linalg.yield %892 : f32
    } -> tensor<1x41xf32>
    %893 = tensor.collapse_shape %888 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41xf32> into tensor<41xf32>
    %894 = tensor.expand_shape %893 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<41xf32> into tensor<1x41x1xf32>
    %895 = arith.constant {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} 1.000000e-05 : f32
    %896 = tensor.splat %895 {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} : tensor<1x41x1xf32>
    %897 = tensor.empty() : tensor<1x41x1xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894, %896 : tensor<1x41x1xf32>, tensor<1x41x1xf32>) outs(%897 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb97(%899: f32, %900: f32, %901: f32):
      %902 = arith.addf %899, %900 : f32
      linalg.yield %902 : f32
    } -> tensor<1x41x1xf32>
    %903 = tensor.empty() : tensor<1x41x1xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%898 : tensor<1x41x1xf32>) outs(%903 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb98(%905: f32, %906: f32):
      %907 = math.rsqrt %905 : f32
      linalg.yield %907 : f32
    } -> tensor<1x41x1xf32>
    %908 = tensor.empty() : tensor<1x41x1536xf32>
    %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%868, %904 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%908 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_2", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.norm3"} {
    ^bb99(%910: f32, %911: f32, %912: f32):
      %913 = arith.mulf %910, %911 : f32
      linalg.yield %913 : f32
    } -> tensor<1x41x1536xf32>
    %914 = tensor.collapse_shape %909 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %915 = tensor.expand_shape %914 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj"} : tensor<62976xf32> into tensor<41x1536xf32>
    %916 = tensor.empty() : tensor<1536x6144xf32>
    %917 = linalg.transpose ins(%31:tensor<6144x1536xf32>) outs(%916:tensor<1536x6144xf32>) permutation = [1, 0]
    %918 = tensor.empty() : tensor<41x6144xf32>
    %919 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %920 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%919 : f32) outs(%918 : tensor<41x6144xf32>) -> tensor<41x6144xf32>
    %921 = linalg.matmul {prov.region_id = "matmul_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj", prov.transposed_b = "true"} ins(%915, %917 : tensor<41x1536xf32>, tensor<1536x6144xf32>) outs(%920 : tensor<41x6144xf32>) -> tensor<41x6144xf32>
    %922 = tensor.empty() : tensor<41x6144xf32>
    %923 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%921, %32 : tensor<41x6144xf32>, tensor<6144xf32>) outs(%922 : tensor<41x6144xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj"} {
    ^bb100(%924: f32, %925: f32, %926: f32):
      %927 = arith.addf %924, %925 : f32
      linalg.yield %927 : f32
    } -> tensor<41x6144xf32>
    %928 = tensor.collapse_shape %923 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj"} : tensor<41x6144xf32> into tensor<251904xf32>
    %929 = tensor.expand_shape %928 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 6144] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0.proj"} : tensor<251904xf32> into tensor<1x41x6144xf32>
    %930 = tensor.empty() : tensor<1x41x6144xf32>
    %931 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%929 : tensor<1x41x6144xf32>) outs(%930 : tensor<1x41x6144xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.0"} {
    ^bb101(%932: f32, %933: f32):
      %934 = arith.constant 5.000000e-01 : f32
      %935 = arith.constant 1.000000e+00 : f32
      %936 = arith.constant 0.707106769 : f32
      %937 = arith.mulf %932, %936 : f32
      %938 = math.erf %937 : f32
      %939 = arith.addf %935, %938 : f32
      %940 = arith.mulf %934, %932 : f32
      %941 = arith.mulf %940, %939 : f32
      linalg.yield %941 : f32
    } -> tensor<1x41x6144xf32>
    %942 = tensor.collapse_shape %931 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2"} : tensor<1x41x6144xf32> into tensor<251904xf32>
    %943 = tensor.expand_shape %942 [[0 : i64, 1 : i64]] output_shape [41, 6144] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2"} : tensor<251904xf32> into tensor<41x6144xf32>
    %944 = tensor.empty() : tensor<6144x1536xf32>
    %945 = linalg.transpose ins(%33:tensor<1536x6144xf32>) outs(%944:tensor<6144x1536xf32>) permutation = [1, 0]
    %946 = tensor.empty() : tensor<41x1536xf32>
    %947 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %948 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%947 : f32) outs(%946 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %949 = linalg.matmul {prov.region_id = "matmul_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2", prov.transposed_b = "true"} ins(%943, %945 : tensor<41x6144xf32>, tensor<6144x1536xf32>) outs(%948 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %950 = tensor.empty() : tensor<41x1536xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%949, %34 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%950 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2"} {
    ^bb102(%952: f32, %953: f32, %954: f32):
      %955 = arith.addf %952, %953 : f32
      linalg.yield %955 : f32
    } -> tensor<41x1536xf32>
    %956 = tensor.collapse_shape %951 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2"} : tensor<41x1536xf32> into tensor<62976xf32>
    %957 = tensor.expand_shape %956 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0.ff.net.2"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %958 = tensor.empty() : tensor<1x41x1536xf32>
    %959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957, %846 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%958 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.0"} {
    ^bb103(%960: f32, %961: f32, %962: f32):
      %963 = arith.addf %960, %961 : f32
      linalg.yield %963 : f32
    } -> tensor<1x41x1536xf32>
    %964 = tensor.empty() : tensor<1x1536xf32>
    %965 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510 : tensor<1x1536xf32>) outs(%964 : tensor<1x1536xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.silu"} {
    ^bb104(%966: f32, %967: f32):
      %968 = arith.constant 1.000000e+00 : f32
      %969 = arith.negf %966 : f32
      %970 = math.exp %969 : f32
      %971 = arith.addf %968, %970 : f32
      %972 = arith.divf %968, %971 : f32
      linalg.yield %972 : f32
    } -> tensor<1x1536xf32>
    %973 = tensor.empty() : tensor<1x1536xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510, %965 : tensor<1x1536xf32>, tensor<1x1536xf32>) outs(%973 : tensor<1x1536xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.silu"} {
    ^bb105(%975: f32, %976: f32, %977: f32):
      %978 = arith.mulf %975, %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x1536xf32>
    %979 = tensor.empty() : tensor<1536x3072xf32>
    %980 = linalg.transpose ins(%35:tensor<3072x1536xf32>) outs(%979:tensor<1536x3072xf32>) permutation = [1, 0]
    %981 = tensor.empty() : tensor<1x3072xf32>
    %982 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %983 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%982 : f32) outs(%981 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %984 = linalg.matmul {prov.region_id = "matmul_16", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.linear", prov.transposed_b = "true"} ins(%974, %980 : tensor<1x1536xf32>, tensor<1536x3072xf32>) outs(%983 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %985 = tensor.empty() : tensor<1x3072xf32>
    %986 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%984, %36 : tensor<1x3072xf32>, tensor<3072xf32>) outs(%985 : tensor<1x3072xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.linear"} {
    ^bb106(%987: f32, %988: f32, %989: f32):
      %990 = arith.addf %987, %988 : f32
      linalg.yield %990 : f32
    } -> tensor<1x3072xf32>
    %991 = "tensor.extract_slice"(%986) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %992 = "tensor.extract_slice"(%986) <{static_offsets = array<i64: 0, 1536>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %993 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} 0.000000e+00 : f32
    %994 = tensor.splat %993 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32>
    %995 = linalg.reduce ins(%959:tensor<1x41x1536xf32>) outs(%994:tensor<1x41xf32>) dimensions = [2]
    (%996: f32, %997: f32) {
      %998 = arith.addf %996, %997 : f32
      linalg.yield %998 : f32
    }
    %999 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} 1.536000e+03 : f32
    %1000 = tensor.splat %999 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32>
    %1001 = tensor.empty() : tensor<1x41xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%995, %1000 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1001 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb107(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.divf %1003, %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x41xf32>
    %1007 = tensor.collapse_shape %1002 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32> into tensor<41xf32>
    %1008 = tensor.expand_shape %1007 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1009 = tensor.empty() : tensor<1x41x1536xf32>
    %1010 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%959, %1008 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1009 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb108(%1011: f32, %1012: f32, %1013: f32):
      %1014 = arith.subf %1011, %1012 : f32
      linalg.yield %1014 : f32
    } -> tensor<1x41x1536xf32>
    %1015 = tensor.empty() : tensor<1x41x1536xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1010, %1010 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1015 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb109(%1017: f32, %1018: f32, %1019: f32):
      %1020 = arith.mulf %1017, %1018 : f32
      linalg.yield %1020 : f32
    } -> tensor<1x41x1536xf32>
    %1021 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} 0.000000e+00 : f32
    %1022 = tensor.splat %1021 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32>
    %1023 = linalg.reduce ins(%1016:tensor<1x41x1536xf32>) outs(%1022:tensor<1x41xf32>) dimensions = [2]
    (%1024: f32, %1025: f32) {
      %1026 = arith.addf %1024, %1025 : f32
      linalg.yield %1026 : f32
    }
    %1027 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} 1.536000e+03 : f32
    %1028 = tensor.splat %1027 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32>
    %1029 = tensor.empty() : tensor<1x41xf32>
    %1030 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1023, %1028 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1029 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb110(%1031: f32, %1032: f32, %1033: f32):
      %1034 = arith.divf %1031, %1032 : f32
      linalg.yield %1034 : f32
    } -> tensor<1x41xf32>
    %1035 = tensor.collapse_shape %1030 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41xf32> into tensor<41xf32>
    %1036 = tensor.expand_shape %1035 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1037 = arith.constant {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} 1.000000e-05 : f32
    %1038 = tensor.splat %1037 {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} : tensor<1x41x1xf32>
    %1039 = tensor.empty() : tensor<1x41x1xf32>
    %1040 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1036, %1038 : tensor<1x41x1xf32>, tensor<1x41x1xf32>) outs(%1039 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb111(%1041: f32, %1042: f32, %1043: f32):
      %1044 = arith.addf %1041, %1042 : f32
      linalg.yield %1044 : f32
    } -> tensor<1x41x1xf32>
    %1045 = tensor.empty() : tensor<1x41x1xf32>
    %1046 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1040 : tensor<1x41x1xf32>) outs(%1045 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb112(%1047: f32, %1048: f32):
      %1049 = math.rsqrt %1047 : f32
      linalg.yield %1049 : f32
    } -> tensor<1x41x1xf32>
    %1050 = tensor.empty() : tensor<1x41x1536xf32>
    %1051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1010, %1046 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1050 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_3", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1.norm"} {
    ^bb113(%1052: f32, %1053: f32, %1054: f32):
      %1055 = arith.mulf %1052, %1053 : f32
      linalg.yield %1055 : f32
    } -> tensor<1x41x1536xf32>
    %1056 = "tensor.extract_slice"(%991) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %1057 = tensor.collapse_shape %1056 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : tensor<1x1536xf32> into tensor<1536xf32>
    %1058 = tensor.expand_shape %1057 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %1059 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} 1.000000e+00 : f32
    %1060 = tensor.splat %1059 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : tensor<1x1x1536xf32>
    %1061 = tensor.empty() : tensor<1x1x1536xf32>
    %1062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1058, %1060 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%1061 : tensor<1x1x1536xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} {
    ^bb114(%1063: f32, %1064: f32, %1065: f32):
      %1066 = arith.addf %1063, %1064 : f32
      linalg.yield %1066 : f32
    } -> tensor<1x1x1536xf32>
    %1067 = tensor.empty() : tensor<1x41x1536xf32>
    %1068 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1051, %1062 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%1067 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} {
    ^bb115(%1069: f32, %1070: f32, %1071: f32):
      %1072 = arith.mulf %1069, %1070 : f32
      linalg.yield %1072 : f32
    } -> tensor<1x41x1536xf32>
    %1073 = "tensor.extract_slice"(%992) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %1074 = tensor.collapse_shape %1073 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : tensor<1x1536xf32> into tensor<1536xf32>
    %1075 = tensor.expand_shape %1074 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %1076 = tensor.empty() : tensor<1x41x1536xf32>
    %1077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1068, %1075 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%1076 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm1"} {
    ^bb116(%1078: f32, %1079: f32, %1080: f32):
      %1081 = arith.addf %1078, %1079 : f32
      linalg.yield %1081 : f32
    } -> tensor<1x41x1536xf32>
    %1082 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1083 = tensor.expand_shape %1082 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1084 = tensor.empty() : tensor<1536x1536xf32>
    %1085 = linalg.transpose ins(%37:tensor<1536x1536xf32>) outs(%1084:tensor<1536x1536xf32>) permutation = [1, 0]
    %1086 = tensor.empty() : tensor<41x1536xf32>
    %1087 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1088 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1087 : f32) outs(%1086 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1089 = linalg.matmul {prov.region_id = "matmul_17", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q", prov.transposed_b = "true"} ins(%1083, %1085 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%1088 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1090 = tensor.empty() : tensor<41x1536xf32>
    %1091 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1089, %38 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%1090 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q"} {
    ^bb117(%1092: f32, %1093: f32, %1094: f32):
      %1095 = arith.addf %1092, %1093 : f32
      linalg.yield %1095 : f32
    } -> tensor<41x1536xf32>
    %1096 = tensor.collapse_shape %1091 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q"} : tensor<41x1536xf32> into tensor<62976xf32>
    %1097 = tensor.expand_shape %1096 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_q"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1098 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1099 = tensor.expand_shape %1098 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1100 = tensor.empty() : tensor<1536x1536xf32>
    %1101 = linalg.transpose ins(%39:tensor<1536x1536xf32>) outs(%1100:tensor<1536x1536xf32>) permutation = [1, 0]
    %1102 = tensor.empty() : tensor<41x1536xf32>
    %1103 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1104 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1103 : f32) outs(%1102 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1105 = linalg.matmul {prov.region_id = "matmul_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k", prov.transposed_b = "true"} ins(%1099, %1101 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%1104 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1106 = tensor.empty() : tensor<41x1536xf32>
    %1107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1105, %40 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%1106 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k"} {
    ^bb118(%1108: f32, %1109: f32, %1110: f32):
      %1111 = arith.addf %1108, %1109 : f32
      linalg.yield %1111 : f32
    } -> tensor<41x1536xf32>
    %1112 = tensor.collapse_shape %1107 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k"} : tensor<41x1536xf32> into tensor<62976xf32>
    %1113 = tensor.expand_shape %1112 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_k"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1114 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1115 = tensor.expand_shape %1114 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1116 = tensor.empty() : tensor<1536x1536xf32>
    %1117 = linalg.transpose ins(%41:tensor<1536x1536xf32>) outs(%1116:tensor<1536x1536xf32>) permutation = [1, 0]
    %1118 = tensor.empty() : tensor<41x1536xf32>
    %1119 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1120 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1119 : f32) outs(%1118 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1121 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v", prov.transposed_b = "true"} ins(%1115, %1117 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%1120 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1122 = tensor.empty() : tensor<41x1536xf32>
    %1123 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1121, %42 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%1122 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v"} {
    ^bb119(%1124: f32, %1125: f32, %1126: f32):
      %1127 = arith.addf %1124, %1125 : f32
      linalg.yield %1127 : f32
    } -> tensor<41x1536xf32>
    %1128 = tensor.collapse_shape %1123 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v"} : tensor<41x1536xf32> into tensor<62976xf32>
    %1129 = tensor.expand_shape %1128 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_v"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1130 = tensor.collapse_shape %1097 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1131 = tensor.expand_shape %1130 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 41, 32, 48] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<1x41x32x48xf32>
    %1132 = tensor.empty() : tensor<1x32x41x48xf32>
    %1133 = linalg.transpose ins(%1131:tensor<1x41x32x48xf32>) outs(%1132:tensor<1x32x41x48xf32>) permutation = [0, 2, 1, 3]
    %1134 = tensor.collapse_shape %1113 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1135 = tensor.expand_shape %1134 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 41, 32, 48] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<1x41x32x48xf32>
    %1136 = tensor.empty() : tensor<1x32x41x48xf32>
    %1137 = linalg.transpose ins(%1135:tensor<1x41x32x48xf32>) outs(%1136:tensor<1x32x41x48xf32>) permutation = [0, 2, 1, 3]
    %1138 = tensor.collapse_shape %1129 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1139 = tensor.expand_shape %1138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 41, 32, 48] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<1x41x32x48xf32>
    %1140 = tensor.empty() : tensor<1x32x41x48xf32>
    %1141 = linalg.transpose ins(%1139:tensor<1x41x32x48xf32>) outs(%1140:tensor<1x32x41x48xf32>) permutation = [0, 2, 1, 3]
    %1142 = tensor.empty() : tensor<1x32x48x41xf32>
    %1143 = linalg.transpose ins(%1137:tensor<1x32x41x48xf32>) outs(%1142:tensor<1x32x48x41xf32>) permutation = [0, 1, 3, 2]
    %1144 = tensor.empty() : tensor<1x32x41x48xf32>
    %1145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1133 : tensor<1x32x41x48xf32>) outs(%1144 : tensor<1x32x41x48xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb120(%1146: f32, %1147: f32):
      linalg.yield %1146 : f32
    } -> tensor<1x32x41x48xf32>
    %1148 = tensor.collapse_shape %1145 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41x48xf32> into tensor<62976xf32>
    %1149 = tensor.expand_shape %1148 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 41, 48] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<32x41x48xf32>
    %1150 = tensor.empty() : tensor<1x32x48x41xf32>
    %1151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1143 : tensor<1x32x48x41xf32>) outs(%1150 : tensor<1x32x48x41xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb121(%1152: f32, %1153: f32):
      linalg.yield %1152 : f32
    } -> tensor<1x32x48x41xf32>
    %1154 = tensor.collapse_shape %1151 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x48x41xf32> into tensor<62976xf32>
    %1155 = tensor.expand_shape %1154 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 48, 41] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<32x48x41xf32>
    %1156 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 0.000000e+00 : f32
    %1157 = tensor.splat %1156 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<32x41x41xf32>
    %1158 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1149, %1155 : tensor<32x41x48xf32>, tensor<32x48x41xf32>) outs(%1157 : tensor<32x41x41xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb122(%1159: f32, %1160: f32, %1161: f32):
      %1162 = arith.mulf %1159, %1160 : f32
      %1163 = arith.addf %1161, %1162 : f32
      linalg.yield %1163 : f32
    } -> tensor<32x41x41xf32>
    %1164 = tensor.collapse_shape %1158 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<32x41x41xf32> into tensor<53792xf32>
    %1165 = tensor.expand_shape %1164 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 41] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<53792xf32> into tensor<1x32x41x41xf32>
    %1166 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 0.144337565 : f32
    %1167 = tensor.splat %1166 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41x41xf32>
    %1168 = tensor.empty() : tensor<1x32x41x41xf32>
    %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1165, %1167 : tensor<1x32x41x41xf32>, tensor<1x32x41x41xf32>) outs(%1168 : tensor<1x32x41x41xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb123(%1170: f32, %1171: f32, %1172: f32):
      %1173 = arith.mulf %1170, %1171 : f32
      linalg.yield %1173 : f32
    } -> tensor<1x32x41x41xf32>
    %1174 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 0xff800000 : f32
    %1175 = tensor.splat %1174 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41xf32>
    %1176 = linalg.reduce ins(%1169:tensor<1x32x41x41xf32>) outs(%1175:tensor<1x32x41xf32>) dimensions = [3]
    (%1177: f32, %1178: f32) {
      %1179 = arith.maximumf %1177, %1178 : f32
      linalg.yield %1179 : f32
    }
    %1180 = tensor.collapse_shape %1176 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41xf32> into tensor<1312xf32>
    %1181 = tensor.expand_shape %1180 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1312xf32> into tensor<1x32x41x1xf32>
    %1182 = tensor.empty() : tensor<1x32x41x41xf32>
    %1183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1169, %1181 : tensor<1x32x41x41xf32>, tensor<1x32x41x1xf32>) outs(%1182 : tensor<1x32x41x41xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb124(%1184: f32, %1185: f32, %1186: f32):
      %1187 = arith.subf %1184, %1185 : f32
      linalg.yield %1187 : f32
    } -> tensor<1x32x41x41xf32>
    %1188 = tensor.empty() : tensor<1x32x41x41xf32>
    %1189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1183 : tensor<1x32x41x41xf32>) outs(%1188 : tensor<1x32x41x41xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb125(%1190: f32, %1191: f32):
      %1192 = math.exp %1190 : f32
      linalg.yield %1192 : f32
    } -> tensor<1x32x41x41xf32>
    %1193 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 0.000000e+00 : f32
    %1194 = tensor.splat %1193 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41xf32>
    %1195 = linalg.reduce ins(%1189:tensor<1x32x41x41xf32>) outs(%1194:tensor<1x32x41xf32>) dimensions = [3]
    (%1196: f32, %1197: f32) {
      %1198 = arith.addf %1196, %1197 : f32
      linalg.yield %1198 : f32
    }
    %1199 = tensor.collapse_shape %1195 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41xf32> into tensor<1312xf32>
    %1200 = tensor.expand_shape %1199 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1312xf32> into tensor<1x32x41x1xf32>
    %1201 = tensor.empty() : tensor<1x32x41x41xf32>
    %1202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1189, %1200 : tensor<1x32x41x41xf32>, tensor<1x32x41x1xf32>) outs(%1201 : tensor<1x32x41x41xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb126(%1203: f32, %1204: f32, %1205: f32):
      %1206 = arith.divf %1203, %1204 : f32
      linalg.yield %1206 : f32
    } -> tensor<1x32x41x41xf32>
    %1207 = tensor.empty() : tensor<1x32x41x41xf32>
    %1208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1202 : tensor<1x32x41x41xf32>) outs(%1207 : tensor<1x32x41x41xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb127(%1209: f32, %1210: f32):
      linalg.yield %1209 : f32
    } -> tensor<1x32x41x41xf32>
    %1211 = tensor.collapse_shape %1208 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41x41xf32> into tensor<53792xf32>
    %1212 = tensor.expand_shape %1211 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 41, 41] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<53792xf32> into tensor<32x41x41xf32>
    %1213 = tensor.empty() : tensor<1x32x41x48xf32>
    %1214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1141 : tensor<1x32x41x48xf32>) outs(%1213 : tensor<1x32x41x48xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb128(%1215: f32, %1216: f32):
      linalg.yield %1215 : f32
    } -> tensor<1x32x41x48xf32>
    %1217 = tensor.collapse_shape %1214 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x32x41x48xf32> into tensor<62976xf32>
    %1218 = tensor.expand_shape %1217 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 41, 48] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<32x41x48xf32>
    %1219 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 0.000000e+00 : f32
    %1220 = tensor.splat %1219 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<32x41x48xf32>
    %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1212, %1218 : tensor<32x41x41xf32>, tensor<32x41x48xf32>) outs(%1220 : tensor<32x41x48xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb129(%1222: f32, %1223: f32, %1224: f32):
      %1225 = arith.mulf %1222, %1223 : f32
      %1226 = arith.addf %1224, %1225 : f32
      linalg.yield %1226 : f32
    } -> tensor<32x41x48xf32>
    %1227 = tensor.collapse_shape %1221 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<32x41x48xf32> into tensor<62976xf32>
    %1228 = tensor.expand_shape %1227 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 41, 48] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<1x32x41x48xf32>
    %1229 = tensor.empty() : tensor<1x41x32x48xf32>
    %1230 = linalg.transpose ins(%1228:tensor<1x32x41x48xf32>) outs(%1229:tensor<1x41x32x48xf32>) permutation = [0, 2, 1, 3]
    %1231 = tensor.collapse_shape %1230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x41x32x48xf32> into tensor<62976xf32>
    %1232 = tensor.expand_shape %1231 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1233 = tensor.collapse_shape %1232 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1235 = tensor.empty() : tensor<1536x1536xf32>
    %1236 = linalg.transpose ins(%43:tensor<1536x1536xf32>) outs(%1235:tensor<1536x1536xf32>) permutation = [1, 0]
    %1237 = tensor.empty() : tensor<41x1536xf32>
    %1238 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1239 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1238 : f32) outs(%1237 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1240 = linalg.matmul {prov.region_id = "matmul_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0", prov.transposed_b = "true"} ins(%1234, %1236 : tensor<41x1536xf32>, tensor<1536x1536xf32>) outs(%1239 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1241 = tensor.empty() : tensor<41x1536xf32>
    %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1240, %44 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%1241 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0"} {
    ^bb130(%1243: f32, %1244: f32, %1245: f32):
      %1246 = arith.addf %1243, %1244 : f32
      linalg.yield %1246 : f32
    } -> tensor<41x1536xf32>
    %1247 = tensor.collapse_shape %1242 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0"} : tensor<41x1536xf32> into tensor<62976xf32>
    %1248 = tensor.expand_shape %1247 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1.to_out.0"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1249 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} 1.000000e+00 : f32
    %1250 = tensor.splat %1249 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} : tensor<1x41x1536xf32>
    %1251 = tensor.empty() : tensor<1x41x1536xf32>
    %1252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1248, %1250 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1251 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.attn1"} {
    ^bb131(%1253: f32, %1254: f32, %1255: f32):
      %1256 = arith.divf %1253, %1254 : f32
      linalg.yield %1256 : f32
    } -> tensor<1x41x1536xf32>
    %1257 = tensor.empty() : tensor<1x41x1536xf32>
    %1258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1252, %959 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1257 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1"} {
    ^bb132(%1259: f32, %1260: f32, %1261: f32):
      %1262 = arith.addf %1259, %1260 : f32
      linalg.yield %1262 : f32
    } -> tensor<1x41x1536xf32>
    %1263 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} 0.000000e+00 : f32
    %1264 = tensor.splat %1263 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32>
    %1265 = linalg.reduce ins(%1258:tensor<1x41x1536xf32>) outs(%1264:tensor<1x41xf32>) dimensions = [2]
    (%1266: f32, %1267: f32) {
      %1268 = arith.addf %1266, %1267 : f32
      linalg.yield %1268 : f32
    }
    %1269 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} 1.536000e+03 : f32
    %1270 = tensor.splat %1269 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32>
    %1271 = tensor.empty() : tensor<1x41xf32>
    %1272 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1265, %1270 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1271 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb133(%1273: f32, %1274: f32, %1275: f32):
      %1276 = arith.divf %1273, %1274 : f32
      linalg.yield %1276 : f32
    } -> tensor<1x41xf32>
    %1277 = tensor.collapse_shape %1272 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32> into tensor<41xf32>
    %1278 = tensor.expand_shape %1277 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1279 = tensor.empty() : tensor<1x41x1536xf32>
    %1280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1258, %1278 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1279 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb134(%1281: f32, %1282: f32, %1283: f32):
      %1284 = arith.subf %1281, %1282 : f32
      linalg.yield %1284 : f32
    } -> tensor<1x41x1536xf32>
    %1285 = tensor.empty() : tensor<1x41x1536xf32>
    %1286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1280, %1280 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1285 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb135(%1287: f32, %1288: f32, %1289: f32):
      %1290 = arith.mulf %1287, %1288 : f32
      linalg.yield %1290 : f32
    } -> tensor<1x41x1536xf32>
    %1291 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} 0.000000e+00 : f32
    %1292 = tensor.splat %1291 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32>
    %1293 = linalg.reduce ins(%1286:tensor<1x41x1536xf32>) outs(%1292:tensor<1x41xf32>) dimensions = [2]
    (%1294: f32, %1295: f32) {
      %1296 = arith.addf %1294, %1295 : f32
      linalg.yield %1296 : f32
    }
    %1297 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} 1.536000e+03 : f32
    %1298 = tensor.splat %1297 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32>
    %1299 = tensor.empty() : tensor<1x41xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1293, %1298 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1299 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb136(%1301: f32, %1302: f32, %1303: f32):
      %1304 = arith.divf %1301, %1302 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x41xf32>
    %1305 = tensor.collapse_shape %1300 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41xf32> into tensor<41xf32>
    %1306 = tensor.expand_shape %1305 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1307 = arith.constant {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} 1.000000e-05 : f32
    %1308 = tensor.splat %1307 {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} : tensor<1x41x1xf32>
    %1309 = tensor.empty() : tensor<1x41x1xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1308 : tensor<1x41x1xf32>, tensor<1x41x1xf32>) outs(%1309 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb137(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.addf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x41x1xf32>
    %1315 = tensor.empty() : tensor<1x41x1xf32>
    %1316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1310 : tensor<1x41x1xf32>) outs(%1315 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb138(%1317: f32, %1318: f32):
      %1319 = math.rsqrt %1317 : f32
      linalg.yield %1319 : f32
    } -> tensor<1x41x1xf32>
    %1320 = tensor.empty() : tensor<1x41x1536xf32>
    %1321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1280, %1316 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1320 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_4", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.norm3"} {
    ^bb139(%1322: f32, %1323: f32, %1324: f32):
      %1325 = arith.mulf %1322, %1323 : f32
      linalg.yield %1325 : f32
    } -> tensor<1x41x1536xf32>
    %1326 = tensor.collapse_shape %1321 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1327 = tensor.expand_shape %1326 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1328 = tensor.empty() : tensor<1536x6144xf32>
    %1329 = linalg.transpose ins(%45:tensor<6144x1536xf32>) outs(%1328:tensor<1536x6144xf32>) permutation = [1, 0]
    %1330 = tensor.empty() : tensor<41x6144xf32>
    %1331 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1332 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1331 : f32) outs(%1330 : tensor<41x6144xf32>) -> tensor<41x6144xf32>
    %1333 = linalg.matmul {prov.region_id = "matmul_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj", prov.transposed_b = "true"} ins(%1327, %1329 : tensor<41x1536xf32>, tensor<1536x6144xf32>) outs(%1332 : tensor<41x6144xf32>) -> tensor<41x6144xf32>
    %1334 = tensor.empty() : tensor<41x6144xf32>
    %1335 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1333, %46 : tensor<41x6144xf32>, tensor<6144xf32>) outs(%1334 : tensor<41x6144xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj"} {
    ^bb140(%1336: f32, %1337: f32, %1338: f32):
      %1339 = arith.addf %1336, %1337 : f32
      linalg.yield %1339 : f32
    } -> tensor<41x6144xf32>
    %1340 = tensor.collapse_shape %1335 [[0 : i64, 1 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj"} : tensor<41x6144xf32> into tensor<251904xf32>
    %1341 = tensor.expand_shape %1340 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 6144] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0.proj"} : tensor<251904xf32> into tensor<1x41x6144xf32>
    %1342 = tensor.empty() : tensor<1x41x6144xf32>
    %1343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1341 : tensor<1x41x6144xf32>) outs(%1342 : tensor<1x41x6144xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.0"} {
    ^bb141(%1344: f32, %1345: f32):
      %1346 = arith.constant 5.000000e-01 : f32
      %1347 = arith.constant 1.000000e+00 : f32
      %1348 = arith.constant 0.707106769 : f32
      %1349 = arith.mulf %1344, %1348 : f32
      %1350 = math.erf %1349 : f32
      %1351 = arith.addf %1347, %1350 : f32
      %1352 = arith.mulf %1346, %1344 : f32
      %1353 = arith.mulf %1352, %1351 : f32
      linalg.yield %1353 : f32
    } -> tensor<1x41x6144xf32>
    %1354 = tensor.collapse_shape %1343 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2"} : tensor<1x41x6144xf32> into tensor<251904xf32>
    %1355 = tensor.expand_shape %1354 [[0 : i64, 1 : i64]] output_shape [41, 6144] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2"} : tensor<251904xf32> into tensor<41x6144xf32>
    %1356 = tensor.empty() : tensor<6144x1536xf32>
    %1357 = linalg.transpose ins(%47:tensor<1536x6144xf32>) outs(%1356:tensor<6144x1536xf32>) permutation = [1, 0]
    %1358 = tensor.empty() : tensor<41x1536xf32>
    %1359 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1360 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1359 : f32) outs(%1358 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1361 = linalg.matmul {prov.region_id = "matmul_24", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2", prov.transposed_b = "true"} ins(%1355, %1357 : tensor<41x6144xf32>, tensor<6144x1536xf32>) outs(%1360 : tensor<41x1536xf32>) -> tensor<41x1536xf32>
    %1362 = tensor.empty() : tensor<41x1536xf32>
    %1363 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1361, %48 : tensor<41x1536xf32>, tensor<1536xf32>) outs(%1362 : tensor<41x1536xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2"} {
    ^bb142(%1364: f32, %1365: f32, %1366: f32):
      %1367 = arith.addf %1364, %1365 : f32
      linalg.yield %1367 : f32
    } -> tensor<41x1536xf32>
    %1368 = tensor.collapse_shape %1363 [[0 : i64, 1 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2"} : tensor<41x1536xf32> into tensor<62976xf32>
    %1369 = tensor.expand_shape %1368 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1536] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1.ff.net.2"} : tensor<62976xf32> into tensor<1x41x1536xf32>
    %1370 = tensor.empty() : tensor<1x41x1536xf32>
    %1371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369, %1258 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1370 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.transformer_blocks.1"} {
    ^bb143(%1372: f32, %1373: f32, %1374: f32):
      %1375 = arith.addf %1372, %1373 : f32
      linalg.yield %1375 : f32
    } -> tensor<1x41x1536xf32>
    %1376 = tensor.empty() : tensor<1x1536xf32>
    %1377 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510 : tensor<1x1536xf32>) outs(%1376 : tensor<1x1536xf32>) attrs =  {prov.region_id = "sigmoid_4", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} {
    ^bb144(%1378: f32, %1379: f32):
      %1380 = arith.constant 1.000000e+00 : f32
      %1381 = arith.negf %1378 : f32
      %1382 = math.exp %1381 : f32
      %1383 = arith.addf %1380, %1382 : f32
      %1384 = arith.divf %1380, %1383 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x1536xf32>
    %1385 = tensor.empty() : tensor<1x1536xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%510, %1377 : tensor<1x1536xf32>, tensor<1x1536xf32>) outs(%1385 : tensor<1x1536xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} {
    ^bb145(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.mulf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x1536xf32>
    %1391 = tensor.empty() : tensor<1536x3072xf32>
    %1392 = linalg.transpose ins(%49:tensor<3072x1536xf32>) outs(%1391:tensor<1536x3072xf32>) permutation = [1, 0]
    %1393 = tensor.empty() : tensor<1x3072xf32>
    %1394 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1395 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1394 : f32) outs(%1393 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %1396 = linalg.matmul {prov.region_id = "matmul_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_1", prov.transposed_b = "true"} ins(%1386, %1392 : tensor<1x1536xf32>, tensor<1536x3072xf32>) outs(%1395 : tensor<1x3072xf32>) -> tensor<1x3072xf32>
    %1397 = tensor.empty() : tensor<1x3072xf32>
    %1398 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1396, %50 : tensor<1x3072xf32>, tensor<3072xf32>) outs(%1397 : tensor<1x3072xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_1"} {
    ^bb146(%1399: f32, %1400: f32, %1401: f32):
      %1402 = arith.addf %1399, %1400 : f32
      linalg.yield %1402 : f32
    } -> tensor<1x3072xf32>
    %1403 = "tensor.extract_slice"(%1398) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %1404 = "tensor.extract_slice"(%1398) <{static_offsets = array<i64: 0, 1536>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : (tensor<1x3072xf32>) -> tensor<1x1536xf32>
    %1405 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} 0.000000e+00 : f32
    %1406 = tensor.splat %1405 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32>
    %1407 = linalg.reduce ins(%1371:tensor<1x41x1536xf32>) outs(%1406:tensor<1x41xf32>) dimensions = [2]
    (%1408: f32, %1409: f32) {
      %1410 = arith.addf %1408, %1409 : f32
      linalg.yield %1410 : f32
    }
    %1411 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} 1.536000e+03 : f32
    %1412 = tensor.splat %1411 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32>
    %1413 = tensor.empty() : tensor<1x41xf32>
    %1414 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1407, %1412 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1413 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb147(%1415: f32, %1416: f32, %1417: f32):
      %1418 = arith.divf %1415, %1416 : f32
      linalg.yield %1418 : f32
    } -> tensor<1x41xf32>
    %1419 = tensor.collapse_shape %1414 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32> into tensor<41xf32>
    %1420 = tensor.expand_shape %1419 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1421 = tensor.empty() : tensor<1x41x1536xf32>
    %1422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1371, %1420 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1421 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb148(%1423: f32, %1424: f32, %1425: f32):
      %1426 = arith.subf %1423, %1424 : f32
      linalg.yield %1426 : f32
    } -> tensor<1x41x1536xf32>
    %1427 = tensor.empty() : tensor<1x41x1536xf32>
    %1428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1422, %1422 : tensor<1x41x1536xf32>, tensor<1x41x1536xf32>) outs(%1427 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb149(%1429: f32, %1430: f32, %1431: f32):
      %1432 = arith.mulf %1429, %1430 : f32
      linalg.yield %1432 : f32
    } -> tensor<1x41x1536xf32>
    %1433 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} 0.000000e+00 : f32
    %1434 = tensor.splat %1433 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32>
    %1435 = linalg.reduce ins(%1428:tensor<1x41x1536xf32>) outs(%1434:tensor<1x41xf32>) dimensions = [2]
    (%1436: f32, %1437: f32) {
      %1438 = arith.addf %1436, %1437 : f32
      linalg.yield %1438 : f32
    }
    %1439 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} 1.536000e+03 : f32
    %1440 = tensor.splat %1439 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32>
    %1441 = tensor.empty() : tensor<1x41xf32>
    %1442 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1435, %1440 : tensor<1x41xf32>, tensor<1x41xf32>) outs(%1441 : tensor<1x41xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb150(%1443: f32, %1444: f32, %1445: f32):
      %1446 = arith.divf %1443, %1444 : f32
      linalg.yield %1446 : f32
    } -> tensor<1x41xf32>
    %1447 = tensor.collapse_shape %1442 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41xf32> into tensor<41xf32>
    %1448 = tensor.expand_shape %1447 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1] {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<41xf32> into tensor<1x41x1xf32>
    %1449 = arith.constant {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} 1.000000e-06 : f32
    %1450 = tensor.splat %1449 {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} : tensor<1x41x1xf32>
    %1451 = tensor.empty() : tensor<1x41x1xf32>
    %1452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1448, %1450 : tensor<1x41x1xf32>, tensor<1x41x1xf32>) outs(%1451 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb151(%1453: f32, %1454: f32, %1455: f32):
      %1456 = arith.addf %1453, %1454 : f32
      linalg.yield %1456 : f32
    } -> tensor<1x41x1xf32>
    %1457 = tensor.empty() : tensor<1x41x1xf32>
    %1458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1452 : tensor<1x41x1xf32>) outs(%1457 : tensor<1x41x1xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb152(%1459: f32, %1460: f32):
      %1461 = math.rsqrt %1459 : f32
      linalg.yield %1461 : f32
    } -> tensor<1x41x1xf32>
    %1462 = tensor.empty() : tensor<1x41x1536xf32>
    %1463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1422, %1458 : tensor<1x41x1536xf32>, tensor<1x41x1xf32>) outs(%1462 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "layer_norm_5", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.norm_out"} {
    ^bb153(%1464: f32, %1465: f32, %1466: f32):
      %1467 = arith.mulf %1464, %1465 : f32
      linalg.yield %1467 : f32
    } -> tensor<1x41x1536xf32>
    %1468 = "tensor.extract_slice"(%1404) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %1469 = tensor.collapse_shape %1468 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : tensor<1x1536xf32> into tensor<1536xf32>
    %1470 = tensor.expand_shape %1469 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %1471 = arith.constant {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} 1.000000e+00 : f32
    %1472 = tensor.splat %1471 {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : tensor<1x1x1536xf32>
    %1473 = tensor.empty() : tensor<1x1x1536xf32>
    %1474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1470, %1472 : tensor<1x1x1536xf32>, tensor<1x1x1536xf32>) outs(%1473 : tensor<1x1x1536xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} {
    ^bb154(%1475: f32, %1476: f32, %1477: f32):
      %1478 = arith.addf %1475, %1476 : f32
      linalg.yield %1478 : f32
    } -> tensor<1x1x1536xf32>
    %1479 = tensor.empty() : tensor<1x41x1536xf32>
    %1480 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1463, %1474 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%1479 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} {
    ^bb155(%1481: f32, %1482: f32, %1483: f32):
      %1484 = arith.mulf %1481, %1482 : f32
      linalg.yield %1484 : f32
    } -> tensor<1x41x1536xf32>
    %1485 = "tensor.extract_slice"(%1403) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1536>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : (tensor<1x1536xf32>) -> tensor<1x1536xf32>
    %1486 = tensor.collapse_shape %1485 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : tensor<1x1536xf32> into tensor<1536xf32>
    %1487 = tensor.expand_shape %1486 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1536] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} : tensor<1536xf32> into tensor<1x1x1536xf32>
    %1488 = tensor.empty() : tensor<1x41x1536xf32>
    %1489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1480, %1487 : tensor<1x41x1536xf32>, tensor<1x1x1536xf32>) outs(%1488 : tensor<1x41x1536xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model"} {
    ^bb156(%1490: f32, %1491: f32, %1492: f32):
      %1493 = arith.addf %1490, %1491 : f32
      linalg.yield %1493 : f32
    } -> tensor<1x41x1536xf32>
    %1494 = tensor.collapse_shape %1489 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2"} : tensor<1x41x1536xf32> into tensor<62976xf32>
    %1495 = tensor.expand_shape %1494 [[0 : i64, 1 : i64]] output_shape [41, 1536] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2"} : tensor<62976xf32> into tensor<41x1536xf32>
    %1496 = tensor.empty() : tensor<1536x1024xf32>
    %1497 = linalg.transpose ins(%51:tensor<1024x1536xf32>) outs(%1496:tensor<1536x1024xf32>) permutation = [1, 0]
    %1498 = tensor.empty() : tensor<41x1024xf32>
    %1499 = arith.constant {prov.module = "head"} 0.000000e+00 : f32
    %1500 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "head"} ins(%1499 : f32) outs(%1498 : tensor<41x1024xf32>) -> tensor<41x1024xf32>
    %1501 = linalg.matmul {prov.region_id = "matmul_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2", prov.transposed_b = "true"} ins(%1495, %1497 : tensor<41x1536xf32>, tensor<1536x1024xf32>) outs(%1500 : tensor<41x1024xf32>) -> tensor<41x1024xf32>
    %1502 = tensor.empty() : tensor<41x1024xf32>
    %1503 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1501, %52 : tensor<41x1024xf32>, tensor<1024xf32>) outs(%1502 : tensor<41x1024xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2"} {
    ^bb157(%1504: f32, %1505: f32, %1506: f32):
      %1507 = arith.addf %1504, %1505 : f32
      linalg.yield %1507 : f32
    } -> tensor<41x1024xf32>
    %1508 = tensor.collapse_shape %1503 [[0 : i64, 1 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2"} : tensor<41x1024xf32> into tensor<41984xf32>
    %1509 = tensor.expand_shape %1508 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 41, 1024] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.model.proj_out_2"} : tensor<41984xf32> into tensor<1x41x1024xf32>
    %1510 = tensor.empty() : tensor<1x1024x1024xf32>
    %1511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%1510 : tensor<1x1024x1024xf32>) attrs =  {prov.region_id = "gather_11", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} {
    ^bb158(%1512: i64, %1513: f32):
      %1514 = arith.index_cast %1512 : i64 to index
      %1515 = linalg.index 1 : index
      %1516 = linalg.index 2 : index
      %1517 = tensor.extract %10[%1514, %1515, %1516] : tensor<32x1024x1024xf32>
      linalg.yield %1517 : f32
    } -> tensor<1x1024x1024xf32>
    %1518 = tensor.empty() : tensor<1x1024xf32>
    %1519 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%1518 : tensor<1x1024xf32>) attrs =  {prov.region_id = "gather_12", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} {
    ^bb159(%1520: i64, %1521: f32):
      %1522 = arith.index_cast %1520 : i64 to index
      %1523 = linalg.index 1 : index
      %1524 = tensor.extract %11[%1522, %1523] : tensor<32x1024xf32>
      linalg.yield %1524 : f32
    } -> tensor<1x1024xf32>
    %1525 = arith.constant {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} 0.000000e+00 : f32
    %1526 = tensor.splat %1525 {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} : tensor<1x41x1024xf32>
    %1527 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1509, %1511 : tensor<1x41x1024xf32>, tensor<1x1024x1024xf32>) outs(%1526 : tensor<1x41x1024xf32>) attrs =  {prov.region_id = "matmul_27", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} {
    ^bb160(%1528: f32, %1529: f32, %1530: f32):
      %1531 = arith.mulf %1528, %1529 : f32
      %1532 = arith.addf %1530, %1531 : f32
      linalg.yield %1532 : f32
    } -> tensor<1x41x1024xf32>
    %1533 = tensor.collapse_shape %1519 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1534 = tensor.expand_shape %1533 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1535 = tensor.empty() : tensor<1x41x1024xf32>
    %1536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1527, %1534 : tensor<1x41x1024xf32>, tensor<1x1x1024xf32>) outs(%1535 : tensor<1x41x1024xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer1"} {
    ^bb161(%1537: f32, %1538: f32, %1539: f32):
      %1540 = arith.addf %1537, %1538 : f32
      linalg.yield %1540 : f32
    } -> tensor<1x41x1024xf32>
    %1541 = tensor.empty() : tensor<1x41x1024xf32>
    %1542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1536 : tensor<1x41x1024xf32>) outs(%1541 : tensor<1x41x1024xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder"} {
    ^bb162(%1543: f32, %1544: f32):
      %1545 = arith.constant 0.000000e+00 : f32
      %1546 = arith.maximumf %1543, %1545 : f32
      linalg.yield %1546 : f32
    } -> tensor<1x41x1024xf32>
    %1547 = tensor.empty() : tensor<1x1024x132xf32>
    %1548 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%1547 : tensor<1x1024x132xf32>) attrs =  {prov.region_id = "gather_13", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} {
    ^bb163(%1549: i64, %1550: f32):
      %1551 = arith.index_cast %1549 : i64 to index
      %1552 = linalg.index 1 : index
      %1553 = linalg.index 2 : index
      %1554 = tensor.extract %12[%1551, %1552, %1553] : tensor<32x1024x132xf32>
      linalg.yield %1554 : f32
    } -> tensor<1x1024x132xf32>
    %1555 = tensor.empty() : tensor<1x132xf32>
    %1556 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<1xi64>) outs(%1555 : tensor<1x132xf32>) attrs =  {prov.region_id = "gather_14", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} {
    ^bb164(%1557: i64, %1558: f32):
      %1559 = arith.index_cast %1557 : i64 to index
      %1560 = linalg.index 1 : index
      %1561 = tensor.extract %13[%1559, %1560] : tensor<32x132xf32>
      linalg.yield %1561 : f32
    } -> tensor<1x132xf32>
    %1562 = arith.constant {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} 0.000000e+00 : f32
    %1563 = tensor.splat %1562 {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} : tensor<1x41x132xf32>
    %1564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1542, %1548 : tensor<1x41x1024xf32>, tensor<1x1024x132xf32>) outs(%1563 : tensor<1x41x132xf32>) attrs =  {prov.region_id = "matmul_28", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} {
    ^bb165(%1565: f32, %1566: f32, %1567: f32):
      %1568 = arith.mulf %1565, %1566 : f32
      %1569 = arith.addf %1567, %1568 : f32
      linalg.yield %1569 : f32
    } -> tensor<1x41x132xf32>
    %1570 = tensor.collapse_shape %1556 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} : tensor<1x132xf32> into tensor<132xf32>
    %1571 = tensor.expand_shape %1570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 132] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} : tensor<132xf32> into tensor<1x1x132xf32>
    %1572 = tensor.empty() : tensor<1x41x132xf32>
    %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1564, %1571 : tensor<1x41x132xf32>, tensor<1x1x132xf32>) outs(%1572 : tensor<1x41x132xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "head", prov.fqn = "head.action_decoder.layer2"} {
    ^bb166(%1574: f32, %1575: f32, %1576: f32):
      %1577 = arith.addf %1574, %1575 : f32
      linalg.yield %1577 : f32
    } -> tensor<1x41x132xf32>
    %1578 = "tensor.extract_slice"(%1573) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 41, 132>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x41x132xf32>) -> tensor<1x41x132xf32>
    %1579 = "tensor.extract_slice"(%1578) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 40, 132>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x41x132xf32>) -> tensor<1x40x132xf32>
    func.return %1579 : tensor<1x40x132xf32>
  }
}
