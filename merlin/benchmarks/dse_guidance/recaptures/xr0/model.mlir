builtin.module attributes {prov.weights_file = "/scratch/agustin/projects/model2MLIR/workloads/xr0/xr0.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<6x1024xf32>, %1: tensor<3072x1024xf32>, %2: tensor<3072xf32>, %3: tensor<1024x1024xf32>, %4: tensor<128xf32>, %5: tensor<128xf32>, %6: tensor<4096x1024xf32>, %7: tensor<4096x1024xf32>, %8: tensor<1024x4096xf32>, %9: tensor<1024xf32>, %10: tensor<1024xf32>, %11: tensor<1024xf32>, %12: tensor<1024xf32>, %13: tensor<6x1024xf32>, %14: tensor<3072x1024xf32>, %15: tensor<3072xf32>, %16: tensor<1024x1024xf32>, %17: tensor<128xf32>, %18: tensor<128xf32>, %19: tensor<4096x1024xf32>, %20: tensor<4096x1024xf32>, %21: tensor<1024x4096xf32>, %22: tensor<1024xf32>, %23: tensor<1024xf32>, %24: tensor<1024xf32>, %25: tensor<1024xf32>, %26: tensor<1024x32xf32>, %27: tensor<1024x1024xf32>, %28: tensor<1024x32xf32>, %29: tensor<1024x1024xf32>, %30: tensor<32x1024xf32>, %31: tensor<32x32xf32>, %32: tensor<1024x256xf32>, %33: tensor<1024x1024xf32>, %34: tensor<6144x1024xf32>, %35: tensor<6144xf32>, %36: tensor<1x1024xf32>, %37: tensor<64xf32>, %38: tensor<1x30x32xf32>, %39: tensor<1x1x1xf32>, %40: tensor<1x30x32xf32>, %41: tensor<1x1x32xf32>, %42: tensor<1x32x128xf32>, %43: tensor<1x32x128xf32>, %44: tensor<1x1x32x48xi1>, %45: tensor<1x8x16x128xf32>, %46: tensor<1x8x16x128xf32>, %47: tensor<1x8x16x128xf32>, %48: tensor<1x8x16x128xf32>) -> tensor<1x30x32xf32> {
    %49 = tensor.empty() : tensor<32x1024xf32>
    %50 = linalg.transpose ins(%26:tensor<1024x32xf32>) outs(%49:tensor<32x1024xf32>) permutation = [1, 0]
    %51 = tensor.empty() : tensor<1x1x1024xf32>
    %52 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %53 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%52 : f32) outs(%51 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %54 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.state_projector.layers.0"} ins(%41, %50 : tensor<1x1x32xf32>, tensor<32x1024xf32>) outs(%53 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %55 = tensor.empty() : tensor<1x1x1024xf32>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54 : tensor<1x1x1024xf32>) outs(%55 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.state_projector.layers.1"} {
    ^bb0(%57: f32, %58: f32):
      %59 = arith.constant 5.000000e-01 : f32
      %60 = arith.constant 1.000000e+00 : f32
      %61 = arith.constant 0.707106769 : f32
      %62 = arith.mulf %57, %61 : f32
      %63 = math.erf %62 : f32
      %64 = arith.addf %60, %63 : f32
      %65 = arith.mulf %59, %57 : f32
      %66 = arith.mulf %65, %64 : f32
      linalg.yield %66 : f32
    } -> tensor<1x1x1024xf32>
    %67 = tensor.empty() : tensor<1024x1024xf32>
    %68 = linalg.transpose ins(%27:tensor<1024x1024xf32>) outs(%67:tensor<1024x1024xf32>) permutation = [1, 0]
    %69 = tensor.empty() : tensor<1x1x1024xf32>
    %70 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %71 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%70 : f32) outs(%69 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %72 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.state_projector.layers.2"} ins(%56, %68 : tensor<1x1x1024xf32>, tensor<1024x1024xf32>) outs(%71 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %73 = tensor.collapse_shape %39 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x1xf32> into tensor<1xf32>
    %74 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+03 : f32
    %75 = tensor.splat %74 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1xf32>
    %76 = tensor.empty() : tensor<1xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%73, %75 : tensor<1xf32>, tensor<1xf32>) outs(%76 : tensor<1xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%78: f32, %79: f32, %80: f32):
      %81 = arith.mulf %78, %79 : f32
      linalg.yield %81 : f32
    } -> tensor<1xf32>
    %82 = tensor.empty() : tensor<128xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%82 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb2(%84: f32):
      %85 = linalg.index 0 : index
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.sitofp %86 : i64 to f32
      %88 = arith.constant 1.000000e+00 : f32
      %89 = arith.mulf %87, %88 : f32
      %90 = arith.constant 0.000000e+00 : f32
      %91 = arith.addf %90, %89 : f32
      linalg.yield %91 : f32
    } -> tensor<128xf32>
    %92 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} -9.2103405 : f32
    %93 = tensor.splat %92 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %94 = tensor.empty() : tensor<128xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%83, %93 : tensor<128xf32>, tensor<128xf32>) outs(%94 : tensor<128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb3(%96: f32, %97: f32, %98: f32):
      %99 = arith.mulf %96, %97 : f32
      linalg.yield %99 : f32
    } -> tensor<128xf32>
    %100 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} 1.280000e+02 : f32
    %101 = tensor.splat %100 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %102 = tensor.empty() : tensor<128xf32>
    %103 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%95, %101 : tensor<128xf32>, tensor<128xf32>) outs(%102 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb4(%104: f32, %105: f32, %106: f32):
      %107 = arith.divf %104, %105 : f32
      linalg.yield %107 : f32
    } -> tensor<128xf32>
    %108 = tensor.empty() : tensor<128xf32>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<128xf32>) outs(%108 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb5(%110: f32, %111: f32):
      %112 = math.exp %110 : f32
      linalg.yield %112 : f32
    } -> tensor<128xf32>
    %113 = "tensor.extract_slice"(%77) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : (tensor<1xf32>) -> tensor<1xf32>
    %114 = tensor.expand_shape %113 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<1xf32> into tensor<1x1xf32>
    %115 = tensor.expand_shape %109 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32> into tensor<1x128xf32>
    %116 = tensor.empty() : tensor<1x128xf32>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%114, %115 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%116 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb6(%118: f32, %119: f32, %120: f32):
      %121 = arith.mulf %118, %119 : f32
      linalg.yield %121 : f32
    } -> tensor<1x128xf32>
    %122 = tensor.empty() : tensor<1x128xf32>
    %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117 : tensor<1x128xf32>) outs(%122 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb7(%124: f32, %125: f32):
      %126 = math.cos %124 : f32
      linalg.yield %126 : f32
    } -> tensor<1x128xf32>
    %127 = tensor.empty() : tensor<1x128xf32>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117 : tensor<1x128xf32>) outs(%127 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb8(%129: f32, %130: f32):
      %131 = math.sin %129 : f32
      linalg.yield %131 : f32
    } -> tensor<1x128xf32>
    %132 = tensor.concat dim(1) %123, %128 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %133 = tensor.empty() : tensor<256x1024xf32>
    %134 = linalg.transpose ins(%32:tensor<1024x256xf32>) outs(%133:tensor<256x1024xf32>) permutation = [1, 0]
    %135 = tensor.empty() : tensor<1x1024xf32>
    %136 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %137 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%136 : f32) outs(%135 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %138 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.0"} ins(%132, %134 : tensor<1x256xf32>, tensor<256x1024xf32>) outs(%137 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %139 = tensor.empty() : tensor<1x1024xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%138 : tensor<1x1024xf32>) outs(%139 : tensor<1x1024xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.1"} {
    ^bb9(%141: f32, %142: f32):
      %143 = arith.constant 1.000000e+00 : f32
      %144 = arith.negf %141 : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %143, %145 : f32
      %147 = arith.divf %143, %146 : f32
      %148 = arith.mulf %141, %147 : f32
      linalg.yield %148 : f32
    } -> tensor<1x1024xf32>
    %149 = tensor.empty() : tensor<1024x1024xf32>
    %150 = linalg.transpose ins(%33:tensor<1024x1024xf32>) outs(%149:tensor<1024x1024xf32>) permutation = [1, 0]
    %151 = tensor.empty() : tensor<1x1024xf32>
    %152 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %153 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%152 : f32) outs(%151 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %154 = linalg.matmul {prov.region_id = "matmul_3", prov.dispatch_id = "matmul_3", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.2"} ins(%140, %150 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%153 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %155 = "tensor.extract_slice"(%154) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %156 = tensor.collapse_shape %155 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<1x1024xf32> into tensor<1024xf32>
    %157 = tensor.expand_shape %156 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %158 = tensor.empty() : tensor<1024x6144xf32>
    %159 = linalg.transpose ins(%34:tensor<6144x1024xf32>) outs(%158:tensor<1024x6144xf32>) permutation = [1, 0]
    %160 = tensor.empty() : tensor<1x1x6144xf32>
    %161 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %162 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%161 : f32) outs(%160 : tensor<1x1x6144xf32>) -> tensor<1x1x6144xf32>
    %163 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_projector.layers.0"} ins(%157, %159 : tensor<1x1x1024xf32>, tensor<1024x6144xf32>) outs(%162 : tensor<1x1x6144xf32>) -> tensor<1x1x6144xf32>
    %164 = tensor.empty() : tensor<1x1x6144xf32>
    %165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%163, %35 : tensor<1x1x6144xf32>, tensor<6144xf32>) outs(%164 : tensor<1x1x6144xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_projector.layers.0"} {
    ^bb10(%166: f32, %167: f32, %168: f32):
      %169 = arith.addf %166, %167 : f32
      linalg.yield %169 : f32
    } -> tensor<1x1x6144xf32>
    %170 = tensor.collapse_shape %165 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6144xf32> into tensor<6144xf32>
    %171 = tensor.expand_shape %170 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x6x1024xf32>
    %172 = tensor.empty() : tensor<1x30x32xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %40 : tensor<1x30x32xf32>, tensor<1x30x32xf32>) outs(%172 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb11(%174: f32, %175: f32, %176: f32):
      %177 = arith.mulf %174, %175 : f32
      linalg.yield %177 : f32
    } -> tensor<1x30x32xf32>
    %178 = tensor.empty() : tensor<32x1024xf32>
    %179 = linalg.transpose ins(%28:tensor<1024x32xf32>) outs(%178:tensor<32x1024xf32>) permutation = [1, 0]
    %180 = tensor.empty() : tensor<1x30x1024xf32>
    %181 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %182 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%181 : f32) outs(%180 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
    %183 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_projector.layers.0"} ins(%173, %179 : tensor<1x30x32xf32>, tensor<32x1024xf32>) outs(%182 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
    %184 = tensor.empty() : tensor<1x30x1024xf32>
    %185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%183 : tensor<1x30x1024xf32>) outs(%184 : tensor<1x30x1024xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_projector.layers.1"} {
    ^bb12(%186: f32, %187: f32):
      %188 = arith.constant 5.000000e-01 : f32
      %189 = arith.constant 1.000000e+00 : f32
      %190 = arith.constant 0.707106769 : f32
      %191 = arith.mulf %186, %190 : f32
      %192 = math.erf %191 : f32
      %193 = arith.addf %189, %192 : f32
      %194 = arith.mulf %188, %186 : f32
      %195 = arith.mulf %194, %193 : f32
      linalg.yield %195 : f32
    } -> tensor<1x30x1024xf32>
    %196 = tensor.empty() : tensor<1024x1024xf32>
    %197 = linalg.transpose ins(%29:tensor<1024x1024xf32>) outs(%196:tensor<1024x1024xf32>) permutation = [1, 0]
    %198 = tensor.empty() : tensor<1x30x1024xf32>
    %199 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %200 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%199 : f32) outs(%198 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
    %201 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_projector.layers.2"} ins(%185, %197 : tensor<1x30x1024xf32>, tensor<1024x1024xf32>) outs(%200 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
    %202 = tensor.collapse_shape %36 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
    %203 = tensor.expand_shape %202 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %204 = tensor.concat dim(1) %203, %72, %201 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>, tensor<1x1x1024xf32>, tensor<1x30x1024xf32>) -> tensor<1x32x1024xf32>
    %205 = tensor.collapse_shape %0 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : tensor<6x1024xf32> into tensor<6144xf32>
    %206 = tensor.expand_shape %205 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : tensor<6144xf32> into tensor<1x6x1024xf32>
    %207 = tensor.empty() : tensor<1x6x1024xf32>
    %208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%206, %171 : tensor<1x6x1024xf32>, tensor<1x6x1024xf32>) outs(%207 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb13(%209: f32, %210: f32, %211: f32):
      %212 = arith.addf %209, %210 : f32
      linalg.yield %212 : f32
    } -> tensor<1x6x1024xf32>
    %213 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %214 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %215 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %216 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %217 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 4, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %218 = "tensor.extract_slice"(%208) <{static_offsets = array<i64: 0, 5, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %219 = tensor.empty() : tensor<1x32x1024xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204 : tensor<1x32x1024xf32>) outs(%219 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb14(%221: f32, %222: f32):
      %223 = arith.constant 2.000000e+00 : f32
      %224 = math.powf %221, %223 : f32
      linalg.yield %224 : f32
    } -> tensor<1x32x1024xf32>
    %225 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} 0.000000e+00 : f32
    %226 = tensor.splat %225 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} : tensor<1x32xf32>
    %227 = linalg.reduce ins(%220:tensor<1x32x1024xf32>) outs(%226:tensor<1x32xf32>) dimensions = [2]
    (%228: f32, %229: f32) {
      %230 = arith.addf %228, %229 : f32
      linalg.yield %230 : f32
    }
    %231 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} 1.024000e+03 : f32
    %232 = tensor.splat %231 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} : tensor<1x32xf32>
    %233 = tensor.empty() : tensor<1x32xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%227, %232 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%233 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb15(%235: f32, %236: f32, %237: f32):
      %238 = arith.divf %235, %236 : f32
      linalg.yield %238 : f32
    } -> tensor<1x32xf32>
    %239 = tensor.collapse_shape %234 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %240 = tensor.expand_shape %239 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %241 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} 1.000000e-06 : f32
    %242 = tensor.splat %241 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} : tensor<1x32x1xf32>
    %243 = tensor.empty() : tensor<1x32x1xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%240, %242 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%243 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb16(%245: f32, %246: f32, %247: f32):
      %248 = arith.addf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x32x1xf32>
    %249 = tensor.empty() : tensor<1x32x1xf32>
    %250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%244 : tensor<1x32x1xf32>) outs(%249 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb17(%251: f32, %252: f32):
      %253 = math.rsqrt %251 : f32
      linalg.yield %253 : f32
    } -> tensor<1x32x1xf32>
    %254 = tensor.empty() : tensor<1x32x1024xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204, %250 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%254 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb18(%256: f32, %257: f32, %258: f32):
      %259 = arith.mulf %256, %257 : f32
      linalg.yield %259 : f32
    } -> tensor<1x32x1024xf32>
    %260 = tensor.empty() : tensor<1x32x1024xf32>
    %261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %255 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%260 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.input_layernorm"} {
    ^bb19(%262: f32, %263: f32, %264: f32):
      %265 = arith.mulf %262, %263 : f32
      linalg.yield %265 : f32
    } -> tensor<1x32x1024xf32>
    %266 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} 1.000000e+00 : f32
    %267 = tensor.splat %266 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : tensor<1x1x1024xf32>
    %268 = tensor.empty() : tensor<1x1x1024xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214, %267 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%268 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb20(%270: f32, %271: f32, %272: f32):
      %273 = arith.addf %270, %271 : f32
      linalg.yield %273 : f32
    } -> tensor<1x1x1024xf32>
    %274 = tensor.empty() : tensor<1x32x1024xf32>
    %275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%261, %269 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%274 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb21(%276: f32, %277: f32, %278: f32):
      %279 = arith.mulf %276, %277 : f32
      linalg.yield %279 : f32
    } -> tensor<1x32x1024xf32>
    %280 = tensor.empty() : tensor<1x32x1024xf32>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275, %213 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%280 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb22(%282: f32, %283: f32, %284: f32):
      %285 = arith.addf %282, %283 : f32
      linalg.yield %285 : f32
    } -> tensor<1x32x1024xf32>
    %286 = tensor.empty() : tensor<1024x3072xf32>
    %287 = linalg.transpose ins(%1:tensor<3072x1024xf32>) outs(%286:tensor<1024x3072xf32>) permutation = [1, 0]
    %288 = tensor.empty() : tensor<1x32x3072xf32>
    %289 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %290 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%289 : f32) outs(%288 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
    %291 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.qkv_proj"} ins(%281, %287 : tensor<1x32x1024xf32>, tensor<1024x3072xf32>) outs(%290 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
    %292 = tensor.empty() : tensor<1x32x3072xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%291, %2 : tensor<1x32x3072xf32>, tensor<3072xf32>) outs(%292 : tensor<1x32x3072xf32>) attrs =  {prov.region_id = "add_5", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.qkv_proj"} {
    ^bb23(%294: f32, %295: f32, %296: f32):
      %297 = arith.addf %294, %295 : f32
      linalg.yield %297 : f32
    } -> tensor<1x32x3072xf32>
    %298 = tensor.collapse_shape %293 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x3072xf32> into tensor<98304xf32>
    %299 = tensor.expand_shape %298 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 3, 8, 128] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<98304xf32> into tensor<1x32x3x8x128xf32>
    %300 = "tensor.extract_slice"(%299) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %301 = tensor.collapse_shape %300 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %302 = tensor.expand_shape %301 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %303 = "tensor.extract_slice"(%299) <{static_offsets = array<i64: 0, 0, 1, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %304 = tensor.collapse_shape %303 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %305 = tensor.expand_shape %304 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %306 = "tensor.extract_slice"(%299) <{static_offsets = array<i64: 0, 0, 2, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %307 = tensor.collapse_shape %306 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %308 = tensor.expand_shape %307 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %309 = tensor.empty() : tensor<1x32x8x128xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%302 : tensor<1x32x8x128xf32>) outs(%309 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb24(%311: f32, %312: f32):
      %313 = arith.constant 2.000000e+00 : f32
      %314 = math.powf %311, %313 : f32
      linalg.yield %314 : f32
    } -> tensor<1x32x8x128xf32>
    %315 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} 0.000000e+00 : f32
    %316 = tensor.splat %315 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} : tensor<1x32x8xf32>
    %317 = linalg.reduce ins(%310:tensor<1x32x8x128xf32>) outs(%316:tensor<1x32x8xf32>) dimensions = [3]
    (%318: f32, %319: f32) {
      %320 = arith.addf %318, %319 : f32
      linalg.yield %320 : f32
    }
    %321 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} 1.280000e+02 : f32
    %322 = tensor.splat %321 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} : tensor<1x32x8xf32>
    %323 = tensor.empty() : tensor<1x32x8xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%317, %322 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%323 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb25(%325: f32, %326: f32, %327: f32):
      %328 = arith.divf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x32x8xf32>
    %329 = tensor.collapse_shape %324 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} : tensor<1x32x8xf32> into tensor<256xf32>
    %330 = tensor.expand_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %331 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} 1.000000e-06 : f32
    %332 = tensor.splat %331 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} : tensor<1x32x8x1xf32>
    %333 = tensor.empty() : tensor<1x32x8x1xf32>
    %334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%330, %332 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%333 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb26(%335: f32, %336: f32, %337: f32):
      %338 = arith.addf %335, %336 : f32
      linalg.yield %338 : f32
    } -> tensor<1x32x8x1xf32>
    %339 = tensor.empty() : tensor<1x32x8x1xf32>
    %340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%334 : tensor<1x32x8x1xf32>) outs(%339 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb27(%341: f32, %342: f32):
      %343 = math.rsqrt %341 : f32
      linalg.yield %343 : f32
    } -> tensor<1x32x8x1xf32>
    %344 = tensor.empty() : tensor<1x32x8x128xf32>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%302, %340 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%344 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb28(%346: f32, %347: f32, %348: f32):
      %349 = arith.mulf %346, %347 : f32
      linalg.yield %349 : f32
    } -> tensor<1x32x8x128xf32>
    %350 = tensor.empty() : tensor<1x32x8x128xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %345 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%350 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.q_norm"} {
    ^bb29(%352: f32, %353: f32, %354: f32):
      %355 = arith.mulf %352, %353 : f32
      linalg.yield %355 : f32
    } -> tensor<1x32x8x128xf32>
    %356 = tensor.empty() : tensor<1x32x8x128xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%305 : tensor<1x32x8x128xf32>) outs(%356 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb30(%358: f32, %359: f32):
      %360 = arith.constant 2.000000e+00 : f32
      %361 = math.powf %358, %360 : f32
      linalg.yield %361 : f32
    } -> tensor<1x32x8x128xf32>
    %362 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} 0.000000e+00 : f32
    %363 = tensor.splat %362 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} : tensor<1x32x8xf32>
    %364 = linalg.reduce ins(%357:tensor<1x32x8x128xf32>) outs(%363:tensor<1x32x8xf32>) dimensions = [3]
    (%365: f32, %366: f32) {
      %367 = arith.addf %365, %366 : f32
      linalg.yield %367 : f32
    }
    %368 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} 1.280000e+02 : f32
    %369 = tensor.splat %368 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} : tensor<1x32x8xf32>
    %370 = tensor.empty() : tensor<1x32x8xf32>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%364, %369 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%370 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb31(%372: f32, %373: f32, %374: f32):
      %375 = arith.divf %372, %373 : f32
      linalg.yield %375 : f32
    } -> tensor<1x32x8xf32>
    %376 = tensor.collapse_shape %371 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} : tensor<1x32x8xf32> into tensor<256xf32>
    %377 = tensor.expand_shape %376 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %378 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} 1.000000e-06 : f32
    %379 = tensor.splat %378 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} : tensor<1x32x8x1xf32>
    %380 = tensor.empty() : tensor<1x32x8x1xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%377, %379 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%380 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb32(%382: f32, %383: f32, %384: f32):
      %385 = arith.addf %382, %383 : f32
      linalg.yield %385 : f32
    } -> tensor<1x32x8x1xf32>
    %386 = tensor.empty() : tensor<1x32x8x1xf32>
    %387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381 : tensor<1x32x8x1xf32>) outs(%386 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb33(%388: f32, %389: f32):
      %390 = math.rsqrt %388 : f32
      linalg.yield %390 : f32
    } -> tensor<1x32x8x1xf32>
    %391 = tensor.empty() : tensor<1x32x8x128xf32>
    %392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%305, %387 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%391 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb34(%393: f32, %394: f32, %395: f32):
      %396 = arith.mulf %393, %394 : f32
      linalg.yield %396 : f32
    } -> tensor<1x32x8x128xf32>
    %397 = tensor.empty() : tensor<1x32x8x128xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %392 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%397 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.k_norm"} {
    ^bb35(%399: f32, %400: f32, %401: f32):
      %402 = arith.mulf %399, %400 : f32
      linalg.yield %402 : f32
    } -> tensor<1x32x8x128xf32>
    %403 = tensor.empty() : tensor<1x8x32x128xf32>
    %404 = linalg.transpose ins(%351:tensor<1x32x8x128xf32>) outs(%403:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %405 = tensor.empty() : tensor<1x8x32x128xf32>
    %406 = linalg.transpose ins(%398:tensor<1x32x8x128xf32>) outs(%405:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %407 = tensor.empty() : tensor<1x8x32x128xf32>
    %408 = linalg.transpose ins(%308:tensor<1x32x8x128xf32>) outs(%407:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %409 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %410 = tensor.expand_shape %409 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
    %411 = tensor.collapse_shape %43 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %412 = tensor.expand_shape %411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
    %413 = tensor.empty() : tensor<1x8x32x128xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%404, %410 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%413 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb36(%415: f32, %416: f32, %417: f32):
      %418 = arith.mulf %415, %416 : f32
      linalg.yield %418 : f32
    } -> tensor<1x8x32x128xf32>
    %419 = "tensor.extract_slice"(%404) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %420 = "tensor.extract_slice"(%404) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %421 = tensor.empty() : tensor<1x8x32x64xf32>
    %422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420 : tensor<1x8x32x64xf32>) outs(%421 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb37(%423: f32, %424: f32):
      %425 = arith.negf %423 : f32
      linalg.yield %425 : f32
    } -> tensor<1x8x32x64xf32>
    %426 = tensor.concat dim(3) %422, %419 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
    %427 = tensor.empty() : tensor<1x8x32x128xf32>
    %428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %412 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%427 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb38(%429: f32, %430: f32, %431: f32):
      %432 = arith.mulf %429, %430 : f32
      linalg.yield %432 : f32
    } -> tensor<1x8x32x128xf32>
    %433 = tensor.empty() : tensor<1x8x32x128xf32>
    %434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414, %428 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%433 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb39(%435: f32, %436: f32, %437: f32):
      %438 = arith.addf %435, %436 : f32
      linalg.yield %438 : f32
    } -> tensor<1x8x32x128xf32>
    %439 = tensor.empty() : tensor<1x8x32x128xf32>
    %440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %410 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%439 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb40(%441: f32, %442: f32, %443: f32):
      %444 = arith.mulf %441, %442 : f32
      linalg.yield %444 : f32
    } -> tensor<1x8x32x128xf32>
    %445 = "tensor.extract_slice"(%406) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %446 = "tensor.extract_slice"(%406) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %447 = tensor.empty() : tensor<1x8x32x64xf32>
    %448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%446 : tensor<1x8x32x64xf32>) outs(%447 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb41(%449: f32, %450: f32):
      %451 = arith.negf %449 : f32
      linalg.yield %451 : f32
    } -> tensor<1x8x32x64xf32>
    %452 = tensor.concat dim(3) %448, %445 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
    %453 = tensor.empty() : tensor<1x8x32x128xf32>
    %454 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452, %412 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%453 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb42(%455: f32, %456: f32, %457: f32):
      %458 = arith.mulf %455, %456 : f32
      linalg.yield %458 : f32
    } -> tensor<1x8x32x128xf32>
    %459 = tensor.empty() : tensor<1x8x32x128xf32>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%440, %454 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%459 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb43(%461: f32, %462: f32, %463: f32):
      %464 = arith.addf %461, %462 : f32
      linalg.yield %464 : f32
    } -> tensor<1x8x32x128xf32>
    %465 = tensor.concat dim(2) %45, %460 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
    %466 = tensor.concat dim(2) %46, %408 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
    %467 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} 0.000000e+00 : f32
    %468 = tensor.splat %467 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32x48xf32>
    %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%434, %465 : tensor<1x8x32x128xf32>, tensor<1x8x48x128xf32>) outs(%468 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb44(%470: f32, %471: f32, %472: f32):
      %473 = arith.mulf %470, %471 : f32
      %474 = arith.addf %472, %473 : f32
      linalg.yield %474 : f32
    } -> tensor<1x8x32x48xf32>
    %475 = tensor.empty() : tensor<1x8x32x48xf32>
    %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%469 : tensor<1x8x32x48xf32>) outs(%475 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb45(%477: f32, %478: f32):
      %479 = arith.constant 0.0883883461 : f32
      %480 = arith.mulf %477, %479 : f32
      linalg.yield %480 : f32
    } -> tensor<1x8x32x48xf32>
    %481 = tensor.empty() : tensor<1x8x32x48xf32>
    %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %44 : tensor<1x8x32x48xf32>, tensor<1x1x32x48xi1>) outs(%481 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb46(%483: f32, %484: i1, %485: f32):
      %486 = arith.constant 0xff800000 : f32
      %487 = arith.select %484, %483, %486 : f32
      linalg.yield %487 : f32
    } -> tensor<1x8x32x48xf32>
    %488 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} 0xff800000 : f32
    %489 = tensor.splat %488 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32xf32>
    %490 = linalg.reduce ins(%482:tensor<1x8x32x48xf32>) outs(%489:tensor<1x8x32xf32>) dimensions = [3]
    (%491: f32, %492: f32) {
      %493 = arith.maximumf %491, %492 : f32
      linalg.yield %493 : f32
    }
    %494 = tensor.collapse_shape %490 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %495 = tensor.expand_shape %494 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %496 = tensor.empty() : tensor<1x8x32x48xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482, %495 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%496 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb47(%498: f32, %499: f32, %500: f32):
      %501 = arith.subf %498, %499 : f32
      linalg.yield %501 : f32
    } -> tensor<1x8x32x48xf32>
    %502 = tensor.empty() : tensor<1x8x32x48xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497 : tensor<1x8x32x48xf32>) outs(%502 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb48(%504: f32, %505: f32):
      %506 = math.exp %504 : f32
      linalg.yield %506 : f32
    } -> tensor<1x8x32x48xf32>
    %507 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} 0.000000e+00 : f32
    %508 = tensor.splat %507 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32xf32>
    %509 = linalg.reduce ins(%503:tensor<1x8x32x48xf32>) outs(%508:tensor<1x8x32xf32>) dimensions = [3]
    (%510: f32, %511: f32) {
      %512 = arith.addf %510, %511 : f32
      linalg.yield %512 : f32
    }
    %513 = tensor.collapse_shape %509 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %514 = tensor.expand_shape %513 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %515 = tensor.empty() : tensor<1x8x32x48xf32>
    %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%503, %514 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%515 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb49(%517: f32, %518: f32, %519: f32):
      %520 = arith.divf %517, %518 : f32
      linalg.yield %520 : f32
    } -> tensor<1x8x32x48xf32>
    %521 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} 0.000000e+00 : f32
    %522 = tensor.splat %521 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x8x32x128xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%516, %466 : tensor<1x8x32x48xf32>, tensor<1x8x48x128xf32>) outs(%522 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} {
    ^bb50(%524: f32, %525: f32, %526: f32):
      %527 = arith.mulf %524, %525 : f32
      %528 = arith.addf %526, %527 : f32
      linalg.yield %528 : f32
    } -> tensor<1x8x32x128xf32>
    %529 = tensor.empty() : tensor<1x32x8x128xf32>
    %530 = linalg.transpose ins(%523:tensor<1x8x32x128xf32>) outs(%529:tensor<1x32x8x128xf32>) permutation = [0, 2, 1, 3]
    %531 = tensor.collapse_shape %530 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<1x32x8x128xf32> into tensor<32768xf32>
    %532 = tensor.expand_shape %531 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn"} : tensor<32768xf32> into tensor<1x32x1024xf32>
    %533 = tensor.empty() : tensor<1024x1024xf32>
    %534 = linalg.transpose ins(%3:tensor<1024x1024xf32>) outs(%533:tensor<1024x1024xf32>) permutation = [1, 0]
    %535 = tensor.empty() : tensor<1x32x1024xf32>
    %536 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %537 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%536 : f32) outs(%535 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %538 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.attn.o_proj"} ins(%532, %534 : tensor<1x32x1024xf32>, tensor<1024x1024xf32>) outs(%537 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %539 = tensor.empty() : tensor<1x32x1024xf32>
    %540 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%215, %538 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%539 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb51(%541: f32, %542: f32, %543: f32):
      %544 = arith.mulf %541, %542 : f32
      linalg.yield %544 : f32
    } -> tensor<1x32x1024xf32>
    %545 = tensor.empty() : tensor<1x32x1024xf32>
    %546 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204, %540 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%545 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb52(%547: f32, %548: f32, %549: f32):
      %550 = arith.addf %547, %548 : f32
      linalg.yield %550 : f32
    } -> tensor<1x32x1024xf32>
    %551 = tensor.empty() : tensor<1x32x1024xf32>
    %552 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546 : tensor<1x32x1024xf32>) outs(%551 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb53(%553: f32, %554: f32):
      %555 = arith.constant 2.000000e+00 : f32
      %556 = math.powf %553, %555 : f32
      linalg.yield %556 : f32
    } -> tensor<1x32x1024xf32>
    %557 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} 0.000000e+00 : f32
    %558 = tensor.splat %557 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} : tensor<1x32xf32>
    %559 = linalg.reduce ins(%552:tensor<1x32x1024xf32>) outs(%558:tensor<1x32xf32>) dimensions = [2]
    (%560: f32, %561: f32) {
      %562 = arith.addf %560, %561 : f32
      linalg.yield %562 : f32
    }
    %563 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} 1.024000e+03 : f32
    %564 = tensor.splat %563 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} : tensor<1x32xf32>
    %565 = tensor.empty() : tensor<1x32xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%559, %564 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%565 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb54(%567: f32, %568: f32, %569: f32):
      %570 = arith.divf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x32xf32>
    %571 = tensor.collapse_shape %566 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %572 = tensor.expand_shape %571 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %573 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} 1.000000e-06 : f32
    %574 = tensor.splat %573 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} : tensor<1x32x1xf32>
    %575 = tensor.empty() : tensor<1x32x1xf32>
    %576 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%572, %574 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%575 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb55(%577: f32, %578: f32, %579: f32):
      %580 = arith.addf %577, %578 : f32
      linalg.yield %580 : f32
    } -> tensor<1x32x1xf32>
    %581 = tensor.empty() : tensor<1x32x1xf32>
    %582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%576 : tensor<1x32x1xf32>) outs(%581 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb56(%583: f32, %584: f32):
      %585 = math.rsqrt %583 : f32
      linalg.yield %585 : f32
    } -> tensor<1x32x1xf32>
    %586 = tensor.empty() : tensor<1x32x1024xf32>
    %587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546, %582 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%586 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb57(%588: f32, %589: f32, %590: f32):
      %591 = arith.mulf %588, %589 : f32
      linalg.yield %591 : f32
    } -> tensor<1x32x1024xf32>
    %592 = tensor.empty() : tensor<1x32x1024xf32>
    %593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %587 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%592 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.middle_layernorm"} {
    ^bb58(%594: f32, %595: f32, %596: f32):
      %597 = arith.mulf %594, %595 : f32
      linalg.yield %597 : f32
    } -> tensor<1x32x1024xf32>
    %598 = tensor.empty() : tensor<1x32x1024xf32>
    %599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%593 : tensor<1x32x1024xf32>) outs(%598 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb59(%600: f32, %601: f32):
      %602 = arith.constant 2.000000e+00 : f32
      %603 = math.powf %600, %602 : f32
      linalg.yield %603 : f32
    } -> tensor<1x32x1024xf32>
    %604 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} 0.000000e+00 : f32
    %605 = tensor.splat %604 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} : tensor<1x32xf32>
    %606 = linalg.reduce ins(%599:tensor<1x32x1024xf32>) outs(%605:tensor<1x32xf32>) dimensions = [2]
    (%607: f32, %608: f32) {
      %609 = arith.addf %607, %608 : f32
      linalg.yield %609 : f32
    }
    %610 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} 1.024000e+03 : f32
    %611 = tensor.splat %610 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} : tensor<1x32xf32>
    %612 = tensor.empty() : tensor<1x32xf32>
    %613 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%606, %611 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%612 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb60(%614: f32, %615: f32, %616: f32):
      %617 = arith.divf %614, %615 : f32
      linalg.yield %617 : f32
    } -> tensor<1x32xf32>
    %618 = tensor.collapse_shape %613 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %619 = tensor.expand_shape %618 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %620 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} 1.000000e-06 : f32
    %621 = tensor.splat %620 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} : tensor<1x32x1xf32>
    %622 = tensor.empty() : tensor<1x32x1xf32>
    %623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%619, %621 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%622 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb61(%624: f32, %625: f32, %626: f32):
      %627 = arith.addf %624, %625 : f32
      linalg.yield %627 : f32
    } -> tensor<1x32x1xf32>
    %628 = tensor.empty() : tensor<1x32x1xf32>
    %629 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%623 : tensor<1x32x1xf32>) outs(%628 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb62(%630: f32, %631: f32):
      %632 = math.rsqrt %630 : f32
      linalg.yield %632 : f32
    } -> tensor<1x32x1xf32>
    %633 = tensor.empty() : tensor<1x32x1024xf32>
    %634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%593, %629 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%633 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb63(%635: f32, %636: f32, %637: f32):
      %638 = arith.mulf %635, %636 : f32
      linalg.yield %638 : f32
    } -> tensor<1x32x1024xf32>
    %639 = tensor.empty() : tensor<1x32x1024xf32>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %634 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%639 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.post_layernorm"} {
    ^bb64(%641: f32, %642: f32, %643: f32):
      %644 = arith.mulf %641, %642 : f32
      linalg.yield %644 : f32
    } -> tensor<1x32x1024xf32>
    %645 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} 1.000000e+00 : f32
    %646 = tensor.splat %645 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} : tensor<1x1x1024xf32>
    %647 = tensor.empty() : tensor<1x1x1024xf32>
    %648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%217, %646 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%647 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb65(%649: f32, %650: f32, %651: f32):
      %652 = arith.addf %649, %650 : f32
      linalg.yield %652 : f32
    } -> tensor<1x1x1024xf32>
    %653 = tensor.empty() : tensor<1x32x1024xf32>
    %654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%640, %648 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%653 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb66(%655: f32, %656: f32, %657: f32):
      %658 = arith.mulf %655, %656 : f32
      linalg.yield %658 : f32
    } -> tensor<1x32x1024xf32>
    %659 = tensor.empty() : tensor<1x32x1024xf32>
    %660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654, %216 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%659 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb67(%661: f32, %662: f32, %663: f32):
      %664 = arith.addf %661, %662 : f32
      linalg.yield %664 : f32
    } -> tensor<1x32x1024xf32>
    %665 = tensor.empty() : tensor<1024x4096xf32>
    %666 = linalg.transpose ins(%6:tensor<4096x1024xf32>) outs(%665:tensor<1024x4096xf32>) permutation = [1, 0]
    %667 = tensor.empty() : tensor<1x32x4096xf32>
    %668 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %669 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%668 : f32) outs(%667 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %670 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.mlp.gate_proj"} ins(%660, %666 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%669 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %671 = tensor.empty() : tensor<1x32x4096xf32>
    %672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%670 : tensor<1x32x4096xf32>) outs(%671 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.mlp.act_fn"} {
    ^bb68(%673: f32, %674: f32):
      %675 = arith.constant 1.000000e+00 : f32
      %676 = arith.negf %673 : f32
      %677 = math.exp %676 : f32
      %678 = arith.addf %675, %677 : f32
      %679 = arith.divf %675, %678 : f32
      %680 = arith.mulf %673, %679 : f32
      linalg.yield %680 : f32
    } -> tensor<1x32x4096xf32>
    %681 = tensor.empty() : tensor<1024x4096xf32>
    %682 = linalg.transpose ins(%7:tensor<4096x1024xf32>) outs(%681:tensor<1024x4096xf32>) permutation = [1, 0]
    %683 = tensor.empty() : tensor<1x32x4096xf32>
    %684 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %685 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%684 : f32) outs(%683 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %686 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.mlp.up_proj"} ins(%660, %682 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%685 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %687 = tensor.empty() : tensor<1x32x4096xf32>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%672, %686 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%687 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.mlp"} {
    ^bb69(%689: f32, %690: f32, %691: f32):
      %692 = arith.mulf %689, %690 : f32
      linalg.yield %692 : f32
    } -> tensor<1x32x4096xf32>
    %693 = tensor.empty() : tensor<4096x1024xf32>
    %694 = linalg.transpose ins(%8:tensor<1024x4096xf32>) outs(%693:tensor<4096x1024xf32>) permutation = [1, 0]
    %695 = tensor.empty() : tensor<1x32x1024xf32>
    %696 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %697 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%696 : f32) outs(%695 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %698 = linalg.matmul {prov.region_id = "matmul_12", prov.dispatch_id = "matmul_12", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.mlp.down_proj"} ins(%688, %694 : tensor<1x32x4096xf32>, tensor<4096x1024xf32>) outs(%697 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %699 = tensor.empty() : tensor<1x32x1024xf32>
    %700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%218, %698 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%699 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb70(%701: f32, %702: f32, %703: f32):
      %704 = arith.mulf %701, %702 : f32
      linalg.yield %704 : f32
    } -> tensor<1x32x1024xf32>
    %705 = tensor.empty() : tensor<1x32x1024xf32>
    %706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%593, %700 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%705 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0"} {
    ^bb71(%707: f32, %708: f32, %709: f32):
      %710 = arith.addf %707, %708 : f32
      linalg.yield %710 : f32
    } -> tensor<1x32x1024xf32>
    %711 = tensor.empty() : tensor<1x32x1024xf32>
    %712 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706 : tensor<1x32x1024xf32>) outs(%711 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb72(%713: f32, %714: f32):
      %715 = arith.constant 2.000000e+00 : f32
      %716 = math.powf %713, %715 : f32
      linalg.yield %716 : f32
    } -> tensor<1x32x1024xf32>
    %717 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} 0.000000e+00 : f32
    %718 = tensor.splat %717 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} : tensor<1x32xf32>
    %719 = linalg.reduce ins(%712:tensor<1x32x1024xf32>) outs(%718:tensor<1x32xf32>) dimensions = [2]
    (%720: f32, %721: f32) {
      %722 = arith.addf %720, %721 : f32
      linalg.yield %722 : f32
    }
    %723 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} 1.024000e+03 : f32
    %724 = tensor.splat %723 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} : tensor<1x32xf32>
    %725 = tensor.empty() : tensor<1x32xf32>
    %726 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%719, %724 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%725 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb73(%727: f32, %728: f32, %729: f32):
      %730 = arith.divf %727, %728 : f32
      linalg.yield %730 : f32
    } -> tensor<1x32xf32>
    %731 = tensor.collapse_shape %726 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %732 = tensor.expand_shape %731 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %733 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} 1.000000e-06 : f32
    %734 = tensor.splat %733 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} : tensor<1x32x1xf32>
    %735 = tensor.empty() : tensor<1x32x1xf32>
    %736 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%732, %734 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%735 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb74(%737: f32, %738: f32, %739: f32):
      %740 = arith.addf %737, %738 : f32
      linalg.yield %740 : f32
    } -> tensor<1x32x1xf32>
    %741 = tensor.empty() : tensor<1x32x1xf32>
    %742 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%736 : tensor<1x32x1xf32>) outs(%741 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb75(%743: f32, %744: f32):
      %745 = math.rsqrt %743 : f32
      linalg.yield %745 : f32
    } -> tensor<1x32x1xf32>
    %746 = tensor.empty() : tensor<1x32x1024xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706, %742 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%746 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb76(%748: f32, %749: f32, %750: f32):
      %751 = arith.mulf %748, %749 : f32
      linalg.yield %751 : f32
    } -> tensor<1x32x1024xf32>
    %752 = tensor.empty() : tensor<1x32x1024xf32>
    %753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %747 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%752 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.0.final_layernorm"} {
    ^bb77(%754: f32, %755: f32, %756: f32):
      %757 = arith.mulf %754, %755 : f32
      linalg.yield %757 : f32
    } -> tensor<1x32x1024xf32>
    %758 = tensor.collapse_shape %13 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : tensor<6x1024xf32> into tensor<6144xf32>
    %759 = tensor.expand_shape %758 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : tensor<6144xf32> into tensor<1x6x1024xf32>
    %760 = tensor.empty() : tensor<1x6x1024xf32>
    %761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%759, %171 : tensor<1x6x1024xf32>, tensor<1x6x1024xf32>) outs(%760 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb78(%762: f32, %763: f32, %764: f32):
      %765 = arith.addf %762, %763 : f32
      linalg.yield %765 : f32
    } -> tensor<1x6x1024xf32>
    %766 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %767 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %768 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %769 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %770 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 4, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %771 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 5, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
    %772 = tensor.empty() : tensor<1x32x1024xf32>
    %773 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%753 : tensor<1x32x1024xf32>) outs(%772 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb79(%774: f32, %775: f32):
      %776 = arith.constant 2.000000e+00 : f32
      %777 = math.powf %774, %776 : f32
      linalg.yield %777 : f32
    } -> tensor<1x32x1024xf32>
    %778 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} 0.000000e+00 : f32
    %779 = tensor.splat %778 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} : tensor<1x32xf32>
    %780 = linalg.reduce ins(%773:tensor<1x32x1024xf32>) outs(%779:tensor<1x32xf32>) dimensions = [2]
    (%781: f32, %782: f32) {
      %783 = arith.addf %781, %782 : f32
      linalg.yield %783 : f32
    }
    %784 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} 1.024000e+03 : f32
    %785 = tensor.splat %784 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} : tensor<1x32xf32>
    %786 = tensor.empty() : tensor<1x32xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%780, %785 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%786 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb80(%788: f32, %789: f32, %790: f32):
      %791 = arith.divf %788, %789 : f32
      linalg.yield %791 : f32
    } -> tensor<1x32xf32>
    %792 = tensor.collapse_shape %787 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %793 = tensor.expand_shape %792 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %794 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} 1.000000e-06 : f32
    %795 = tensor.splat %794 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} : tensor<1x32x1xf32>
    %796 = tensor.empty() : tensor<1x32x1xf32>
    %797 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793, %795 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%796 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb81(%798: f32, %799: f32, %800: f32):
      %801 = arith.addf %798, %799 : f32
      linalg.yield %801 : f32
    } -> tensor<1x32x1xf32>
    %802 = tensor.empty() : tensor<1x32x1xf32>
    %803 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%797 : tensor<1x32x1xf32>) outs(%802 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb82(%804: f32, %805: f32):
      %806 = math.rsqrt %804 : f32
      linalg.yield %806 : f32
    } -> tensor<1x32x1xf32>
    %807 = tensor.empty() : tensor<1x32x1024xf32>
    %808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%753, %803 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%807 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb83(%809: f32, %810: f32, %811: f32):
      %812 = arith.mulf %809, %810 : f32
      linalg.yield %812 : f32
    } -> tensor<1x32x1024xf32>
    %813 = tensor.empty() : tensor<1x32x1024xf32>
    %814 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %808 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%813 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.input_layernorm"} {
    ^bb84(%815: f32, %816: f32, %817: f32):
      %818 = arith.mulf %815, %816 : f32
      linalg.yield %818 : f32
    } -> tensor<1x32x1024xf32>
    %819 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} 1.000000e+00 : f32
    %820 = tensor.splat %819 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : tensor<1x1x1024xf32>
    %821 = tensor.empty() : tensor<1x1x1024xf32>
    %822 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%767, %820 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%821 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb85(%823: f32, %824: f32, %825: f32):
      %826 = arith.addf %823, %824 : f32
      linalg.yield %826 : f32
    } -> tensor<1x1x1024xf32>
    %827 = tensor.empty() : tensor<1x32x1024xf32>
    %828 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%814, %822 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%827 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb86(%829: f32, %830: f32, %831: f32):
      %832 = arith.mulf %829, %830 : f32
      linalg.yield %832 : f32
    } -> tensor<1x32x1024xf32>
    %833 = tensor.empty() : tensor<1x32x1024xf32>
    %834 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%828, %766 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%833 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb87(%835: f32, %836: f32, %837: f32):
      %838 = arith.addf %835, %836 : f32
      linalg.yield %838 : f32
    } -> tensor<1x32x1024xf32>
    %839 = tensor.empty() : tensor<1024x3072xf32>
    %840 = linalg.transpose ins(%14:tensor<3072x1024xf32>) outs(%839:tensor<1024x3072xf32>) permutation = [1, 0]
    %841 = tensor.empty() : tensor<1x32x3072xf32>
    %842 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %843 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%842 : f32) outs(%841 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
    %844 = linalg.matmul {prov.region_id = "matmul_13", prov.dispatch_id = "matmul_13", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.qkv_proj"} ins(%834, %840 : tensor<1x32x1024xf32>, tensor<1024x3072xf32>) outs(%843 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
    %845 = tensor.empty() : tensor<1x32x3072xf32>
    %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%844, %15 : tensor<1x32x3072xf32>, tensor<3072xf32>) outs(%845 : tensor<1x32x3072xf32>) attrs =  {prov.region_id = "add_21", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.qkv_proj"} {
    ^bb88(%847: f32, %848: f32, %849: f32):
      %850 = arith.addf %847, %848 : f32
      linalg.yield %850 : f32
    } -> tensor<1x32x3072xf32>
    %851 = tensor.collapse_shape %846 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x3072xf32> into tensor<98304xf32>
    %852 = tensor.expand_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 3, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<98304xf32> into tensor<1x32x3x8x128xf32>
    %853 = "tensor.extract_slice"(%852) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %854 = tensor.collapse_shape %853 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %855 = tensor.expand_shape %854 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %856 = "tensor.extract_slice"(%852) <{static_offsets = array<i64: 0, 0, 1, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %857 = tensor.collapse_shape %856 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %858 = tensor.expand_shape %857 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %859 = "tensor.extract_slice"(%852) <{static_offsets = array<i64: 0, 0, 2, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
    %860 = tensor.collapse_shape %859 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
    %861 = tensor.expand_shape %860 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
    %862 = tensor.empty() : tensor<1x32x8x128xf32>
    %863 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%855 : tensor<1x32x8x128xf32>) outs(%862 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb89(%864: f32, %865: f32):
      %866 = arith.constant 2.000000e+00 : f32
      %867 = math.powf %864, %866 : f32
      linalg.yield %867 : f32
    } -> tensor<1x32x8x128xf32>
    %868 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} 0.000000e+00 : f32
    %869 = tensor.splat %868 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} : tensor<1x32x8xf32>
    %870 = linalg.reduce ins(%863:tensor<1x32x8x128xf32>) outs(%869:tensor<1x32x8xf32>) dimensions = [3]
    (%871: f32, %872: f32) {
      %873 = arith.addf %871, %872 : f32
      linalg.yield %873 : f32
    }
    %874 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} 1.280000e+02 : f32
    %875 = tensor.splat %874 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} : tensor<1x32x8xf32>
    %876 = tensor.empty() : tensor<1x32x8xf32>
    %877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%870, %875 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%876 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb90(%878: f32, %879: f32, %880: f32):
      %881 = arith.divf %878, %879 : f32
      linalg.yield %881 : f32
    } -> tensor<1x32x8xf32>
    %882 = tensor.collapse_shape %877 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} : tensor<1x32x8xf32> into tensor<256xf32>
    %883 = tensor.expand_shape %882 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %884 = arith.constant {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} 1.000000e-06 : f32
    %885 = tensor.splat %884 {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} : tensor<1x32x8x1xf32>
    %886 = tensor.empty() : tensor<1x32x8x1xf32>
    %887 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%883, %885 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%886 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb91(%888: f32, %889: f32, %890: f32):
      %891 = arith.addf %888, %889 : f32
      linalg.yield %891 : f32
    } -> tensor<1x32x8x1xf32>
    %892 = tensor.empty() : tensor<1x32x8x1xf32>
    %893 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%887 : tensor<1x32x8x1xf32>) outs(%892 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb92(%894: f32, %895: f32):
      %896 = math.rsqrt %894 : f32
      linalg.yield %896 : f32
    } -> tensor<1x32x8x1xf32>
    %897 = tensor.empty() : tensor<1x32x8x128xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%855, %893 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%897 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb93(%899: f32, %900: f32, %901: f32):
      %902 = arith.mulf %899, %900 : f32
      linalg.yield %902 : f32
    } -> tensor<1x32x8x128xf32>
    %903 = tensor.empty() : tensor<1x32x8x128xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %898 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%903 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.q_norm"} {
    ^bb94(%905: f32, %906: f32, %907: f32):
      %908 = arith.mulf %905, %906 : f32
      linalg.yield %908 : f32
    } -> tensor<1x32x8x128xf32>
    %909 = tensor.empty() : tensor<1x32x8x128xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858 : tensor<1x32x8x128xf32>) outs(%909 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb95(%911: f32, %912: f32):
      %913 = arith.constant 2.000000e+00 : f32
      %914 = math.powf %911, %913 : f32
      linalg.yield %914 : f32
    } -> tensor<1x32x8x128xf32>
    %915 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} 0.000000e+00 : f32
    %916 = tensor.splat %915 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} : tensor<1x32x8xf32>
    %917 = linalg.reduce ins(%910:tensor<1x32x8x128xf32>) outs(%916:tensor<1x32x8xf32>) dimensions = [3]
    (%918: f32, %919: f32) {
      %920 = arith.addf %918, %919 : f32
      linalg.yield %920 : f32
    }
    %921 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} 1.280000e+02 : f32
    %922 = tensor.splat %921 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} : tensor<1x32x8xf32>
    %923 = tensor.empty() : tensor<1x32x8xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%917, %922 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%923 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb96(%925: f32, %926: f32, %927: f32):
      %928 = arith.divf %925, %926 : f32
      linalg.yield %928 : f32
    } -> tensor<1x32x8xf32>
    %929 = tensor.collapse_shape %924 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} : tensor<1x32x8xf32> into tensor<256xf32>
    %930 = tensor.expand_shape %929 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %931 = arith.constant {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} 1.000000e-06 : f32
    %932 = tensor.splat %931 {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} : tensor<1x32x8x1xf32>
    %933 = tensor.empty() : tensor<1x32x8x1xf32>
    %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%930, %932 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%933 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb97(%935: f32, %936: f32, %937: f32):
      %938 = arith.addf %935, %936 : f32
      linalg.yield %938 : f32
    } -> tensor<1x32x8x1xf32>
    %939 = tensor.empty() : tensor<1x32x8x1xf32>
    %940 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%934 : tensor<1x32x8x1xf32>) outs(%939 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb98(%941: f32, %942: f32):
      %943 = math.rsqrt %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x32x8x1xf32>
    %944 = tensor.empty() : tensor<1x32x8x128xf32>
    %945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%858, %940 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%944 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb99(%946: f32, %947: f32, %948: f32):
      %949 = arith.mulf %946, %947 : f32
      linalg.yield %949 : f32
    } -> tensor<1x32x8x128xf32>
    %950 = tensor.empty() : tensor<1x32x8x128xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %945 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%950 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.k_norm"} {
    ^bb100(%952: f32, %953: f32, %954: f32):
      %955 = arith.mulf %952, %953 : f32
      linalg.yield %955 : f32
    } -> tensor<1x32x8x128xf32>
    %956 = tensor.empty() : tensor<1x8x32x128xf32>
    %957 = linalg.transpose ins(%904:tensor<1x32x8x128xf32>) outs(%956:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %958 = tensor.empty() : tensor<1x8x32x128xf32>
    %959 = linalg.transpose ins(%951:tensor<1x32x8x128xf32>) outs(%958:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %960 = tensor.empty() : tensor<1x8x32x128xf32>
    %961 = linalg.transpose ins(%861:tensor<1x32x8x128xf32>) outs(%960:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
    %962 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %963 = tensor.expand_shape %962 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
    %964 = tensor.collapse_shape %43 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %965 = tensor.expand_shape %964 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
    %966 = tensor.empty() : tensor<1x8x32x128xf32>
    %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%957, %963 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%966 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb101(%968: f32, %969: f32, %970: f32):
      %971 = arith.mulf %968, %969 : f32
      linalg.yield %971 : f32
    } -> tensor<1x8x32x128xf32>
    %972 = "tensor.extract_slice"(%957) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %973 = "tensor.extract_slice"(%957) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %974 = tensor.empty() : tensor<1x8x32x64xf32>
    %975 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%973 : tensor<1x8x32x64xf32>) outs(%974 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb102(%976: f32, %977: f32):
      %978 = arith.negf %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x8x32x64xf32>
    %979 = tensor.concat dim(3) %975, %972 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
    %980 = tensor.empty() : tensor<1x8x32x128xf32>
    %981 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%979, %965 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%980 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb103(%982: f32, %983: f32, %984: f32):
      %985 = arith.mulf %982, %983 : f32
      linalg.yield %985 : f32
    } -> tensor<1x8x32x128xf32>
    %986 = tensor.empty() : tensor<1x8x32x128xf32>
    %987 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%967, %981 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%986 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb104(%988: f32, %989: f32, %990: f32):
      %991 = arith.addf %988, %989 : f32
      linalg.yield %991 : f32
    } -> tensor<1x8x32x128xf32>
    %992 = tensor.empty() : tensor<1x8x32x128xf32>
    %993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%959, %963 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%992 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb105(%994: f32, %995: f32, %996: f32):
      %997 = arith.mulf %994, %995 : f32
      linalg.yield %997 : f32
    } -> tensor<1x8x32x128xf32>
    %998 = "tensor.extract_slice"(%959) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %999 = "tensor.extract_slice"(%959) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
    %1000 = tensor.empty() : tensor<1x8x32x64xf32>
    %1001 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%999 : tensor<1x8x32x64xf32>) outs(%1000 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb106(%1002: f32, %1003: f32):
      %1004 = arith.negf %1002 : f32
      linalg.yield %1004 : f32
    } -> tensor<1x8x32x64xf32>
    %1005 = tensor.concat dim(3) %1001, %998 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
    %1006 = tensor.empty() : tensor<1x8x32x128xf32>
    %1007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1005, %965 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%1006 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb107(%1008: f32, %1009: f32, %1010: f32):
      %1011 = arith.mulf %1008, %1009 : f32
      linalg.yield %1011 : f32
    } -> tensor<1x8x32x128xf32>
    %1012 = tensor.empty() : tensor<1x8x32x128xf32>
    %1013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%993, %1007 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%1012 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb108(%1014: f32, %1015: f32, %1016: f32):
      %1017 = arith.addf %1014, %1015 : f32
      linalg.yield %1017 : f32
    } -> tensor<1x8x32x128xf32>
    %1018 = tensor.concat dim(2) %47, %1013 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
    %1019 = tensor.concat dim(2) %48, %961 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
    %1020 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} 0.000000e+00 : f32
    %1021 = tensor.splat %1020 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32x48xf32>
    %1022 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%987, %1018 : tensor<1x8x32x128xf32>, tensor<1x8x48x128xf32>) outs(%1021 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb109(%1023: f32, %1024: f32, %1025: f32):
      %1026 = arith.mulf %1023, %1024 : f32
      %1027 = arith.addf %1025, %1026 : f32
      linalg.yield %1027 : f32
    } -> tensor<1x8x32x48xf32>
    %1028 = tensor.empty() : tensor<1x8x32x48xf32>
    %1029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1022 : tensor<1x8x32x48xf32>) outs(%1028 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb110(%1030: f32, %1031: f32):
      %1032 = arith.constant 0.0883883461 : f32
      %1033 = arith.mulf %1030, %1032 : f32
      linalg.yield %1033 : f32
    } -> tensor<1x8x32x48xf32>
    %1034 = tensor.empty() : tensor<1x8x32x48xf32>
    %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1029, %44 : tensor<1x8x32x48xf32>, tensor<1x1x32x48xi1>) outs(%1034 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb111(%1036: f32, %1037: i1, %1038: f32):
      %1039 = arith.constant 0xff800000 : f32
      %1040 = arith.select %1037, %1036, %1039 : f32
      linalg.yield %1040 : f32
    } -> tensor<1x8x32x48xf32>
    %1041 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} 0xff800000 : f32
    %1042 = tensor.splat %1041 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32xf32>
    %1043 = linalg.reduce ins(%1035:tensor<1x8x32x48xf32>) outs(%1042:tensor<1x8x32xf32>) dimensions = [3]
    (%1044: f32, %1045: f32) {
      %1046 = arith.maximumf %1044, %1045 : f32
      linalg.yield %1046 : f32
    }
    %1047 = tensor.collapse_shape %1043 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1048 = tensor.expand_shape %1047 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1049 = tensor.empty() : tensor<1x8x32x48xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1035, %1048 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%1049 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb112(%1051: f32, %1052: f32, %1053: f32):
      %1054 = arith.subf %1051, %1052 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x8x32x48xf32>
    %1055 = tensor.empty() : tensor<1x8x32x48xf32>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1050 : tensor<1x8x32x48xf32>) outs(%1055 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb113(%1057: f32, %1058: f32):
      %1059 = math.exp %1057 : f32
      linalg.yield %1059 : f32
    } -> tensor<1x8x32x48xf32>
    %1060 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} 0.000000e+00 : f32
    %1061 = tensor.splat %1060 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32xf32>
    %1062 = linalg.reduce ins(%1056:tensor<1x8x32x48xf32>) outs(%1061:tensor<1x8x32xf32>) dimensions = [3]
    (%1063: f32, %1064: f32) {
      %1065 = arith.addf %1063, %1064 : f32
      linalg.yield %1065 : f32
    }
    %1066 = tensor.collapse_shape %1062 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1067 = tensor.expand_shape %1066 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1068 = tensor.empty() : tensor<1x8x32x48xf32>
    %1069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1056, %1067 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%1068 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb114(%1070: f32, %1071: f32, %1072: f32):
      %1073 = arith.divf %1070, %1071 : f32
      linalg.yield %1073 : f32
    } -> tensor<1x8x32x48xf32>
    %1074 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} 0.000000e+00 : f32
    %1075 = tensor.splat %1074 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x8x32x128xf32>
    %1076 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1069, %1019 : tensor<1x8x32x48xf32>, tensor<1x8x48x128xf32>) outs(%1075 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} {
    ^bb115(%1077: f32, %1078: f32, %1079: f32):
      %1080 = arith.mulf %1077, %1078 : f32
      %1081 = arith.addf %1079, %1080 : f32
      linalg.yield %1081 : f32
    } -> tensor<1x8x32x128xf32>
    %1082 = tensor.empty() : tensor<1x32x8x128xf32>
    %1083 = linalg.transpose ins(%1076:tensor<1x8x32x128xf32>) outs(%1082:tensor<1x32x8x128xf32>) permutation = [0, 2, 1, 3]
    %1084 = tensor.collapse_shape %1083 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<1x32x8x128xf32> into tensor<32768xf32>
    %1085 = tensor.expand_shape %1084 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn"} : tensor<32768xf32> into tensor<1x32x1024xf32>
    %1086 = tensor.empty() : tensor<1024x1024xf32>
    %1087 = linalg.transpose ins(%16:tensor<1024x1024xf32>) outs(%1086:tensor<1024x1024xf32>) permutation = [1, 0]
    %1088 = tensor.empty() : tensor<1x32x1024xf32>
    %1089 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1090 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1089 : f32) outs(%1088 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %1091 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.attn.o_proj"} ins(%1085, %1087 : tensor<1x32x1024xf32>, tensor<1024x1024xf32>) outs(%1090 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %1092 = tensor.empty() : tensor<1x32x1024xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%768, %1091 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%1092 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb116(%1094: f32, %1095: f32, %1096: f32):
      %1097 = arith.mulf %1094, %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x32x1024xf32>
    %1098 = tensor.empty() : tensor<1x32x1024xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%753, %1093 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%1098 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb117(%1100: f32, %1101: f32, %1102: f32):
      %1103 = arith.addf %1100, %1101 : f32
      linalg.yield %1103 : f32
    } -> tensor<1x32x1024xf32>
    %1104 = tensor.empty() : tensor<1x32x1024xf32>
    %1105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099 : tensor<1x32x1024xf32>) outs(%1104 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb118(%1106: f32, %1107: f32):
      %1108 = arith.constant 2.000000e+00 : f32
      %1109 = math.powf %1106, %1108 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x32x1024xf32>
    %1110 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} 0.000000e+00 : f32
    %1111 = tensor.splat %1110 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} : tensor<1x32xf32>
    %1112 = linalg.reduce ins(%1105:tensor<1x32x1024xf32>) outs(%1111:tensor<1x32xf32>) dimensions = [2]
    (%1113: f32, %1114: f32) {
      %1115 = arith.addf %1113, %1114 : f32
      linalg.yield %1115 : f32
    }
    %1116 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} 1.024000e+03 : f32
    %1117 = tensor.splat %1116 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} : tensor<1x32xf32>
    %1118 = tensor.empty() : tensor<1x32xf32>
    %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1112, %1117 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1118 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb119(%1120: f32, %1121: f32, %1122: f32):
      %1123 = arith.divf %1120, %1121 : f32
      linalg.yield %1123 : f32
    } -> tensor<1x32xf32>
    %1124 = tensor.collapse_shape %1119 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1125 = tensor.expand_shape %1124 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1126 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} 1.000000e-06 : f32
    %1127 = tensor.splat %1126 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} : tensor<1x32x1xf32>
    %1128 = tensor.empty() : tensor<1x32x1xf32>
    %1129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1125, %1127 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1128 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb120(%1130: f32, %1131: f32, %1132: f32):
      %1133 = arith.addf %1130, %1131 : f32
      linalg.yield %1133 : f32
    } -> tensor<1x32x1xf32>
    %1134 = tensor.empty() : tensor<1x32x1xf32>
    %1135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1129 : tensor<1x32x1xf32>) outs(%1134 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_9", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb121(%1136: f32, %1137: f32):
      %1138 = math.rsqrt %1136 : f32
      linalg.yield %1138 : f32
    } -> tensor<1x32x1xf32>
    %1139 = tensor.empty() : tensor<1x32x1024xf32>
    %1140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1099, %1135 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1139 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb122(%1141: f32, %1142: f32, %1143: f32):
      %1144 = arith.mulf %1141, %1142 : f32
      linalg.yield %1144 : f32
    } -> tensor<1x32x1024xf32>
    %1145 = tensor.empty() : tensor<1x32x1024xf32>
    %1146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %1140 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1145 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.middle_layernorm"} {
    ^bb123(%1147: f32, %1148: f32, %1149: f32):
      %1150 = arith.mulf %1147, %1148 : f32
      linalg.yield %1150 : f32
    } -> tensor<1x32x1024xf32>
    %1151 = tensor.empty() : tensor<1x32x1024xf32>
    %1152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1146 : tensor<1x32x1024xf32>) outs(%1151 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb124(%1153: f32, %1154: f32):
      %1155 = arith.constant 2.000000e+00 : f32
      %1156 = math.powf %1153, %1155 : f32
      linalg.yield %1156 : f32
    } -> tensor<1x32x1024xf32>
    %1157 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} 0.000000e+00 : f32
    %1158 = tensor.splat %1157 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} : tensor<1x32xf32>
    %1159 = linalg.reduce ins(%1152:tensor<1x32x1024xf32>) outs(%1158:tensor<1x32xf32>) dimensions = [2]
    (%1160: f32, %1161: f32) {
      %1162 = arith.addf %1160, %1161 : f32
      linalg.yield %1162 : f32
    }
    %1163 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} 1.024000e+03 : f32
    %1164 = tensor.splat %1163 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} : tensor<1x32xf32>
    %1165 = tensor.empty() : tensor<1x32xf32>
    %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1159, %1164 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1165 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb125(%1167: f32, %1168: f32, %1169: f32):
      %1170 = arith.divf %1167, %1168 : f32
      linalg.yield %1170 : f32
    } -> tensor<1x32xf32>
    %1171 = tensor.collapse_shape %1166 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1172 = tensor.expand_shape %1171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1173 = arith.constant {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} 1.000000e-06 : f32
    %1174 = tensor.splat %1173 {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} : tensor<1x32x1xf32>
    %1175 = tensor.empty() : tensor<1x32x1xf32>
    %1176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1172, %1174 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1175 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb126(%1177: f32, %1178: f32, %1179: f32):
      %1180 = arith.addf %1177, %1178 : f32
      linalg.yield %1180 : f32
    } -> tensor<1x32x1xf32>
    %1181 = tensor.empty() : tensor<1x32x1xf32>
    %1182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1176 : tensor<1x32x1xf32>) outs(%1181 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_10", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb127(%1183: f32, %1184: f32):
      %1185 = math.rsqrt %1183 : f32
      linalg.yield %1185 : f32
    } -> tensor<1x32x1xf32>
    %1186 = tensor.empty() : tensor<1x32x1024xf32>
    %1187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1146, %1182 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1186 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb128(%1188: f32, %1189: f32, %1190: f32):
      %1191 = arith.mulf %1188, %1189 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x32x1024xf32>
    %1192 = tensor.empty() : tensor<1x32x1024xf32>
    %1193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %1187 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1192 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.post_layernorm"} {
    ^bb129(%1194: f32, %1195: f32, %1196: f32):
      %1197 = arith.mulf %1194, %1195 : f32
      linalg.yield %1197 : f32
    } -> tensor<1x32x1024xf32>
    %1198 = arith.constant {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} 1.000000e+00 : f32
    %1199 = tensor.splat %1198 {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} : tensor<1x1x1024xf32>
    %1200 = tensor.empty() : tensor<1x1x1024xf32>
    %1201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%770, %1199 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1200 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb130(%1202: f32, %1203: f32, %1204: f32):
      %1205 = arith.addf %1202, %1203 : f32
      linalg.yield %1205 : f32
    } -> tensor<1x1x1024xf32>
    %1206 = tensor.empty() : tensor<1x32x1024xf32>
    %1207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1193, %1201 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%1206 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb131(%1208: f32, %1209: f32, %1210: f32):
      %1211 = arith.mulf %1208, %1209 : f32
      linalg.yield %1211 : f32
    } -> tensor<1x32x1024xf32>
    %1212 = tensor.empty() : tensor<1x32x1024xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1207, %769 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%1212 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb132(%1214: f32, %1215: f32, %1216: f32):
      %1217 = arith.addf %1214, %1215 : f32
      linalg.yield %1217 : f32
    } -> tensor<1x32x1024xf32>
    %1218 = tensor.empty() : tensor<1024x4096xf32>
    %1219 = linalg.transpose ins(%19:tensor<4096x1024xf32>) outs(%1218:tensor<1024x4096xf32>) permutation = [1, 0]
    %1220 = tensor.empty() : tensor<1x32x4096xf32>
    %1221 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1222 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1221 : f32) outs(%1220 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %1223 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.mlp.gate_proj"} ins(%1213, %1219 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%1222 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %1224 = tensor.empty() : tensor<1x32x4096xf32>
    %1225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1223 : tensor<1x32x4096xf32>) outs(%1224 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "silu_2", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.mlp.act_fn"} {
    ^bb133(%1226: f32, %1227: f32):
      %1228 = arith.constant 1.000000e+00 : f32
      %1229 = arith.negf %1226 : f32
      %1230 = math.exp %1229 : f32
      %1231 = arith.addf %1228, %1230 : f32
      %1232 = arith.divf %1228, %1231 : f32
      %1233 = arith.mulf %1226, %1232 : f32
      linalg.yield %1233 : f32
    } -> tensor<1x32x4096xf32>
    %1234 = tensor.empty() : tensor<1024x4096xf32>
    %1235 = linalg.transpose ins(%20:tensor<4096x1024xf32>) outs(%1234:tensor<1024x4096xf32>) permutation = [1, 0]
    %1236 = tensor.empty() : tensor<1x32x4096xf32>
    %1237 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1238 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1237 : f32) outs(%1236 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %1239 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.mlp.up_proj"} ins(%1213, %1235 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%1238 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
    %1240 = tensor.empty() : tensor<1x32x4096xf32>
    %1241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1225, %1239 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%1240 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.mlp"} {
    ^bb134(%1242: f32, %1243: f32, %1244: f32):
      %1245 = arith.mulf %1242, %1243 : f32
      linalg.yield %1245 : f32
    } -> tensor<1x32x4096xf32>
    %1246 = tensor.empty() : tensor<4096x1024xf32>
    %1247 = linalg.transpose ins(%21:tensor<1024x4096xf32>) outs(%1246:tensor<4096x1024xf32>) permutation = [1, 0]
    %1248 = tensor.empty() : tensor<1x32x1024xf32>
    %1249 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1250 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1249 : f32) outs(%1248 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %1251 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.mlp.down_proj"} ins(%1241, %1247 : tensor<1x32x4096xf32>, tensor<4096x1024xf32>) outs(%1250 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %1252 = tensor.empty() : tensor<1x32x1024xf32>
    %1253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%771, %1251 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%1252 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb135(%1254: f32, %1255: f32, %1256: f32):
      %1257 = arith.mulf %1254, %1255 : f32
      linalg.yield %1257 : f32
    } -> tensor<1x32x1024xf32>
    %1258 = tensor.empty() : tensor<1x32x1024xf32>
    %1259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1146, %1253 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%1258 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1"} {
    ^bb136(%1260: f32, %1261: f32, %1262: f32):
      %1263 = arith.addf %1260, %1261 : f32
      linalg.yield %1263 : f32
    } -> tensor<1x32x1024xf32>
    %1264 = tensor.empty() : tensor<1x32x1024xf32>
    %1265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1259 : tensor<1x32x1024xf32>) outs(%1264 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_11", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb137(%1266: f32, %1267: f32):
      %1268 = arith.constant 2.000000e+00 : f32
      %1269 = math.powf %1266, %1268 : f32
      linalg.yield %1269 : f32
    } -> tensor<1x32x1024xf32>
    %1270 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} 0.000000e+00 : f32
    %1271 = tensor.splat %1270 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} : tensor<1x32xf32>
    %1272 = linalg.reduce ins(%1265:tensor<1x32x1024xf32>) outs(%1271:tensor<1x32xf32>) dimensions = [2]
    (%1273: f32, %1274: f32) {
      %1275 = arith.addf %1273, %1274 : f32
      linalg.yield %1275 : f32
    }
    %1276 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} 1.024000e+03 : f32
    %1277 = tensor.splat %1276 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} : tensor<1x32xf32>
    %1278 = tensor.empty() : tensor<1x32xf32>
    %1279 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1272, %1277 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1278 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb138(%1280: f32, %1281: f32, %1282: f32):
      %1283 = arith.divf %1280, %1281 : f32
      linalg.yield %1283 : f32
    } -> tensor<1x32xf32>
    %1284 = tensor.collapse_shape %1279 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1285 = tensor.expand_shape %1284 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1286 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} 1.000000e-06 : f32
    %1287 = tensor.splat %1286 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} : tensor<1x32x1xf32>
    %1288 = tensor.empty() : tensor<1x32x1xf32>
    %1289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1285, %1287 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1288 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb139(%1290: f32, %1291: f32, %1292: f32):
      %1293 = arith.addf %1290, %1291 : f32
      linalg.yield %1293 : f32
    } -> tensor<1x32x1xf32>
    %1294 = tensor.empty() : tensor<1x32x1xf32>
    %1295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1289 : tensor<1x32x1xf32>) outs(%1294 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_11", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb140(%1296: f32, %1297: f32):
      %1298 = math.rsqrt %1296 : f32
      linalg.yield %1298 : f32
    } -> tensor<1x32x1xf32>
    %1299 = tensor.empty() : tensor<1x32x1024xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1259, %1295 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1299 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb141(%1301: f32, %1302: f32, %1303: f32):
      %1304 = arith.mulf %1301, %1302 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x32x1024xf32>
    %1305 = tensor.empty() : tensor<1x32x1024xf32>
    %1306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %1300 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1305 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.dit.layers.1.final_layernorm"} {
    ^bb142(%1307: f32, %1308: f32, %1309: f32):
      %1310 = arith.mulf %1307, %1308 : f32
      linalg.yield %1310 : f32
    } -> tensor<1x32x1024xf32>
    %1311 = "tensor.extract_slice"(%1306) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %1312 = "tensor.extract_slice"(%1311) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 30, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1024xf32>) -> tensor<1x30x1024xf32>
    %1313 = "tensor.extract_slice"(%1312) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 30, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
    %1314 = tensor.empty() : tensor<1024x32xf32>
    %1315 = linalg.transpose ins(%30:tensor<32x1024xf32>) outs(%1314:tensor<1024x32xf32>) permutation = [1, 0]
    %1316 = tensor.empty() : tensor<1x30x32xf32>
    %1317 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1318 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1317 : f32) outs(%1316 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
    %1319 = linalg.matmul {prov.region_id = "matmul_19", prov.dispatch_id = "matmul_19", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_output_layer.layers.0"} ins(%1313, %1315 : tensor<1x30x1024xf32>, tensor<1024x32xf32>) outs(%1318 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
    %1320 = tensor.empty() : tensor<1x30x32xf32>
    %1321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1319 : tensor<1x30x32xf32>) outs(%1320 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_output_layer.layers.1"} {
    ^bb143(%1322: f32, %1323: f32):
      %1324 = arith.constant 5.000000e-01 : f32
      %1325 = arith.constant 1.000000e+00 : f32
      %1326 = arith.constant 0.707106769 : f32
      %1327 = arith.mulf %1322, %1326 : f32
      %1328 = math.erf %1327 : f32
      %1329 = arith.addf %1325, %1328 : f32
      %1330 = arith.mulf %1324, %1322 : f32
      %1331 = arith.mulf %1330, %1329 : f32
      linalg.yield %1331 : f32
    } -> tensor<1x30x32xf32>
    %1332 = tensor.empty() : tensor<32x32xf32>
    %1333 = linalg.transpose ins(%31:tensor<32x32xf32>) outs(%1332:tensor<32x32xf32>) permutation = [1, 0]
    %1334 = tensor.empty() : tensor<1x30x32xf32>
    %1335 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1336 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1335 : f32) outs(%1334 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
    %1337 = linalg.matmul {prov.region_id = "matmul_20", prov.dispatch_id = "matmul_20", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.action_output_layer.layers.2"} ins(%1321, %1333 : tensor<1x30x32xf32>, tensor<32x32xf32>) outs(%1336 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
    func.return %1337 : tensor<1x30x32xf32>
  }
}
