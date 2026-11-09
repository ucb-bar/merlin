builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "float8_weight_only_e4m3"} {
  func.func @forward(%0: tensor<256x128xf32>, %1: tensor<128xf32>, %2: tensor<128x128xf32>, %3: tensor<128x1xf32>, %4: tensor<128x128xf32>, %5: tensor<128x1xf32>, %6: tensor<128x128xf32>, %7: tensor<128x1xf32>, %8: tensor<128x128xf32>, %9: tensor<128x1xf32>, %10: tensor<128xf32>, %11: tensor<344x128xf32>, %12: tensor<344x1xf32>, %13: tensor<344x128xf32>, %14: tensor<344x1xf32>, %15: tensor<128x344xf32>, %16: tensor<128x1xf32>, %17: tensor<128xf32>, %18: tensor<128x128xf32>, %19: tensor<128x1xf32>, %20: tensor<128x128xf32>, %21: tensor<128x1xf32>, %22: tensor<128x128xf32>, %23: tensor<128x1xf32>, %24: tensor<128x128xf32>, %25: tensor<128x1xf32>, %26: tensor<128xf32>, %27: tensor<344x128xf32>, %28: tensor<344x1xf32>, %29: tensor<344x128xf32>, %30: tensor<344x1xf32>, %31: tensor<128x344xf32>, %32: tensor<128x1xf32>, %33: tensor<128xf32>, %34: tensor<256x128xf32>, %35: tensor<256x1xf32>, %36: tensor<1x8xi64>) -> tensor<1x8x256xf32> {
    %37 = tensor.empty() : tensor<8xi64>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%37 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%39: i64):
      %40 = linalg.index 0 : index
      %41 = arith.index_cast %40 : index to i64
      %42 = arith.constant 1 : i64
      %43 = arith.muli %41, %42 : i64
      %44 = arith.constant 0 : i64
      %45 = arith.addi %44, %43 : i64
      linalg.yield %45 : i64
    } -> tensor<8xi64>
    %46 = tensor.empty() : tensor<1x8x128xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<1x8xi64>) outs(%46 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "emb", prov.fqn = "emb"} {
    ^bb1(%48: i64, %49: f32):
      %50 = arith.index_cast %48 : i64 to index
      %51 = linalg.index 2 : index
      %52 = tensor.extract %0[%50, %51] : tensor<256x128xf32>
      linalg.yield %52 : f32
    } -> tensor<1x8x128xf32>
    %53 = tensor.empty() : tensor<1x8x128xf32>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : tensor<1x8x128xf32>) outs(%53 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb2(%55: f32, %56: f32):
      %57 = arith.constant 2.000000e+00 : f32
      %58 = math.powf %55, %57 : f32
      linalg.yield %58 : f32
    } -> tensor<1x8x128xf32>
    %59 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 0.000000e+00 : f32
    %60 = tensor.splat %59 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %61 = linalg.reduce ins(%54:tensor<1x8x128xf32>) outs(%60:tensor<1x8xf32>) dimensions = [2]
    (%62: f32, %63: f32) {
      %64 = arith.addf %62, %63 : f32
      linalg.yield %64 : f32
    }
    %65 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.280000e+02 : f32
    %66 = tensor.splat %65 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %67 = tensor.empty() : tensor<1x8xf32>
    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %66 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%67 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb3(%69: f32, %70: f32, %71: f32):
      %72 = arith.divf %69, %70 : f32
      linalg.yield %72 : f32
    } -> tensor<1x8xf32>
    %73 = tensor.collapse_shape %68 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %74 = tensor.expand_shape %73 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %75 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.000000e-05 : f32
    %76 = tensor.splat %75 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8x1xf32>
    %77 = tensor.empty() : tensor<1x8x1xf32>
    %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %76 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%77 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb4(%79: f32, %80: f32, %81: f32):
      %82 = arith.addf %79, %80 : f32
      linalg.yield %82 : f32
    } -> tensor<1x8x1xf32>
    %83 = tensor.empty() : tensor<1x8x1xf32>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<1x8x1xf32>) outs(%83 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb5(%85: f32, %86: f32):
      %87 = math.rsqrt %85 : f32
      linalg.yield %87 : f32
    } -> tensor<1x8x1xf32>
    %88 = tensor.empty() : tensor<1x8x128xf32>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %84 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%88 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb6(%90: f32, %91: f32, %92: f32):
      %93 = arith.mulf %90, %91 : f32
      linalg.yield %93 : f32
    } -> tensor<1x8x128xf32>
    %94 = tensor.empty() : tensor<1x8x128xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %1 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%94 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb7(%96: f32, %97: f32, %98: f32):
      %99 = arith.mulf %96, %97 : f32
      linalg.yield %99 : f32
    } -> tensor<1x8x128xf32>
    %100 = tensor.empty() : tensor<128x128xf32>
    %101 = linalg.transpose ins(%2:tensor<128x128xf32>) outs(%100:tensor<128x128xf32>) permutation = [1, 0]
    %102 = tensor.empty() : tensor<1x128xf32>
    %103 = linalg.transpose ins(%3:tensor<128x1xf32>) outs(%102:tensor<1x128xf32>) permutation = [1, 0]
    %104 = tensor.empty() : tensor<128x128xf32>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %103 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%104 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} {
    ^bb8(%106: f32, %107: f32, %108: f32):
      %109 = arith.mulf %106, %107 : f32
      linalg.yield %109 : f32
    } -> tensor<128x128xf32>
    %110 = tensor.collapse_shape %95 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %111 = tensor.expand_shape %110 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %112 = tensor.empty() : tensor<8x128xf32>
    %113 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %114 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%113 : f32) outs(%112 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %115 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} ins(%111, %105 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%114 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %116 = tensor.collapse_shape %115 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %117 = tensor.expand_shape %116 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %118 = tensor.collapse_shape %117 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %119 = tensor.expand_shape %118 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %120 = tensor.empty() : tensor<1x4x8x32xf32>
    %121 = linalg.transpose ins(%119:tensor<1x8x4x32xf32>) outs(%120:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %122 = tensor.empty() : tensor<128x128xf32>
    %123 = linalg.transpose ins(%4:tensor<128x128xf32>) outs(%122:tensor<128x128xf32>) permutation = [1, 0]
    %124 = tensor.empty() : tensor<1x128xf32>
    %125 = linalg.transpose ins(%5:tensor<128x1xf32>) outs(%124:tensor<1x128xf32>) permutation = [1, 0]
    %126 = tensor.empty() : tensor<128x128xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%123, %125 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%126 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} {
    ^bb9(%128: f32, %129: f32, %130: f32):
      %131 = arith.mulf %128, %129 : f32
      linalg.yield %131 : f32
    } -> tensor<128x128xf32>
    %132 = tensor.collapse_shape %95 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %134 = tensor.empty() : tensor<8x128xf32>
    %135 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %136 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%135 : f32) outs(%134 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %137 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} ins(%133, %127 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%136 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %138 = tensor.collapse_shape %137 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %139 = tensor.expand_shape %138 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %140 = tensor.collapse_shape %139 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %141 = tensor.expand_shape %140 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %142 = tensor.empty() : tensor<1x4x8x32xf32>
    %143 = linalg.transpose ins(%141:tensor<1x8x4x32xf32>) outs(%142:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %144 = tensor.empty() : tensor<128x128xf32>
    %145 = linalg.transpose ins(%6:tensor<128x128xf32>) outs(%144:tensor<128x128xf32>) permutation = [1, 0]
    %146 = tensor.empty() : tensor<1x128xf32>
    %147 = linalg.transpose ins(%7:tensor<128x1xf32>) outs(%146:tensor<1x128xf32>) permutation = [1, 0]
    %148 = tensor.empty() : tensor<128x128xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %147 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%148 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} {
    ^bb10(%150: f32, %151: f32, %152: f32):
      %153 = arith.mulf %150, %151 : f32
      linalg.yield %153 : f32
    } -> tensor<128x128xf32>
    %154 = tensor.collapse_shape %95 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %155 = tensor.expand_shape %154 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %156 = tensor.empty() : tensor<8x128xf32>
    %157 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %158 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%157 : f32) outs(%156 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %159 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} ins(%155, %149 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%158 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %160 = tensor.collapse_shape %159 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %161 = tensor.expand_shape %160 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %162 = tensor.collapse_shape %161 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %163 = tensor.expand_shape %162 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %164 = tensor.empty() : tensor<1x4x8x32xf32>
    %165 = linalg.transpose ins(%163:tensor<1x8x4x32xf32>) outs(%164:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %166 = tensor.empty() : tensor<16xf32>
    %167 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%166 : tensor<16xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb11(%168: f32):
      %169 = linalg.index 0 : index
      %170 = arith.index_cast %169 : index to i64
      %171 = arith.sitofp %170 : i64 to f32
      %172 = arith.constant 1.000000e+00 : f32
      %173 = arith.mulf %171, %172 : f32
      %174 = arith.constant 0.000000e+00 : f32
      %175 = arith.addf %174, %173 : f32
      linalg.yield %175 : f32
    } -> tensor<16xf32>
    %176 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %177 = tensor.splat %176 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %178 = tensor.empty() : tensor<16xf32>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%167, %177 : tensor<16xf32>, tensor<16xf32>) outs(%178 : tensor<16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb12(%180: f32, %181: f32, %182: f32):
      %183 = arith.divf %180, %181 : f32
      linalg.yield %183 : f32
    } -> tensor<16xf32>
    %184 = tensor.empty() : tensor<16xf32>
    %185 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%179 : tensor<16xf32>) outs(%184 : tensor<16xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb13(%186: f32, %187: f32):
      %188 = arith.constant 1.000000e+04 : f32
      %189 = math.powf %188, %186 : f32
      linalg.yield %189 : f32
    } -> tensor<16xf32>
    %190 = tensor.empty() : tensor<16xf32>
    %191 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%185 : tensor<16xf32>) outs(%190 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb14(%192: f32, %193: f32):
      %194 = arith.constant 1.000000e+00 : f32
      %195 = arith.divf %194, %192 : f32
      linalg.yield %195 : f32
    } -> tensor<16xf32>
    %196 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %197 = tensor.splat %196 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %198 = tensor.empty() : tensor<16xf32>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%191, %197 : tensor<16xf32>, tensor<16xf32>) outs(%198 : tensor<16xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb15(%200: f32, %201: f32, %202: f32):
      %203 = arith.mulf %200, %201 : f32
      linalg.yield %203 : f32
    } -> tensor<16xf32>
    %204 = tensor.expand_shape %38 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %205 = tensor.empty() : tensor<8x1xf32>
    %206 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%204 : tensor<8x1xi64>) outs(%205 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb16(%207: i64, %208: f32):
      %209 = arith.sitofp %207 : i64 to f32
      linalg.yield %209 : f32
    } -> tensor<8x1xf32>
    %210 = tensor.expand_shape %199 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %211 = tensor.empty() : tensor<8x16xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%206, %210 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%211 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb17(%213: f32, %214: f32, %215: f32):
      %216 = arith.mulf %213, %214 : f32
      linalg.yield %216 : f32
    } -> tensor<8x16xf32>
    %217 = tensor.empty() : tensor<8x16xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212 : tensor<8x16xf32>) outs(%217 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb18(%219: f32, %220: f32):
      %221 = math.cos %219 : f32
      linalg.yield %221 : f32
    } -> tensor<8x16xf32>
    %222 = tensor.empty() : tensor<8x16xf32>
    %223 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212 : tensor<8x16xf32>) outs(%222 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb19(%224: f32, %225: f32):
      %226 = math.cos %224 : f32
      linalg.yield %226 : f32
    } -> tensor<8x16xf32>
    %227 = tensor.concat dim(1) %218, %223 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %228 = tensor.collapse_shape %227 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %229 = tensor.expand_shape %228 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %230 = tensor.collapse_shape %229 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %231 = tensor.expand_shape %230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %232 = tensor.empty() : tensor<8x16xf32>
    %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212 : tensor<8x16xf32>) outs(%232 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb20(%234: f32, %235: f32):
      %236 = math.sin %234 : f32
      linalg.yield %236 : f32
    } -> tensor<8x16xf32>
    %237 = tensor.empty() : tensor<8x16xf32>
    %238 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212 : tensor<8x16xf32>) outs(%237 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb21(%239: f32, %240: f32):
      %241 = math.sin %239 : f32
      linalg.yield %241 : f32
    } -> tensor<8x16xf32>
    %242 = tensor.concat dim(1) %233, %238 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %243 = tensor.collapse_shape %242 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %244 = tensor.expand_shape %243 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %245 = tensor.collapse_shape %244 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %246 = tensor.expand_shape %245 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %247 = "tensor.extract_slice"(%121) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %248 = "tensor.extract_slice"(%121) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %249 = tensor.empty() : tensor<1x4x8x16xf32>
    %250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%248 : tensor<1x4x8x16xf32>) outs(%249 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb22(%251: f32, %252: f32):
      %253 = arith.negf %251 : f32
      linalg.yield %253 : f32
    } -> tensor<1x4x8x16xf32>
    %254 = tensor.concat dim(3) %250, %247 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %255 = tensor.empty() : tensor<1x4x8x32xf32>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121, %231 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%255 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb23(%257: f32, %258: f32, %259: f32):
      %260 = arith.mulf %257, %258 : f32
      linalg.yield %260 : f32
    } -> tensor<1x4x8x32xf32>
    %261 = tensor.empty() : tensor<1x4x8x32xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254, %246 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%261 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb24(%263: f32, %264: f32, %265: f32):
      %266 = arith.mulf %263, %264 : f32
      linalg.yield %266 : f32
    } -> tensor<1x4x8x32xf32>
    %267 = tensor.empty() : tensor<1x4x8x32xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%256, %262 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%267 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb25(%269: f32, %270: f32, %271: f32):
      %272 = arith.addf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<1x4x8x32xf32>
    %273 = tensor.empty() : tensor<16xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%273 : tensor<16xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb26(%275: f32):
      %276 = linalg.index 0 : index
      %277 = arith.index_cast %276 : index to i64
      %278 = arith.sitofp %277 : i64 to f32
      %279 = arith.constant 1.000000e+00 : f32
      %280 = arith.mulf %278, %279 : f32
      %281 = arith.constant 0.000000e+00 : f32
      %282 = arith.addf %281, %280 : f32
      linalg.yield %282 : f32
    } -> tensor<16xf32>
    %283 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %284 = tensor.splat %283 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %285 = tensor.empty() : tensor<16xf32>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%274, %284 : tensor<16xf32>, tensor<16xf32>) outs(%285 : tensor<16xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb27(%287: f32, %288: f32, %289: f32):
      %290 = arith.divf %287, %288 : f32
      linalg.yield %290 : f32
    } -> tensor<16xf32>
    %291 = tensor.empty() : tensor<16xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%286 : tensor<16xf32>) outs(%291 : tensor<16xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb28(%293: f32, %294: f32):
      %295 = arith.constant 1.000000e+04 : f32
      %296 = math.powf %295, %293 : f32
      linalg.yield %296 : f32
    } -> tensor<16xf32>
    %297 = tensor.empty() : tensor<16xf32>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%292 : tensor<16xf32>) outs(%297 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb29(%299: f32, %300: f32):
      %301 = arith.constant 1.000000e+00 : f32
      %302 = arith.divf %301, %299 : f32
      linalg.yield %302 : f32
    } -> tensor<16xf32>
    %303 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %304 = tensor.splat %303 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %305 = tensor.empty() : tensor<16xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%298, %304 : tensor<16xf32>, tensor<16xf32>) outs(%305 : tensor<16xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb30(%307: f32, %308: f32, %309: f32):
      %310 = arith.mulf %307, %308 : f32
      linalg.yield %310 : f32
    } -> tensor<16xf32>
    %311 = tensor.expand_shape %38 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %312 = tensor.empty() : tensor<8x1xf32>
    %313 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%311 : tensor<8x1xi64>) outs(%312 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb31(%314: i64, %315: f32):
      %316 = arith.sitofp %314 : i64 to f32
      linalg.yield %316 : f32
    } -> tensor<8x1xf32>
    %317 = tensor.expand_shape %306 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %318 = tensor.empty() : tensor<8x16xf32>
    %319 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%313, %317 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%318 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb32(%320: f32, %321: f32, %322: f32):
      %323 = arith.mulf %320, %321 : f32
      linalg.yield %323 : f32
    } -> tensor<8x16xf32>
    %324 = tensor.empty() : tensor<8x16xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%319 : tensor<8x16xf32>) outs(%324 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb33(%326: f32, %327: f32):
      %328 = math.cos %326 : f32
      linalg.yield %328 : f32
    } -> tensor<8x16xf32>
    %329 = tensor.empty() : tensor<8x16xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%319 : tensor<8x16xf32>) outs(%329 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb34(%331: f32, %332: f32):
      %333 = math.cos %331 : f32
      linalg.yield %333 : f32
    } -> tensor<8x16xf32>
    %334 = tensor.concat dim(1) %325, %330 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %335 = tensor.collapse_shape %334 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %336 = tensor.expand_shape %335 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %337 = tensor.collapse_shape %336 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %338 = tensor.expand_shape %337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %339 = tensor.empty() : tensor<8x16xf32>
    %340 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%319 : tensor<8x16xf32>) outs(%339 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb35(%341: f32, %342: f32):
      %343 = math.sin %341 : f32
      linalg.yield %343 : f32
    } -> tensor<8x16xf32>
    %344 = tensor.empty() : tensor<8x16xf32>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%319 : tensor<8x16xf32>) outs(%344 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb36(%346: f32, %347: f32):
      %348 = math.sin %346 : f32
      linalg.yield %348 : f32
    } -> tensor<8x16xf32>
    %349 = tensor.concat dim(1) %340, %345 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %350 = tensor.collapse_shape %349 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %351 = tensor.expand_shape %350 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %352 = tensor.collapse_shape %351 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %353 = tensor.expand_shape %352 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %354 = "tensor.extract_slice"(%143) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %355 = "tensor.extract_slice"(%143) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %356 = tensor.empty() : tensor<1x4x8x16xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355 : tensor<1x4x8x16xf32>) outs(%356 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb37(%358: f32, %359: f32):
      %360 = arith.negf %358 : f32
      linalg.yield %360 : f32
    } -> tensor<1x4x8x16xf32>
    %361 = tensor.concat dim(3) %357, %354 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %362 = tensor.empty() : tensor<1x4x8x32xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%143, %338 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%362 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb38(%364: f32, %365: f32, %366: f32):
      %367 = arith.mulf %364, %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x4x8x32xf32>
    %368 = tensor.empty() : tensor<1x4x8x32xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361, %353 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%368 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb39(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<1x4x8x32xf32>
    %374 = tensor.empty() : tensor<1x4x8x32xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363, %369 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%374 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb40(%376: f32, %377: f32, %378: f32):
      %379 = arith.addf %376, %377 : f32
      linalg.yield %379 : f32
    } -> tensor<1x4x8x32xf32>
    %380 = tensor.empty() : tensor<1x4x32x8xf32>
    %381 = linalg.transpose ins(%375:tensor<1x4x8x32xf32>) outs(%380:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %382 = tensor.empty() : tensor<1x4x8x32xf32>
    %383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%268 : tensor<1x4x8x32xf32>) outs(%382 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb41(%384: f32, %385: f32):
      linalg.yield %384 : f32
    } -> tensor<1x4x8x32xf32>
    %386 = tensor.collapse_shape %383 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %387 = tensor.expand_shape %386 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %388 = tensor.empty() : tensor<1x4x32x8xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381 : tensor<1x4x32x8xf32>) outs(%388 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb42(%390: f32, %391: f32):
      linalg.yield %390 : f32
    } -> tensor<1x4x32x8xf32>
    %392 = tensor.collapse_shape %389 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %393 = tensor.expand_shape %392 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %394 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %395 = tensor.splat %394 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32>
    %396 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%387, %393 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%395 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb43(%397: f32, %398: f32, %399: f32):
      %400 = arith.mulf %397, %398 : f32
      %401 = arith.addf %399, %400 : f32
      linalg.yield %401 : f32
    } -> tensor<4x8x8xf32>
    %402 = tensor.collapse_shape %396 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %404 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 5.65685415 : f32
    %405 = tensor.splat %404 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32>
    %406 = tensor.empty() : tensor<1x4x8x8xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%403, %405 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%406 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb44(%408: f32, %409: f32, %410: f32):
      %411 = arith.divf %408, %409 : f32
      linalg.yield %411 : f32
    } -> tensor<1x4x8x8xf32>
    %412 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %413 = tensor.splat %412 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xf32>
    %414 = tensor.empty() : tensor<8xi64>
    %415 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%414 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb45(%416: i64):
      %417 = linalg.index 0 : index
      %418 = arith.index_cast %417 : index to i64
      %419 = arith.constant 1 : i64
      %420 = arith.muli %418, %419 : i64
      %421 = arith.constant 0 : i64
      %422 = arith.addi %421, %420 : i64
      linalg.yield %422 : i64
    } -> tensor<8xi64>
    %423 = tensor.expand_shape %415 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %424 = tensor.empty() : tensor<8xi64>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%424 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb46(%426: i64):
      %427 = linalg.index 0 : index
      %428 = arith.index_cast %427 : index to i64
      %429 = arith.constant 1 : i64
      %430 = arith.muli %428, %429 : i64
      %431 = arith.constant 0 : i64
      %432 = arith.addi %431, %430 : i64
      linalg.yield %432 : i64
    } -> tensor<8xi64>
    %433 = tensor.expand_shape %425 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %434 = tensor.empty() : tensor<8x8xi64>
    %435 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%423, %433 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%434 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb47(%436: i64, %437: i64, %438: i64):
      %439 = arith.subi %436, %437 : i64
      linalg.yield %439 : i64
    } -> tensor<8x8xi64>
    %440 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1 : i64
    %441 = tensor.splat %440 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xi64>
    %442 = tensor.empty() : tensor<8x8xi1>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%435, %441 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%442 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb48(%444: i64, %445: i64, %446: i1):
      %447 = arith.cmpi sge, %444, %445 : i64
      linalg.yield %447 : i1
    } -> tensor<8x8xi1>
    %448 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %449 = tensor.splat %448 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<f32>
    %450 = tensor.empty() : tensor<8x8xf32>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%443, %413, %449 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%450 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb49(%452: i1, %453: f32, %454: f32, %455: f32):
      %456 = arith.select %452, %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<8x8xf32>
    %457 = tensor.empty() : tensor<1x4x8x8xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407, %451 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%457 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb50(%459: f32, %460: f32, %461: f32):
      %462 = arith.addf %459, %460 : f32
      linalg.yield %462 : f32
    } -> tensor<1x4x8x8xf32>
    %463 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %464 = tensor.splat %463 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %465 = linalg.reduce ins(%458:tensor<1x4x8x8xf32>) outs(%464:tensor<1x4x8xf32>) dimensions = [3]
    (%466: f32, %467: f32) {
      %468 = arith.maximumf %466, %467 : f32
      linalg.yield %468 : f32
    }
    %469 = tensor.collapse_shape %465 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %470 = tensor.expand_shape %469 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %471 = tensor.empty() : tensor<1x4x8x8xf32>
    %472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458, %470 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%471 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb51(%473: f32, %474: f32, %475: f32):
      %476 = arith.subf %473, %474 : f32
      linalg.yield %476 : f32
    } -> tensor<1x4x8x8xf32>
    %477 = tensor.empty() : tensor<1x4x8x8xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472 : tensor<1x4x8x8xf32>) outs(%477 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb52(%479: f32, %480: f32):
      %481 = math.exp %479 : f32
      linalg.yield %481 : f32
    } -> tensor<1x4x8x8xf32>
    %482 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %483 = tensor.splat %482 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %484 = linalg.reduce ins(%478:tensor<1x4x8x8xf32>) outs(%483:tensor<1x4x8xf32>) dimensions = [3]
    (%485: f32, %486: f32) {
      %487 = arith.addf %485, %486 : f32
      linalg.yield %487 : f32
    }
    %488 = tensor.collapse_shape %484 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %489 = tensor.expand_shape %488 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %490 = tensor.empty() : tensor<1x4x8x8xf32>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478, %489 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%490 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb53(%492: f32, %493: f32, %494: f32):
      %495 = arith.divf %492, %493 : f32
      linalg.yield %495 : f32
    } -> tensor<1x4x8x8xf32>
    %496 = tensor.empty() : tensor<1x4x8x8xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%491 : tensor<1x4x8x8xf32>) outs(%496 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb54(%498: f32, %499: f32):
      linalg.yield %498 : f32
    } -> tensor<1x4x8x8xf32>
    %500 = tensor.collapse_shape %497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %501 = tensor.expand_shape %500 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %502 = tensor.empty() : tensor<1x4x8x32xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%165 : tensor<1x4x8x32xf32>) outs(%502 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb55(%504: f32, %505: f32):
      linalg.yield %504 : f32
    } -> tensor<1x4x8x32xf32>
    %506 = tensor.collapse_shape %503 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %507 = tensor.expand_shape %506 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %508 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %509 = tensor.splat %508 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%501, %507 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%509 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb56(%511: f32, %512: f32, %513: f32):
      %514 = arith.mulf %511, %512 : f32
      %515 = arith.addf %513, %514 : f32
      linalg.yield %515 : f32
    } -> tensor<4x8x32xf32>
    %516 = tensor.collapse_shape %510 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %517 = tensor.expand_shape %516 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %518 = tensor.empty() : tensor<1x8x4x32xf32>
    %519 = linalg.transpose ins(%517:tensor<1x4x8x32xf32>) outs(%518:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %520 = tensor.collapse_shape %519 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %521 = tensor.expand_shape %520 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %522 = tensor.empty() : tensor<128x128xf32>
    %523 = linalg.transpose ins(%8:tensor<128x128xf32>) outs(%522:tensor<128x128xf32>) permutation = [1, 0]
    %524 = tensor.empty() : tensor<1x128xf32>
    %525 = linalg.transpose ins(%9:tensor<128x1xf32>) outs(%524:tensor<1x128xf32>) permutation = [1, 0]
    %526 = tensor.empty() : tensor<128x128xf32>
    %527 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%523, %525 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%526 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} {
    ^bb57(%528: f32, %529: f32, %530: f32):
      %531 = arith.mulf %528, %529 : f32
      linalg.yield %531 : f32
    } -> tensor<128x128xf32>
    %532 = tensor.collapse_shape %521 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %533 = tensor.expand_shape %532 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %534 = tensor.empty() : tensor<8x128xf32>
    %535 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %536 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%535 : f32) outs(%534 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %537 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} ins(%533, %527 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%536 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %538 = tensor.collapse_shape %537 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %539 = tensor.expand_shape %538 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %540 = tensor.empty() : tensor<1x8x128xf32>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %539 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%540 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb58(%542: f32, %543: f32, %544: f32):
      %545 = arith.addf %542, %543 : f32
      linalg.yield %545 : f32
    } -> tensor<1x8x128xf32>
    %546 = tensor.empty() : tensor<1x8x128xf32>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541 : tensor<1x8x128xf32>) outs(%546 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb59(%548: f32, %549: f32):
      %550 = arith.constant 2.000000e+00 : f32
      %551 = math.powf %548, %550 : f32
      linalg.yield %551 : f32
    } -> tensor<1x8x128xf32>
    %552 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 0.000000e+00 : f32
    %553 = tensor.splat %552 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %554 = linalg.reduce ins(%547:tensor<1x8x128xf32>) outs(%553:tensor<1x8xf32>) dimensions = [2]
    (%555: f32, %556: f32) {
      %557 = arith.addf %555, %556 : f32
      linalg.yield %557 : f32
    }
    %558 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.280000e+02 : f32
    %559 = tensor.splat %558 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %560 = tensor.empty() : tensor<1x8xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%554, %559 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%560 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb60(%562: f32, %563: f32, %564: f32):
      %565 = arith.divf %562, %563 : f32
      linalg.yield %565 : f32
    } -> tensor<1x8xf32>
    %566 = tensor.collapse_shape %561 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %567 = tensor.expand_shape %566 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %568 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.000000e-05 : f32
    %569 = tensor.splat %568 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8x1xf32>
    %570 = tensor.empty() : tensor<1x8x1xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%567, %569 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%570 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb61(%572: f32, %573: f32, %574: f32):
      %575 = arith.addf %572, %573 : f32
      linalg.yield %575 : f32
    } -> tensor<1x8x1xf32>
    %576 = tensor.empty() : tensor<1x8x1xf32>
    %577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%571 : tensor<1x8x1xf32>) outs(%576 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb62(%578: f32, %579: f32):
      %580 = math.rsqrt %578 : f32
      linalg.yield %580 : f32
    } -> tensor<1x8x1xf32>
    %581 = tensor.empty() : tensor<1x8x128xf32>
    %582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541, %577 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%581 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb63(%583: f32, %584: f32, %585: f32):
      %586 = arith.mulf %583, %584 : f32
      linalg.yield %586 : f32
    } -> tensor<1x8x128xf32>
    %587 = tensor.empty() : tensor<1x8x128xf32>
    %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%582, %10 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%587 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb64(%589: f32, %590: f32, %591: f32):
      %592 = arith.mulf %589, %590 : f32
      linalg.yield %592 : f32
    } -> tensor<1x8x128xf32>
    %593 = tensor.empty() : tensor<128x344xf32>
    %594 = linalg.transpose ins(%11:tensor<344x128xf32>) outs(%593:tensor<128x344xf32>) permutation = [1, 0]
    %595 = tensor.empty() : tensor<1x344xf32>
    %596 = linalg.transpose ins(%12:tensor<344x1xf32>) outs(%595:tensor<1x344xf32>) permutation = [1, 0]
    %597 = tensor.empty() : tensor<128x344xf32>
    %598 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%594, %596 : tensor<128x344xf32>, tensor<1x344xf32>) outs(%597 : tensor<128x344xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} {
    ^bb65(%599: f32, %600: f32, %601: f32):
      %602 = arith.mulf %599, %600 : f32
      linalg.yield %602 : f32
    } -> tensor<128x344xf32>
    %603 = tensor.collapse_shape %588 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %604 = tensor.expand_shape %603 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %605 = tensor.empty() : tensor<8x344xf32>
    %606 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %607 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%606 : f32) outs(%605 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %608 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} ins(%604, %598 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%607 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %609 = tensor.collapse_shape %608 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %610 = tensor.expand_shape %609 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %611 = tensor.empty() : tensor<1x8x344xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%610 : tensor<1x8x344xf32>) outs(%611 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb66(%613: f32, %614: f32):
      %615 = arith.constant 1.000000e+00 : f32
      %616 = arith.negf %613 : f32
      %617 = math.exp %616 : f32
      %618 = arith.addf %615, %617 : f32
      %619 = arith.divf %615, %618 : f32
      linalg.yield %619 : f32
    } -> tensor<1x8x344xf32>
    %620 = tensor.empty() : tensor<1x8x344xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%610, %612 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%620 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb67(%622: f32, %623: f32, %624: f32):
      %625 = arith.mulf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<1x8x344xf32>
    %626 = tensor.empty() : tensor<128x344xf32>
    %627 = linalg.transpose ins(%13:tensor<344x128xf32>) outs(%626:tensor<128x344xf32>) permutation = [1, 0]
    %628 = tensor.empty() : tensor<1x344xf32>
    %629 = linalg.transpose ins(%14:tensor<344x1xf32>) outs(%628:tensor<1x344xf32>) permutation = [1, 0]
    %630 = tensor.empty() : tensor<128x344xf32>
    %631 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%627, %629 : tensor<128x344xf32>, tensor<1x344xf32>) outs(%630 : tensor<128x344xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} {
    ^bb68(%632: f32, %633: f32, %634: f32):
      %635 = arith.mulf %632, %633 : f32
      linalg.yield %635 : f32
    } -> tensor<128x344xf32>
    %636 = tensor.collapse_shape %588 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %637 = tensor.expand_shape %636 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %638 = tensor.empty() : tensor<8x344xf32>
    %639 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %640 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%639 : f32) outs(%638 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %641 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} ins(%637, %631 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%640 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %642 = tensor.collapse_shape %641 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %643 = tensor.expand_shape %642 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %644 = tensor.empty() : tensor<1x8x344xf32>
    %645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %643 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%644 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb69(%646: f32, %647: f32, %648: f32):
      %649 = arith.mulf %646, %647 : f32
      linalg.yield %649 : f32
    } -> tensor<1x8x344xf32>
    %650 = tensor.empty() : tensor<344x128xf32>
    %651 = linalg.transpose ins(%15:tensor<128x344xf32>) outs(%650:tensor<344x128xf32>) permutation = [1, 0]
    %652 = tensor.empty() : tensor<1x128xf32>
    %653 = linalg.transpose ins(%16:tensor<128x1xf32>) outs(%652:tensor<1x128xf32>) permutation = [1, 0]
    %654 = tensor.empty() : tensor<344x128xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%651, %653 : tensor<344x128xf32>, tensor<1x128xf32>) outs(%654 : tensor<344x128xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} {
    ^bb70(%656: f32, %657: f32, %658: f32):
      %659 = arith.mulf %656, %657 : f32
      linalg.yield %659 : f32
    } -> tensor<344x128xf32>
    %660 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %662 = tensor.empty() : tensor<8x128xf32>
    %663 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %664 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%663 : f32) outs(%662 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %665 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} ins(%661, %655 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%664 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %666 = tensor.collapse_shape %665 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %667 = tensor.expand_shape %666 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %668 = tensor.empty() : tensor<1x8x128xf32>
    %669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541, %667 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%668 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb71(%670: f32, %671: f32, %672: f32):
      %673 = arith.addf %670, %671 : f32
      linalg.yield %673 : f32
    } -> tensor<1x8x128xf32>
    %674 = tensor.empty() : tensor<1x8x128xf32>
    %675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%669 : tensor<1x8x128xf32>) outs(%674 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb72(%676: f32, %677: f32):
      %678 = arith.constant 2.000000e+00 : f32
      %679 = math.powf %676, %678 : f32
      linalg.yield %679 : f32
    } -> tensor<1x8x128xf32>
    %680 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 0.000000e+00 : f32
    %681 = tensor.splat %680 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %682 = linalg.reduce ins(%675:tensor<1x8x128xf32>) outs(%681:tensor<1x8xf32>) dimensions = [2]
    (%683: f32, %684: f32) {
      %685 = arith.addf %683, %684 : f32
      linalg.yield %685 : f32
    }
    %686 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.280000e+02 : f32
    %687 = tensor.splat %686 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %688 = tensor.empty() : tensor<1x8xf32>
    %689 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%682, %687 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%688 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb73(%690: f32, %691: f32, %692: f32):
      %693 = arith.divf %690, %691 : f32
      linalg.yield %693 : f32
    } -> tensor<1x8xf32>
    %694 = tensor.collapse_shape %689 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %695 = tensor.expand_shape %694 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %696 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.000000e-05 : f32
    %697 = tensor.splat %696 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8x1xf32>
    %698 = tensor.empty() : tensor<1x8x1xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%695, %697 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%698 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb74(%700: f32, %701: f32, %702: f32):
      %703 = arith.addf %700, %701 : f32
      linalg.yield %703 : f32
    } -> tensor<1x8x1xf32>
    %704 = tensor.empty() : tensor<1x8x1xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%699 : tensor<1x8x1xf32>) outs(%704 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb75(%706: f32, %707: f32):
      %708 = math.rsqrt %706 : f32
      linalg.yield %708 : f32
    } -> tensor<1x8x1xf32>
    %709 = tensor.empty() : tensor<1x8x128xf32>
    %710 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%669, %705 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%709 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb76(%711: f32, %712: f32, %713: f32):
      %714 = arith.mulf %711, %712 : f32
      linalg.yield %714 : f32
    } -> tensor<1x8x128xf32>
    %715 = tensor.empty() : tensor<1x8x128xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%710, %17 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%715 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb77(%717: f32, %718: f32, %719: f32):
      %720 = arith.mulf %717, %718 : f32
      linalg.yield %720 : f32
    } -> tensor<1x8x128xf32>
    %721 = tensor.empty() : tensor<128x128xf32>
    %722 = linalg.transpose ins(%18:tensor<128x128xf32>) outs(%721:tensor<128x128xf32>) permutation = [1, 0]
    %723 = tensor.empty() : tensor<1x128xf32>
    %724 = linalg.transpose ins(%19:tensor<128x1xf32>) outs(%723:tensor<1x128xf32>) permutation = [1, 0]
    %725 = tensor.empty() : tensor<128x128xf32>
    %726 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%722, %724 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%725 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} {
    ^bb78(%727: f32, %728: f32, %729: f32):
      %730 = arith.mulf %727, %728 : f32
      linalg.yield %730 : f32
    } -> tensor<128x128xf32>
    %731 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %732 = tensor.expand_shape %731 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %733 = tensor.empty() : tensor<8x128xf32>
    %734 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %735 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%734 : f32) outs(%733 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %736 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} ins(%732, %726 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%735 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %737 = tensor.collapse_shape %736 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %738 = tensor.expand_shape %737 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %739 = tensor.collapse_shape %738 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %740 = tensor.expand_shape %739 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %741 = tensor.empty() : tensor<1x4x8x32xf32>
    %742 = linalg.transpose ins(%740:tensor<1x8x4x32xf32>) outs(%741:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %743 = tensor.empty() : tensor<128x128xf32>
    %744 = linalg.transpose ins(%20:tensor<128x128xf32>) outs(%743:tensor<128x128xf32>) permutation = [1, 0]
    %745 = tensor.empty() : tensor<1x128xf32>
    %746 = linalg.transpose ins(%21:tensor<128x1xf32>) outs(%745:tensor<1x128xf32>) permutation = [1, 0]
    %747 = tensor.empty() : tensor<128x128xf32>
    %748 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%744, %746 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%747 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} {
    ^bb79(%749: f32, %750: f32, %751: f32):
      %752 = arith.mulf %749, %750 : f32
      linalg.yield %752 : f32
    } -> tensor<128x128xf32>
    %753 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %754 = tensor.expand_shape %753 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %755 = tensor.empty() : tensor<8x128xf32>
    %756 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %757 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%756 : f32) outs(%755 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %758 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} ins(%754, %748 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%757 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %759 = tensor.collapse_shape %758 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %760 = tensor.expand_shape %759 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %761 = tensor.collapse_shape %760 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %762 = tensor.expand_shape %761 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %763 = tensor.empty() : tensor<1x4x8x32xf32>
    %764 = linalg.transpose ins(%762:tensor<1x8x4x32xf32>) outs(%763:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %765 = tensor.empty() : tensor<128x128xf32>
    %766 = linalg.transpose ins(%22:tensor<128x128xf32>) outs(%765:tensor<128x128xf32>) permutation = [1, 0]
    %767 = tensor.empty() : tensor<1x128xf32>
    %768 = linalg.transpose ins(%23:tensor<128x1xf32>) outs(%767:tensor<1x128xf32>) permutation = [1, 0]
    %769 = tensor.empty() : tensor<128x128xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%766, %768 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%769 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} {
    ^bb80(%771: f32, %772: f32, %773: f32):
      %774 = arith.mulf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<128x128xf32>
    %775 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %776 = tensor.expand_shape %775 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %777 = tensor.empty() : tensor<8x128xf32>
    %778 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %779 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%778 : f32) outs(%777 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %780 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} ins(%776, %770 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%779 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %781 = tensor.collapse_shape %780 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %782 = tensor.expand_shape %781 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %783 = tensor.collapse_shape %782 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %784 = tensor.expand_shape %783 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %785 = tensor.empty() : tensor<1x4x8x32xf32>
    %786 = linalg.transpose ins(%784:tensor<1x8x4x32xf32>) outs(%785:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %787 = tensor.empty() : tensor<16xf32>
    %788 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%787 : tensor<16xf32>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb81(%789: f32):
      %790 = linalg.index 0 : index
      %791 = arith.index_cast %790 : index to i64
      %792 = arith.sitofp %791 : i64 to f32
      %793 = arith.constant 1.000000e+00 : f32
      %794 = arith.mulf %792, %793 : f32
      %795 = arith.constant 0.000000e+00 : f32
      %796 = arith.addf %795, %794 : f32
      linalg.yield %796 : f32
    } -> tensor<16xf32>
    %797 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %798 = tensor.splat %797 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %799 = tensor.empty() : tensor<16xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%788, %798 : tensor<16xf32>, tensor<16xf32>) outs(%799 : tensor<16xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb82(%801: f32, %802: f32, %803: f32):
      %804 = arith.divf %801, %802 : f32
      linalg.yield %804 : f32
    } -> tensor<16xf32>
    %805 = tensor.empty() : tensor<16xf32>
    %806 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%800 : tensor<16xf32>) outs(%805 : tensor<16xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb83(%807: f32, %808: f32):
      %809 = arith.constant 1.000000e+04 : f32
      %810 = math.powf %809, %807 : f32
      linalg.yield %810 : f32
    } -> tensor<16xf32>
    %811 = tensor.empty() : tensor<16xf32>
    %812 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%806 : tensor<16xf32>) outs(%811 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb84(%813: f32, %814: f32):
      %815 = arith.constant 1.000000e+00 : f32
      %816 = arith.divf %815, %813 : f32
      linalg.yield %816 : f32
    } -> tensor<16xf32>
    %817 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %818 = tensor.splat %817 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %819 = tensor.empty() : tensor<16xf32>
    %820 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%812, %818 : tensor<16xf32>, tensor<16xf32>) outs(%819 : tensor<16xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb85(%821: f32, %822: f32, %823: f32):
      %824 = arith.mulf %821, %822 : f32
      linalg.yield %824 : f32
    } -> tensor<16xf32>
    %825 = tensor.expand_shape %38 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %826 = tensor.empty() : tensor<8x1xf32>
    %827 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%825 : tensor<8x1xi64>) outs(%826 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb86(%828: i64, %829: f32):
      %830 = arith.sitofp %828 : i64 to f32
      linalg.yield %830 : f32
    } -> tensor<8x1xf32>
    %831 = tensor.expand_shape %820 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %832 = tensor.empty() : tensor<8x16xf32>
    %833 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%827, %831 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%832 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb87(%834: f32, %835: f32, %836: f32):
      %837 = arith.mulf %834, %835 : f32
      linalg.yield %837 : f32
    } -> tensor<8x16xf32>
    %838 = tensor.empty() : tensor<8x16xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%833 : tensor<8x16xf32>) outs(%838 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb88(%840: f32, %841: f32):
      %842 = math.cos %840 : f32
      linalg.yield %842 : f32
    } -> tensor<8x16xf32>
    %843 = tensor.empty() : tensor<8x16xf32>
    %844 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%833 : tensor<8x16xf32>) outs(%843 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb89(%845: f32, %846: f32):
      %847 = math.cos %845 : f32
      linalg.yield %847 : f32
    } -> tensor<8x16xf32>
    %848 = tensor.concat dim(1) %839, %844 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %849 = tensor.collapse_shape %848 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %850 = tensor.expand_shape %849 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %851 = tensor.collapse_shape %850 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %852 = tensor.expand_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %853 = tensor.empty() : tensor<8x16xf32>
    %854 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%833 : tensor<8x16xf32>) outs(%853 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb90(%855: f32, %856: f32):
      %857 = math.sin %855 : f32
      linalg.yield %857 : f32
    } -> tensor<8x16xf32>
    %858 = tensor.empty() : tensor<8x16xf32>
    %859 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%833 : tensor<8x16xf32>) outs(%858 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb91(%860: f32, %861: f32):
      %862 = math.sin %860 : f32
      linalg.yield %862 : f32
    } -> tensor<8x16xf32>
    %863 = tensor.concat dim(1) %854, %859 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %864 = tensor.collapse_shape %863 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %865 = tensor.expand_shape %864 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %866 = tensor.collapse_shape %865 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %867 = tensor.expand_shape %866 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %868 = "tensor.extract_slice"(%742) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %869 = "tensor.extract_slice"(%742) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %870 = tensor.empty() : tensor<1x4x8x16xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%869 : tensor<1x4x8x16xf32>) outs(%870 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb92(%872: f32, %873: f32):
      %874 = arith.negf %872 : f32
      linalg.yield %874 : f32
    } -> tensor<1x4x8x16xf32>
    %875 = tensor.concat dim(3) %871, %868 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %876 = tensor.empty() : tensor<1x4x8x32xf32>
    %877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%742, %852 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%876 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb93(%878: f32, %879: f32, %880: f32):
      %881 = arith.mulf %878, %879 : f32
      linalg.yield %881 : f32
    } -> tensor<1x4x8x32xf32>
    %882 = tensor.empty() : tensor<1x4x8x32xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%875, %867 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%882 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb94(%884: f32, %885: f32, %886: f32):
      %887 = arith.mulf %884, %885 : f32
      linalg.yield %887 : f32
    } -> tensor<1x4x8x32xf32>
    %888 = tensor.empty() : tensor<1x4x8x32xf32>
    %889 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%877, %883 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%888 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb95(%890: f32, %891: f32, %892: f32):
      %893 = arith.addf %890, %891 : f32
      linalg.yield %893 : f32
    } -> tensor<1x4x8x32xf32>
    %894 = tensor.empty() : tensor<16xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%894 : tensor<16xf32>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb96(%896: f32):
      %897 = linalg.index 0 : index
      %898 = arith.index_cast %897 : index to i64
      %899 = arith.sitofp %898 : i64 to f32
      %900 = arith.constant 1.000000e+00 : f32
      %901 = arith.mulf %899, %900 : f32
      %902 = arith.constant 0.000000e+00 : f32
      %903 = arith.addf %902, %901 : f32
      linalg.yield %903 : f32
    } -> tensor<16xf32>
    %904 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %905 = tensor.splat %904 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %906 = tensor.empty() : tensor<16xf32>
    %907 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%895, %905 : tensor<16xf32>, tensor<16xf32>) outs(%906 : tensor<16xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb97(%908: f32, %909: f32, %910: f32):
      %911 = arith.divf %908, %909 : f32
      linalg.yield %911 : f32
    } -> tensor<16xf32>
    %912 = tensor.empty() : tensor<16xf32>
    %913 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%907 : tensor<16xf32>) outs(%912 : tensor<16xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb98(%914: f32, %915: f32):
      %916 = arith.constant 1.000000e+04 : f32
      %917 = math.powf %916, %914 : f32
      linalg.yield %917 : f32
    } -> tensor<16xf32>
    %918 = tensor.empty() : tensor<16xf32>
    %919 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%913 : tensor<16xf32>) outs(%918 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb99(%920: f32, %921: f32):
      %922 = arith.constant 1.000000e+00 : f32
      %923 = arith.divf %922, %920 : f32
      linalg.yield %923 : f32
    } -> tensor<16xf32>
    %924 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %925 = tensor.splat %924 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %926 = tensor.empty() : tensor<16xf32>
    %927 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%919, %925 : tensor<16xf32>, tensor<16xf32>) outs(%926 : tensor<16xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb100(%928: f32, %929: f32, %930: f32):
      %931 = arith.mulf %928, %929 : f32
      linalg.yield %931 : f32
    } -> tensor<16xf32>
    %932 = tensor.expand_shape %38 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %933 = tensor.empty() : tensor<8x1xf32>
    %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%932 : tensor<8x1xi64>) outs(%933 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb101(%935: i64, %936: f32):
      %937 = arith.sitofp %935 : i64 to f32
      linalg.yield %937 : f32
    } -> tensor<8x1xf32>
    %938 = tensor.expand_shape %927 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %939 = tensor.empty() : tensor<8x16xf32>
    %940 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%934, %938 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%939 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb102(%941: f32, %942: f32, %943: f32):
      %944 = arith.mulf %941, %942 : f32
      linalg.yield %944 : f32
    } -> tensor<8x16xf32>
    %945 = tensor.empty() : tensor<8x16xf32>
    %946 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%940 : tensor<8x16xf32>) outs(%945 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb103(%947: f32, %948: f32):
      %949 = math.cos %947 : f32
      linalg.yield %949 : f32
    } -> tensor<8x16xf32>
    %950 = tensor.empty() : tensor<8x16xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%940 : tensor<8x16xf32>) outs(%950 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb104(%952: f32, %953: f32):
      %954 = math.cos %952 : f32
      linalg.yield %954 : f32
    } -> tensor<8x16xf32>
    %955 = tensor.concat dim(1) %946, %951 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %956 = tensor.collapse_shape %955 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %957 = tensor.expand_shape %956 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %958 = tensor.collapse_shape %957 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %959 = tensor.expand_shape %958 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %960 = tensor.empty() : tensor<8x16xf32>
    %961 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%940 : tensor<8x16xf32>) outs(%960 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb105(%962: f32, %963: f32):
      %964 = math.sin %962 : f32
      linalg.yield %964 : f32
    } -> tensor<8x16xf32>
    %965 = tensor.empty() : tensor<8x16xf32>
    %966 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%940 : tensor<8x16xf32>) outs(%965 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb106(%967: f32, %968: f32):
      %969 = math.sin %967 : f32
      linalg.yield %969 : f32
    } -> tensor<8x16xf32>
    %970 = tensor.concat dim(1) %961, %966 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %971 = tensor.collapse_shape %970 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %972 = tensor.expand_shape %971 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %973 = tensor.collapse_shape %972 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %974 = tensor.expand_shape %973 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %975 = "tensor.extract_slice"(%764) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %976 = "tensor.extract_slice"(%764) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %977 = tensor.empty() : tensor<1x4x8x16xf32>
    %978 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%976 : tensor<1x4x8x16xf32>) outs(%977 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb107(%979: f32, %980: f32):
      %981 = arith.negf %979 : f32
      linalg.yield %981 : f32
    } -> tensor<1x4x8x16xf32>
    %982 = tensor.concat dim(3) %978, %975 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %983 = tensor.empty() : tensor<1x4x8x32xf32>
    %984 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%764, %959 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%983 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb108(%985: f32, %986: f32, %987: f32):
      %988 = arith.mulf %985, %986 : f32
      linalg.yield %988 : f32
    } -> tensor<1x4x8x32xf32>
    %989 = tensor.empty() : tensor<1x4x8x32xf32>
    %990 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%982, %974 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%989 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb109(%991: f32, %992: f32, %993: f32):
      %994 = arith.mulf %991, %992 : f32
      linalg.yield %994 : f32
    } -> tensor<1x4x8x32xf32>
    %995 = tensor.empty() : tensor<1x4x8x32xf32>
    %996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%984, %990 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%995 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb110(%997: f32, %998: f32, %999: f32):
      %1000 = arith.addf %997, %998 : f32
      linalg.yield %1000 : f32
    } -> tensor<1x4x8x32xf32>
    %1001 = tensor.empty() : tensor<1x4x32x8xf32>
    %1002 = linalg.transpose ins(%996:tensor<1x4x8x32xf32>) outs(%1001:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %1003 = tensor.empty() : tensor<1x4x8x32xf32>
    %1004 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%889 : tensor<1x4x8x32xf32>) outs(%1003 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb111(%1005: f32, %1006: f32):
      linalg.yield %1005 : f32
    } -> tensor<1x4x8x32xf32>
    %1007 = tensor.collapse_shape %1004 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1008 = tensor.expand_shape %1007 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1009 = tensor.empty() : tensor<1x4x32x8xf32>
    %1010 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1002 : tensor<1x4x32x8xf32>) outs(%1009 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb112(%1011: f32, %1012: f32):
      linalg.yield %1011 : f32
    } -> tensor<1x4x32x8xf32>
    %1013 = tensor.collapse_shape %1010 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %1014 = tensor.expand_shape %1013 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %1015 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1016 = tensor.splat %1015 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32>
    %1017 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1008, %1014 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%1016 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb113(%1018: f32, %1019: f32, %1020: f32):
      %1021 = arith.mulf %1018, %1019 : f32
      %1022 = arith.addf %1020, %1021 : f32
      linalg.yield %1022 : f32
    } -> tensor<4x8x8xf32>
    %1023 = tensor.collapse_shape %1017 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %1024 = tensor.expand_shape %1023 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %1025 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 5.65685415 : f32
    %1026 = tensor.splat %1025 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32>
    %1027 = tensor.empty() : tensor<1x4x8x8xf32>
    %1028 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1024, %1026 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%1027 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb114(%1029: f32, %1030: f32, %1031: f32):
      %1032 = arith.divf %1029, %1030 : f32
      linalg.yield %1032 : f32
    } -> tensor<1x4x8x8xf32>
    %1033 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1034 = tensor.splat %1033 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xf32>
    %1035 = tensor.empty() : tensor<8xi64>
    %1036 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1035 : tensor<8xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb115(%1037: i64):
      %1038 = linalg.index 0 : index
      %1039 = arith.index_cast %1038 : index to i64
      %1040 = arith.constant 1 : i64
      %1041 = arith.muli %1039, %1040 : i64
      %1042 = arith.constant 0 : i64
      %1043 = arith.addi %1042, %1041 : i64
      linalg.yield %1043 : i64
    } -> tensor<8xi64>
    %1044 = tensor.expand_shape %1036 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %1045 = tensor.empty() : tensor<8xi64>
    %1046 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1045 : tensor<8xi64>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb116(%1047: i64):
      %1048 = linalg.index 0 : index
      %1049 = arith.index_cast %1048 : index to i64
      %1050 = arith.constant 1 : i64
      %1051 = arith.muli %1049, %1050 : i64
      %1052 = arith.constant 0 : i64
      %1053 = arith.addi %1052, %1051 : i64
      linalg.yield %1053 : i64
    } -> tensor<8xi64>
    %1054 = tensor.expand_shape %1046 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %1055 = tensor.empty() : tensor<8x8xi64>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1044, %1054 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%1055 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb117(%1057: i64, %1058: i64, %1059: i64):
      %1060 = arith.subi %1057, %1058 : i64
      linalg.yield %1060 : i64
    } -> tensor<8x8xi64>
    %1061 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1 : i64
    %1062 = tensor.splat %1061 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xi64>
    %1063 = tensor.empty() : tensor<8x8xi1>
    %1064 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1056, %1062 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%1063 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb118(%1065: i64, %1066: i64, %1067: i1):
      %1068 = arith.cmpi sge, %1065, %1066 : i64
      linalg.yield %1068 : i1
    } -> tensor<8x8xi1>
    %1069 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<f32>
    %1071 = tensor.empty() : tensor<8x8xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1064, %1034, %1070 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%1071 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb119(%1073: i1, %1074: f32, %1075: f32, %1076: f32):
      %1077 = arith.select %1073, %1074, %1075 : f32
      linalg.yield %1077 : f32
    } -> tensor<8x8xf32>
    %1078 = tensor.empty() : tensor<1x4x8x8xf32>
    %1079 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1028, %1072 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%1078 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb120(%1080: f32, %1081: f32, %1082: f32):
      %1083 = arith.addf %1080, %1081 : f32
      linalg.yield %1083 : f32
    } -> tensor<1x4x8x8xf32>
    %1084 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1085 = tensor.splat %1084 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1086 = linalg.reduce ins(%1079:tensor<1x4x8x8xf32>) outs(%1085:tensor<1x4x8xf32>) dimensions = [3]
    (%1087: f32, %1088: f32) {
      %1089 = arith.maximumf %1087, %1088 : f32
      linalg.yield %1089 : f32
    }
    %1090 = tensor.collapse_shape %1086 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1091 = tensor.expand_shape %1090 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1092 = tensor.empty() : tensor<1x4x8x8xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1079, %1091 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1092 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb121(%1094: f32, %1095: f32, %1096: f32):
      %1097 = arith.subf %1094, %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x4x8x8xf32>
    %1098 = tensor.empty() : tensor<1x4x8x8xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1093 : tensor<1x4x8x8xf32>) outs(%1098 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb122(%1100: f32, %1101: f32):
      %1102 = math.exp %1100 : f32
      linalg.yield %1102 : f32
    } -> tensor<1x4x8x8xf32>
    %1103 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1104 = tensor.splat %1103 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1105 = linalg.reduce ins(%1099:tensor<1x4x8x8xf32>) outs(%1104:tensor<1x4x8xf32>) dimensions = [3]
    (%1106: f32, %1107: f32) {
      %1108 = arith.addf %1106, %1107 : f32
      linalg.yield %1108 : f32
    }
    %1109 = tensor.collapse_shape %1105 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1110 = tensor.expand_shape %1109 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1111 = tensor.empty() : tensor<1x4x8x8xf32>
    %1112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1099, %1110 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1111 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb123(%1113: f32, %1114: f32, %1115: f32):
      %1116 = arith.divf %1113, %1114 : f32
      linalg.yield %1116 : f32
    } -> tensor<1x4x8x8xf32>
    %1117 = tensor.empty() : tensor<1x4x8x8xf32>
    %1118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1112 : tensor<1x4x8x8xf32>) outs(%1117 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb124(%1119: f32, %1120: f32):
      linalg.yield %1119 : f32
    } -> tensor<1x4x8x8xf32>
    %1121 = tensor.collapse_shape %1118 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %1122 = tensor.expand_shape %1121 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %1123 = tensor.empty() : tensor<1x4x8x32xf32>
    %1124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%786 : tensor<1x4x8x32xf32>) outs(%1123 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb125(%1125: f32, %1126: f32):
      linalg.yield %1125 : f32
    } -> tensor<1x4x8x32xf32>
    %1127 = tensor.collapse_shape %1124 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1129 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1130 = tensor.splat %1129 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32>
    %1131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1122, %1128 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%1130 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb126(%1132: f32, %1133: f32, %1134: f32):
      %1135 = arith.mulf %1132, %1133 : f32
      %1136 = arith.addf %1134, %1135 : f32
      linalg.yield %1136 : f32
    } -> tensor<4x8x32xf32>
    %1137 = tensor.collapse_shape %1131 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %1138 = tensor.expand_shape %1137 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %1139 = tensor.empty() : tensor<1x8x4x32xf32>
    %1140 = linalg.transpose ins(%1138:tensor<1x4x8x32xf32>) outs(%1139:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %1141 = tensor.collapse_shape %1140 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %1142 = tensor.expand_shape %1141 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1143 = tensor.empty() : tensor<128x128xf32>
    %1144 = linalg.transpose ins(%24:tensor<128x128xf32>) outs(%1143:tensor<128x128xf32>) permutation = [1, 0]
    %1145 = tensor.empty() : tensor<1x128xf32>
    %1146 = linalg.transpose ins(%25:tensor<128x1xf32>) outs(%1145:tensor<1x128xf32>) permutation = [1, 0]
    %1147 = tensor.empty() : tensor<128x128xf32>
    %1148 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1144, %1146 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%1147 : tensor<128x128xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} {
    ^bb127(%1149: f32, %1150: f32, %1151: f32):
      %1152 = arith.mulf %1149, %1150 : f32
      linalg.yield %1152 : f32
    } -> tensor<128x128xf32>
    %1153 = tensor.collapse_shape %1142 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1154 = tensor.expand_shape %1153 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %1155 = tensor.empty() : tensor<8x128xf32>
    %1156 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1157 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1156 : f32) outs(%1155 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1158 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} ins(%1154, %1148 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%1157 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1159 = tensor.collapse_shape %1158 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %1160 = tensor.expand_shape %1159 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1161 = tensor.empty() : tensor<1x8x128xf32>
    %1162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%669, %1160 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1161 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb128(%1163: f32, %1164: f32, %1165: f32):
      %1166 = arith.addf %1163, %1164 : f32
      linalg.yield %1166 : f32
    } -> tensor<1x8x128xf32>
    %1167 = tensor.empty() : tensor<1x8x128xf32>
    %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1162 : tensor<1x8x128xf32>) outs(%1167 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb129(%1169: f32, %1170: f32):
      %1171 = arith.constant 2.000000e+00 : f32
      %1172 = math.powf %1169, %1171 : f32
      linalg.yield %1172 : f32
    } -> tensor<1x8x128xf32>
    %1173 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 0.000000e+00 : f32
    %1174 = tensor.splat %1173 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1175 = linalg.reduce ins(%1168:tensor<1x8x128xf32>) outs(%1174:tensor<1x8xf32>) dimensions = [2]
    (%1176: f32, %1177: f32) {
      %1178 = arith.addf %1176, %1177 : f32
      linalg.yield %1178 : f32
    }
    %1179 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.280000e+02 : f32
    %1180 = tensor.splat %1179 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1181 = tensor.empty() : tensor<1x8xf32>
    %1182 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1175, %1180 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1181 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb130(%1183: f32, %1184: f32, %1185: f32):
      %1186 = arith.divf %1183, %1184 : f32
      linalg.yield %1186 : f32
    } -> tensor<1x8xf32>
    %1187 = tensor.collapse_shape %1182 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %1188 = tensor.expand_shape %1187 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1189 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.000000e-05 : f32
    %1190 = tensor.splat %1189 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8x1xf32>
    %1191 = tensor.empty() : tensor<1x8x1xf32>
    %1192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1188, %1190 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1191 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb131(%1193: f32, %1194: f32, %1195: f32):
      %1196 = arith.addf %1193, %1194 : f32
      linalg.yield %1196 : f32
    } -> tensor<1x8x1xf32>
    %1197 = tensor.empty() : tensor<1x8x1xf32>
    %1198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1192 : tensor<1x8x1xf32>) outs(%1197 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb132(%1199: f32, %1200: f32):
      %1201 = math.rsqrt %1199 : f32
      linalg.yield %1201 : f32
    } -> tensor<1x8x1xf32>
    %1202 = tensor.empty() : tensor<1x8x128xf32>
    %1203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1162, %1198 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1202 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb133(%1204: f32, %1205: f32, %1206: f32):
      %1207 = arith.mulf %1204, %1205 : f32
      linalg.yield %1207 : f32
    } -> tensor<1x8x128xf32>
    %1208 = tensor.empty() : tensor<1x8x128xf32>
    %1209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1203, %26 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1208 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb134(%1210: f32, %1211: f32, %1212: f32):
      %1213 = arith.mulf %1210, %1211 : f32
      linalg.yield %1213 : f32
    } -> tensor<1x8x128xf32>
    %1214 = tensor.empty() : tensor<128x344xf32>
    %1215 = linalg.transpose ins(%27:tensor<344x128xf32>) outs(%1214:tensor<128x344xf32>) permutation = [1, 0]
    %1216 = tensor.empty() : tensor<1x344xf32>
    %1217 = linalg.transpose ins(%28:tensor<344x1xf32>) outs(%1216:tensor<1x344xf32>) permutation = [1, 0]
    %1218 = tensor.empty() : tensor<128x344xf32>
    %1219 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1215, %1217 : tensor<128x344xf32>, tensor<1x344xf32>) outs(%1218 : tensor<128x344xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} {
    ^bb135(%1220: f32, %1221: f32, %1222: f32):
      %1223 = arith.mulf %1220, %1221 : f32
      linalg.yield %1223 : f32
    } -> tensor<128x344xf32>
    %1224 = tensor.collapse_shape %1209 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1225 = tensor.expand_shape %1224 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %1226 = tensor.empty() : tensor<8x344xf32>
    %1227 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1228 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1227 : f32) outs(%1226 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1229 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} ins(%1225, %1219 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1228 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1230 = tensor.collapse_shape %1229 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %1231 = tensor.expand_shape %1230 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1232 = tensor.empty() : tensor<1x8x344xf32>
    %1233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1231 : tensor<1x8x344xf32>) outs(%1232 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb136(%1234: f32, %1235: f32):
      %1236 = arith.constant 1.000000e+00 : f32
      %1237 = arith.negf %1234 : f32
      %1238 = math.exp %1237 : f32
      %1239 = arith.addf %1236, %1238 : f32
      %1240 = arith.divf %1236, %1239 : f32
      linalg.yield %1240 : f32
    } -> tensor<1x8x344xf32>
    %1241 = tensor.empty() : tensor<1x8x344xf32>
    %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1231, %1233 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1241 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb137(%1243: f32, %1244: f32, %1245: f32):
      %1246 = arith.mulf %1243, %1244 : f32
      linalg.yield %1246 : f32
    } -> tensor<1x8x344xf32>
    %1247 = tensor.empty() : tensor<128x344xf32>
    %1248 = linalg.transpose ins(%29:tensor<344x128xf32>) outs(%1247:tensor<128x344xf32>) permutation = [1, 0]
    %1249 = tensor.empty() : tensor<1x344xf32>
    %1250 = linalg.transpose ins(%30:tensor<344x1xf32>) outs(%1249:tensor<1x344xf32>) permutation = [1, 0]
    %1251 = tensor.empty() : tensor<128x344xf32>
    %1252 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1248, %1250 : tensor<128x344xf32>, tensor<1x344xf32>) outs(%1251 : tensor<128x344xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} {
    ^bb138(%1253: f32, %1254: f32, %1255: f32):
      %1256 = arith.mulf %1253, %1254 : f32
      linalg.yield %1256 : f32
    } -> tensor<128x344xf32>
    %1257 = tensor.collapse_shape %1209 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1258 = tensor.expand_shape %1257 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %1259 = tensor.empty() : tensor<8x344xf32>
    %1260 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1261 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1260 : f32) outs(%1259 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1262 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} ins(%1258, %1252 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1261 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1263 = tensor.collapse_shape %1262 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %1264 = tensor.expand_shape %1263 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1265 = tensor.empty() : tensor<1x8x344xf32>
    %1266 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1242, %1264 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1265 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb139(%1267: f32, %1268: f32, %1269: f32):
      %1270 = arith.mulf %1267, %1268 : f32
      linalg.yield %1270 : f32
    } -> tensor<1x8x344xf32>
    %1271 = tensor.empty() : tensor<344x128xf32>
    %1272 = linalg.transpose ins(%31:tensor<128x344xf32>) outs(%1271:tensor<344x128xf32>) permutation = [1, 0]
    %1273 = tensor.empty() : tensor<1x128xf32>
    %1274 = linalg.transpose ins(%32:tensor<128x1xf32>) outs(%1273:tensor<1x128xf32>) permutation = [1, 0]
    %1275 = tensor.empty() : tensor<344x128xf32>
    %1276 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1272, %1274 : tensor<344x128xf32>, tensor<1x128xf32>) outs(%1275 : tensor<344x128xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} {
    ^bb140(%1277: f32, %1278: f32, %1279: f32):
      %1280 = arith.mulf %1277, %1278 : f32
      linalg.yield %1280 : f32
    } -> tensor<344x128xf32>
    %1281 = tensor.collapse_shape %1266 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %1282 = tensor.expand_shape %1281 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %1283 = tensor.empty() : tensor<8x128xf32>
    %1284 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1285 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1284 : f32) outs(%1283 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1286 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} ins(%1282, %1276 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%1285 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1287 = tensor.collapse_shape %1286 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %1288 = tensor.expand_shape %1287 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1289 = tensor.empty() : tensor<1x8x128xf32>
    %1290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1162, %1288 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1289 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb141(%1291: f32, %1292: f32, %1293: f32):
      %1294 = arith.addf %1291, %1292 : f32
      linalg.yield %1294 : f32
    } -> tensor<1x8x128xf32>
    %1295 = tensor.empty() : tensor<1x8x128xf32>
    %1296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1290 : tensor<1x8x128xf32>) outs(%1295 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb142(%1297: f32, %1298: f32):
      %1299 = arith.constant 2.000000e+00 : f32
      %1300 = math.powf %1297, %1299 : f32
      linalg.yield %1300 : f32
    } -> tensor<1x8x128xf32>
    %1301 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %1302 = tensor.splat %1301 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1303 = linalg.reduce ins(%1296:tensor<1x8x128xf32>) outs(%1302:tensor<1x8xf32>) dimensions = [2]
    (%1304: f32, %1305: f32) {
      %1306 = arith.addf %1304, %1305 : f32
      linalg.yield %1306 : f32
    }
    %1307 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.280000e+02 : f32
    %1308 = tensor.splat %1307 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1309 = tensor.empty() : tensor<1x8xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1303, %1308 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1309 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb143(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.divf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x8xf32>
    %1315 = tensor.collapse_shape %1310 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32> into tensor<8xf32>
    %1316 = tensor.expand_shape %1315 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1317 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.000000e-05 : f32
    %1318 = tensor.splat %1317 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8x1xf32>
    %1319 = tensor.empty() : tensor<1x8x1xf32>
    %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316, %1318 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1319 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb144(%1321: f32, %1322: f32, %1323: f32):
      %1324 = arith.addf %1321, %1322 : f32
      linalg.yield %1324 : f32
    } -> tensor<1x8x1xf32>
    %1325 = tensor.empty() : tensor<1x8x1xf32>
    %1326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1320 : tensor<1x8x1xf32>) outs(%1325 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb145(%1327: f32, %1328: f32):
      %1329 = math.rsqrt %1327 : f32
      linalg.yield %1329 : f32
    } -> tensor<1x8x1xf32>
    %1330 = tensor.empty() : tensor<1x8x128xf32>
    %1331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1290, %1326 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1330 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb146(%1332: f32, %1333: f32, %1334: f32):
      %1335 = arith.mulf %1332, %1333 : f32
      linalg.yield %1335 : f32
    } -> tensor<1x8x128xf32>
    %1336 = tensor.empty() : tensor<1x8x128xf32>
    %1337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1331, %33 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1336 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb147(%1338: f32, %1339: f32, %1340: f32):
      %1341 = arith.mulf %1338, %1339 : f32
      linalg.yield %1341 : f32
    } -> tensor<1x8x128xf32>
    %1342 = tensor.empty() : tensor<128x256xf32>
    %1343 = linalg.transpose ins(%34:tensor<256x128xf32>) outs(%1342:tensor<128x256xf32>) permutation = [1, 0]
    %1344 = tensor.empty() : tensor<1x256xf32>
    %1345 = linalg.transpose ins(%35:tensor<256x1xf32>) outs(%1344:tensor<1x256xf32>) permutation = [1, 0]
    %1346 = tensor.empty() : tensor<128x256xf32>
    %1347 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1343, %1345 : tensor<128x256xf32>, tensor<1x256xf32>) outs(%1346 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} {
    ^bb148(%1348: f32, %1349: f32, %1350: f32):
      %1351 = arith.mulf %1348, %1349 : f32
      linalg.yield %1351 : f32
    } -> tensor<128x256xf32>
    %1352 = tensor.collapse_shape %1337 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1353 = tensor.expand_shape %1352 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1024xf32> into tensor<8x128xf32>
    %1354 = tensor.empty() : tensor<8x256xf32>
    %1355 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1356 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1355 : f32) outs(%1354 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1357 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} ins(%1353, %1347 : tensor<8x128xf32>, tensor<128x256xf32>) outs(%1356 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1358 = tensor.collapse_shape %1357 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<8x256xf32> into tensor<2048xf32>
    %1359 = tensor.expand_shape %1358 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<2048xf32> into tensor<1x8x256xf32>
    func.return %1359 : tensor<1x8x256xf32>
  }
}
