builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x67x2048xf32>, %1: tensor<1x1024x2048xf32>, %2: tensor<1x4096x2048xf32>, %3: tensor<2048x256xf32>, %4: tensor<2048xf32>, %5: tensor<2048x2048xf32>, %6: tensor<2048xf32>, %7: tensor<2048x256xf32>, %8: tensor<2048xf32>, %9: tensor<2048x2048xf32>, %10: tensor<2048xf32>, %11: tensor<2048xf32>, %12: tensor<6144x2048xf32>, %13: tensor<6144xf32>, %14: tensor<64xf32>, %15: tensor<64xf32>, %16: tensor<2048x2048xf32>, %17: tensor<2048xf32>, %18: tensor<2048x2048xf32>, %19: tensor<2048xf32>, %20: tensor<4096x2048xf32>, %21: tensor<4096xf32>, %22: tensor<64xf32>, %23: tensor<64xf32>, %24: tensor<2048x2048xf32>, %25: tensor<2048xf32>, %26: tensor<2048xf32>, %27: tensor<2048x2048xf32>, %28: tensor<2048xf32>, %29: tensor<2048x2048xf32>, %30: tensor<2048xf32>, %31: tensor<2048xf32>, %32: tensor<2048xf32>, %33: tensor<6144x2048xf32>, %34: tensor<6144xf32>, %35: tensor<64xf32>, %36: tensor<64xf32>, %37: tensor<2048x2048xf32>, %38: tensor<2048xf32>, %39: tensor<2048x2048xf32>, %40: tensor<2048xf32>, %41: tensor<4096x2048xf32>, %42: tensor<4096xf32>, %43: tensor<64xf32>, %44: tensor<64xf32>, %45: tensor<2048x2048xf32>, %46: tensor<2048xf32>, %47: tensor<2048xf32>, %48: tensor<2048x2048xf32>, %49: tensor<2048xf32>, %50: tensor<2048x2048xf32>, %51: tensor<2048xf32>, %52: tensor<2048xf32>, %53: tensor<2048xf32>, %54: tensor<2048x2048xf32>, %55: tensor<2048xf32>, %56: tensor<128x2048xf32>, %57: tensor<128xf32>, %58: tensor<1x65x2048xf32>, %59: tensor<1xf32>, %60: tensor<1xf32>, %61: tensor<1x32x2048xf32>, %62: tensor<1x4096x2048xf32>, %63: tensor<1x32xi1>) -> tensor<1x64x128xf32> {
    %64 = tensor.empty() : tensor<128xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%64 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb0(%66: f32):
      %67 = linalg.index 0 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.sitofp %68 : i64 to f32
      %70 = arith.constant 1.000000e+00 : f32
      %71 = arith.mulf %69, %70 : f32
      %72 = arith.constant 0.000000e+00 : f32
      %73 = arith.addf %72, %71 : f32
      linalg.yield %73 : f32
    } -> tensor<128xf32>
    %74 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} -9.2103405 : f32
    %75 = tensor.splat %74 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %76 = tensor.empty() : tensor<128xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%65, %75 : tensor<128xf32>, tensor<128xf32>) outs(%76 : tensor<128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb1(%78: f32, %79: f32, %80: f32):
      %81 = arith.mulf %78, %79 : f32
      linalg.yield %81 : f32
    } -> tensor<128xf32>
    %82 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} 1.280000e+02 : f32
    %83 = tensor.splat %82 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %84 = tensor.empty() : tensor<128xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%77, %83 : tensor<128xf32>, tensor<128xf32>) outs(%84 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb2(%86: f32, %87: f32, %88: f32):
      %89 = arith.divf %86, %87 : f32
      linalg.yield %89 : f32
    } -> tensor<128xf32>
    %90 = tensor.empty() : tensor<128xf32>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%85 : tensor<128xf32>) outs(%90 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb3(%92: f32, %93: f32):
      %94 = math.exp %92 : f32
      linalg.yield %94 : f32
    } -> tensor<128xf32>
    %95 = tensor.expand_shape %60 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<1xf32> into tensor<1x1xf32>
    %96 = tensor.expand_shape %91 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32> into tensor<1x128xf32>
    %97 = tensor.empty() : tensor<1x128xf32>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%95, %96 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%97 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb4(%99: f32, %100: f32, %101: f32):
      %102 = arith.mulf %99, %100 : f32
      linalg.yield %102 : f32
    } -> tensor<1x128xf32>
    %103 = tensor.empty() : tensor<1x128xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%98 : tensor<1x128xf32>) outs(%103 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb5(%105: f32, %106: f32):
      %107 = math.cos %105 : f32
      linalg.yield %107 : f32
    } -> tensor<1x128xf32>
    %108 = tensor.empty() : tensor<1x128xf32>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%98 : tensor<1x128xf32>) outs(%108 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb6(%110: f32, %111: f32):
      %112 = math.sin %110 : f32
      linalg.yield %112 : f32
    } -> tensor<1x128xf32>
    %113 = tensor.concat dim(1) %104, %109 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %114 = tensor.empty() : tensor<256x2048xf32>
    %115 = linalg.transpose ins(%3:tensor<2048x256xf32>) outs(%114:tensor<256x2048xf32>) permutation = [1, 0]
    %116 = tensor.empty() : tensor<1x2048xf32>
    %117 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %118 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%117 : f32) outs(%116 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %119 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.0", prov.transposed_b = "true"} ins(%113, %115 : tensor<1x256xf32>, tensor<256x2048xf32>) outs(%118 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %120 = tensor.empty() : tensor<1x2048xf32>
    %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%119, %4 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%120 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.0"} {
    ^bb7(%122: f32, %123: f32, %124: f32):
      %125 = arith.addf %122, %123 : f32
      linalg.yield %125 : f32
    } -> tensor<1x2048xf32>
    %126 = tensor.empty() : tensor<1x2048xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%121 : tensor<1x2048xf32>) outs(%126 : tensor<1x2048xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.1"} {
    ^bb8(%128: f32, %129: f32):
      %130 = arith.constant 1.000000e+00 : f32
      %131 = arith.negf %128 : f32
      %132 = math.exp %131 : f32
      %133 = arith.addf %130, %132 : f32
      %134 = arith.divf %130, %133 : f32
      linalg.yield %134 : f32
    } -> tensor<1x2048xf32>
    %135 = tensor.empty() : tensor<1x2048xf32>
    %136 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%121, %127 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%135 : tensor<1x2048xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.1"} {
    ^bb9(%137: f32, %138: f32, %139: f32):
      %140 = arith.mulf %137, %138 : f32
      linalg.yield %140 : f32
    } -> tensor<1x2048xf32>
    %141 = tensor.empty() : tensor<2048x2048xf32>
    %142 = linalg.transpose ins(%5:tensor<2048x2048xf32>) outs(%141:tensor<2048x2048xf32>) permutation = [1, 0]
    %143 = tensor.empty() : tensor<1x2048xf32>
    %144 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %145 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%144 : f32) outs(%143 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %146 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.2", prov.transposed_b = "true"} ins(%136, %142 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%145 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %147 = tensor.empty() : tensor<1x2048xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%146, %6 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%147 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.2"} {
    ^bb10(%149: f32, %150: f32, %151: f32):
      %152 = arith.addf %149, %150 : f32
      linalg.yield %152 : f32
    } -> tensor<1x2048xf32>
    %153 = tensor.collapse_shape %148 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1x2048xf32> into tensor<2048xf32>
    %154 = tensor.expand_shape %153 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<2048xf32> into tensor<1x1x2048xf32>
    %155 = tensor.empty() : tensor<128xf32>
    %156 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%155 : tensor<128xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb11(%157: f32):
      %158 = linalg.index 0 : index
      %159 = arith.index_cast %158 : index to i64
      %160 = arith.sitofp %159 : i64 to f32
      %161 = arith.constant 1.000000e+00 : f32
      %162 = arith.mulf %160, %161 : f32
      %163 = arith.constant 0.000000e+00 : f32
      %164 = arith.addf %163, %162 : f32
      linalg.yield %164 : f32
    } -> tensor<128xf32>
    %165 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} -9.2103405 : f32
    %166 = tensor.splat %165 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} : tensor<128xf32>
    %167 = tensor.empty() : tensor<128xf32>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%156, %166 : tensor<128xf32>, tensor<128xf32>) outs(%167 : tensor<128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb12(%169: f32, %170: f32, %171: f32):
      %172 = arith.mulf %169, %170 : f32
      linalg.yield %172 : f32
    } -> tensor<128xf32>
    %173 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} 1.280000e+02 : f32
    %174 = tensor.splat %173 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} : tensor<128xf32>
    %175 = tensor.empty() : tensor<128xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%168, %174 : tensor<128xf32>, tensor<128xf32>) outs(%175 : tensor<128xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb13(%177: f32, %178: f32, %179: f32):
      %180 = arith.divf %177, %178 : f32
      linalg.yield %180 : f32
    } -> tensor<128xf32>
    %181 = tensor.empty() : tensor<128xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%176 : tensor<128xf32>) outs(%181 : tensor<128xf32>) attrs =  {prov.region_id = "exp_1", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb14(%183: f32, %184: f32):
      %185 = math.exp %183 : f32
      linalg.yield %185 : f32
    } -> tensor<128xf32>
    %186 = tensor.expand_shape %59 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} : tensor<1xf32> into tensor<1x1xf32>
    %187 = tensor.expand_shape %182 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} : tensor<128xf32> into tensor<1x128xf32>
    %188 = tensor.empty() : tensor<1x128xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%186, %187 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%188 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb15(%190: f32, %191: f32, %192: f32):
      %193 = arith.mulf %190, %191 : f32
      linalg.yield %193 : f32
    } -> tensor<1x128xf32>
    %194 = tensor.empty() : tensor<1x128xf32>
    %195 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%189 : tensor<1x128xf32>) outs(%194 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb16(%196: f32, %197: f32):
      %198 = math.cos %196 : f32
      linalg.yield %198 : f32
    } -> tensor<1x128xf32>
    %199 = tensor.empty() : tensor<1x128xf32>
    %200 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%189 : tensor<1x128xf32>) outs(%199 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} {
    ^bb17(%201: f32, %202: f32):
      %203 = math.sin %201 : f32
      linalg.yield %203 : f32
    } -> tensor<1x128xf32>
    %204 = tensor.concat dim(1) %195, %200 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %205 = tensor.empty() : tensor<256x2048xf32>
    %206 = linalg.transpose ins(%7:tensor<2048x256xf32>) outs(%205:tensor<256x2048xf32>) permutation = [1, 0]
    %207 = tensor.empty() : tensor<1x2048xf32>
    %208 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %209 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%208 : f32) outs(%207 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %210 = linalg.matmul {prov.region_id = "matmul_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.0", prov.transposed_b = "true"} ins(%204, %206 : tensor<1x256xf32>, tensor<256x2048xf32>) outs(%209 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %211 = tensor.empty() : tensor<1x2048xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%210, %8 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%211 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.0"} {
    ^bb18(%213: f32, %214: f32, %215: f32):
      %216 = arith.addf %213, %214 : f32
      linalg.yield %216 : f32
    } -> tensor<1x2048xf32>
    %217 = tensor.empty() : tensor<1x2048xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212 : tensor<1x2048xf32>) outs(%217 : tensor<1x2048xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.1"} {
    ^bb19(%219: f32, %220: f32):
      %221 = arith.constant 1.000000e+00 : f32
      %222 = arith.negf %219 : f32
      %223 = math.exp %222 : f32
      %224 = arith.addf %221, %223 : f32
      %225 = arith.divf %221, %224 : f32
      linalg.yield %225 : f32
    } -> tensor<1x2048xf32>
    %226 = tensor.empty() : tensor<1x2048xf32>
    %227 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212, %218 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%226 : tensor<1x2048xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.1"} {
    ^bb20(%228: f32, %229: f32, %230: f32):
      %231 = arith.mulf %228, %229 : f32
      linalg.yield %231 : f32
    } -> tensor<1x2048xf32>
    %232 = tensor.empty() : tensor<2048x2048xf32>
    %233 = linalg.transpose ins(%9:tensor<2048x2048xf32>) outs(%232:tensor<2048x2048xf32>) permutation = [1, 0]
    %234 = tensor.empty() : tensor<1x2048xf32>
    %235 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %236 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%235 : f32) outs(%234 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %237 = linalg.matmul {prov.region_id = "matmul_3", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.2", prov.transposed_b = "true"} ins(%227, %233 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%236 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %238 = tensor.empty() : tensor<1x2048xf32>
    %239 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%237, %10 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%238 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.freq_embedder.mlp.2"} {
    ^bb21(%240: f32, %241: f32, %242: f32):
      %243 = arith.addf %240, %241 : f32
      linalg.yield %243 : f32
    } -> tensor<1x2048xf32>
    %244 = tensor.collapse_shape %239 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1x2048xf32> into tensor<2048xf32>
    %245 = tensor.expand_shape %244 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<2048xf32> into tensor<1x1x2048xf32>
    %246 = tensor.empty() : tensor<1x1x2048xf32>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<1x1x2048xf32>) outs(%246 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb22(%248: f32, %249: f32):
      linalg.yield %248 : f32
    } -> tensor<1x1x2048xf32>
    %250 = tensor.concat dim(1) %247, %245, %58 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x1x2048xf32>, tensor<1x1x2048xf32>, tensor<1x65x2048xf32>) -> tensor<1x67x2048xf32>
    %251 = tensor.empty() : tensor<1x67x2048xf32>
    %252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%250, %0 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%251 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb23(%253: f32, %254: f32, %255: f32):
      %256 = arith.addf %253, %254 : f32
      linalg.yield %256 : f32
    } -> tensor<1x67x2048xf32>
    %257 = "tensor.extract_slice"(%1) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 2048>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x1024x2048xf32>) -> tensor<1x32x2048xf32>
    %258 = tensor.empty() : tensor<1x32x2048xf32>
    %259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %257 : tensor<1x32x2048xf32>, tensor<1x32x2048xf32>) outs(%258 : tensor<1x32x2048xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb24(%260: f32, %261: f32, %262: f32):
      %263 = arith.addf %260, %261 : f32
      linalg.yield %263 : f32
    } -> tensor<1x32x2048xf32>
    %264 = tensor.empty() : tensor<1x4096x2048xf32>
    %265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %2 : tensor<1x4096x2048xf32>, tensor<1x4096x2048xf32>) outs(%264 : tensor<1x4096x2048xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb25(%266: f32, %267: f32, %268: f32):
      %269 = arith.addf %266, %267 : f32
      linalg.yield %269 : f32
    } -> tensor<1x4096x2048xf32>
    %270 = tensor.empty() : tensor<1x67x2048xf32>
    %271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252 : tensor<1x67x2048xf32>) outs(%270 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb26(%272: f32, %273: f32):
      %274 = arith.constant 2.000000e+00 : f32
      %275 = math.powf %272, %274 : f32
      linalg.yield %275 : f32
    } -> tensor<1x67x2048xf32>
    %276 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} 0.000000e+00 : f32
    %277 = tensor.splat %276 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} : tensor<1x67xf32>
    %278 = linalg.reduce ins(%271:tensor<1x67x2048xf32>) outs(%277:tensor<1x67xf32>) dimensions = [2]
    (%279: f32, %280: f32) {
      %281 = arith.addf %279, %280 : f32
      linalg.yield %281 : f32
    }
    %282 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} 2.048000e+03 : f32
    %283 = tensor.splat %282 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} : tensor<1x67xf32>
    %284 = tensor.empty() : tensor<1x67xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%278, %283 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%284 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb27(%286: f32, %287: f32, %288: f32):
      %289 = arith.divf %286, %287 : f32
      linalg.yield %289 : f32
    } -> tensor<1x67xf32>
    %290 = tensor.collapse_shape %285 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} : tensor<1x67xf32> into tensor<67xf32>
    %291 = tensor.expand_shape %290 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} : tensor<67xf32> into tensor<1x67x1xf32>
    %292 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} 1.000000e-06 : f32
    %293 = tensor.splat %292 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} : tensor<1x67x1xf32>
    %294 = tensor.empty() : tensor<1x67x1xf32>
    %295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%291, %293 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%294 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb28(%296: f32, %297: f32, %298: f32):
      %299 = arith.addf %296, %297 : f32
      linalg.yield %299 : f32
    } -> tensor<1x67x1xf32>
    %300 = tensor.empty() : tensor<1x67x1xf32>
    %301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%295 : tensor<1x67x1xf32>) outs(%300 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb29(%302: f32, %303: f32):
      %304 = math.rsqrt %302 : f32
      linalg.yield %304 : f32
    } -> tensor<1x67x1xf32>
    %305 = tensor.empty() : tensor<1x67x2048xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252, %301 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%305 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb30(%307: f32, %308: f32, %309: f32):
      %310 = arith.mulf %307, %308 : f32
      linalg.yield %310 : f32
    } -> tensor<1x67x2048xf32>
    %311 = tensor.empty() : tensor<1x67x2048xf32>
    %312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%306, %11 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%311 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm1"} {
    ^bb31(%313: f32, %314: f32, %315: f32):
      %316 = arith.mulf %313, %314 : f32
      linalg.yield %316 : f32
    } -> tensor<1x67x2048xf32>
    %317 = tensor.collapse_shape %312 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %318 = tensor.expand_shape %317 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv"} : tensor<137216xf32> into tensor<67x2048xf32>
    %319 = tensor.empty() : tensor<2048x6144xf32>
    %320 = linalg.transpose ins(%12:tensor<6144x2048xf32>) outs(%319:tensor<2048x6144xf32>) permutation = [1, 0]
    %321 = tensor.empty() : tensor<67x6144xf32>
    %322 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %323 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%322 : f32) outs(%321 : tensor<67x6144xf32>) -> tensor<67x6144xf32>
    %324 = linalg.matmul {prov.region_id = "matmul_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv", prov.transposed_b = "true"} ins(%318, %320 : tensor<67x2048xf32>, tensor<2048x6144xf32>) outs(%323 : tensor<67x6144xf32>) -> tensor<67x6144xf32>
    %325 = tensor.empty() : tensor<67x6144xf32>
    %326 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%324, %13 : tensor<67x6144xf32>, tensor<6144xf32>) outs(%325 : tensor<67x6144xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv"} {
    ^bb32(%327: f32, %328: f32, %329: f32):
      %330 = arith.addf %327, %328 : f32
      linalg.yield %330 : f32
    } -> tensor<67x6144xf32>
    %331 = tensor.collapse_shape %326 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv"} : tensor<67x6144xf32> into tensor<411648xf32>
    %332 = tensor.expand_shape %331 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 6144] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.qkv"} : tensor<411648xf32> into tensor<1x67x6144xf32>
    %333 = tensor.collapse_shape %332 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x67x6144xf32> into tensor<411648xf32>
    %334 = tensor.expand_shape %333 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 67, 3, 32, 64] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<411648xf32> into tensor<1x67x3x32x64xf32>
    %335 = tensor.empty() : tensor<3x1x32x67x64xf32>
    %336 = linalg.transpose ins(%334:tensor<1x67x3x32x64xf32>) outs(%335:tensor<3x1x32x67x64xf32>) permutation = [2, 0, 3, 1, 4]
    %337 = "tensor.extract_slice"(%336) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %338 = "tensor.extract_slice"(%336) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %339 = "tensor.extract_slice"(%336) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %340 = tensor.collapse_shape %337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %341 = tensor.expand_shape %340 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %342 = tensor.collapse_shape %338 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %343 = tensor.expand_shape %342 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %344 = tensor.collapse_shape %339 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %345 = tensor.expand_shape %344 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %346 = tensor.empty() : tensor<1x32x67x64xf32>
    %347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341 : tensor<1x32x67x64xf32>) outs(%346 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb33(%348: f32, %349: f32):
      %350 = arith.constant 2.000000e+00 : f32
      %351 = math.powf %348, %350 : f32
      linalg.yield %351 : f32
    } -> tensor<1x32x67x64xf32>
    %352 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} 0.000000e+00 : f32
    %353 = tensor.splat %352 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} : tensor<1x32x67xf32>
    %354 = linalg.reduce ins(%347:tensor<1x32x67x64xf32>) outs(%353:tensor<1x32x67xf32>) dimensions = [3]
    (%355: f32, %356: f32) {
      %357 = arith.addf %355, %356 : f32
      linalg.yield %357 : f32
    }
    %358 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} 6.400000e+01 : f32
    %359 = tensor.splat %358 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} : tensor<1x32x67xf32>
    %360 = tensor.empty() : tensor<1x32x67xf32>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%354, %359 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%360 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb34(%362: f32, %363: f32, %364: f32):
      %365 = arith.divf %362, %363 : f32
      linalg.yield %365 : f32
    } -> tensor<1x32x67xf32>
    %366 = tensor.collapse_shape %361 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %367 = tensor.expand_shape %366 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %368 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} 1.000000e-06 : f32
    %369 = tensor.splat %368 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} : tensor<1x32x67x1xf32>
    %370 = tensor.empty() : tensor<1x32x67x1xf32>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %369 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%370 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb35(%372: f32, %373: f32, %374: f32):
      %375 = arith.addf %372, %373 : f32
      linalg.yield %375 : f32
    } -> tensor<1x32x67x1xf32>
    %376 = tensor.empty() : tensor<1x32x67x1xf32>
    %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371 : tensor<1x32x67x1xf32>) outs(%376 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb36(%378: f32, %379: f32):
      %380 = math.rsqrt %378 : f32
      linalg.yield %380 : f32
    } -> tensor<1x32x67x1xf32>
    %381 = tensor.empty() : tensor<1x32x67x64xf32>
    %382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341, %377 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%381 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb37(%383: f32, %384: f32, %385: f32):
      %386 = arith.mulf %383, %384 : f32
      linalg.yield %386 : f32
    } -> tensor<1x32x67x64xf32>
    %387 = tensor.empty() : tensor<1x32x67x64xf32>
    %388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %14 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%387 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.q_norm"} {
    ^bb38(%389: f32, %390: f32, %391: f32):
      %392 = arith.mulf %389, %390 : f32
      linalg.yield %392 : f32
    } -> tensor<1x32x67x64xf32>
    %393 = tensor.empty() : tensor<1x32x67x64xf32>
    %394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343 : tensor<1x32x67x64xf32>) outs(%393 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb39(%395: f32, %396: f32):
      %397 = arith.constant 2.000000e+00 : f32
      %398 = math.powf %395, %397 : f32
      linalg.yield %398 : f32
    } -> tensor<1x32x67x64xf32>
    %399 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} 0.000000e+00 : f32
    %400 = tensor.splat %399 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} : tensor<1x32x67xf32>
    %401 = linalg.reduce ins(%394:tensor<1x32x67x64xf32>) outs(%400:tensor<1x32x67xf32>) dimensions = [3]
    (%402: f32, %403: f32) {
      %404 = arith.addf %402, %403 : f32
      linalg.yield %404 : f32
    }
    %405 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} 6.400000e+01 : f32
    %406 = tensor.splat %405 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} : tensor<1x32x67xf32>
    %407 = tensor.empty() : tensor<1x32x67xf32>
    %408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401, %406 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%407 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb40(%409: f32, %410: f32, %411: f32):
      %412 = arith.divf %409, %410 : f32
      linalg.yield %412 : f32
    } -> tensor<1x32x67xf32>
    %413 = tensor.collapse_shape %408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %414 = tensor.expand_shape %413 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %415 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} 1.000000e-06 : f32
    %416 = tensor.splat %415 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} : tensor<1x32x67x1xf32>
    %417 = tensor.empty() : tensor<1x32x67x1xf32>
    %418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%414, %416 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%417 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb41(%419: f32, %420: f32, %421: f32):
      %422 = arith.addf %419, %420 : f32
      linalg.yield %422 : f32
    } -> tensor<1x32x67x1xf32>
    %423 = tensor.empty() : tensor<1x32x67x1xf32>
    %424 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%418 : tensor<1x32x67x1xf32>) outs(%423 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb42(%425: f32, %426: f32):
      %427 = math.rsqrt %425 : f32
      linalg.yield %427 : f32
    } -> tensor<1x32x67x1xf32>
    %428 = tensor.empty() : tensor<1x32x67x64xf32>
    %429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %424 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%428 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb43(%430: f32, %431: f32, %432: f32):
      %433 = arith.mulf %430, %431 : f32
      linalg.yield %433 : f32
    } -> tensor<1x32x67x64xf32>
    %434 = tensor.empty() : tensor<1x32x67x64xf32>
    %435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%429, %15 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%434 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.k_norm"} {
    ^bb44(%436: f32, %437: f32, %438: f32):
      %439 = arith.mulf %436, %437 : f32
      linalg.yield %439 : f32
    } -> tensor<1x32x67x64xf32>
    %440 = tensor.empty() : tensor<1x32x64x67xf32>
    %441 = linalg.transpose ins(%435:tensor<1x32x67x64xf32>) outs(%440:tensor<1x32x64x67xf32>) permutation = [0, 1, 3, 2]
    %442 = tensor.empty() : tensor<1x32x67x64xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388 : tensor<1x32x67x64xf32>) outs(%442 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb45(%444: f32, %445: f32):
      linalg.yield %444 : f32
    } -> tensor<1x32x67x64xf32>
    %446 = tensor.collapse_shape %443 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %447 = tensor.expand_shape %446 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %448 = tensor.empty() : tensor<1x32x64x67xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441 : tensor<1x32x64x67xf32>) outs(%448 : tensor<1x32x64x67xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb46(%450: f32, %451: f32):
      linalg.yield %450 : f32
    } -> tensor<1x32x64x67xf32>
    %452 = tensor.collapse_shape %449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x64x67xf32> into tensor<137216xf32>
    %453 = tensor.expand_shape %452 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 67] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<32x64x67xf32>
    %454 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %455 = tensor.splat %454 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<32x67x67xf32>
    %456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%447, %453 : tensor<32x67x64xf32>, tensor<32x64x67xf32>) outs(%455 : tensor<32x67x67xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb47(%457: f32, %458: f32, %459: f32):
      %460 = arith.mulf %457, %458 : f32
      %461 = arith.addf %459, %460 : f32
      linalg.yield %461 : f32
    } -> tensor<32x67x67xf32>
    %462 = tensor.collapse_shape %456 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<32x67x67xf32> into tensor<143648xf32>
    %463 = tensor.expand_shape %462 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 67] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<143648xf32> into tensor<1x32x67x67xf32>
    %464 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 1.250000e-01 : f32
    %465 = tensor.splat %464 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67x67xf32>
    %466 = tensor.empty() : tensor<1x32x67x67xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%463, %465 : tensor<1x32x67x67xf32>, tensor<1x32x67x67xf32>) outs(%466 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb48(%468: f32, %469: f32, %470: f32):
      %471 = arith.mulf %468, %469 : f32
      linalg.yield %471 : f32
    } -> tensor<1x32x67x67xf32>
    %472 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0xff800000 : f32
    %473 = tensor.splat %472 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67xf32>
    %474 = linalg.reduce ins(%467:tensor<1x32x67x67xf32>) outs(%473:tensor<1x32x67xf32>) dimensions = [3]
    (%475: f32, %476: f32) {
      %477 = arith.maximumf %475, %476 : f32
      linalg.yield %477 : f32
    }
    %478 = tensor.collapse_shape %474 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %479 = tensor.expand_shape %478 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %480 = tensor.empty() : tensor<1x32x67x67xf32>
    %481 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%467, %479 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%480 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb49(%482: f32, %483: f32, %484: f32):
      %485 = arith.subf %482, %483 : f32
      linalg.yield %485 : f32
    } -> tensor<1x32x67x67xf32>
    %486 = tensor.empty() : tensor<1x32x67x67xf32>
    %487 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%481 : tensor<1x32x67x67xf32>) outs(%486 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb50(%488: f32, %489: f32):
      %490 = math.exp %488 : f32
      linalg.yield %490 : f32
    } -> tensor<1x32x67x67xf32>
    %491 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %492 = tensor.splat %491 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67xf32>
    %493 = linalg.reduce ins(%487:tensor<1x32x67x67xf32>) outs(%492:tensor<1x32x67xf32>) dimensions = [3]
    (%494: f32, %495: f32) {
      %496 = arith.addf %494, %495 : f32
      linalg.yield %496 : f32
    }
    %497 = tensor.collapse_shape %493 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %498 = tensor.expand_shape %497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %499 = tensor.empty() : tensor<1x32x67x67xf32>
    %500 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%487, %498 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%499 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb51(%501: f32, %502: f32, %503: f32):
      %504 = arith.divf %501, %502 : f32
      linalg.yield %504 : f32
    } -> tensor<1x32x67x67xf32>
    %505 = tensor.empty() : tensor<1x32x67x67xf32>
    %506 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500 : tensor<1x32x67x67xf32>) outs(%505 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb52(%507: f32, %508: f32):
      linalg.yield %507 : f32
    } -> tensor<1x32x67x67xf32>
    %509 = tensor.collapse_shape %506 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67x67xf32> into tensor<143648xf32>
    %510 = tensor.expand_shape %509 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 67] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<143648xf32> into tensor<32x67x67xf32>
    %511 = tensor.empty() : tensor<1x32x67x64xf32>
    %512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345 : tensor<1x32x67x64xf32>) outs(%511 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb53(%513: f32, %514: f32):
      linalg.yield %513 : f32
    } -> tensor<1x32x67x64xf32>
    %515 = tensor.collapse_shape %512 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %516 = tensor.expand_shape %515 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %517 = arith.constant {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %518 = tensor.splat %517 {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<32x67x64xf32>
    %519 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%510, %516 : tensor<32x67x67xf32>, tensor<32x67x64xf32>) outs(%518 : tensor<32x67x64xf32>) attrs =  {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb54(%520: f32, %521: f32, %522: f32):
      %523 = arith.mulf %520, %521 : f32
      %524 = arith.addf %522, %523 : f32
      linalg.yield %524 : f32
    } -> tensor<32x67x64xf32>
    %525 = tensor.collapse_shape %519 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<32x67x64xf32> into tensor<137216xf32>
    %526 = tensor.expand_shape %525 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %527 = tensor.empty() : tensor<1x67x32x64xf32>
    %528 = linalg.transpose ins(%526:tensor<1x32x67x64xf32>) outs(%527:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
    %529 = tensor.collapse_shape %528 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
    %530 = tensor.expand_shape %529 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %531 = tensor.collapse_shape %530 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %532 = tensor.expand_shape %531 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj"} : tensor<137216xf32> into tensor<67x2048xf32>
    %533 = tensor.empty() : tensor<2048x2048xf32>
    %534 = linalg.transpose ins(%16:tensor<2048x2048xf32>) outs(%533:tensor<2048x2048xf32>) permutation = [1, 0]
    %535 = tensor.empty() : tensor<67x2048xf32>
    %536 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %537 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%536 : f32) outs(%535 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %538 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj", prov.transposed_b = "true"} ins(%532, %534 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%537 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %539 = tensor.empty() : tensor<67x2048xf32>
    %540 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%538, %17 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%539 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj"} {
    ^bb55(%541: f32, %542: f32, %543: f32):
      %544 = arith.addf %541, %542 : f32
      linalg.yield %544 : f32
    } -> tensor<67x2048xf32>
    %545 = tensor.collapse_shape %540 [[0 : i64, 1 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj"} : tensor<67x2048xf32> into tensor<137216xf32>
    %546 = tensor.expand_shape %545 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.proj"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %547 = tensor.empty() : tensor<1x67x2048xf32>
    %548 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%546, %252 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%547 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb56(%549: f32, %550: f32, %551: f32):
      %552 = arith.addf %549, %550 : f32
      linalg.yield %552 : f32
    } -> tensor<1x67x2048xf32>
    %553 = tensor.empty() : tensor<1x67x2048xf32>
    %554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%548 : tensor<1x67x2048xf32>) outs(%553 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb57(%555: f32, %556: f32):
      %557 = arith.constant 2.000000e+00 : f32
      %558 = math.powf %555, %557 : f32
      linalg.yield %558 : f32
    } -> tensor<1x67x2048xf32>
    %559 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} 0.000000e+00 : f32
    %560 = tensor.splat %559 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} : tensor<1x67xf32>
    %561 = linalg.reduce ins(%554:tensor<1x67x2048xf32>) outs(%560:tensor<1x67xf32>) dimensions = [2]
    (%562: f32, %563: f32) {
      %564 = arith.addf %562, %563 : f32
      linalg.yield %564 : f32
    }
    %565 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} 2.048000e+03 : f32
    %566 = tensor.splat %565 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} : tensor<1x67xf32>
    %567 = tensor.empty() : tensor<1x67xf32>
    %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%561, %566 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%567 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb58(%569: f32, %570: f32, %571: f32):
      %572 = arith.divf %569, %570 : f32
      linalg.yield %572 : f32
    } -> tensor<1x67xf32>
    %573 = tensor.collapse_shape %568 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} : tensor<1x67xf32> into tensor<67xf32>
    %574 = tensor.expand_shape %573 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} : tensor<67xf32> into tensor<1x67x1xf32>
    %575 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} 1.000000e-06 : f32
    %576 = tensor.splat %575 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} : tensor<1x67x1xf32>
    %577 = tensor.empty() : tensor<1x67x1xf32>
    %578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%574, %576 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%577 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb59(%579: f32, %580: f32, %581: f32):
      %582 = arith.addf %579, %580 : f32
      linalg.yield %582 : f32
    } -> tensor<1x67x1xf32>
    %583 = tensor.empty() : tensor<1x67x1xf32>
    %584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%578 : tensor<1x67x1xf32>) outs(%583 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb60(%585: f32, %586: f32):
      %587 = math.rsqrt %585 : f32
      linalg.yield %587 : f32
    } -> tensor<1x67x1xf32>
    %588 = tensor.empty() : tensor<1x67x2048xf32>
    %589 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%548, %584 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%588 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb61(%590: f32, %591: f32, %592: f32):
      %593 = arith.mulf %590, %591 : f32
      linalg.yield %593 : f32
    } -> tensor<1x67x2048xf32>
    %594 = tensor.empty() : tensor<1x67x2048xf32>
    %595 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%589, %26 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%594 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm2"} {
    ^bb62(%596: f32, %597: f32, %598: f32):
      %599 = arith.mulf %596, %597 : f32
      linalg.yield %599 : f32
    } -> tensor<1x67x2048xf32>
    %600 = tensor.collapse_shape %595 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %601 = tensor.expand_shape %600 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q"} : tensor<137216xf32> into tensor<67x2048xf32>
    %602 = tensor.empty() : tensor<2048x2048xf32>
    %603 = linalg.transpose ins(%18:tensor<2048x2048xf32>) outs(%602:tensor<2048x2048xf32>) permutation = [1, 0]
    %604 = tensor.empty() : tensor<67x2048xf32>
    %605 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %606 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%605 : f32) outs(%604 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %607 = linalg.matmul {prov.region_id = "matmul_8", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q", prov.transposed_b = "true"} ins(%601, %603 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%606 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %608 = tensor.empty() : tensor<67x2048xf32>
    %609 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%607, %19 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%608 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q"} {
    ^bb63(%610: f32, %611: f32, %612: f32):
      %613 = arith.addf %610, %611 : f32
      linalg.yield %613 : f32
    } -> tensor<67x2048xf32>
    %614 = tensor.collapse_shape %609 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q"} : tensor<67x2048xf32> into tensor<137216xf32>
    %615 = tensor.expand_shape %614 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %616 = tensor.collapse_shape %615 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %617 = tensor.expand_shape %616 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 67, 32, 64] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<137216xf32> into tensor<1x67x32x64xf32>
    %618 = tensor.empty() : tensor<1x32x67x64xf32>
    %619 = linalg.transpose ins(%617:tensor<1x67x32x64xf32>) outs(%618:tensor<1x32x67x64xf32>) permutation = [0, 2, 1, 3]
    %620 = tensor.collapse_shape %259 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv"} : tensor<1x32x2048xf32> into tensor<65536xf32>
    %621 = tensor.expand_shape %620 [[0 : i64, 1 : i64]] output_shape [32, 2048] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv"} : tensor<65536xf32> into tensor<32x2048xf32>
    %622 = tensor.empty() : tensor<2048x4096xf32>
    %623 = linalg.transpose ins(%20:tensor<4096x2048xf32>) outs(%622:tensor<2048x4096xf32>) permutation = [1, 0]
    %624 = tensor.empty() : tensor<32x4096xf32>
    %625 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %626 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%625 : f32) outs(%624 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    %627 = linalg.matmul {prov.region_id = "matmul_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv", prov.transposed_b = "true"} ins(%621, %623 : tensor<32x2048xf32>, tensor<2048x4096xf32>) outs(%626 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    %628 = tensor.empty() : tensor<32x4096xf32>
    %629 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%627, %21 : tensor<32x4096xf32>, tensor<4096xf32>) outs(%628 : tensor<32x4096xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv"} {
    ^bb64(%630: f32, %631: f32, %632: f32):
      %633 = arith.addf %630, %631 : f32
      linalg.yield %633 : f32
    } -> tensor<32x4096xf32>
    %634 = tensor.collapse_shape %629 [[0 : i64, 1 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv"} : tensor<32x4096xf32> into tensor<131072xf32>
    %635 = tensor.expand_shape %634 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 4096] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.kv"} : tensor<131072xf32> into tensor<1x32x4096xf32>
    %636 = tensor.collapse_shape %635 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x4096xf32> into tensor<131072xf32>
    %637 = tensor.expand_shape %636 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 2, 32, 64] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<131072xf32> into tensor<1x32x2x32x64xf32>
    %638 = tensor.empty() : tensor<2x1x32x32x64xf32>
    %639 = linalg.transpose ins(%637:tensor<1x32x2x32x64xf32>) outs(%638:tensor<2x1x32x32x64xf32>) permutation = [2, 0, 3, 1, 4]
    %640 = "tensor.extract_slice"(%639) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : (tensor<2x1x32x32x64xf32>) -> tensor<1x1x32x32x64xf32>
    %641 = "tensor.extract_slice"(%639) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : (tensor<2x1x32x32x64xf32>) -> tensor<1x1x32x32x64xf32>
    %642 = tensor.collapse_shape %640 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x1x32x32x64xf32> into tensor<65536xf32>
    %643 = tensor.expand_shape %642 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 32, 64] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<1x32x32x64xf32>
    %644 = tensor.collapse_shape %641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x1x32x32x64xf32> into tensor<65536xf32>
    %645 = tensor.expand_shape %644 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 32, 64] {prov.region_id = "squeeze_4", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<1x32x32x64xf32>
    %646 = tensor.empty() : tensor<1x32x67x64xf32>
    %647 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%619 : tensor<1x32x67x64xf32>) outs(%646 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb65(%648: f32, %649: f32):
      %650 = arith.constant 2.000000e+00 : f32
      %651 = math.powf %648, %650 : f32
      linalg.yield %651 : f32
    } -> tensor<1x32x67x64xf32>
    %652 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} 0.000000e+00 : f32
    %653 = tensor.splat %652 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} : tensor<1x32x67xf32>
    %654 = linalg.reduce ins(%647:tensor<1x32x67x64xf32>) outs(%653:tensor<1x32x67xf32>) dimensions = [3]
    (%655: f32, %656: f32) {
      %657 = arith.addf %655, %656 : f32
      linalg.yield %657 : f32
    }
    %658 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} 6.400000e+01 : f32
    %659 = tensor.splat %658 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} : tensor<1x32x67xf32>
    %660 = tensor.empty() : tensor<1x32x67xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654, %659 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%660 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb66(%662: f32, %663: f32, %664: f32):
      %665 = arith.divf %662, %663 : f32
      linalg.yield %665 : f32
    } -> tensor<1x32x67xf32>
    %666 = tensor.collapse_shape %661 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %667 = tensor.expand_shape %666 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %668 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} 1.000000e-06 : f32
    %669 = tensor.splat %668 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} : tensor<1x32x67x1xf32>
    %670 = tensor.empty() : tensor<1x32x67x1xf32>
    %671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%667, %669 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%670 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb67(%672: f32, %673: f32, %674: f32):
      %675 = arith.addf %672, %673 : f32
      linalg.yield %675 : f32
    } -> tensor<1x32x67x1xf32>
    %676 = tensor.empty() : tensor<1x32x67x1xf32>
    %677 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%671 : tensor<1x32x67x1xf32>) outs(%676 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb68(%678: f32, %679: f32):
      %680 = math.rsqrt %678 : f32
      linalg.yield %680 : f32
    } -> tensor<1x32x67x1xf32>
    %681 = tensor.empty() : tensor<1x32x67x64xf32>
    %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%619, %677 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%681 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb69(%683: f32, %684: f32, %685: f32):
      %686 = arith.mulf %683, %684 : f32
      linalg.yield %686 : f32
    } -> tensor<1x32x67x64xf32>
    %687 = tensor.empty() : tensor<1x32x67x64xf32>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%682, %22 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%687 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.q_norm"} {
    ^bb70(%689: f32, %690: f32, %691: f32):
      %692 = arith.mulf %689, %690 : f32
      linalg.yield %692 : f32
    } -> tensor<1x32x67x64xf32>
    %693 = tensor.empty() : tensor<1x32x32x64xf32>
    %694 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643 : tensor<1x32x32x64xf32>) outs(%693 : tensor<1x32x32x64xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb71(%695: f32, %696: f32):
      %697 = arith.constant 2.000000e+00 : f32
      %698 = math.powf %695, %697 : f32
      linalg.yield %698 : f32
    } -> tensor<1x32x32x64xf32>
    %699 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} 0.000000e+00 : f32
    %700 = tensor.splat %699 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} : tensor<1x32x32xf32>
    %701 = linalg.reduce ins(%694:tensor<1x32x32x64xf32>) outs(%700:tensor<1x32x32xf32>) dimensions = [3]
    (%702: f32, %703: f32) {
      %704 = arith.addf %702, %703 : f32
      linalg.yield %704 : f32
    }
    %705 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} 6.400000e+01 : f32
    %706 = tensor.splat %705 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} : tensor<1x32x32xf32>
    %707 = tensor.empty() : tensor<1x32x32xf32>
    %708 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%701, %706 : tensor<1x32x32xf32>, tensor<1x32x32xf32>) outs(%707 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb72(%709: f32, %710: f32, %711: f32):
      %712 = arith.divf %709, %710 : f32
      linalg.yield %712 : f32
    } -> tensor<1x32x32xf32>
    %713 = tensor.collapse_shape %708 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %714 = tensor.expand_shape %713 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} : tensor<1024xf32> into tensor<1x32x32x1xf32>
    %715 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} 1.000000e-06 : f32
    %716 = tensor.splat %715 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} : tensor<1x32x32x1xf32>
    %717 = tensor.empty() : tensor<1x32x32x1xf32>
    %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%714, %716 : tensor<1x32x32x1xf32>, tensor<1x32x32x1xf32>) outs(%717 : tensor<1x32x32x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb73(%719: f32, %720: f32, %721: f32):
      %722 = arith.addf %719, %720 : f32
      linalg.yield %722 : f32
    } -> tensor<1x32x32x1xf32>
    %723 = tensor.empty() : tensor<1x32x32x1xf32>
    %724 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%718 : tensor<1x32x32x1xf32>) outs(%723 : tensor<1x32x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb74(%725: f32, %726: f32):
      %727 = math.rsqrt %725 : f32
      linalg.yield %727 : f32
    } -> tensor<1x32x32x1xf32>
    %728 = tensor.empty() : tensor<1x32x32x64xf32>
    %729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %724 : tensor<1x32x32x64xf32>, tensor<1x32x32x1xf32>) outs(%728 : tensor<1x32x32x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb75(%730: f32, %731: f32, %732: f32):
      %733 = arith.mulf %730, %731 : f32
      linalg.yield %733 : f32
    } -> tensor<1x32x32x64xf32>
    %734 = tensor.empty() : tensor<1x32x32x64xf32>
    %735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%729, %23 : tensor<1x32x32x64xf32>, tensor<64xf32>) outs(%734 : tensor<1x32x32x64xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.k_norm"} {
    ^bb76(%736: f32, %737: f32, %738: f32):
      %739 = arith.mulf %736, %737 : f32
      linalg.yield %739 : f32
    } -> tensor<1x32x32x64xf32>
    %740 = tensor.collapse_shape %63 [[0 : i64, 1 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32xi1> into tensor<32xi1>
    %741 = tensor.expand_shape %740 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32xi1> into tensor<1x1x1x32xi1>
    %742 = tensor.empty() : tensor<1x1x67x32xi1>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741 : tensor<1x1x1x32xi1>) outs(%742 : tensor<1x1x67x32xi1>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb77(%744: i1, %745: i1):
      linalg.yield %744 : i1
    } -> tensor<1x1x67x32xi1>
    %746 = tensor.empty() : tensor<1x32x64x32xf32>
    %747 = linalg.transpose ins(%735:tensor<1x32x32x64xf32>) outs(%746:tensor<1x32x64x32xf32>) permutation = [0, 1, 3, 2]
    %748 = tensor.empty() : tensor<1x32x67x64xf32>
    %749 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%688 : tensor<1x32x67x64xf32>) outs(%748 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb78(%750: f32, %751: f32):
      linalg.yield %750 : f32
    } -> tensor<1x32x67x64xf32>
    %752 = tensor.collapse_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %753 = tensor.expand_shape %752 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %754 = tensor.empty() : tensor<1x32x64x32xf32>
    %755 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747 : tensor<1x32x64x32xf32>) outs(%754 : tensor<1x32x64x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb79(%756: f32, %757: f32):
      linalg.yield %756 : f32
    } -> tensor<1x32x64x32xf32>
    %758 = tensor.collapse_shape %755 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x64x32xf32> into tensor<65536xf32>
    %759 = tensor.expand_shape %758 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 32] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<32x64x32xf32>
    %760 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %761 = tensor.splat %760 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32x67x32xf32>
    %762 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%753, %759 : tensor<32x67x64xf32>, tensor<32x64x32xf32>) outs(%761 : tensor<32x67x32xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb80(%763: f32, %764: f32, %765: f32):
      %766 = arith.mulf %763, %764 : f32
      %767 = arith.addf %765, %766 : f32
      linalg.yield %767 : f32
    } -> tensor<32x67x32xf32>
    %768 = tensor.collapse_shape %762 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32x67x32xf32> into tensor<68608xf32>
    %769 = tensor.expand_shape %768 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 32] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<68608xf32> into tensor<1x32x67x32xf32>
    %770 = arith.constant {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 1.250000e-01 : f32
    %771 = tensor.splat %770 {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67x32xf32>
    %772 = tensor.empty() : tensor<1x32x67x32xf32>
    %773 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%769, %771 : tensor<1x32x67x32xf32>, tensor<1x32x67x32xf32>) outs(%772 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb81(%774: f32, %775: f32, %776: f32):
      %777 = arith.mulf %774, %775 : f32
      linalg.yield %777 : f32
    } -> tensor<1x32x67x32xf32>
    %778 = tensor.empty() : tensor<1x1x67x32xi1>
    %779 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%743 : tensor<1x1x67x32xi1>) outs(%778 : tensor<1x1x67x32xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb82(%780: i1, %781: i1):
      %782 = arith.constant true
      %783 = arith.xori %780, %782 : i1
      linalg.yield %783 : i1
    } -> tensor<1x1x67x32xi1>
    %784 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0xff800000 : f32
    %785 = tensor.splat %784 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<f32>
    %786 = tensor.empty() : tensor<1x32x67x32xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%779, %785, %773 : tensor<1x1x67x32xi1>, tensor<f32>, tensor<1x32x67x32xf32>) outs(%786 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb83(%788: i1, %789: f32, %790: f32, %791: f32):
      %792 = arith.select %788, %789, %790 : f32
      linalg.yield %792 : f32
    } -> tensor<1x32x67x32xf32>
    %793 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0xff800000 : f32
    %794 = tensor.splat %793 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67xf32>
    %795 = linalg.reduce ins(%787:tensor<1x32x67x32xf32>) outs(%794:tensor<1x32x67xf32>) dimensions = [3]
    (%796: f32, %797: f32) {
      %798 = arith.maximumf %796, %797 : f32
      linalg.yield %798 : f32
    }
    %799 = tensor.collapse_shape %795 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %800 = tensor.expand_shape %799 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %801 = tensor.empty() : tensor<1x32x67x32xf32>
    %802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%787, %800 : tensor<1x32x67x32xf32>, tensor<1x32x67x1xf32>) outs(%801 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb84(%803: f32, %804: f32, %805: f32):
      %806 = arith.subf %803, %804 : f32
      linalg.yield %806 : f32
    } -> tensor<1x32x67x32xf32>
    %807 = tensor.empty() : tensor<1x32x67x32xf32>
    %808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%802 : tensor<1x32x67x32xf32>) outs(%807 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb85(%809: f32, %810: f32):
      %811 = math.exp %809 : f32
      linalg.yield %811 : f32
    } -> tensor<1x32x67x32xf32>
    %812 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %813 = tensor.splat %812 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67xf32>
    %814 = linalg.reduce ins(%808:tensor<1x32x67x32xf32>) outs(%813:tensor<1x32x67xf32>) dimensions = [3]
    (%815: f32, %816: f32) {
      %817 = arith.addf %815, %816 : f32
      linalg.yield %817 : f32
    }
    %818 = tensor.collapse_shape %814 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %819 = tensor.expand_shape %818 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %820 = tensor.empty() : tensor<1x32x67x32xf32>
    %821 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%808, %819 : tensor<1x32x67x32xf32>, tensor<1x32x67x1xf32>) outs(%820 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb86(%822: f32, %823: f32, %824: f32):
      %825 = arith.divf %822, %823 : f32
      linalg.yield %825 : f32
    } -> tensor<1x32x67x32xf32>
    %826 = tensor.empty() : tensor<1x32x67x32xf32>
    %827 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%821 : tensor<1x32x67x32xf32>) outs(%826 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb87(%828: f32, %829: f32):
      linalg.yield %828 : f32
    } -> tensor<1x32x67x32xf32>
    %830 = tensor.collapse_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x67x32xf32> into tensor<68608xf32>
    %831 = tensor.expand_shape %830 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 32] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<68608xf32> into tensor<32x67x32xf32>
    %832 = tensor.empty() : tensor<1x32x32x64xf32>
    %833 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645 : tensor<1x32x32x64xf32>) outs(%832 : tensor<1x32x32x64xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb88(%834: f32, %835: f32):
      linalg.yield %834 : f32
    } -> tensor<1x32x32x64xf32>
    %836 = tensor.collapse_shape %833 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x32x32x64xf32> into tensor<65536xf32>
    %837 = tensor.expand_shape %836 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 32, 64] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<32x32x64xf32>
    %838 = arith.constant {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %839 = tensor.splat %838 {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32x67x64xf32>
    %840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%831, %837 : tensor<32x67x32xf32>, tensor<32x32x64xf32>) outs(%839 : tensor<32x67x64xf32>) attrs =  {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb89(%841: f32, %842: f32, %843: f32):
      %844 = arith.mulf %841, %842 : f32
      %845 = arith.addf %843, %844 : f32
      linalg.yield %845 : f32
    } -> tensor<32x67x64xf32>
    %846 = tensor.collapse_shape %840 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32x67x64xf32> into tensor<137216xf32>
    %847 = tensor.expand_shape %846 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %848 = tensor.empty() : tensor<1x67x32x64xf32>
    %849 = linalg.transpose ins(%847:tensor<1x32x67x64xf32>) outs(%848:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
    %850 = tensor.collapse_shape %849 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
    %851 = tensor.expand_shape %850 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %852 = tensor.collapse_shape %851 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %853 = tensor.expand_shape %852 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj"} : tensor<137216xf32> into tensor<67x2048xf32>
    %854 = tensor.empty() : tensor<2048x2048xf32>
    %855 = linalg.transpose ins(%24:tensor<2048x2048xf32>) outs(%854:tensor<2048x2048xf32>) permutation = [1, 0]
    %856 = tensor.empty() : tensor<67x2048xf32>
    %857 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %858 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%857 : f32) outs(%856 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %859 = linalg.matmul {prov.region_id = "matmul_12", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj", prov.transposed_b = "true"} ins(%853, %855 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%858 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %860 = tensor.empty() : tensor<67x2048xf32>
    %861 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%859, %25 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%860 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj"} {
    ^bb90(%862: f32, %863: f32, %864: f32):
      %865 = arith.addf %862, %863 : f32
      linalg.yield %865 : f32
    } -> tensor<67x2048xf32>
    %866 = tensor.collapse_shape %861 [[0 : i64, 1 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj"} : tensor<67x2048xf32> into tensor<137216xf32>
    %867 = tensor.expand_shape %866 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.proj"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %868 = tensor.empty() : tensor<1x67x2048xf32>
    %869 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%867, %548 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%868 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb91(%870: f32, %871: f32, %872: f32):
      %873 = arith.addf %870, %871 : f32
      linalg.yield %873 : f32
    } -> tensor<1x67x2048xf32>
    %874 = tensor.empty() : tensor<1x67x2048xf32>
    %875 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%869 : tensor<1x67x2048xf32>) outs(%874 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb92(%876: f32, %877: f32):
      %878 = arith.constant 2.000000e+00 : f32
      %879 = math.powf %876, %878 : f32
      linalg.yield %879 : f32
    } -> tensor<1x67x2048xf32>
    %880 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} 0.000000e+00 : f32
    %881 = tensor.splat %880 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} : tensor<1x67xf32>
    %882 = linalg.reduce ins(%875:tensor<1x67x2048xf32>) outs(%881:tensor<1x67xf32>) dimensions = [2]
    (%883: f32, %884: f32) {
      %885 = arith.addf %883, %884 : f32
      linalg.yield %885 : f32
    }
    %886 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} 2.048000e+03 : f32
    %887 = tensor.splat %886 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} : tensor<1x67xf32>
    %888 = tensor.empty() : tensor<1x67xf32>
    %889 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%882, %887 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%888 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb93(%890: f32, %891: f32, %892: f32):
      %893 = arith.divf %890, %891 : f32
      linalg.yield %893 : f32
    } -> tensor<1x67xf32>
    %894 = tensor.collapse_shape %889 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} : tensor<1x67xf32> into tensor<67xf32>
    %895 = tensor.expand_shape %894 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} : tensor<67xf32> into tensor<1x67x1xf32>
    %896 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} 1.000000e-06 : f32
    %897 = tensor.splat %896 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} : tensor<1x67x1xf32>
    %898 = tensor.empty() : tensor<1x67x1xf32>
    %899 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895, %897 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%898 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb94(%900: f32, %901: f32, %902: f32):
      %903 = arith.addf %900, %901 : f32
      linalg.yield %903 : f32
    } -> tensor<1x67x1xf32>
    %904 = tensor.empty() : tensor<1x67x1xf32>
    %905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%899 : tensor<1x67x1xf32>) outs(%904 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb95(%906: f32, %907: f32):
      %908 = math.rsqrt %906 : f32
      linalg.yield %908 : f32
    } -> tensor<1x67x1xf32>
    %909 = tensor.empty() : tensor<1x67x2048xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%869, %905 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%909 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb96(%911: f32, %912: f32, %913: f32):
      %914 = arith.mulf %911, %912 : f32
      linalg.yield %914 : f32
    } -> tensor<1x67x2048xf32>
    %915 = tensor.empty() : tensor<1x67x2048xf32>
    %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%910, %31 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%915 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.norm3"} {
    ^bb97(%917: f32, %918: f32, %919: f32):
      %920 = arith.mulf %917, %918 : f32
      linalg.yield %920 : f32
    } -> tensor<1x67x2048xf32>
    %921 = tensor.collapse_shape %916 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %922 = tensor.expand_shape %921 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1"} : tensor<137216xf32> into tensor<67x2048xf32>
    %923 = tensor.empty() : tensor<2048x2048xf32>
    %924 = linalg.transpose ins(%27:tensor<2048x2048xf32>) outs(%923:tensor<2048x2048xf32>) permutation = [1, 0]
    %925 = tensor.empty() : tensor<67x2048xf32>
    %926 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %927 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%926 : f32) outs(%925 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %928 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1", prov.transposed_b = "true"} ins(%922, %924 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%927 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %929 = tensor.empty() : tensor<67x2048xf32>
    %930 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%928, %28 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%929 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1"} {
    ^bb98(%931: f32, %932: f32, %933: f32):
      %934 = arith.addf %931, %932 : f32
      linalg.yield %934 : f32
    } -> tensor<67x2048xf32>
    %935 = tensor.collapse_shape %930 [[0 : i64, 1 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1"} : tensor<67x2048xf32> into tensor<137216xf32>
    %936 = tensor.expand_shape %935 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc1"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %937 = tensor.empty() : tensor<1x67x2048xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%936 : tensor<1x67x2048xf32>) outs(%937 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.act"} {
    ^bb99(%939: f32, %940: f32):
      %941 = arith.constant 5.000000e-01 : f32
      %942 = arith.constant 1.000000e+00 : f32
      %943 = arith.constant 0.707106769 : f32
      %944 = arith.mulf %939, %943 : f32
      %945 = math.erf %944 : f32
      %946 = arith.addf %942, %945 : f32
      %947 = arith.mulf %941, %939 : f32
      %948 = arith.mulf %947, %946 : f32
      linalg.yield %948 : f32
    } -> tensor<1x67x2048xf32>
    %949 = tensor.collapse_shape %938 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %950 = tensor.expand_shape %949 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2"} : tensor<137216xf32> into tensor<67x2048xf32>
    %951 = tensor.empty() : tensor<2048x2048xf32>
    %952 = linalg.transpose ins(%29:tensor<2048x2048xf32>) outs(%951:tensor<2048x2048xf32>) permutation = [1, 0]
    %953 = tensor.empty() : tensor<67x2048xf32>
    %954 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %955 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%954 : f32) outs(%953 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %956 = linalg.matmul {prov.region_id = "matmul_14", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2", prov.transposed_b = "true"} ins(%950, %952 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%955 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %957 = tensor.empty() : tensor<67x2048xf32>
    %958 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%956, %30 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%957 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2"} {
    ^bb100(%959: f32, %960: f32, %961: f32):
      %962 = arith.addf %959, %960 : f32
      linalg.yield %962 : f32
    } -> tensor<67x2048xf32>
    %963 = tensor.collapse_shape %958 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2"} : tensor<67x2048xf32> into tensor<137216xf32>
    %964 = tensor.expand_shape %963 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.fc2"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %965 = tensor.empty() : tensor<1x67x2048xf32>
    %966 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%964, %869 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%965 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb101(%967: f32, %968: f32, %969: f32):
      %970 = arith.addf %967, %968 : f32
      linalg.yield %970 : f32
    } -> tensor<1x67x2048xf32>
    %971 = tensor.empty() : tensor<1x67x2048xf32>
    %972 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%966 : tensor<1x67x2048xf32>) outs(%971 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb102(%973: f32, %974: f32):
      %975 = arith.constant 2.000000e+00 : f32
      %976 = math.powf %973, %975 : f32
      linalg.yield %976 : f32
    } -> tensor<1x67x2048xf32>
    %977 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} 0.000000e+00 : f32
    %978 = tensor.splat %977 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} : tensor<1x67xf32>
    %979 = linalg.reduce ins(%972:tensor<1x67x2048xf32>) outs(%978:tensor<1x67xf32>) dimensions = [2]
    (%980: f32, %981: f32) {
      %982 = arith.addf %980, %981 : f32
      linalg.yield %982 : f32
    }
    %983 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} 2.048000e+03 : f32
    %984 = tensor.splat %983 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} : tensor<1x67xf32>
    %985 = tensor.empty() : tensor<1x67xf32>
    %986 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%979, %984 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%985 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb103(%987: f32, %988: f32, %989: f32):
      %990 = arith.divf %987, %988 : f32
      linalg.yield %990 : f32
    } -> tensor<1x67xf32>
    %991 = tensor.collapse_shape %986 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} : tensor<1x67xf32> into tensor<67xf32>
    %992 = tensor.expand_shape %991 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} : tensor<67xf32> into tensor<1x67x1xf32>
    %993 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} 1.000000e-06 : f32
    %994 = tensor.splat %993 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} : tensor<1x67x1xf32>
    %995 = tensor.empty() : tensor<1x67x1xf32>
    %996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%992, %994 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%995 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb104(%997: f32, %998: f32, %999: f32):
      %1000 = arith.addf %997, %998 : f32
      linalg.yield %1000 : f32
    } -> tensor<1x67x1xf32>
    %1001 = tensor.empty() : tensor<1x67x1xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%996 : tensor<1x67x1xf32>) outs(%1001 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb105(%1003: f32, %1004: f32):
      %1005 = math.rsqrt %1003 : f32
      linalg.yield %1005 : f32
    } -> tensor<1x67x1xf32>
    %1006 = tensor.empty() : tensor<1x67x2048xf32>
    %1007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%966, %1002 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%1006 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb106(%1008: f32, %1009: f32, %1010: f32):
      %1011 = arith.mulf %1008, %1009 : f32
      linalg.yield %1011 : f32
    } -> tensor<1x67x2048xf32>
    %1012 = tensor.empty() : tensor<1x67x2048xf32>
    %1013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1007, %32 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%1012 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm1"} {
    ^bb107(%1014: f32, %1015: f32, %1016: f32):
      %1017 = arith.mulf %1014, %1015 : f32
      linalg.yield %1017 : f32
    } -> tensor<1x67x2048xf32>
    %1018 = tensor.collapse_shape %1013 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1019 = tensor.expand_shape %1018 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1020 = tensor.empty() : tensor<2048x6144xf32>
    %1021 = linalg.transpose ins(%33:tensor<6144x2048xf32>) outs(%1020:tensor<2048x6144xf32>) permutation = [1, 0]
    %1022 = tensor.empty() : tensor<67x6144xf32>
    %1023 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1024 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1023 : f32) outs(%1022 : tensor<67x6144xf32>) -> tensor<67x6144xf32>
    %1025 = linalg.matmul {prov.region_id = "matmul_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv", prov.transposed_b = "true"} ins(%1019, %1021 : tensor<67x2048xf32>, tensor<2048x6144xf32>) outs(%1024 : tensor<67x6144xf32>) -> tensor<67x6144xf32>
    %1026 = tensor.empty() : tensor<67x6144xf32>
    %1027 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1025, %34 : tensor<67x6144xf32>, tensor<6144xf32>) outs(%1026 : tensor<67x6144xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv"} {
    ^bb108(%1028: f32, %1029: f32, %1030: f32):
      %1031 = arith.addf %1028, %1029 : f32
      linalg.yield %1031 : f32
    } -> tensor<67x6144xf32>
    %1032 = tensor.collapse_shape %1027 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv"} : tensor<67x6144xf32> into tensor<411648xf32>
    %1033 = tensor.expand_shape %1032 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 6144] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.qkv"} : tensor<411648xf32> into tensor<1x67x6144xf32>
    %1034 = tensor.collapse_shape %1033 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x67x6144xf32> into tensor<411648xf32>
    %1035 = tensor.expand_shape %1034 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 67, 3, 32, 64] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<411648xf32> into tensor<1x67x3x32x64xf32>
    %1036 = tensor.empty() : tensor<3x1x32x67x64xf32>
    %1037 = linalg.transpose ins(%1035:tensor<1x67x3x32x64xf32>) outs(%1036:tensor<3x1x32x67x64xf32>) permutation = [2, 0, 3, 1, 4]
    %1038 = "tensor.extract_slice"(%1037) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %1039 = "tensor.extract_slice"(%1037) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %1040 = "tensor.extract_slice"(%1037) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
    %1041 = tensor.collapse_shape %1038 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %1042 = tensor.expand_shape %1041 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_5", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %1043 = tensor.collapse_shape %1039 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %1044 = tensor.expand_shape %1043 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_6", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %1045 = tensor.collapse_shape %1040 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
    %1046 = tensor.expand_shape %1045 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "squeeze_7", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %1047 = tensor.empty() : tensor<1x32x67x64xf32>
    %1048 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1042 : tensor<1x32x67x64xf32>) outs(%1047 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb109(%1049: f32, %1050: f32):
      %1051 = arith.constant 2.000000e+00 : f32
      %1052 = math.powf %1049, %1051 : f32
      linalg.yield %1052 : f32
    } -> tensor<1x32x67x64xf32>
    %1053 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} 0.000000e+00 : f32
    %1054 = tensor.splat %1053 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} : tensor<1x32x67xf32>
    %1055 = linalg.reduce ins(%1048:tensor<1x32x67x64xf32>) outs(%1054:tensor<1x32x67xf32>) dimensions = [3]
    (%1056: f32, %1057: f32) {
      %1058 = arith.addf %1056, %1057 : f32
      linalg.yield %1058 : f32
    }
    %1059 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} 6.400000e+01 : f32
    %1060 = tensor.splat %1059 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} : tensor<1x32x67xf32>
    %1061 = tensor.empty() : tensor<1x32x67xf32>
    %1062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1055, %1060 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%1061 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb110(%1063: f32, %1064: f32, %1065: f32):
      %1066 = arith.divf %1063, %1064 : f32
      linalg.yield %1066 : f32
    } -> tensor<1x32x67xf32>
    %1067 = tensor.collapse_shape %1062 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1068 = tensor.expand_shape %1067 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1069 = arith.constant {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} 1.000000e-06 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} : tensor<1x32x67x1xf32>
    %1071 = tensor.empty() : tensor<1x32x67x1xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1068, %1070 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%1071 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb111(%1073: f32, %1074: f32, %1075: f32):
      %1076 = arith.addf %1073, %1074 : f32
      linalg.yield %1076 : f32
    } -> tensor<1x32x67x1xf32>
    %1077 = tensor.empty() : tensor<1x32x67x1xf32>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1072 : tensor<1x32x67x1xf32>) outs(%1077 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb112(%1079: f32, %1080: f32):
      %1081 = math.rsqrt %1079 : f32
      linalg.yield %1081 : f32
    } -> tensor<1x32x67x1xf32>
    %1082 = tensor.empty() : tensor<1x32x67x64xf32>
    %1083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1042, %1078 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%1082 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb113(%1084: f32, %1085: f32, %1086: f32):
      %1087 = arith.mulf %1084, %1085 : f32
      linalg.yield %1087 : f32
    } -> tensor<1x32x67x64xf32>
    %1088 = tensor.empty() : tensor<1x32x67x64xf32>
    %1089 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1083, %35 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%1088 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.q_norm"} {
    ^bb114(%1090: f32, %1091: f32, %1092: f32):
      %1093 = arith.mulf %1090, %1091 : f32
      linalg.yield %1093 : f32
    } -> tensor<1x32x67x64xf32>
    %1094 = tensor.empty() : tensor<1x32x67x64xf32>
    %1095 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1044 : tensor<1x32x67x64xf32>) outs(%1094 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb115(%1096: f32, %1097: f32):
      %1098 = arith.constant 2.000000e+00 : f32
      %1099 = math.powf %1096, %1098 : f32
      linalg.yield %1099 : f32
    } -> tensor<1x32x67x64xf32>
    %1100 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} 0.000000e+00 : f32
    %1101 = tensor.splat %1100 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} : tensor<1x32x67xf32>
    %1102 = linalg.reduce ins(%1095:tensor<1x32x67x64xf32>) outs(%1101:tensor<1x32x67xf32>) dimensions = [3]
    (%1103: f32, %1104: f32) {
      %1105 = arith.addf %1103, %1104 : f32
      linalg.yield %1105 : f32
    }
    %1106 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} 6.400000e+01 : f32
    %1107 = tensor.splat %1106 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} : tensor<1x32x67xf32>
    %1108 = tensor.empty() : tensor<1x32x67xf32>
    %1109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1102, %1107 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%1108 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb116(%1110: f32, %1111: f32, %1112: f32):
      %1113 = arith.divf %1110, %1111 : f32
      linalg.yield %1113 : f32
    } -> tensor<1x32x67xf32>
    %1114 = tensor.collapse_shape %1109 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1115 = tensor.expand_shape %1114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1116 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} 1.000000e-06 : f32
    %1117 = tensor.splat %1116 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} : tensor<1x32x67x1xf32>
    %1118 = tensor.empty() : tensor<1x32x67x1xf32>
    %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1115, %1117 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%1118 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb117(%1120: f32, %1121: f32, %1122: f32):
      %1123 = arith.addf %1120, %1121 : f32
      linalg.yield %1123 : f32
    } -> tensor<1x32x67x1xf32>
    %1124 = tensor.empty() : tensor<1x32x67x1xf32>
    %1125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1119 : tensor<1x32x67x1xf32>) outs(%1124 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_9", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb118(%1126: f32, %1127: f32):
      %1128 = math.rsqrt %1126 : f32
      linalg.yield %1128 : f32
    } -> tensor<1x32x67x1xf32>
    %1129 = tensor.empty() : tensor<1x32x67x64xf32>
    %1130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1044, %1125 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%1129 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb119(%1131: f32, %1132: f32, %1133: f32):
      %1134 = arith.mulf %1131, %1132 : f32
      linalg.yield %1134 : f32
    } -> tensor<1x32x67x64xf32>
    %1135 = tensor.empty() : tensor<1x32x67x64xf32>
    %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1130, %36 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%1135 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.k_norm"} {
    ^bb120(%1137: f32, %1138: f32, %1139: f32):
      %1140 = arith.mulf %1137, %1138 : f32
      linalg.yield %1140 : f32
    } -> tensor<1x32x67x64xf32>
    %1141 = tensor.empty() : tensor<1x32x64x67xf32>
    %1142 = linalg.transpose ins(%1136:tensor<1x32x67x64xf32>) outs(%1141:tensor<1x32x64x67xf32>) permutation = [0, 1, 3, 2]
    %1143 = tensor.empty() : tensor<1x32x67x64xf32>
    %1144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x32x67x64xf32>) outs(%1143 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb121(%1145: f32, %1146: f32):
      linalg.yield %1145 : f32
    } -> tensor<1x32x67x64xf32>
    %1147 = tensor.collapse_shape %1144 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %1148 = tensor.expand_shape %1147 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %1149 = tensor.empty() : tensor<1x32x64x67xf32>
    %1150 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1142 : tensor<1x32x64x67xf32>) outs(%1149 : tensor<1x32x64x67xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb122(%1151: f32, %1152: f32):
      linalg.yield %1151 : f32
    } -> tensor<1x32x64x67xf32>
    %1153 = tensor.collapse_shape %1150 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x64x67xf32> into tensor<137216xf32>
    %1154 = tensor.expand_shape %1153 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 67] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<32x64x67xf32>
    %1155 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1156 = tensor.splat %1155 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<32x67x67xf32>
    %1157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1148, %1154 : tensor<32x67x64xf32>, tensor<32x64x67xf32>) outs(%1156 : tensor<32x67x67xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb123(%1158: f32, %1159: f32, %1160: f32):
      %1161 = arith.mulf %1158, %1159 : f32
      %1162 = arith.addf %1160, %1161 : f32
      linalg.yield %1162 : f32
    } -> tensor<32x67x67xf32>
    %1163 = tensor.collapse_shape %1157 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<32x67x67xf32> into tensor<143648xf32>
    %1164 = tensor.expand_shape %1163 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 67] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<143648xf32> into tensor<1x32x67x67xf32>
    %1165 = arith.constant {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 1.250000e-01 : f32
    %1166 = tensor.splat %1165 {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67x67xf32>
    %1167 = tensor.empty() : tensor<1x32x67x67xf32>
    %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1164, %1166 : tensor<1x32x67x67xf32>, tensor<1x32x67x67xf32>) outs(%1167 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb124(%1169: f32, %1170: f32, %1171: f32):
      %1172 = arith.mulf %1169, %1170 : f32
      linalg.yield %1172 : f32
    } -> tensor<1x32x67x67xf32>
    %1173 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0xff800000 : f32
    %1174 = tensor.splat %1173 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67xf32>
    %1175 = linalg.reduce ins(%1168:tensor<1x32x67x67xf32>) outs(%1174:tensor<1x32x67xf32>) dimensions = [3]
    (%1176: f32, %1177: f32) {
      %1178 = arith.maximumf %1176, %1177 : f32
      linalg.yield %1178 : f32
    }
    %1179 = tensor.collapse_shape %1175 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1180 = tensor.expand_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1181 = tensor.empty() : tensor<1x32x67x67xf32>
    %1182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1168, %1180 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%1181 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb125(%1183: f32, %1184: f32, %1185: f32):
      %1186 = arith.subf %1183, %1184 : f32
      linalg.yield %1186 : f32
    } -> tensor<1x32x67x67xf32>
    %1187 = tensor.empty() : tensor<1x32x67x67xf32>
    %1188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1182 : tensor<1x32x67x67xf32>) outs(%1187 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb126(%1189: f32, %1190: f32):
      %1191 = math.exp %1189 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x32x67x67xf32>
    %1192 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1193 = tensor.splat %1192 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67xf32>
    %1194 = linalg.reduce ins(%1188:tensor<1x32x67x67xf32>) outs(%1193:tensor<1x32x67xf32>) dimensions = [3]
    (%1195: f32, %1196: f32) {
      %1197 = arith.addf %1195, %1196 : f32
      linalg.yield %1197 : f32
    }
    %1198 = tensor.collapse_shape %1194 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1199 = tensor.expand_shape %1198 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1200 = tensor.empty() : tensor<1x32x67x67xf32>
    %1201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1188, %1199 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%1200 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb127(%1202: f32, %1203: f32, %1204: f32):
      %1205 = arith.divf %1202, %1203 : f32
      linalg.yield %1205 : f32
    } -> tensor<1x32x67x67xf32>
    %1206 = tensor.empty() : tensor<1x32x67x67xf32>
    %1207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1201 : tensor<1x32x67x67xf32>) outs(%1206 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb128(%1208: f32, %1209: f32):
      linalg.yield %1208 : f32
    } -> tensor<1x32x67x67xf32>
    %1210 = tensor.collapse_shape %1207 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67x67xf32> into tensor<143648xf32>
    %1211 = tensor.expand_shape %1210 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 67] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<143648xf32> into tensor<32x67x67xf32>
    %1212 = tensor.empty() : tensor<1x32x67x64xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1046 : tensor<1x32x67x64xf32>) outs(%1212 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb129(%1214: f32, %1215: f32):
      linalg.yield %1214 : f32
    } -> tensor<1x32x67x64xf32>
    %1216 = tensor.collapse_shape %1213 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %1217 = tensor.expand_shape %1216 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %1218 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1219 = tensor.splat %1218 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<32x67x64xf32>
    %1220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1211, %1217 : tensor<32x67x67xf32>, tensor<32x67x64xf32>) outs(%1219 : tensor<32x67x64xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb130(%1221: f32, %1222: f32, %1223: f32):
      %1224 = arith.mulf %1221, %1222 : f32
      %1225 = arith.addf %1223, %1224 : f32
      linalg.yield %1225 : f32
    } -> tensor<32x67x64xf32>
    %1226 = tensor.collapse_shape %1220 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<32x67x64xf32> into tensor<137216xf32>
    %1227 = tensor.expand_shape %1226 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %1228 = tensor.empty() : tensor<1x67x32x64xf32>
    %1229 = linalg.transpose ins(%1227:tensor<1x32x67x64xf32>) outs(%1228:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
    %1230 = tensor.collapse_shape %1229 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
    %1231 = tensor.expand_shape %1230 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1232 = tensor.collapse_shape %1231 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1233 = tensor.expand_shape %1232 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1234 = tensor.empty() : tensor<2048x2048xf32>
    %1235 = linalg.transpose ins(%37:tensor<2048x2048xf32>) outs(%1234:tensor<2048x2048xf32>) permutation = [1, 0]
    %1236 = tensor.empty() : tensor<67x2048xf32>
    %1237 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1238 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1237 : f32) outs(%1236 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1239 = linalg.matmul {prov.region_id = "matmul_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj", prov.transposed_b = "true"} ins(%1233, %1235 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1238 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1240 = tensor.empty() : tensor<67x2048xf32>
    %1241 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1239, %38 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1240 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj"} {
    ^bb131(%1242: f32, %1243: f32, %1244: f32):
      %1245 = arith.addf %1242, %1243 : f32
      linalg.yield %1245 : f32
    } -> tensor<67x2048xf32>
    %1246 = tensor.collapse_shape %1241 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1247 = tensor.expand_shape %1246 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.proj"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1248 = tensor.empty() : tensor<1x67x2048xf32>
    %1249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1247, %966 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%1248 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb132(%1250: f32, %1251: f32, %1252: f32):
      %1253 = arith.addf %1250, %1251 : f32
      linalg.yield %1253 : f32
    } -> tensor<1x67x2048xf32>
    %1254 = tensor.empty() : tensor<1x67x2048xf32>
    %1255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1249 : tensor<1x67x2048xf32>) outs(%1254 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb133(%1256: f32, %1257: f32):
      %1258 = arith.constant 2.000000e+00 : f32
      %1259 = math.powf %1256, %1258 : f32
      linalg.yield %1259 : f32
    } -> tensor<1x67x2048xf32>
    %1260 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} 0.000000e+00 : f32
    %1261 = tensor.splat %1260 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} : tensor<1x67xf32>
    %1262 = linalg.reduce ins(%1255:tensor<1x67x2048xf32>) outs(%1261:tensor<1x67xf32>) dimensions = [2]
    (%1263: f32, %1264: f32) {
      %1265 = arith.addf %1263, %1264 : f32
      linalg.yield %1265 : f32
    }
    %1266 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} 2.048000e+03 : f32
    %1267 = tensor.splat %1266 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} : tensor<1x67xf32>
    %1268 = tensor.empty() : tensor<1x67xf32>
    %1269 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1262, %1267 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%1268 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb134(%1270: f32, %1271: f32, %1272: f32):
      %1273 = arith.divf %1270, %1271 : f32
      linalg.yield %1273 : f32
    } -> tensor<1x67xf32>
    %1274 = tensor.collapse_shape %1269 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} : tensor<1x67xf32> into tensor<67xf32>
    %1275 = tensor.expand_shape %1274 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} : tensor<67xf32> into tensor<1x67x1xf32>
    %1276 = arith.constant {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} 1.000000e-06 : f32
    %1277 = tensor.splat %1276 {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} : tensor<1x67x1xf32>
    %1278 = tensor.empty() : tensor<1x67x1xf32>
    %1279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %1277 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%1278 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb135(%1280: f32, %1281: f32, %1282: f32):
      %1283 = arith.addf %1280, %1281 : f32
      linalg.yield %1283 : f32
    } -> tensor<1x67x1xf32>
    %1284 = tensor.empty() : tensor<1x67x1xf32>
    %1285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1279 : tensor<1x67x1xf32>) outs(%1284 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_10", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb136(%1286: f32, %1287: f32):
      %1288 = math.rsqrt %1286 : f32
      linalg.yield %1288 : f32
    } -> tensor<1x67x1xf32>
    %1289 = tensor.empty() : tensor<1x67x2048xf32>
    %1290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1249, %1285 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%1289 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb137(%1291: f32, %1292: f32, %1293: f32):
      %1294 = arith.mulf %1291, %1292 : f32
      linalg.yield %1294 : f32
    } -> tensor<1x67x2048xf32>
    %1295 = tensor.empty() : tensor<1x67x2048xf32>
    %1296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1290, %47 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%1295 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm2"} {
    ^bb138(%1297: f32, %1298: f32, %1299: f32):
      %1300 = arith.mulf %1297, %1298 : f32
      linalg.yield %1300 : f32
    } -> tensor<1x67x2048xf32>
    %1301 = tensor.collapse_shape %1296 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1302 = tensor.expand_shape %1301 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1303 = tensor.empty() : tensor<2048x2048xf32>
    %1304 = linalg.transpose ins(%39:tensor<2048x2048xf32>) outs(%1303:tensor<2048x2048xf32>) permutation = [1, 0]
    %1305 = tensor.empty() : tensor<67x2048xf32>
    %1306 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1307 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1306 : f32) outs(%1305 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1308 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q", prov.transposed_b = "true"} ins(%1302, %1304 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1307 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1309 = tensor.empty() : tensor<67x2048xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1308, %40 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1309 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q"} {
    ^bb139(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.addf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<67x2048xf32>
    %1315 = tensor.collapse_shape %1310 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1316 = tensor.expand_shape %1315 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1317 = tensor.collapse_shape %1316 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1318 = tensor.expand_shape %1317 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 67, 32, 64] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<137216xf32> into tensor<1x67x32x64xf32>
    %1319 = tensor.empty() : tensor<1x32x67x64xf32>
    %1320 = linalg.transpose ins(%1318:tensor<1x67x32x64xf32>) outs(%1319:tensor<1x32x67x64xf32>) permutation = [0, 2, 1, 3]
    %1321 = tensor.collapse_shape %265 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv"} : tensor<1x4096x2048xf32> into tensor<8388608xf32>
    %1322 = tensor.expand_shape %1321 [[0 : i64, 1 : i64]] output_shape [4096, 2048] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv"} : tensor<8388608xf32> into tensor<4096x2048xf32>
    %1323 = tensor.empty() : tensor<2048x4096xf32>
    %1324 = linalg.transpose ins(%41:tensor<4096x2048xf32>) outs(%1323:tensor<2048x4096xf32>) permutation = [1, 0]
    %1325 = tensor.empty() : tensor<4096x4096xf32>
    %1326 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1327 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1326 : f32) outs(%1325 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1328 = linalg.matmul {prov.region_id = "matmul_20", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv", prov.transposed_b = "true"} ins(%1322, %1324 : tensor<4096x2048xf32>, tensor<2048x4096xf32>) outs(%1327 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1329 = tensor.empty() : tensor<4096x4096xf32>
    %1330 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1328, %42 : tensor<4096x4096xf32>, tensor<4096xf32>) outs(%1329 : tensor<4096x4096xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv"} {
    ^bb140(%1331: f32, %1332: f32, %1333: f32):
      %1334 = arith.addf %1331, %1332 : f32
      linalg.yield %1334 : f32
    } -> tensor<4096x4096xf32>
    %1335 = tensor.collapse_shape %1330 [[0 : i64, 1 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv"} : tensor<4096x4096xf32> into tensor<16777216xf32>
    %1336 = tensor.expand_shape %1335 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4096, 4096] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.kv"} : tensor<16777216xf32> into tensor<1x4096x4096xf32>
    %1337 = tensor.collapse_shape %1336 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x4096x4096xf32> into tensor<16777216xf32>
    %1338 = tensor.expand_shape %1337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4096, 2, 32, 64] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<16777216xf32> into tensor<1x4096x2x32x64xf32>
    %1339 = tensor.empty() : tensor<2x1x32x4096x64xf32>
    %1340 = linalg.transpose ins(%1338:tensor<1x4096x2x32x64xf32>) outs(%1339:tensor<2x1x32x4096x64xf32>) permutation = [2, 0, 3, 1, 4]
    %1341 = "tensor.extract_slice"(%1340) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 4096, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : (tensor<2x1x32x4096x64xf32>) -> tensor<1x1x32x4096x64xf32>
    %1342 = "tensor.extract_slice"(%1340) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 4096, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : (tensor<2x1x32x4096x64xf32>) -> tensor<1x1x32x4096x64xf32>
    %1343 = tensor.collapse_shape %1341 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x1x32x4096x64xf32> into tensor<8388608xf32>
    %1344 = tensor.expand_shape %1343 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4096, 64] {prov.region_id = "squeeze_8", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8388608xf32> into tensor<1x32x4096x64xf32>
    %1345 = tensor.collapse_shape %1342 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x1x32x4096x64xf32> into tensor<8388608xf32>
    %1346 = tensor.expand_shape %1345 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4096, 64] {prov.region_id = "squeeze_9", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8388608xf32> into tensor<1x32x4096x64xf32>
    %1347 = tensor.empty() : tensor<1x32x67x64xf32>
    %1348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1320 : tensor<1x32x67x64xf32>) outs(%1347 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "pow_11", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb141(%1349: f32, %1350: f32):
      %1351 = arith.constant 2.000000e+00 : f32
      %1352 = math.powf %1349, %1351 : f32
      linalg.yield %1352 : f32
    } -> tensor<1x32x67x64xf32>
    %1353 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} 0.000000e+00 : f32
    %1354 = tensor.splat %1353 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} : tensor<1x32x67xf32>
    %1355 = linalg.reduce ins(%1348:tensor<1x32x67x64xf32>) outs(%1354:tensor<1x32x67xf32>) dimensions = [3]
    (%1356: f32, %1357: f32) {
      %1358 = arith.addf %1356, %1357 : f32
      linalg.yield %1358 : f32
    }
    %1359 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} 6.400000e+01 : f32
    %1360 = tensor.splat %1359 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} : tensor<1x32x67xf32>
    %1361 = tensor.empty() : tensor<1x32x67xf32>
    %1362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1355, %1360 : tensor<1x32x67xf32>, tensor<1x32x67xf32>) outs(%1361 : tensor<1x32x67xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb142(%1363: f32, %1364: f32, %1365: f32):
      %1366 = arith.divf %1363, %1364 : f32
      linalg.yield %1366 : f32
    } -> tensor<1x32x67xf32>
    %1367 = tensor.collapse_shape %1362 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1368 = tensor.expand_shape %1367 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1369 = arith.constant {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} 1.000000e-06 : f32
    %1370 = tensor.splat %1369 {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} : tensor<1x32x67x1xf32>
    %1371 = tensor.empty() : tensor<1x32x67x1xf32>
    %1372 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1368, %1370 : tensor<1x32x67x1xf32>, tensor<1x32x67x1xf32>) outs(%1371 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb143(%1373: f32, %1374: f32, %1375: f32):
      %1376 = arith.addf %1373, %1374 : f32
      linalg.yield %1376 : f32
    } -> tensor<1x32x67x1xf32>
    %1377 = tensor.empty() : tensor<1x32x67x1xf32>
    %1378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1372 : tensor<1x32x67x1xf32>) outs(%1377 : tensor<1x32x67x1xf32>) attrs =  {prov.region_id = "rsqrt_11", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb144(%1379: f32, %1380: f32):
      %1381 = math.rsqrt %1379 : f32
      linalg.yield %1381 : f32
    } -> tensor<1x32x67x1xf32>
    %1382 = tensor.empty() : tensor<1x32x67x64xf32>
    %1383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1320, %1378 : tensor<1x32x67x64xf32>, tensor<1x32x67x1xf32>) outs(%1382 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb145(%1384: f32, %1385: f32, %1386: f32):
      %1387 = arith.mulf %1384, %1385 : f32
      linalg.yield %1387 : f32
    } -> tensor<1x32x67x64xf32>
    %1388 = tensor.empty() : tensor<1x32x67x64xf32>
    %1389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1383, %43 : tensor<1x32x67x64xf32>, tensor<64xf32>) outs(%1388 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.q_norm"} {
    ^bb146(%1390: f32, %1391: f32, %1392: f32):
      %1393 = arith.mulf %1390, %1391 : f32
      linalg.yield %1393 : f32
    } -> tensor<1x32x67x64xf32>
    %1394 = tensor.empty() : tensor<1x32x4096x64xf32>
    %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1344 : tensor<1x32x4096x64xf32>) outs(%1394 : tensor<1x32x4096x64xf32>) attrs =  {prov.region_id = "pow_12", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb147(%1396: f32, %1397: f32):
      %1398 = arith.constant 2.000000e+00 : f32
      %1399 = math.powf %1396, %1398 : f32
      linalg.yield %1399 : f32
    } -> tensor<1x32x4096x64xf32>
    %1400 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} 0.000000e+00 : f32
    %1401 = tensor.splat %1400 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} : tensor<1x32x4096xf32>
    %1402 = linalg.reduce ins(%1395:tensor<1x32x4096x64xf32>) outs(%1401:tensor<1x32x4096xf32>) dimensions = [3]
    (%1403: f32, %1404: f32) {
      %1405 = arith.addf %1403, %1404 : f32
      linalg.yield %1405 : f32
    }
    %1406 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} 6.400000e+01 : f32
    %1407 = tensor.splat %1406 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} : tensor<1x32x4096xf32>
    %1408 = tensor.empty() : tensor<1x32x4096xf32>
    %1409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1402, %1407 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%1408 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb148(%1410: f32, %1411: f32, %1412: f32):
      %1413 = arith.divf %1410, %1411 : f32
      linalg.yield %1413 : f32
    } -> tensor<1x32x4096xf32>
    %1414 = tensor.collapse_shape %1409 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} : tensor<1x32x4096xf32> into tensor<131072xf32>
    %1415 = tensor.expand_shape %1414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4096, 1] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} : tensor<131072xf32> into tensor<1x32x4096x1xf32>
    %1416 = arith.constant {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} 1.000000e-06 : f32
    %1417 = tensor.splat %1416 {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} : tensor<1x32x4096x1xf32>
    %1418 = tensor.empty() : tensor<1x32x4096x1xf32>
    %1419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1415, %1417 : tensor<1x32x4096x1xf32>, tensor<1x32x4096x1xf32>) outs(%1418 : tensor<1x32x4096x1xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb149(%1420: f32, %1421: f32, %1422: f32):
      %1423 = arith.addf %1420, %1421 : f32
      linalg.yield %1423 : f32
    } -> tensor<1x32x4096x1xf32>
    %1424 = tensor.empty() : tensor<1x32x4096x1xf32>
    %1425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1419 : tensor<1x32x4096x1xf32>) outs(%1424 : tensor<1x32x4096x1xf32>) attrs =  {prov.region_id = "rsqrt_12", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb150(%1426: f32, %1427: f32):
      %1428 = math.rsqrt %1426 : f32
      linalg.yield %1428 : f32
    } -> tensor<1x32x4096x1xf32>
    %1429 = tensor.empty() : tensor<1x32x4096x64xf32>
    %1430 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1344, %1425 : tensor<1x32x4096x64xf32>, tensor<1x32x4096x1xf32>) outs(%1429 : tensor<1x32x4096x64xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb151(%1431: f32, %1432: f32, %1433: f32):
      %1434 = arith.mulf %1431, %1432 : f32
      linalg.yield %1434 : f32
    } -> tensor<1x32x4096x64xf32>
    %1435 = tensor.empty() : tensor<1x32x4096x64xf32>
    %1436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1430, %44 : tensor<1x32x4096x64xf32>, tensor<64xf32>) outs(%1435 : tensor<1x32x4096x64xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.k_norm"} {
    ^bb152(%1437: f32, %1438: f32, %1439: f32):
      %1440 = arith.mulf %1437, %1438 : f32
      linalg.yield %1440 : f32
    } -> tensor<1x32x4096x64xf32>
    %1441 = tensor.empty() : tensor<1x32x64x4096xf32>
    %1442 = linalg.transpose ins(%1436:tensor<1x32x4096x64xf32>) outs(%1441:tensor<1x32x64x4096xf32>) permutation = [0, 1, 3, 2]
    %1443 = tensor.empty() : tensor<1x32x67x64xf32>
    %1444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1389 : tensor<1x32x67x64xf32>) outs(%1443 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb153(%1445: f32, %1446: f32):
      linalg.yield %1445 : f32
    } -> tensor<1x32x67x64xf32>
    %1447 = tensor.collapse_shape %1444 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67x64xf32> into tensor<137216xf32>
    %1448 = tensor.expand_shape %1447 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 64] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<137216xf32> into tensor<32x67x64xf32>
    %1449 = tensor.empty() : tensor<1x32x64x4096xf32>
    %1450 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1442 : tensor<1x32x64x4096xf32>) outs(%1449 : tensor<1x32x64x4096xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb154(%1451: f32, %1452: f32):
      linalg.yield %1451 : f32
    } -> tensor<1x32x64x4096xf32>
    %1453 = tensor.collapse_shape %1450 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x64x4096xf32> into tensor<8388608xf32>
    %1454 = tensor.expand_shape %1453 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 4096] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8388608xf32> into tensor<32x64x4096xf32>
    %1455 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1456 = tensor.splat %1455 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32x67x4096xf32>
    %1457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1448, %1454 : tensor<32x67x64xf32>, tensor<32x64x4096xf32>) outs(%1456 : tensor<32x67x4096xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb155(%1458: f32, %1459: f32, %1460: f32):
      %1461 = arith.mulf %1458, %1459 : f32
      %1462 = arith.addf %1460, %1461 : f32
      linalg.yield %1462 : f32
    } -> tensor<32x67x4096xf32>
    %1463 = tensor.collapse_shape %1457 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32x67x4096xf32> into tensor<8781824xf32>
    %1464 = tensor.expand_shape %1463 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 4096] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8781824xf32> into tensor<1x32x67x4096xf32>
    %1465 = arith.constant {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 1.250000e-01 : f32
    %1466 = tensor.splat %1465 {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67x4096xf32>
    %1467 = tensor.empty() : tensor<1x32x67x4096xf32>
    %1468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1464, %1466 : tensor<1x32x67x4096xf32>, tensor<1x32x67x4096xf32>) outs(%1467 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb156(%1469: f32, %1470: f32, %1471: f32):
      %1472 = arith.mulf %1469, %1470 : f32
      linalg.yield %1472 : f32
    } -> tensor<1x32x67x4096xf32>
    %1473 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0xff800000 : f32
    %1474 = tensor.splat %1473 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67xf32>
    %1475 = linalg.reduce ins(%1468:tensor<1x32x67x4096xf32>) outs(%1474:tensor<1x32x67xf32>) dimensions = [3]
    (%1476: f32, %1477: f32) {
      %1478 = arith.maximumf %1476, %1477 : f32
      linalg.yield %1478 : f32
    }
    %1479 = tensor.collapse_shape %1475 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1480 = tensor.expand_shape %1479 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1481 = tensor.empty() : tensor<1x32x67x4096xf32>
    %1482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1468, %1480 : tensor<1x32x67x4096xf32>, tensor<1x32x67x1xf32>) outs(%1481 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb157(%1483: f32, %1484: f32, %1485: f32):
      %1486 = arith.subf %1483, %1484 : f32
      linalg.yield %1486 : f32
    } -> tensor<1x32x67x4096xf32>
    %1487 = tensor.empty() : tensor<1x32x67x4096xf32>
    %1488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1482 : tensor<1x32x67x4096xf32>) outs(%1487 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb158(%1489: f32, %1490: f32):
      %1491 = math.exp %1489 : f32
      linalg.yield %1491 : f32
    } -> tensor<1x32x67x4096xf32>
    %1492 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1493 = tensor.splat %1492 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67xf32>
    %1494 = linalg.reduce ins(%1488:tensor<1x32x67x4096xf32>) outs(%1493:tensor<1x32x67xf32>) dimensions = [3]
    (%1495: f32, %1496: f32) {
      %1497 = arith.addf %1495, %1496 : f32
      linalg.yield %1497 : f32
    }
    %1498 = tensor.collapse_shape %1494 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67xf32> into tensor<2144xf32>
    %1499 = tensor.expand_shape %1498 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
    %1500 = tensor.empty() : tensor<1x32x67x4096xf32>
    %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1488, %1499 : tensor<1x32x67x4096xf32>, tensor<1x32x67x1xf32>) outs(%1500 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb159(%1502: f32, %1503: f32, %1504: f32):
      %1505 = arith.divf %1502, %1503 : f32
      linalg.yield %1505 : f32
    } -> tensor<1x32x67x4096xf32>
    %1506 = tensor.empty() : tensor<1x32x67x4096xf32>
    %1507 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1501 : tensor<1x32x67x4096xf32>) outs(%1506 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb160(%1508: f32, %1509: f32):
      linalg.yield %1508 : f32
    } -> tensor<1x32x67x4096xf32>
    %1510 = tensor.collapse_shape %1507 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x67x4096xf32> into tensor<8781824xf32>
    %1511 = tensor.expand_shape %1510 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 67, 4096] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8781824xf32> into tensor<32x67x4096xf32>
    %1512 = tensor.empty() : tensor<1x32x4096x64xf32>
    %1513 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1346 : tensor<1x32x4096x64xf32>) outs(%1512 : tensor<1x32x4096x64xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb161(%1514: f32, %1515: f32):
      linalg.yield %1514 : f32
    } -> tensor<1x32x4096x64xf32>
    %1516 = tensor.collapse_shape %1513 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x32x4096x64xf32> into tensor<8388608xf32>
    %1517 = tensor.expand_shape %1516 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4096, 64] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8388608xf32> into tensor<32x4096x64xf32>
    %1518 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1519 = tensor.splat %1518 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32x67x64xf32>
    %1520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1511, %1517 : tensor<32x67x4096xf32>, tensor<32x4096x64xf32>) outs(%1519 : tensor<32x67x64xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb162(%1521: f32, %1522: f32, %1523: f32):
      %1524 = arith.mulf %1521, %1522 : f32
      %1525 = arith.addf %1523, %1524 : f32
      linalg.yield %1525 : f32
    } -> tensor<32x67x64xf32>
    %1526 = tensor.collapse_shape %1520 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32x67x64xf32> into tensor<137216xf32>
    %1527 = tensor.expand_shape %1526 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
    %1528 = tensor.empty() : tensor<1x67x32x64xf32>
    %1529 = linalg.transpose ins(%1527:tensor<1x32x67x64xf32>) outs(%1528:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
    %1530 = tensor.collapse_shape %1529 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
    %1531 = tensor.expand_shape %1530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1532 = tensor.collapse_shape %1531 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1533 = tensor.expand_shape %1532 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1534 = tensor.empty() : tensor<2048x2048xf32>
    %1535 = linalg.transpose ins(%45:tensor<2048x2048xf32>) outs(%1534:tensor<2048x2048xf32>) permutation = [1, 0]
    %1536 = tensor.empty() : tensor<67x2048xf32>
    %1537 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1538 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1537 : f32) outs(%1536 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1539 = linalg.matmul {prov.region_id = "matmul_23", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj", prov.transposed_b = "true"} ins(%1533, %1535 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1538 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1540 = tensor.empty() : tensor<67x2048xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1539, %46 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1540 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj"} {
    ^bb163(%1542: f32, %1543: f32, %1544: f32):
      %1545 = arith.addf %1542, %1543 : f32
      linalg.yield %1545 : f32
    } -> tensor<67x2048xf32>
    %1546 = tensor.collapse_shape %1541 [[0 : i64, 1 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1547 = tensor.expand_shape %1546 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.proj"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1548 = tensor.empty() : tensor<1x67x2048xf32>
    %1549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1547, %1249 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%1548 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb164(%1550: f32, %1551: f32, %1552: f32):
      %1553 = arith.addf %1550, %1551 : f32
      linalg.yield %1553 : f32
    } -> tensor<1x67x2048xf32>
    %1554 = tensor.empty() : tensor<1x67x2048xf32>
    %1555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1549 : tensor<1x67x2048xf32>) outs(%1554 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_13", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb165(%1556: f32, %1557: f32):
      %1558 = arith.constant 2.000000e+00 : f32
      %1559 = math.powf %1556, %1558 : f32
      linalg.yield %1559 : f32
    } -> tensor<1x67x2048xf32>
    %1560 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} 0.000000e+00 : f32
    %1561 = tensor.splat %1560 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} : tensor<1x67xf32>
    %1562 = linalg.reduce ins(%1555:tensor<1x67x2048xf32>) outs(%1561:tensor<1x67xf32>) dimensions = [2]
    (%1563: f32, %1564: f32) {
      %1565 = arith.addf %1563, %1564 : f32
      linalg.yield %1565 : f32
    }
    %1566 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} 2.048000e+03 : f32
    %1567 = tensor.splat %1566 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} : tensor<1x67xf32>
    %1568 = tensor.empty() : tensor<1x67xf32>
    %1569 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1562, %1567 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%1568 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb166(%1570: f32, %1571: f32, %1572: f32):
      %1573 = arith.divf %1570, %1571 : f32
      linalg.yield %1573 : f32
    } -> tensor<1x67xf32>
    %1574 = tensor.collapse_shape %1569 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} : tensor<1x67xf32> into tensor<67xf32>
    %1575 = tensor.expand_shape %1574 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} : tensor<67xf32> into tensor<1x67x1xf32>
    %1576 = arith.constant {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} 1.000000e-06 : f32
    %1577 = tensor.splat %1576 {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} : tensor<1x67x1xf32>
    %1578 = tensor.empty() : tensor<1x67x1xf32>
    %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1575, %1577 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%1578 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb167(%1580: f32, %1581: f32, %1582: f32):
      %1583 = arith.addf %1580, %1581 : f32
      linalg.yield %1583 : f32
    } -> tensor<1x67x1xf32>
    %1584 = tensor.empty() : tensor<1x67x1xf32>
    %1585 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1579 : tensor<1x67x1xf32>) outs(%1584 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_13", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb168(%1586: f32, %1587: f32):
      %1588 = math.rsqrt %1586 : f32
      linalg.yield %1588 : f32
    } -> tensor<1x67x1xf32>
    %1589 = tensor.empty() : tensor<1x67x2048xf32>
    %1590 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1549, %1585 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%1589 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb169(%1591: f32, %1592: f32, %1593: f32):
      %1594 = arith.mulf %1591, %1592 : f32
      linalg.yield %1594 : f32
    } -> tensor<1x67x2048xf32>
    %1595 = tensor.empty() : tensor<1x67x2048xf32>
    %1596 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1590, %52 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%1595 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.norm3"} {
    ^bb170(%1597: f32, %1598: f32, %1599: f32):
      %1600 = arith.mulf %1597, %1598 : f32
      linalg.yield %1600 : f32
    } -> tensor<1x67x2048xf32>
    %1601 = tensor.collapse_shape %1596 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1602 = tensor.expand_shape %1601 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1603 = tensor.empty() : tensor<2048x2048xf32>
    %1604 = linalg.transpose ins(%48:tensor<2048x2048xf32>) outs(%1603:tensor<2048x2048xf32>) permutation = [1, 0]
    %1605 = tensor.empty() : tensor<67x2048xf32>
    %1606 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1607 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1606 : f32) outs(%1605 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1608 = linalg.matmul {prov.region_id = "matmul_24", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1", prov.transposed_b = "true"} ins(%1602, %1604 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1607 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1609 = tensor.empty() : tensor<67x2048xf32>
    %1610 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1608, %49 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1609 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1"} {
    ^bb171(%1611: f32, %1612: f32, %1613: f32):
      %1614 = arith.addf %1611, %1612 : f32
      linalg.yield %1614 : f32
    } -> tensor<67x2048xf32>
    %1615 = tensor.collapse_shape %1610 [[0 : i64, 1 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1616 = tensor.expand_shape %1615 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc1"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1617 = tensor.empty() : tensor<1x67x2048xf32>
    %1618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1616 : tensor<1x67x2048xf32>) outs(%1617 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.act"} {
    ^bb172(%1619: f32, %1620: f32):
      %1621 = arith.constant 5.000000e-01 : f32
      %1622 = arith.constant 1.000000e+00 : f32
      %1623 = arith.constant 0.707106769 : f32
      %1624 = arith.mulf %1619, %1623 : f32
      %1625 = math.erf %1624 : f32
      %1626 = arith.addf %1622, %1625 : f32
      %1627 = arith.mulf %1621, %1619 : f32
      %1628 = arith.mulf %1627, %1626 : f32
      linalg.yield %1628 : f32
    } -> tensor<1x67x2048xf32>
    %1629 = tensor.collapse_shape %1618 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1630 = tensor.expand_shape %1629 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1631 = tensor.empty() : tensor<2048x2048xf32>
    %1632 = linalg.transpose ins(%50:tensor<2048x2048xf32>) outs(%1631:tensor<2048x2048xf32>) permutation = [1, 0]
    %1633 = tensor.empty() : tensor<67x2048xf32>
    %1634 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1635 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1634 : f32) outs(%1633 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1636 = linalg.matmul {prov.region_id = "matmul_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2", prov.transposed_b = "true"} ins(%1630, %1632 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1635 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1637 = tensor.empty() : tensor<67x2048xf32>
    %1638 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1636, %51 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1637 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2"} {
    ^bb173(%1639: f32, %1640: f32, %1641: f32):
      %1642 = arith.addf %1639, %1640 : f32
      linalg.yield %1642 : f32
    } -> tensor<67x2048xf32>
    %1643 = tensor.collapse_shape %1638 [[0 : i64, 1 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1644 = tensor.expand_shape %1643 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.fc2"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1645 = tensor.empty() : tensor<1x67x2048xf32>
    %1646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1644, %1549 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%1645 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb174(%1647: f32, %1648: f32, %1649: f32):
      %1650 = arith.addf %1647, %1648 : f32
      linalg.yield %1650 : f32
    } -> tensor<1x67x2048xf32>
    %1651 = tensor.empty() : tensor<1x67x2048xf32>
    %1652 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1646 : tensor<1x67x2048xf32>) outs(%1651 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "pow_14", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb175(%1653: f32, %1654: f32):
      %1655 = arith.constant 2.000000e+00 : f32
      %1656 = math.powf %1653, %1655 : f32
      linalg.yield %1656 : f32
    } -> tensor<1x67x2048xf32>
    %1657 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} 0.000000e+00 : f32
    %1658 = tensor.splat %1657 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} : tensor<1x67xf32>
    %1659 = linalg.reduce ins(%1652:tensor<1x67x2048xf32>) outs(%1658:tensor<1x67xf32>) dimensions = [2]
    (%1660: f32, %1661: f32) {
      %1662 = arith.addf %1660, %1661 : f32
      linalg.yield %1662 : f32
    }
    %1663 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} 2.048000e+03 : f32
    %1664 = tensor.splat %1663 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} : tensor<1x67xf32>
    %1665 = tensor.empty() : tensor<1x67xf32>
    %1666 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1659, %1664 : tensor<1x67xf32>, tensor<1x67xf32>) outs(%1665 : tensor<1x67xf32>) attrs =  {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb176(%1667: f32, %1668: f32, %1669: f32):
      %1670 = arith.divf %1667, %1668 : f32
      linalg.yield %1670 : f32
    } -> tensor<1x67xf32>
    %1671 = tensor.collapse_shape %1666 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} : tensor<1x67xf32> into tensor<67xf32>
    %1672 = tensor.expand_shape %1671 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 1] {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} : tensor<67xf32> into tensor<1x67x1xf32>
    %1673 = arith.constant {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} 1.000000e-06 : f32
    %1674 = tensor.splat %1673 {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} : tensor<1x67x1xf32>
    %1675 = tensor.empty() : tensor<1x67x1xf32>
    %1676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1672, %1674 : tensor<1x67x1xf32>, tensor<1x67x1xf32>) outs(%1675 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb177(%1677: f32, %1678: f32, %1679: f32):
      %1680 = arith.addf %1677, %1678 : f32
      linalg.yield %1680 : f32
    } -> tensor<1x67x1xf32>
    %1681 = tensor.empty() : tensor<1x67x1xf32>
    %1682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1676 : tensor<1x67x1xf32>) outs(%1681 : tensor<1x67x1xf32>) attrs =  {prov.region_id = "rsqrt_14", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb178(%1683: f32, %1684: f32):
      %1685 = math.rsqrt %1683 : f32
      linalg.yield %1685 : f32
    } -> tensor<1x67x1xf32>
    %1686 = tensor.empty() : tensor<1x67x2048xf32>
    %1687 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1646, %1682 : tensor<1x67x2048xf32>, tensor<1x67x1xf32>) outs(%1686 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb179(%1688: f32, %1689: f32, %1690: f32):
      %1691 = arith.mulf %1688, %1689 : f32
      linalg.yield %1691 : f32
    } -> tensor<1x67x2048xf32>
    %1692 = tensor.empty() : tensor<1x67x2048xf32>
    %1693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1687, %53 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%1692 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.norm_final"} {
    ^bb180(%1694: f32, %1695: f32, %1696: f32):
      %1697 = arith.mulf %1694, %1695 : f32
      linalg.yield %1697 : f32
    } -> tensor<1x67x2048xf32>
    %1698 = tensor.collapse_shape %1693 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1699 = tensor.expand_shape %1698 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1700 = tensor.empty() : tensor<2048x2048xf32>
    %1701 = linalg.transpose ins(%54:tensor<2048x2048xf32>) outs(%1700:tensor<2048x2048xf32>) permutation = [1, 0]
    %1702 = tensor.empty() : tensor<67x2048xf32>
    %1703 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1704 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1703 : f32) outs(%1702 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1705 = linalg.matmul {prov.region_id = "matmul_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1", prov.transposed_b = "true"} ins(%1699, %1701 : tensor<67x2048xf32>, tensor<2048x2048xf32>) outs(%1704 : tensor<67x2048xf32>) -> tensor<67x2048xf32>
    %1706 = tensor.empty() : tensor<67x2048xf32>
    %1707 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1705, %55 : tensor<67x2048xf32>, tensor<2048xf32>) outs(%1706 : tensor<67x2048xf32>) attrs =  {prov.region_id = "add_42", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1"} {
    ^bb181(%1708: f32, %1709: f32, %1710: f32):
      %1711 = arith.addf %1708, %1709 : f32
      linalg.yield %1711 : f32
    } -> tensor<67x2048xf32>
    %1712 = tensor.collapse_shape %1707 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1"} : tensor<67x2048xf32> into tensor<137216xf32>
    %1713 = tensor.expand_shape %1712 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc1"} : tensor<137216xf32> into tensor<1x67x2048xf32>
    %1714 = tensor.empty() : tensor<1x67x2048xf32>
    %1715 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1713 : tensor<1x67x2048xf32>) outs(%1714 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.act"} {
    ^bb182(%1716: f32, %1717: f32):
      %1718 = arith.constant 5.000000e-01 : f32
      %1719 = arith.constant 1.000000e+00 : f32
      %1720 = arith.constant 0.707106769 : f32
      %1721 = arith.mulf %1716, %1720 : f32
      %1722 = math.erf %1721 : f32
      %1723 = arith.addf %1719, %1722 : f32
      %1724 = arith.mulf %1718, %1716 : f32
      %1725 = arith.mulf %1724, %1723 : f32
      linalg.yield %1725 : f32
    } -> tensor<1x67x2048xf32>
    %1726 = tensor.collapse_shape %1715 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2"} : tensor<1x67x2048xf32> into tensor<137216xf32>
    %1727 = tensor.expand_shape %1726 [[0 : i64, 1 : i64]] output_shape [67, 2048] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2"} : tensor<137216xf32> into tensor<67x2048xf32>
    %1728 = tensor.empty() : tensor<2048x128xf32>
    %1729 = linalg.transpose ins(%56:tensor<128x2048xf32>) outs(%1728:tensor<2048x128xf32>) permutation = [1, 0]
    %1730 = tensor.empty() : tensor<67x128xf32>
    %1731 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1732 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1731 : f32) outs(%1730 : tensor<67x128xf32>) -> tensor<67x128xf32>
    %1733 = linalg.matmul {prov.region_id = "matmul_27", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2", prov.transposed_b = "true"} ins(%1727, %1729 : tensor<67x2048xf32>, tensor<2048x128xf32>) outs(%1732 : tensor<67x128xf32>) -> tensor<67x128xf32>
    %1734 = tensor.empty() : tensor<67x128xf32>
    %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1733, %57 : tensor<67x128xf32>, tensor<128xf32>) outs(%1734 : tensor<67x128xf32>) attrs =  {prov.region_id = "add_43", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2"} {
    ^bb183(%1736: f32, %1737: f32, %1738: f32):
      %1739 = arith.addf %1736, %1737 : f32
      linalg.yield %1739 : f32
    } -> tensor<67x128xf32>
    %1740 = tensor.collapse_shape %1735 [[0 : i64, 1 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2"} : tensor<67x128xf32> into tensor<8576xf32>
    %1741 = tensor.expand_shape %1740 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 128] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_final.fc2"} : tensor<8576xf32> into tensor<1x67x128xf32>
    %1742 = "tensor.extract_slice"(%1741) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 64, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x67x128xf32>) -> tensor<1x64x128xf32>
    func.return %1742 : tensor<1x64x128xf32>
  }
}
