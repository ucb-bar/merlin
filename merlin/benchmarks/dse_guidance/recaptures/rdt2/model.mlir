builtin.module attributes {prov.weights_file = "/path/to/model2MLIR/workloads/rdt2/rdt2.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x4x1024xf32>, %1: tensor<1x28x1024xf32>, %2: tensor<1x1x1024xf32>, %3: tensor<1024x256xf32>, %4: tensor<1024xf32>, %5: tensor<1024x1024xf32>, %6: tensor<1024xf32>, %7: tensor<1024xf32>, %8: tensor<1024x1024xf32>, %9: tensor<1024x1024xf32>, %10: tensor<1024x1024xf32>, %11: tensor<128xf32>, %12: tensor<128xf32>, %13: tensor<1024xf32>, %14: tensor<1024xf32>, %15: tensor<1024x1024xf32>, %16: tensor<1024x1024xf32>, %17: tensor<1024x1024xf32>, %18: tensor<128xf32>, %19: tensor<128xf32>, %20: tensor<1024xf32>, %21: tensor<2816x1024xf32>, %22: tensor<1024x2816xf32>, %23: tensor<2816x1024xf32>, %24: tensor<9216x2048xf32>, %25: tensor<9216xf32>, %26: tensor<1024xf32>, %27: tensor<1024x1024xf32>, %28: tensor<1024x1024xf32>, %29: tensor<1024x1024xf32>, %30: tensor<128xf32>, %31: tensor<128xf32>, %32: tensor<1024xf32>, %33: tensor<1024xf32>, %34: tensor<1024x1024xf32>, %35: tensor<1024x1024xf32>, %36: tensor<1024x1024xf32>, %37: tensor<128xf32>, %38: tensor<128xf32>, %39: tensor<1024xf32>, %40: tensor<2816x1024xf32>, %41: tensor<1024x2816xf32>, %42: tensor<2816x1024xf32>, %43: tensor<9216x2048xf32>, %44: tensor<9216xf32>, %45: tensor<1024xf32>, %46: tensor<4096x1024xf32>, %47: tensor<4096xf32>, %48: tensor<20x4096xf32>, %49: tensor<20xf32>, %50: tensor<2048x2048xf32>, %51: tensor<2048xf32>, %52: tensor<1x24x1024xf32>, %53: tensor<1xf32>, %54: tensor<1x1x1024xf32>, %55: tensor<1x4x64x128xf32>, %56: tensor<1x4x64x128xf32>, %57: tensor<1x4x64x128xf32>, %58: tensor<1x4x64x128xf32>) -> tensor<1x24x20xf32> {
    %59 = tensor.empty() : tensor<128xf32>
    %60 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%59 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb0(%61: f32):
      %62 = linalg.index 0 : index
      %63 = arith.index_cast %62 : index to i64
      %64 = arith.sitofp %63 : i64 to f32
      %65 = arith.constant 1.000000e+00 : f32
      %66 = arith.mulf %64, %65 : f32
      %67 = arith.constant 0.000000e+00 : f32
      %68 = arith.addf %67, %66 : f32
      linalg.yield %68 : f32
    } -> tensor<128xf32>
    %69 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} -9.2103405 : f32
    %70 = tensor.splat %69 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %71 = tensor.empty() : tensor<128xf32>
    %72 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%60, %70 : tensor<128xf32>, tensor<128xf32>) outs(%71 : tensor<128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb1(%73: f32, %74: f32, %75: f32):
      %76 = arith.mulf %73, %74 : f32
      linalg.yield %76 : f32
    } -> tensor<128xf32>
    %77 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} 1.280000e+02 : f32
    %78 = tensor.splat %77 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32>
    %79 = tensor.empty() : tensor<128xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%72, %78 : tensor<128xf32>, tensor<128xf32>) outs(%79 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb2(%81: f32, %82: f32, %83: f32):
      %84 = arith.divf %81, %82 : f32
      linalg.yield %84 : f32
    } -> tensor<128xf32>
    %85 = tensor.empty() : tensor<128xf32>
    %86 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%80 : tensor<128xf32>) outs(%85 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb3(%87: f32, %88: f32):
      %89 = math.exp %87 : f32
      linalg.yield %89 : f32
    } -> tensor<128xf32>
    %90 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<1xf32> into tensor<1x1xf32>
    %91 = tensor.expand_shape %86 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : tensor<128xf32> into tensor<1x128xf32>
    %92 = tensor.empty() : tensor<1x128xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%90, %91 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%92 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb4(%94: f32, %95: f32, %96: f32):
      %97 = arith.mulf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<1x128xf32>
    %98 = tensor.empty() : tensor<1x128xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93 : tensor<1x128xf32>) outs(%98 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb5(%100: f32, %101: f32):
      %102 = math.cos %100 : f32
      linalg.yield %102 : f32
    } -> tensor<1x128xf32>
    %103 = tensor.empty() : tensor<1x128xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93 : tensor<1x128xf32>) outs(%103 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} {
    ^bb6(%105: f32, %106: f32):
      %107 = math.sin %105 : f32
      linalg.yield %107 : f32
    } -> tensor<1x128xf32>
    %108 = tensor.concat dim(1) %99, %104 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
    %109 = tensor.empty() : tensor<256x1024xf32>
    %110 = linalg.transpose ins(%3:tensor<1024x256xf32>) outs(%109:tensor<256x1024xf32>) permutation = [1, 0]
    %111 = tensor.empty() : tensor<1x1024xf32>
    %112 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %113 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%112 : f32) outs(%111 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %114 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.0", prov.transposed_b = "true"} ins(%108, %110 : tensor<1x256xf32>, tensor<256x1024xf32>) outs(%113 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %115 = tensor.empty() : tensor<1x1024xf32>
    %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%114, %4 : tensor<1x1024xf32>, tensor<1024xf32>) outs(%115 : tensor<1x1024xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.0"} {
    ^bb7(%117: f32, %118: f32, %119: f32):
      %120 = arith.addf %117, %118 : f32
      linalg.yield %120 : f32
    } -> tensor<1x1024xf32>
    %121 = tensor.empty() : tensor<1x1024xf32>
    %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116 : tensor<1x1024xf32>) outs(%121 : tensor<1x1024xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.1"} {
    ^bb8(%123: f32, %124: f32):
      %125 = arith.constant 1.000000e+00 : f32
      %126 = arith.negf %123 : f32
      %127 = math.exp %126 : f32
      %128 = arith.addf %125, %127 : f32
      %129 = arith.divf %125, %128 : f32
      linalg.yield %129 : f32
    } -> tensor<1x1024xf32>
    %130 = tensor.empty() : tensor<1x1024xf32>
    %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %122 : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%130 : tensor<1x1024xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.1"} {
    ^bb9(%132: f32, %133: f32, %134: f32):
      %135 = arith.mulf %132, %133 : f32
      linalg.yield %135 : f32
    } -> tensor<1x1024xf32>
    %136 = tensor.empty() : tensor<1024x1024xf32>
    %137 = linalg.transpose ins(%5:tensor<1024x1024xf32>) outs(%136:tensor<1024x1024xf32>) permutation = [1, 0]
    %138 = tensor.empty() : tensor<1x1024xf32>
    %139 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %140 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%139 : f32) outs(%138 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %141 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.2", prov.transposed_b = "true"} ins(%131, %137 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%140 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %142 = tensor.empty() : tensor<1x1024xf32>
    %143 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%141, %6 : tensor<1x1024xf32>, tensor<1024xf32>) outs(%142 : tensor<1x1024xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.t_embedder.mlp.2"} {
    ^bb10(%144: f32, %145: f32, %146: f32):
      %147 = arith.addf %144, %145 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1024xf32>
    %148 = tensor.empty() : tensor<1x1024xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%143 : tensor<1x1024xf32>) outs(%148 : tensor<1x1024xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb11(%150: f32, %151: f32):
      linalg.yield %150 : f32
    } -> tensor<1x1024xf32>
    %152 = tensor.empty() : tensor<1x1x1024xf32>
    %153 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54, %2 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%152 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb12(%154: f32, %155: f32, %156: f32):
      %157 = arith.addf %154, %155 : f32
      linalg.yield %157 : f32
    } -> tensor<1x1x1024xf32>
    %158 = tensor.collapse_shape %149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1x1024xf32> into tensor<1024xf32>
    %159 = tensor.expand_shape %158 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %160 = tensor.concat dim(1) %159, %153 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x2x1024xf32>
    %161 = tensor.collapse_shape %160 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1x2x1024xf32> into tensor<2048xf32>
    %162 = tensor.expand_shape %161 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<2048xf32> into tensor<1x2048xf32>
    %163 = tensor.empty() : tensor<1x4x1024xf32>
    %164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x4x1024xf32>) outs(%163 : tensor<1x4x1024xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb13(%165: f32, %166: f32):
      linalg.yield %165 : f32
    } -> tensor<1x4x1024xf32>
    %167 = tensor.concat dim(1) %52, %164 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x24x1024xf32>, tensor<1x4x1024xf32>) -> tensor<1x28x1024xf32>
    %168 = tensor.empty() : tensor<1x28x1024xf32>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%167, %1 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%168 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} {
    ^bb14(%170: f32, %171: f32, %172: f32):
      %173 = arith.addf %170, %171 : f32
      linalg.yield %173 : f32
    } -> tensor<1x28x1024xf32>
    %174 = tensor.empty() : tensor<1x64x4x128xf32>
    %175 = linalg.transpose ins(%55:tensor<1x4x64x128xf32>) outs(%174:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
    %176 = tensor.empty() : tensor<1x64x4x128xf32>
    %177 = linalg.transpose ins(%56:tensor<1x4x64x128xf32>) outs(%176:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
    %178 = tensor.empty() : tensor<1x2048xf32>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162 : tensor<1x2048xf32>) outs(%178 : tensor<1x2048xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.adaLN_modulation.0"} {
    ^bb15(%180: f32, %181: f32):
      %182 = arith.constant 1.000000e+00 : f32
      %183 = arith.negf %180 : f32
      %184 = math.exp %183 : f32
      %185 = arith.addf %182, %184 : f32
      %186 = arith.divf %182, %185 : f32
      linalg.yield %186 : f32
    } -> tensor<1x2048xf32>
    %187 = tensor.empty() : tensor<1x2048xf32>
    %188 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162, %179 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%187 : tensor<1x2048xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.adaLN_modulation.0"} {
    ^bb16(%189: f32, %190: f32, %191: f32):
      %192 = arith.mulf %189, %190 : f32
      linalg.yield %192 : f32
    } -> tensor<1x2048xf32>
    %193 = tensor.empty() : tensor<2048x9216xf32>
    %194 = linalg.transpose ins(%24:tensor<9216x2048xf32>) outs(%193:tensor<2048x9216xf32>) permutation = [1, 0]
    %195 = tensor.empty() : tensor<1x9216xf32>
    %196 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %197 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%196 : f32) outs(%195 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
    %198 = linalg.matmul {prov.region_id = "matmul_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.adaLN_modulation.1", prov.transposed_b = "true"} ins(%188, %194 : tensor<1x2048xf32>, tensor<2048x9216xf32>) outs(%197 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
    %199 = tensor.empty() : tensor<1x9216xf32>
    %200 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%198, %25 : tensor<1x9216xf32>, tensor<9216xf32>) outs(%199 : tensor<1x9216xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.adaLN_modulation.1"} {
    ^bb17(%201: f32, %202: f32, %203: f32):
      %204 = arith.addf %201, %202 : f32
      linalg.yield %204 : f32
    } -> tensor<1x9216xf32>
    %205 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %206 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %207 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 2048>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %208 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 3072>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %209 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 4096>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %210 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 5120>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %211 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 6144>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %212 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 7168>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %213 = "tensor.extract_slice"(%200) <{static_offsets = array<i64: 0, 8192>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %214 = tensor.collapse_shape %207 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %215 = tensor.expand_shape %214 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %216 = tensor.empty() : tensor<1x28x1024xf32>
    %217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<1x28x1024xf32>) outs(%216 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb18(%218: f32, %219: f32):
      %220 = arith.constant 2.000000e+00 : f32
      %221 = math.powf %218, %220 : f32
      linalg.yield %221 : f32
    } -> tensor<1x28x1024xf32>
    %222 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} 0.000000e+00 : f32
    %223 = tensor.splat %222 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} : tensor<1x28xf32>
    %224 = linalg.reduce ins(%217:tensor<1x28x1024xf32>) outs(%223:tensor<1x28xf32>) dimensions = [2]
    (%225: f32, %226: f32) {
      %227 = arith.addf %225, %226 : f32
      linalg.yield %227 : f32
    }
    %228 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} 1.024000e+03 : f32
    %229 = tensor.splat %228 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} : tensor<1x28xf32>
    %230 = tensor.empty() : tensor<1x28xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%224, %229 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%230 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb19(%232: f32, %233: f32, %234: f32):
      %235 = arith.divf %232, %233 : f32
      linalg.yield %235 : f32
    } -> tensor<1x28xf32>
    %236 = tensor.collapse_shape %231 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %237 = tensor.expand_shape %236 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %238 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} 1.000000e-05 : f32
    %239 = tensor.splat %238 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} : tensor<1x28x1xf32>
    %240 = tensor.empty() : tensor<1x28x1xf32>
    %241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%237, %239 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%240 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb20(%242: f32, %243: f32, %244: f32):
      %245 = arith.addf %242, %243 : f32
      linalg.yield %245 : f32
    } -> tensor<1x28x1xf32>
    %246 = tensor.empty() : tensor<1x28x1xf32>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%241 : tensor<1x28x1xf32>) outs(%246 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb21(%248: f32, %249: f32):
      %250 = math.rsqrt %248 : f32
      linalg.yield %250 : f32
    } -> tensor<1x28x1xf32>
    %251 = tensor.empty() : tensor<1x28x1024xf32>
    %252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169, %247 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%251 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb22(%253: f32, %254: f32, %255: f32):
      %256 = arith.mulf %253, %254 : f32
      linalg.yield %256 : f32
    } -> tensor<1x28x1024xf32>
    %257 = tensor.empty() : tensor<1x28x1024xf32>
    %258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%252, %7 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%257 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn_norm"} {
    ^bb23(%259: f32, %260: f32, %261: f32):
      %262 = arith.mulf %259, %260 : f32
      linalg.yield %262 : f32
    } -> tensor<1x28x1024xf32>
    %263 = tensor.collapse_shape %206 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %264 = tensor.expand_shape %263 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %265 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} 1.000000e+00 : f32
    %266 = tensor.splat %265 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1x1024xf32>
    %267 = tensor.empty() : tensor<1x1x1024xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%264, %266 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%267 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb24(%269: f32, %270: f32, %271: f32):
      %272 = arith.addf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<1x1x1024xf32>
    %273 = tensor.empty() : tensor<1x28x1024xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%258, %268 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%273 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb25(%275: f32, %276: f32, %277: f32):
      %278 = arith.mulf %275, %276 : f32
      linalg.yield %278 : f32
    } -> tensor<1x28x1024xf32>
    %279 = tensor.collapse_shape %205 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %280 = tensor.expand_shape %279 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %281 = tensor.empty() : tensor<1x28x1024xf32>
    %282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%274, %280 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%281 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb26(%283: f32, %284: f32, %285: f32):
      %286 = arith.addf %283, %284 : f32
      linalg.yield %286 : f32
    } -> tensor<1x28x1024xf32>
    %287 = tensor.empty() : tensor<1024x1024xf32>
    %288 = linalg.transpose ins(%8:tensor<1024x1024xf32>) outs(%287:tensor<1024x1024xf32>) permutation = [1, 0]
    %289 = tensor.collapse_shape %282 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wq"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %290 = tensor.expand_shape %289 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wq"} : tensor<28672xf32> into tensor<28x1024xf32>
    %291 = tensor.empty() : tensor<28x1024xf32>
    %292 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %293 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%292 : f32) outs(%291 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %294 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wq", prov.transposed_b = "true"} ins(%290, %288 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%293 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %295 = tensor.collapse_shape %294 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wq"} : tensor<28x1024xf32> into tensor<28672xf32>
    %296 = tensor.expand_shape %295 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wq"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %297 = tensor.collapse_shape %296 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %298 = tensor.expand_shape %297 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %299 = tensor.empty() : tensor<1024x1024xf32>
    %300 = linalg.transpose ins(%9:tensor<1024x1024xf32>) outs(%299:tensor<1024x1024xf32>) permutation = [1, 0]
    %301 = tensor.collapse_shape %282 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wkv"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %302 = tensor.expand_shape %301 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wkv"} : tensor<28672xf32> into tensor<28x1024xf32>
    %303 = tensor.empty() : tensor<28x1024xf32>
    %304 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %305 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%304 : f32) outs(%303 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %306 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wkv", prov.transposed_b = "true"} ins(%302, %300 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%305 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %307 = tensor.collapse_shape %306 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wkv"} : tensor<28x1024xf32> into tensor<28672xf32>
    %308 = tensor.expand_shape %307 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wkv"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %309 = tensor.collapse_shape %308 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %310 = tensor.expand_shape %309 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 128, 2] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x28x4x128x2xf32>
    %311 = "tensor.extract_slice"(%310) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
    %312 = "tensor.extract_slice"(%310) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
    %313 = tensor.collapse_shape %311 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
    %314 = tensor.expand_shape %313 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "squeeze_0", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
    %315 = tensor.collapse_shape %312 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
    %316 = tensor.expand_shape %315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "squeeze_1", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
    %317 = tensor.empty() : tensor<1x28x8x128xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298 : tensor<1x28x8x128xf32>) outs(%317 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb27(%319: f32, %320: f32):
      %321 = arith.constant 2.000000e+00 : f32
      %322 = math.powf %319, %321 : f32
      linalg.yield %322 : f32
    } -> tensor<1x28x8x128xf32>
    %323 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} 0.000000e+00 : f32
    %324 = tensor.splat %323 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} : tensor<1x28x8xf32>
    %325 = linalg.reduce ins(%318:tensor<1x28x8x128xf32>) outs(%324:tensor<1x28x8xf32>) dimensions = [3]
    (%326: f32, %327: f32) {
      %328 = arith.addf %326, %327 : f32
      linalg.yield %328 : f32
    }
    %329 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} 1.280000e+02 : f32
    %330 = tensor.splat %329 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} : tensor<1x28x8xf32>
    %331 = tensor.empty() : tensor<1x28x8xf32>
    %332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%325, %330 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%331 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb28(%333: f32, %334: f32, %335: f32):
      %336 = arith.divf %333, %334 : f32
      linalg.yield %336 : f32
    } -> tensor<1x28x8xf32>
    %337 = tensor.collapse_shape %332 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} : tensor<1x28x8xf32> into tensor<224xf32>
    %338 = tensor.expand_shape %337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %339 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} 1.000000e-05 : f32
    %340 = tensor.splat %339 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} : tensor<1x28x8x1xf32>
    %341 = tensor.empty() : tensor<1x28x8x1xf32>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%338, %340 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%341 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb29(%343: f32, %344: f32, %345: f32):
      %346 = arith.addf %343, %344 : f32
      linalg.yield %346 : f32
    } -> tensor<1x28x8x1xf32>
    %347 = tensor.empty() : tensor<1x28x8x1xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%342 : tensor<1x28x8x1xf32>) outs(%347 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb30(%349: f32, %350: f32):
      %351 = math.rsqrt %349 : f32
      linalg.yield %351 : f32
    } -> tensor<1x28x8x1xf32>
    %352 = tensor.empty() : tensor<1x28x8x128xf32>
    %353 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298, %348 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%352 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb31(%354: f32, %355: f32, %356: f32):
      %357 = arith.mulf %354, %355 : f32
      linalg.yield %357 : f32
    } -> tensor<1x28x8x128xf32>
    %358 = tensor.empty() : tensor<1x28x8x128xf32>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%353, %11 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%358 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_q"} {
    ^bb32(%360: f32, %361: f32, %362: f32):
      %363 = arith.mulf %360, %361 : f32
      linalg.yield %363 : f32
    } -> tensor<1x28x8x128xf32>
    %364 = tensor.empty() : tensor<1x28x4x128xf32>
    %365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%314 : tensor<1x28x4x128xf32>) outs(%364 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb33(%366: f32, %367: f32):
      %368 = arith.constant 2.000000e+00 : f32
      %369 = math.powf %366, %368 : f32
      linalg.yield %369 : f32
    } -> tensor<1x28x4x128xf32>
    %370 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} 0.000000e+00 : f32
    %371 = tensor.splat %370 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} : tensor<1x28x4xf32>
    %372 = linalg.reduce ins(%365:tensor<1x28x4x128xf32>) outs(%371:tensor<1x28x4xf32>) dimensions = [3]
    (%373: f32, %374: f32) {
      %375 = arith.addf %373, %374 : f32
      linalg.yield %375 : f32
    }
    %376 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} 1.280000e+02 : f32
    %377 = tensor.splat %376 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} : tensor<1x28x4xf32>
    %378 = tensor.empty() : tensor<1x28x4xf32>
    %379 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%372, %377 : tensor<1x28x4xf32>, tensor<1x28x4xf32>) outs(%378 : tensor<1x28x4xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb34(%380: f32, %381: f32, %382: f32):
      %383 = arith.divf %380, %381 : f32
      linalg.yield %383 : f32
    } -> tensor<1x28x4xf32>
    %384 = tensor.collapse_shape %379 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} : tensor<1x28x4xf32> into tensor<112xf32>
    %385 = tensor.expand_shape %384 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} : tensor<112xf32> into tensor<1x28x4x1xf32>
    %386 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} 1.000000e-05 : f32
    %387 = tensor.splat %386 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} : tensor<1x28x4x1xf32>
    %388 = tensor.empty() : tensor<1x28x4x1xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%385, %387 : tensor<1x28x4x1xf32>, tensor<1x28x4x1xf32>) outs(%388 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb35(%390: f32, %391: f32, %392: f32):
      %393 = arith.addf %390, %391 : f32
      linalg.yield %393 : f32
    } -> tensor<1x28x4x1xf32>
    %394 = tensor.empty() : tensor<1x28x4x1xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389 : tensor<1x28x4x1xf32>) outs(%394 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb36(%396: f32, %397: f32):
      %398 = math.rsqrt %396 : f32
      linalg.yield %398 : f32
    } -> tensor<1x28x4x1xf32>
    %399 = tensor.empty() : tensor<1x28x4x128xf32>
    %400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%314, %395 : tensor<1x28x4x128xf32>, tensor<1x28x4x1xf32>) outs(%399 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb37(%401: f32, %402: f32, %403: f32):
      %404 = arith.mulf %401, %402 : f32
      linalg.yield %404 : f32
    } -> tensor<1x28x4x128xf32>
    %405 = tensor.empty() : tensor<1x28x4x128xf32>
    %406 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%400, %12 : tensor<1x28x4x128xf32>, tensor<128xf32>) outs(%405 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.norm_k"} {
    ^bb38(%407: f32, %408: f32, %409: f32):
      %410 = arith.mulf %407, %408 : f32
      linalg.yield %410 : f32
    } -> tensor<1x28x4x128xf32>
    %411 = tensor.collapse_shape %406 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
    %412 = tensor.expand_shape %411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
    %413 = tensor.empty() : tensor<1x28x4x2x128xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%412 : tensor<1x28x4x1x128xf32>) outs(%413 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb39(%415: f32, %416: f32):
      linalg.yield %415 : f32
    } -> tensor<1x28x4x2x128xf32>
    %417 = tensor.collapse_shape %414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
    %418 = tensor.expand_shape %417 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %419 = tensor.collapse_shape %316 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
    %420 = tensor.expand_shape %419 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
    %421 = tensor.empty() : tensor<1x28x4x2x128xf32>
    %422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%420 : tensor<1x28x4x1x128xf32>) outs(%421 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb40(%423: f32, %424: f32):
      linalg.yield %423 : f32
    } -> tensor<1x28x4x2x128xf32>
    %425 = tensor.collapse_shape %422 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
    %426 = tensor.expand_shape %425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %427 = tensor.empty() : tensor<1x8x28x128xf32>
    %428 = linalg.transpose ins(%359:tensor<1x28x8x128xf32>) outs(%427:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %429 = tensor.empty() : tensor<1x8x28x128xf32>
    %430 = linalg.transpose ins(%418:tensor<1x28x8x128xf32>) outs(%429:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %431 = tensor.empty() : tensor<1x8x28x128xf32>
    %432 = linalg.transpose ins(%426:tensor<1x28x8x128xf32>) outs(%431:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %433 = tensor.empty() : tensor<1x8x128x28xf32>
    %434 = linalg.transpose ins(%430:tensor<1x8x28x128xf32>) outs(%433:tensor<1x8x128x28xf32>) permutation = [0, 1, 3, 2]
    %435 = tensor.empty() : tensor<1x8x28x128xf32>
    %436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%428 : tensor<1x8x28x128xf32>) outs(%435 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb41(%437: f32, %438: f32):
      linalg.yield %437 : f32
    } -> tensor<1x8x28x128xf32>
    %439 = tensor.collapse_shape %436 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %440 = tensor.expand_shape %439 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %441 = tensor.empty() : tensor<1x8x128x28xf32>
    %442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%434 : tensor<1x8x128x28xf32>) outs(%441 : tensor<1x8x128x28xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb42(%443: f32, %444: f32):
      linalg.yield %443 : f32
    } -> tensor<1x8x128x28xf32>
    %445 = tensor.collapse_shape %442 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x128x28xf32> into tensor<28672xf32>
    %446 = tensor.expand_shape %445 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 128, 28] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<8x128x28xf32>
    %447 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %448 = tensor.splat %447 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<8x28x28xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%440, %446 : tensor<8x28x128xf32>, tensor<8x128x28xf32>) outs(%448 : tensor<8x28x28xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb43(%450: f32, %451: f32, %452: f32):
      %453 = arith.mulf %450, %451 : f32
      %454 = arith.addf %452, %453 : f32
      linalg.yield %454 : f32
    } -> tensor<8x28x28xf32>
    %455 = tensor.collapse_shape %449 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<8x28x28xf32> into tensor<6272xf32>
    %456 = tensor.expand_shape %455 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 28] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<6272xf32> into tensor<1x8x28x28xf32>
    %457 = arith.constant {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.0883883461 : f32
    %458 = tensor.splat %457 {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28x28xf32>
    %459 = tensor.empty() : tensor<1x8x28x28xf32>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%456, %458 : tensor<1x8x28x28xf32>, tensor<1x8x28x28xf32>) outs(%459 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb44(%461: f32, %462: f32, %463: f32):
      %464 = arith.mulf %461, %462 : f32
      linalg.yield %464 : f32
    } -> tensor<1x8x28x28xf32>
    %465 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0xff800000 : f32
    %466 = tensor.splat %465 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28xf32>
    %467 = linalg.reduce ins(%460:tensor<1x8x28x28xf32>) outs(%466:tensor<1x8x28xf32>) dimensions = [3]
    (%468: f32, %469: f32) {
      %470 = arith.maximumf %468, %469 : f32
      linalg.yield %470 : f32
    }
    %471 = tensor.collapse_shape %467 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %472 = tensor.expand_shape %471 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %473 = tensor.empty() : tensor<1x8x28x28xf32>
    %474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%460, %472 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%473 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb45(%475: f32, %476: f32, %477: f32):
      %478 = arith.subf %475, %476 : f32
      linalg.yield %478 : f32
    } -> tensor<1x8x28x28xf32>
    %479 = tensor.empty() : tensor<1x8x28x28xf32>
    %480 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474 : tensor<1x8x28x28xf32>) outs(%479 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb46(%481: f32, %482: f32):
      %483 = math.exp %481 : f32
      linalg.yield %483 : f32
    } -> tensor<1x8x28x28xf32>
    %484 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %485 = tensor.splat %484 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28xf32>
    %486 = linalg.reduce ins(%480:tensor<1x8x28x28xf32>) outs(%485:tensor<1x8x28xf32>) dimensions = [3]
    (%487: f32, %488: f32) {
      %489 = arith.addf %487, %488 : f32
      linalg.yield %489 : f32
    }
    %490 = tensor.collapse_shape %486 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %491 = tensor.expand_shape %490 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %492 = tensor.empty() : tensor<1x8x28x28xf32>
    %493 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %491 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%492 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb47(%494: f32, %495: f32, %496: f32):
      %497 = arith.divf %494, %495 : f32
      linalg.yield %497 : f32
    } -> tensor<1x8x28x28xf32>
    %498 = tensor.empty() : tensor<1x8x28x28xf32>
    %499 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%493 : tensor<1x8x28x28xf32>) outs(%498 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb48(%500: f32, %501: f32):
      linalg.yield %500 : f32
    } -> tensor<1x8x28x28xf32>
    %502 = tensor.collapse_shape %499 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28x28xf32> into tensor<6272xf32>
    %503 = tensor.expand_shape %502 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 28] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<6272xf32> into tensor<8x28x28xf32>
    %504 = tensor.empty() : tensor<1x8x28x128xf32>
    %505 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%432 : tensor<1x8x28x128xf32>) outs(%504 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb49(%506: f32, %507: f32):
      linalg.yield %506 : f32
    } -> tensor<1x8x28x128xf32>
    %508 = tensor.collapse_shape %505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %509 = tensor.expand_shape %508 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %510 = arith.constant {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} 0.000000e+00 : f32
    %511 = tensor.splat %510 {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<8x28x128xf32>
    %512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%503, %509 : tensor<8x28x28xf32>, tensor<8x28x128xf32>) outs(%511 : tensor<8x28x128xf32>) attrs =  {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} {
    ^bb50(%513: f32, %514: f32, %515: f32):
      %516 = arith.mulf %513, %514 : f32
      %517 = arith.addf %515, %516 : f32
      linalg.yield %517 : f32
    } -> tensor<8x28x128xf32>
    %518 = tensor.collapse_shape %512 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<8x28x128xf32> into tensor<28672xf32>
    %519 = tensor.expand_shape %518 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %520 = tensor.empty() : tensor<1x28x8x128xf32>
    %521 = linalg.transpose ins(%519:tensor<1x8x28x128xf32>) outs(%520:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %522 = tensor.collapse_shape %521 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %524 = tensor.empty() : tensor<1024x1024xf32>
    %525 = linalg.transpose ins(%10:tensor<1024x1024xf32>) outs(%524:tensor<1024x1024xf32>) permutation = [1, 0]
    %526 = tensor.collapse_shape %523 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wo"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %527 = tensor.expand_shape %526 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wo"} : tensor<28672xf32> into tensor<28x1024xf32>
    %528 = tensor.empty() : tensor<28x1024xf32>
    %529 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %530 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%529 : f32) outs(%528 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %531 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wo", prov.transposed_b = "true"} ins(%527, %525 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%530 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %532 = tensor.collapse_shape %531 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wo"} : tensor<28x1024xf32> into tensor<28672xf32>
    %533 = tensor.expand_shape %532 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.attn.wo"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %534 = tensor.empty() : tensor<1x28x1024xf32>
    %535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%215, %533 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%534 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb51(%536: f32, %537: f32, %538: f32):
      %539 = arith.mulf %536, %537 : f32
      linalg.yield %539 : f32
    } -> tensor<1x28x1024xf32>
    %540 = tensor.empty() : tensor<1x28x1024xf32>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169, %535 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%540 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb52(%542: f32, %543: f32, %544: f32):
      %545 = arith.addf %542, %543 : f32
      linalg.yield %545 : f32
    } -> tensor<1x28x1024xf32>
    %546 = tensor.collapse_shape %210 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %547 = tensor.expand_shape %546 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %548 = tensor.empty() : tensor<1x28x1024xf32>
    %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541 : tensor<1x28x1024xf32>) outs(%548 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb53(%550: f32, %551: f32):
      %552 = arith.constant 2.000000e+00 : f32
      %553 = math.powf %550, %552 : f32
      linalg.yield %553 : f32
    } -> tensor<1x28x1024xf32>
    %554 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} 0.000000e+00 : f32
    %555 = tensor.splat %554 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} : tensor<1x28xf32>
    %556 = linalg.reduce ins(%549:tensor<1x28x1024xf32>) outs(%555:tensor<1x28xf32>) dimensions = [2]
    (%557: f32, %558: f32) {
      %559 = arith.addf %557, %558 : f32
      linalg.yield %559 : f32
    }
    %560 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} 1.024000e+03 : f32
    %561 = tensor.splat %560 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} : tensor<1x28xf32>
    %562 = tensor.empty() : tensor<1x28xf32>
    %563 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%556, %561 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%562 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb54(%564: f32, %565: f32, %566: f32):
      %567 = arith.divf %564, %565 : f32
      linalg.yield %567 : f32
    } -> tensor<1x28xf32>
    %568 = tensor.collapse_shape %563 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %569 = tensor.expand_shape %568 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %570 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} 1.000000e-05 : f32
    %571 = tensor.splat %570 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} : tensor<1x28x1xf32>
    %572 = tensor.empty() : tensor<1x28x1xf32>
    %573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%569, %571 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%572 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb55(%574: f32, %575: f32, %576: f32):
      %577 = arith.addf %574, %575 : f32
      linalg.yield %577 : f32
    } -> tensor<1x28x1xf32>
    %578 = tensor.empty() : tensor<1x28x1xf32>
    %579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573 : tensor<1x28x1xf32>) outs(%578 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb56(%580: f32, %581: f32):
      %582 = math.rsqrt %580 : f32
      linalg.yield %582 : f32
    } -> tensor<1x28x1xf32>
    %583 = tensor.empty() : tensor<1x28x1024xf32>
    %584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541, %579 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%583 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb57(%585: f32, %586: f32, %587: f32):
      %588 = arith.mulf %585, %586 : f32
      linalg.yield %588 : f32
    } -> tensor<1x28x1024xf32>
    %589 = tensor.empty() : tensor<1x28x1024xf32>
    %590 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%584, %13 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%589 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_norm"} {
    ^bb58(%591: f32, %592: f32, %593: f32):
      %594 = arith.mulf %591, %592 : f32
      linalg.yield %594 : f32
    } -> tensor<1x28x1024xf32>
    %595 = tensor.collapse_shape %209 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %596 = tensor.expand_shape %595 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %597 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} 1.000000e+00 : f32
    %598 = tensor.splat %597 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1x1024xf32>
    %599 = tensor.empty() : tensor<1x1x1024xf32>
    %600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%596, %598 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%599 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb59(%601: f32, %602: f32, %603: f32):
      %604 = arith.addf %601, %602 : f32
      linalg.yield %604 : f32
    } -> tensor<1x1x1024xf32>
    %605 = tensor.empty() : tensor<1x28x1024xf32>
    %606 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%590, %600 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%605 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb60(%607: f32, %608: f32, %609: f32):
      %610 = arith.mulf %607, %608 : f32
      linalg.yield %610 : f32
    } -> tensor<1x28x1024xf32>
    %611 = tensor.collapse_shape %208 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %612 = tensor.expand_shape %611 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %613 = tensor.empty() : tensor<1x28x1024xf32>
    %614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606, %612 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%613 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb61(%615: f32, %616: f32, %617: f32):
      %618 = arith.addf %615, %616 : f32
      linalg.yield %618 : f32
    } -> tensor<1x28x1024xf32>
    %619 = tensor.empty() : tensor<1024x1024xf32>
    %620 = linalg.transpose ins(%15:tensor<1024x1024xf32>) outs(%619:tensor<1024x1024xf32>) permutation = [1, 0]
    %621 = tensor.collapse_shape %614 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wq"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %622 = tensor.expand_shape %621 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wq"} : tensor<28672xf32> into tensor<28x1024xf32>
    %623 = tensor.empty() : tensor<28x1024xf32>
    %624 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %625 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%624 : f32) outs(%623 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %626 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wq", prov.transposed_b = "true"} ins(%622, %620 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%625 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %627 = tensor.collapse_shape %626 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wq"} : tensor<28x1024xf32> into tensor<28672xf32>
    %628 = tensor.expand_shape %627 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wq"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %629 = tensor.collapse_shape %628 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %630 = tensor.expand_shape %629 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %631 = tensor.empty() : tensor<1x28x8x128xf32>
    %632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%630 : tensor<1x28x8x128xf32>) outs(%631 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb62(%633: f32, %634: f32):
      %635 = arith.constant 2.000000e+00 : f32
      %636 = math.powf %633, %635 : f32
      linalg.yield %636 : f32
    } -> tensor<1x28x8x128xf32>
    %637 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} 0.000000e+00 : f32
    %638 = tensor.splat %637 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} : tensor<1x28x8xf32>
    %639 = linalg.reduce ins(%632:tensor<1x28x8x128xf32>) outs(%638:tensor<1x28x8xf32>) dimensions = [3]
    (%640: f32, %641: f32) {
      %642 = arith.addf %640, %641 : f32
      linalg.yield %642 : f32
    }
    %643 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} 1.280000e+02 : f32
    %644 = tensor.splat %643 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} : tensor<1x28x8xf32>
    %645 = tensor.empty() : tensor<1x28x8xf32>
    %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639, %644 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%645 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb63(%647: f32, %648: f32, %649: f32):
      %650 = arith.divf %647, %648 : f32
      linalg.yield %650 : f32
    } -> tensor<1x28x8xf32>
    %651 = tensor.collapse_shape %646 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} : tensor<1x28x8xf32> into tensor<224xf32>
    %652 = tensor.expand_shape %651 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %653 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} 1.000000e-05 : f32
    %654 = tensor.splat %653 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} : tensor<1x28x8x1xf32>
    %655 = tensor.empty() : tensor<1x28x8x1xf32>
    %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%652, %654 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%655 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb64(%657: f32, %658: f32, %659: f32):
      %660 = arith.addf %657, %658 : f32
      linalg.yield %660 : f32
    } -> tensor<1x28x8x1xf32>
    %661 = tensor.empty() : tensor<1x28x8x1xf32>
    %662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656 : tensor<1x28x8x1xf32>) outs(%661 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb65(%663: f32, %664: f32):
      %665 = math.rsqrt %663 : f32
      linalg.yield %665 : f32
    } -> tensor<1x28x8x1xf32>
    %666 = tensor.empty() : tensor<1x28x8x128xf32>
    %667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%630, %662 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%666 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb66(%668: f32, %669: f32, %670: f32):
      %671 = arith.mulf %668, %669 : f32
      linalg.yield %671 : f32
    } -> tensor<1x28x8x128xf32>
    %672 = tensor.empty() : tensor<1x28x8x128xf32>
    %673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%667, %18 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%672 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.norm_q"} {
    ^bb67(%674: f32, %675: f32, %676: f32):
      %677 = arith.mulf %674, %675 : f32
      linalg.yield %677 : f32
    } -> tensor<1x28x8x128xf32>
    %678 = tensor.collapse_shape %175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
    %679 = tensor.expand_shape %678 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
    %680 = tensor.empty() : tensor<1x64x4x2x128xf32>
    %681 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%679 : tensor<1x64x4x1x128xf32>) outs(%680 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb68(%682: f32, %683: f32):
      linalg.yield %682 : f32
    } -> tensor<1x64x4x2x128xf32>
    %684 = tensor.collapse_shape %681 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
    %685 = tensor.expand_shape %684 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
    %686 = tensor.collapse_shape %177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
    %687 = tensor.expand_shape %686 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
    %688 = tensor.empty() : tensor<1x64x4x2x128xf32>
    %689 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%687 : tensor<1x64x4x1x128xf32>) outs(%688 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb69(%690: f32, %691: f32):
      linalg.yield %690 : f32
    } -> tensor<1x64x4x2x128xf32>
    %692 = tensor.collapse_shape %689 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
    %693 = tensor.expand_shape %692 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
    %694 = tensor.empty() : tensor<1x8x28x128xf32>
    %695 = linalg.transpose ins(%673:tensor<1x28x8x128xf32>) outs(%694:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %696 = tensor.empty() : tensor<1x8x64x128xf32>
    %697 = linalg.transpose ins(%685:tensor<1x64x8x128xf32>) outs(%696:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
    %698 = tensor.empty() : tensor<1x8x64x128xf32>
    %699 = linalg.transpose ins(%693:tensor<1x64x8x128xf32>) outs(%698:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
    %700 = tensor.empty() : tensor<1x8x128x64xf32>
    %701 = linalg.transpose ins(%697:tensor<1x8x64x128xf32>) outs(%700:tensor<1x8x128x64xf32>) permutation = [0, 1, 3, 2]
    %702 = tensor.empty() : tensor<1x8x28x128xf32>
    %703 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%695 : tensor<1x8x28x128xf32>) outs(%702 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb70(%704: f32, %705: f32):
      linalg.yield %704 : f32
    } -> tensor<1x8x28x128xf32>
    %706 = tensor.collapse_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %707 = tensor.expand_shape %706 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %708 = tensor.empty() : tensor<1x8x128x64xf32>
    %709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%701 : tensor<1x8x128x64xf32>) outs(%708 : tensor<1x8x128x64xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb71(%710: f32, %711: f32):
      linalg.yield %710 : f32
    } -> tensor<1x8x128x64xf32>
    %712 = tensor.collapse_shape %709 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x128x64xf32> into tensor<65536xf32>
    %713 = tensor.expand_shape %712 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 128, 64] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<8x128x64xf32>
    %714 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %715 = tensor.splat %714 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<8x28x64xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%707, %713 : tensor<8x28x128xf32>, tensor<8x128x64xf32>) outs(%715 : tensor<8x28x64xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb72(%717: f32, %718: f32, %719: f32):
      %720 = arith.mulf %717, %718 : f32
      %721 = arith.addf %719, %720 : f32
      linalg.yield %721 : f32
    } -> tensor<8x28x64xf32>
    %722 = tensor.collapse_shape %716 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<8x28x64xf32> into tensor<14336xf32>
    %723 = tensor.expand_shape %722 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 64] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<14336xf32> into tensor<1x8x28x64xf32>
    %724 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.0883883461 : f32
    %725 = tensor.splat %724 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28x64xf32>
    %726 = tensor.empty() : tensor<1x8x28x64xf32>
    %727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%723, %725 : tensor<1x8x28x64xf32>, tensor<1x8x28x64xf32>) outs(%726 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb73(%728: f32, %729: f32, %730: f32):
      %731 = arith.mulf %728, %729 : f32
      linalg.yield %731 : f32
    } -> tensor<1x8x28x64xf32>
    %732 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0xff800000 : f32
    %733 = tensor.splat %732 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28xf32>
    %734 = linalg.reduce ins(%727:tensor<1x8x28x64xf32>) outs(%733:tensor<1x8x28xf32>) dimensions = [3]
    (%735: f32, %736: f32) {
      %737 = arith.maximumf %735, %736 : f32
      linalg.yield %737 : f32
    }
    %738 = tensor.collapse_shape %734 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %739 = tensor.expand_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %740 = tensor.empty() : tensor<1x8x28x64xf32>
    %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %739 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%740 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb74(%742: f32, %743: f32, %744: f32):
      %745 = arith.subf %742, %743 : f32
      linalg.yield %745 : f32
    } -> tensor<1x8x28x64xf32>
    %746 = tensor.empty() : tensor<1x8x28x64xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741 : tensor<1x8x28x64xf32>) outs(%746 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb75(%748: f32, %749: f32):
      %750 = math.exp %748 : f32
      linalg.yield %750 : f32
    } -> tensor<1x8x28x64xf32>
    %751 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %752 = tensor.splat %751 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28xf32>
    %753 = linalg.reduce ins(%747:tensor<1x8x28x64xf32>) outs(%752:tensor<1x8x28xf32>) dimensions = [3]
    (%754: f32, %755: f32) {
      %756 = arith.addf %754, %755 : f32
      linalg.yield %756 : f32
    }
    %757 = tensor.collapse_shape %753 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %758 = tensor.expand_shape %757 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %759 = tensor.empty() : tensor<1x8x28x64xf32>
    %760 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %758 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%759 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb76(%761: f32, %762: f32, %763: f32):
      %764 = arith.divf %761, %762 : f32
      linalg.yield %764 : f32
    } -> tensor<1x8x28x64xf32>
    %765 = tensor.empty() : tensor<1x8x28x64xf32>
    %766 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%760 : tensor<1x8x28x64xf32>) outs(%765 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb77(%767: f32, %768: f32):
      linalg.yield %767 : f32
    } -> tensor<1x8x28x64xf32>
    %769 = tensor.collapse_shape %766 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x28x64xf32> into tensor<14336xf32>
    %770 = tensor.expand_shape %769 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 64] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<14336xf32> into tensor<8x28x64xf32>
    %771 = tensor.empty() : tensor<1x8x64x128xf32>
    %772 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699 : tensor<1x8x64x128xf32>) outs(%771 : tensor<1x8x64x128xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb78(%773: f32, %774: f32):
      linalg.yield %773 : f32
    } -> tensor<1x8x64x128xf32>
    %775 = tensor.collapse_shape %772 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x8x64x128xf32> into tensor<65536xf32>
    %776 = tensor.expand_shape %775 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 64, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<65536xf32> into tensor<8x64x128xf32>
    %777 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} 0.000000e+00 : f32
    %778 = tensor.splat %777 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<8x28x128xf32>
    %779 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%770, %776 : tensor<8x28x64xf32>, tensor<8x64x128xf32>) outs(%778 : tensor<8x28x128xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} {
    ^bb79(%780: f32, %781: f32, %782: f32):
      %783 = arith.mulf %780, %781 : f32
      %784 = arith.addf %782, %783 : f32
      linalg.yield %784 : f32
    } -> tensor<8x28x128xf32>
    %785 = tensor.collapse_shape %779 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<8x28x128xf32> into tensor<28672xf32>
    %786 = tensor.expand_shape %785 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %787 = tensor.empty() : tensor<1x28x8x128xf32>
    %788 = linalg.transpose ins(%786:tensor<1x8x28x128xf32>) outs(%787:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %789 = tensor.collapse_shape %788 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %790 = tensor.expand_shape %789 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %791 = tensor.empty() : tensor<1024x1024xf32>
    %792 = linalg.transpose ins(%17:tensor<1024x1024xf32>) outs(%791:tensor<1024x1024xf32>) permutation = [1, 0]
    %793 = tensor.collapse_shape %790 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wo"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %794 = tensor.expand_shape %793 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wo"} : tensor<28672xf32> into tensor<28x1024xf32>
    %795 = tensor.empty() : tensor<28x1024xf32>
    %796 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %797 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%796 : f32) outs(%795 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %798 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wo", prov.transposed_b = "true"} ins(%794, %792 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%797 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %799 = tensor.collapse_shape %798 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wo"} : tensor<28x1024xf32> into tensor<28672xf32>
    %800 = tensor.expand_shape %799 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.cross_attn.wo"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %801 = tensor.empty() : tensor<1x28x1024xf32>
    %802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%547, %800 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%801 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb80(%803: f32, %804: f32, %805: f32):
      %806 = arith.mulf %803, %804 : f32
      linalg.yield %806 : f32
    } -> tensor<1x28x1024xf32>
    %807 = tensor.empty() : tensor<1x28x1024xf32>
    %808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541, %802 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%807 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb81(%809: f32, %810: f32, %811: f32):
      %812 = arith.addf %809, %810 : f32
      linalg.yield %812 : f32
    } -> tensor<1x28x1024xf32>
    %813 = tensor.collapse_shape %213 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %814 = tensor.expand_shape %813 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %815 = tensor.empty() : tensor<1x28x1024xf32>
    %816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%808 : tensor<1x28x1024xf32>) outs(%815 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb82(%817: f32, %818: f32):
      %819 = arith.constant 2.000000e+00 : f32
      %820 = math.powf %817, %819 : f32
      linalg.yield %820 : f32
    } -> tensor<1x28x1024xf32>
    %821 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} 0.000000e+00 : f32
    %822 = tensor.splat %821 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} : tensor<1x28xf32>
    %823 = linalg.reduce ins(%816:tensor<1x28x1024xf32>) outs(%822:tensor<1x28xf32>) dimensions = [2]
    (%824: f32, %825: f32) {
      %826 = arith.addf %824, %825 : f32
      linalg.yield %826 : f32
    }
    %827 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} 1.024000e+03 : f32
    %828 = tensor.splat %827 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} : tensor<1x28xf32>
    %829 = tensor.empty() : tensor<1x28xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%823, %828 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%829 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb83(%831: f32, %832: f32, %833: f32):
      %834 = arith.divf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<1x28xf32>
    %835 = tensor.collapse_shape %830 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %836 = tensor.expand_shape %835 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %837 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} 1.000000e-05 : f32
    %838 = tensor.splat %837 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} : tensor<1x28x1xf32>
    %839 = tensor.empty() : tensor<1x28x1xf32>
    %840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%836, %838 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%839 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb84(%841: f32, %842: f32, %843: f32):
      %844 = arith.addf %841, %842 : f32
      linalg.yield %844 : f32
    } -> tensor<1x28x1xf32>
    %845 = tensor.empty() : tensor<1x28x1xf32>
    %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%840 : tensor<1x28x1xf32>) outs(%845 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb85(%847: f32, %848: f32):
      %849 = math.rsqrt %847 : f32
      linalg.yield %849 : f32
    } -> tensor<1x28x1xf32>
    %850 = tensor.empty() : tensor<1x28x1024xf32>
    %851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%808, %846 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%850 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb86(%852: f32, %853: f32, %854: f32):
      %855 = arith.mulf %852, %853 : f32
      linalg.yield %855 : f32
    } -> tensor<1x28x1024xf32>
    %856 = tensor.empty() : tensor<1x28x1024xf32>
    %857 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%851, %20 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%856 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn_norm"} {
    ^bb87(%858: f32, %859: f32, %860: f32):
      %861 = arith.mulf %858, %859 : f32
      linalg.yield %861 : f32
    } -> tensor<1x28x1024xf32>
    %862 = tensor.collapse_shape %212 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %863 = tensor.expand_shape %862 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %864 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} 1.000000e+00 : f32
    %865 = tensor.splat %864 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1x1024xf32>
    %866 = tensor.empty() : tensor<1x1x1024xf32>
    %867 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%863, %865 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%866 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb88(%868: f32, %869: f32, %870: f32):
      %871 = arith.addf %868, %869 : f32
      linalg.yield %871 : f32
    } -> tensor<1x1x1024xf32>
    %872 = tensor.empty() : tensor<1x28x1024xf32>
    %873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%857, %867 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%872 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb89(%874: f32, %875: f32, %876: f32):
      %877 = arith.mulf %874, %875 : f32
      linalg.yield %877 : f32
    } -> tensor<1x28x1024xf32>
    %878 = tensor.collapse_shape %211 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1x1024xf32> into tensor<1024xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %880 = tensor.empty() : tensor<1x28x1024xf32>
    %881 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%873, %879 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%880 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb90(%882: f32, %883: f32, %884: f32):
      %885 = arith.addf %882, %883 : f32
      linalg.yield %885 : f32
    } -> tensor<1x28x1024xf32>
    %886 = tensor.empty() : tensor<1024x2816xf32>
    %887 = linalg.transpose ins(%21:tensor<2816x1024xf32>) outs(%886:tensor<1024x2816xf32>) permutation = [1, 0]
    %888 = tensor.collapse_shape %881 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w1"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %889 = tensor.expand_shape %888 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w1"} : tensor<28672xf32> into tensor<28x1024xf32>
    %890 = tensor.empty() : tensor<28x2816xf32>
    %891 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %892 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%891 : f32) outs(%890 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %893 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w1", prov.transposed_b = "true"} ins(%889, %887 : tensor<28x1024xf32>, tensor<1024x2816xf32>) outs(%892 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %894 = tensor.collapse_shape %893 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w1"} : tensor<28x2816xf32> into tensor<78848xf32>
    %895 = tensor.expand_shape %894 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 2816] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w1"} : tensor<78848xf32> into tensor<1x28x2816xf32>
    %896 = tensor.empty() : tensor<1x28x2816xf32>
    %897 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895 : tensor<1x28x2816xf32>) outs(%896 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn"} {
    ^bb91(%898: f32, %899: f32):
      %900 = arith.constant 1.000000e+00 : f32
      %901 = arith.negf %898 : f32
      %902 = math.exp %901 : f32
      %903 = arith.addf %900, %902 : f32
      %904 = arith.divf %900, %903 : f32
      linalg.yield %904 : f32
    } -> tensor<1x28x2816xf32>
    %905 = tensor.empty() : tensor<1x28x2816xf32>
    %906 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895, %897 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%905 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn"} {
    ^bb92(%907: f32, %908: f32, %909: f32):
      %910 = arith.mulf %907, %908 : f32
      linalg.yield %910 : f32
    } -> tensor<1x28x2816xf32>
    %911 = tensor.empty() : tensor<1024x2816xf32>
    %912 = linalg.transpose ins(%23:tensor<2816x1024xf32>) outs(%911:tensor<1024x2816xf32>) permutation = [1, 0]
    %913 = tensor.collapse_shape %881 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w3"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %914 = tensor.expand_shape %913 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w3"} : tensor<28672xf32> into tensor<28x1024xf32>
    %915 = tensor.empty() : tensor<28x2816xf32>
    %916 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %917 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%916 : f32) outs(%915 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %918 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w3", prov.transposed_b = "true"} ins(%914, %912 : tensor<28x1024xf32>, tensor<1024x2816xf32>) outs(%917 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %919 = tensor.collapse_shape %918 [[0 : i64, 1 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w3"} : tensor<28x2816xf32> into tensor<78848xf32>
    %920 = tensor.expand_shape %919 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 2816] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w3"} : tensor<78848xf32> into tensor<1x28x2816xf32>
    %921 = tensor.empty() : tensor<1x28x2816xf32>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%906, %920 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%921 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn"} {
    ^bb93(%923: f32, %924: f32, %925: f32):
      %926 = arith.mulf %923, %924 : f32
      linalg.yield %926 : f32
    } -> tensor<1x28x2816xf32>
    %927 = tensor.empty() : tensor<2816x1024xf32>
    %928 = linalg.transpose ins(%22:tensor<1024x2816xf32>) outs(%927:tensor<2816x1024xf32>) permutation = [1, 0]
    %929 = tensor.collapse_shape %922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w2"} : tensor<1x28x2816xf32> into tensor<78848xf32>
    %930 = tensor.expand_shape %929 [[0 : i64, 1 : i64]] output_shape [28, 2816] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w2"} : tensor<78848xf32> into tensor<28x2816xf32>
    %931 = tensor.empty() : tensor<28x1024xf32>
    %932 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %933 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%932 : f32) outs(%931 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %934 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w2", prov.transposed_b = "true"} ins(%930, %928 : tensor<28x2816xf32>, tensor<2816x1024xf32>) outs(%933 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %935 = tensor.collapse_shape %934 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w2"} : tensor<28x1024xf32> into tensor<28672xf32>
    %936 = tensor.expand_shape %935 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0.ffn.w2"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %937 = tensor.empty() : tensor<1x28x1024xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%814, %936 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%937 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb94(%939: f32, %940: f32, %941: f32):
      %942 = arith.mulf %939, %940 : f32
      linalg.yield %942 : f32
    } -> tensor<1x28x1024xf32>
    %943 = tensor.empty() : tensor<1x28x1024xf32>
    %944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%808, %938 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%943 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.0"} {
    ^bb95(%945: f32, %946: f32, %947: f32):
      %948 = arith.addf %945, %946 : f32
      linalg.yield %948 : f32
    } -> tensor<1x28x1024xf32>
    %949 = tensor.empty() : tensor<1x64x4x128xf32>
    %950 = linalg.transpose ins(%57:tensor<1x4x64x128xf32>) outs(%949:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
    %951 = tensor.empty() : tensor<1x64x4x128xf32>
    %952 = linalg.transpose ins(%58:tensor<1x4x64x128xf32>) outs(%951:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
    %953 = tensor.empty() : tensor<1x2048xf32>
    %954 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162 : tensor<1x2048xf32>) outs(%953 : tensor<1x2048xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.adaLN_modulation.0"} {
    ^bb96(%955: f32, %956: f32):
      %957 = arith.constant 1.000000e+00 : f32
      %958 = arith.negf %955 : f32
      %959 = math.exp %958 : f32
      %960 = arith.addf %957, %959 : f32
      %961 = arith.divf %957, %960 : f32
      linalg.yield %961 : f32
    } -> tensor<1x2048xf32>
    %962 = tensor.empty() : tensor<1x2048xf32>
    %963 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162, %954 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%962 : tensor<1x2048xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.adaLN_modulation.0"} {
    ^bb97(%964: f32, %965: f32, %966: f32):
      %967 = arith.mulf %964, %965 : f32
      linalg.yield %967 : f32
    } -> tensor<1x2048xf32>
    %968 = tensor.empty() : tensor<2048x9216xf32>
    %969 = linalg.transpose ins(%43:tensor<9216x2048xf32>) outs(%968:tensor<2048x9216xf32>) permutation = [1, 0]
    %970 = tensor.empty() : tensor<1x9216xf32>
    %971 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %972 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%971 : f32) outs(%970 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
    %973 = linalg.matmul {prov.region_id = "matmul_15", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.adaLN_modulation.1", prov.transposed_b = "true"} ins(%963, %969 : tensor<1x2048xf32>, tensor<2048x9216xf32>) outs(%972 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
    %974 = tensor.empty() : tensor<1x9216xf32>
    %975 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%973, %44 : tensor<1x9216xf32>, tensor<9216xf32>) outs(%974 : tensor<1x9216xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.adaLN_modulation.1"} {
    ^bb98(%976: f32, %977: f32, %978: f32):
      %979 = arith.addf %976, %977 : f32
      linalg.yield %979 : f32
    } -> tensor<1x9216xf32>
    %980 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %981 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %982 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 2048>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %983 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 3072>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %984 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 4096>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %985 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 5120>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %986 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 6144>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %987 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 7168>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %988 = "tensor.extract_slice"(%975) <{static_offsets = array<i64: 0, 8192>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
    %989 = tensor.collapse_shape %982 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %990 = tensor.expand_shape %989 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %991 = tensor.empty() : tensor<1x28x1024xf32>
    %992 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944 : tensor<1x28x1024xf32>) outs(%991 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb99(%993: f32, %994: f32):
      %995 = arith.constant 2.000000e+00 : f32
      %996 = math.powf %993, %995 : f32
      linalg.yield %996 : f32
    } -> tensor<1x28x1024xf32>
    %997 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} 0.000000e+00 : f32
    %998 = tensor.splat %997 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} : tensor<1x28xf32>
    %999 = linalg.reduce ins(%992:tensor<1x28x1024xf32>) outs(%998:tensor<1x28xf32>) dimensions = [2]
    (%1000: f32, %1001: f32) {
      %1002 = arith.addf %1000, %1001 : f32
      linalg.yield %1002 : f32
    }
    %1003 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} 1.024000e+03 : f32
    %1004 = tensor.splat %1003 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} : tensor<1x28xf32>
    %1005 = tensor.empty() : tensor<1x28xf32>
    %1006 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%999, %1004 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1005 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb100(%1007: f32, %1008: f32, %1009: f32):
      %1010 = arith.divf %1007, %1008 : f32
      linalg.yield %1010 : f32
    } -> tensor<1x28xf32>
    %1011 = tensor.collapse_shape %1006 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %1012 = tensor.expand_shape %1011 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %1013 = arith.constant {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} 1.000000e-05 : f32
    %1014 = tensor.splat %1013 {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} : tensor<1x28x1xf32>
    %1015 = tensor.empty() : tensor<1x28x1xf32>
    %1016 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1012, %1014 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1015 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb101(%1017: f32, %1018: f32, %1019: f32):
      %1020 = arith.addf %1017, %1018 : f32
      linalg.yield %1020 : f32
    } -> tensor<1x28x1xf32>
    %1021 = tensor.empty() : tensor<1x28x1xf32>
    %1022 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1016 : tensor<1x28x1xf32>) outs(%1021 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb102(%1023: f32, %1024: f32):
      %1025 = math.rsqrt %1023 : f32
      linalg.yield %1025 : f32
    } -> tensor<1x28x1xf32>
    %1026 = tensor.empty() : tensor<1x28x1024xf32>
    %1027 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944, %1022 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1026 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb103(%1028: f32, %1029: f32, %1030: f32):
      %1031 = arith.mulf %1028, %1029 : f32
      linalg.yield %1031 : f32
    } -> tensor<1x28x1024xf32>
    %1032 = tensor.empty() : tensor<1x28x1024xf32>
    %1033 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1027, %26 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1032 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn_norm"} {
    ^bb104(%1034: f32, %1035: f32, %1036: f32):
      %1037 = arith.mulf %1034, %1035 : f32
      linalg.yield %1037 : f32
    } -> tensor<1x28x1024xf32>
    %1038 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1039 = tensor.expand_shape %1038 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1040 = arith.constant {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} 1.000000e+00 : f32
    %1041 = tensor.splat %1040 {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1x1024xf32>
    %1042 = tensor.empty() : tensor<1x1x1024xf32>
    %1043 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1039, %1041 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1042 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb105(%1044: f32, %1045: f32, %1046: f32):
      %1047 = arith.addf %1044, %1045 : f32
      linalg.yield %1047 : f32
    } -> tensor<1x1x1024xf32>
    %1048 = tensor.empty() : tensor<1x28x1024xf32>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1033, %1043 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1048 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb106(%1050: f32, %1051: f32, %1052: f32):
      %1053 = arith.mulf %1050, %1051 : f32
      linalg.yield %1053 : f32
    } -> tensor<1x28x1024xf32>
    %1054 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1055 = tensor.expand_shape %1054 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1056 = tensor.empty() : tensor<1x28x1024xf32>
    %1057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1049, %1055 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1056 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb107(%1058: f32, %1059: f32, %1060: f32):
      %1061 = arith.addf %1058, %1059 : f32
      linalg.yield %1061 : f32
    } -> tensor<1x28x1024xf32>
    %1062 = tensor.empty() : tensor<1024x1024xf32>
    %1063 = linalg.transpose ins(%27:tensor<1024x1024xf32>) outs(%1062:tensor<1024x1024xf32>) permutation = [1, 0]
    %1064 = tensor.collapse_shape %1057 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wq"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1065 = tensor.expand_shape %1064 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wq"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1066 = tensor.empty() : tensor<28x1024xf32>
    %1067 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1068 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1067 : f32) outs(%1066 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1069 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wq", prov.transposed_b = "true"} ins(%1065, %1063 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%1068 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1070 = tensor.collapse_shape %1069 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wq"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1071 = tensor.expand_shape %1070 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wq"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1072 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1073 = tensor.expand_shape %1072 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1074 = tensor.empty() : tensor<1024x1024xf32>
    %1075 = linalg.transpose ins(%28:tensor<1024x1024xf32>) outs(%1074:tensor<1024x1024xf32>) permutation = [1, 0]
    %1076 = tensor.collapse_shape %1057 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wkv"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1077 = tensor.expand_shape %1076 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wkv"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1078 = tensor.empty() : tensor<28x1024xf32>
    %1079 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1080 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1079 : f32) outs(%1078 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1081 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wkv", prov.transposed_b = "true"} ins(%1077, %1075 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%1080 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1082 = tensor.collapse_shape %1081 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wkv"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1083 = tensor.expand_shape %1082 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wkv"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1084 = tensor.collapse_shape %1083 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1085 = tensor.expand_shape %1084 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 128, 2] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x28x4x128x2xf32>
    %1086 = "tensor.extract_slice"(%1085) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
    %1087 = "tensor.extract_slice"(%1085) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
    %1088 = tensor.collapse_shape %1086 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
    %1089 = tensor.expand_shape %1088 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "squeeze_2", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
    %1090 = tensor.collapse_shape %1087 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
    %1091 = tensor.expand_shape %1090 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "squeeze_3", prov._pattern_hint = "squeeze", prov.op = "squeeze", prov.family = "layout", prov.aten = "aten.squeeze.dims", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
    %1092 = tensor.empty() : tensor<1x28x8x128xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1073 : tensor<1x28x8x128xf32>) outs(%1092 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb108(%1094: f32, %1095: f32):
      %1096 = arith.constant 2.000000e+00 : f32
      %1097 = math.powf %1094, %1096 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x28x8x128xf32>
    %1098 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} 0.000000e+00 : f32
    %1099 = tensor.splat %1098 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} : tensor<1x28x8xf32>
    %1100 = linalg.reduce ins(%1093:tensor<1x28x8x128xf32>) outs(%1099:tensor<1x28x8xf32>) dimensions = [3]
    (%1101: f32, %1102: f32) {
      %1103 = arith.addf %1101, %1102 : f32
      linalg.yield %1103 : f32
    }
    %1104 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} 1.280000e+02 : f32
    %1105 = tensor.splat %1104 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} : tensor<1x28x8xf32>
    %1106 = tensor.empty() : tensor<1x28x8xf32>
    %1107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100, %1105 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%1106 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb109(%1108: f32, %1109: f32, %1110: f32):
      %1111 = arith.divf %1108, %1109 : f32
      linalg.yield %1111 : f32
    } -> tensor<1x28x8xf32>
    %1112 = tensor.collapse_shape %1107 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1113 = tensor.expand_shape %1112 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1114 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} 1.000000e-05 : f32
    %1115 = tensor.splat %1114 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} : tensor<1x28x8x1xf32>
    %1116 = tensor.empty() : tensor<1x28x8x1xf32>
    %1117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1113, %1115 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%1116 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb110(%1118: f32, %1119: f32, %1120: f32):
      %1121 = arith.addf %1118, %1119 : f32
      linalg.yield %1121 : f32
    } -> tensor<1x28x8x1xf32>
    %1122 = tensor.empty() : tensor<1x28x8x1xf32>
    %1123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1117 : tensor<1x28x8x1xf32>) outs(%1122 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb111(%1124: f32, %1125: f32):
      %1126 = math.rsqrt %1124 : f32
      linalg.yield %1126 : f32
    } -> tensor<1x28x8x1xf32>
    %1127 = tensor.empty() : tensor<1x28x8x128xf32>
    %1128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1073, %1123 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%1127 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb112(%1129: f32, %1130: f32, %1131: f32):
      %1132 = arith.mulf %1129, %1130 : f32
      linalg.yield %1132 : f32
    } -> tensor<1x28x8x128xf32>
    %1133 = tensor.empty() : tensor<1x28x8x128xf32>
    %1134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1128, %30 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%1133 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_q"} {
    ^bb113(%1135: f32, %1136: f32, %1137: f32):
      %1138 = arith.mulf %1135, %1136 : f32
      linalg.yield %1138 : f32
    } -> tensor<1x28x8x128xf32>
    %1139 = tensor.empty() : tensor<1x28x4x128xf32>
    %1140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x28x4x128xf32>) outs(%1139 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb114(%1141: f32, %1142: f32):
      %1143 = arith.constant 2.000000e+00 : f32
      %1144 = math.powf %1141, %1143 : f32
      linalg.yield %1144 : f32
    } -> tensor<1x28x4x128xf32>
    %1145 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} 0.000000e+00 : f32
    %1146 = tensor.splat %1145 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} : tensor<1x28x4xf32>
    %1147 = linalg.reduce ins(%1140:tensor<1x28x4x128xf32>) outs(%1146:tensor<1x28x4xf32>) dimensions = [3]
    (%1148: f32, %1149: f32) {
      %1150 = arith.addf %1148, %1149 : f32
      linalg.yield %1150 : f32
    }
    %1151 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} 1.280000e+02 : f32
    %1152 = tensor.splat %1151 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} : tensor<1x28x4xf32>
    %1153 = tensor.empty() : tensor<1x28x4xf32>
    %1154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %1152 : tensor<1x28x4xf32>, tensor<1x28x4xf32>) outs(%1153 : tensor<1x28x4xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb115(%1155: f32, %1156: f32, %1157: f32):
      %1158 = arith.divf %1155, %1156 : f32
      linalg.yield %1158 : f32
    } -> tensor<1x28x4xf32>
    %1159 = tensor.collapse_shape %1154 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} : tensor<1x28x4xf32> into tensor<112xf32>
    %1160 = tensor.expand_shape %1159 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} : tensor<112xf32> into tensor<1x28x4x1xf32>
    %1161 = arith.constant {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} 1.000000e-05 : f32
    %1162 = tensor.splat %1161 {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} : tensor<1x28x4x1xf32>
    %1163 = tensor.empty() : tensor<1x28x4x1xf32>
    %1164 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1160, %1162 : tensor<1x28x4x1xf32>, tensor<1x28x4x1xf32>) outs(%1163 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb116(%1165: f32, %1166: f32, %1167: f32):
      %1168 = arith.addf %1165, %1166 : f32
      linalg.yield %1168 : f32
    } -> tensor<1x28x4x1xf32>
    %1169 = tensor.empty() : tensor<1x28x4x1xf32>
    %1170 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1164 : tensor<1x28x4x1xf32>) outs(%1169 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb117(%1171: f32, %1172: f32):
      %1173 = math.rsqrt %1171 : f32
      linalg.yield %1173 : f32
    } -> tensor<1x28x4x1xf32>
    %1174 = tensor.empty() : tensor<1x28x4x128xf32>
    %1175 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089, %1170 : tensor<1x28x4x128xf32>, tensor<1x28x4x1xf32>) outs(%1174 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb118(%1176: f32, %1177: f32, %1178: f32):
      %1179 = arith.mulf %1176, %1177 : f32
      linalg.yield %1179 : f32
    } -> tensor<1x28x4x128xf32>
    %1180 = tensor.empty() : tensor<1x28x4x128xf32>
    %1181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1175, %31 : tensor<1x28x4x128xf32>, tensor<128xf32>) outs(%1180 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.norm_k"} {
    ^bb119(%1182: f32, %1183: f32, %1184: f32):
      %1185 = arith.mulf %1182, %1183 : f32
      linalg.yield %1185 : f32
    } -> tensor<1x28x4x128xf32>
    %1186 = tensor.collapse_shape %1181 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
    %1187 = tensor.expand_shape %1186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
    %1188 = tensor.empty() : tensor<1x28x4x2x128xf32>
    %1189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1187 : tensor<1x28x4x1x128xf32>) outs(%1188 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb120(%1190: f32, %1191: f32):
      linalg.yield %1190 : f32
    } -> tensor<1x28x4x2x128xf32>
    %1192 = tensor.collapse_shape %1189 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
    %1193 = tensor.expand_shape %1192 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1194 = tensor.collapse_shape %1091 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
    %1195 = tensor.expand_shape %1194 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
    %1196 = tensor.empty() : tensor<1x28x4x2x128xf32>
    %1197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1195 : tensor<1x28x4x1x128xf32>) outs(%1196 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb121(%1198: f32, %1199: f32):
      linalg.yield %1198 : f32
    } -> tensor<1x28x4x2x128xf32>
    %1200 = tensor.collapse_shape %1197 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
    %1201 = tensor.expand_shape %1200 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1202 = tensor.empty() : tensor<1x8x28x128xf32>
    %1203 = linalg.transpose ins(%1134:tensor<1x28x8x128xf32>) outs(%1202:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1204 = tensor.empty() : tensor<1x8x28x128xf32>
    %1205 = linalg.transpose ins(%1193:tensor<1x28x8x128xf32>) outs(%1204:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1206 = tensor.empty() : tensor<1x8x28x128xf32>
    %1207 = linalg.transpose ins(%1201:tensor<1x28x8x128xf32>) outs(%1206:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1208 = tensor.empty() : tensor<1x8x128x28xf32>
    %1209 = linalg.transpose ins(%1205:tensor<1x8x28x128xf32>) outs(%1208:tensor<1x8x128x28xf32>) permutation = [0, 1, 3, 2]
    %1210 = tensor.empty() : tensor<1x8x28x128xf32>
    %1211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1203 : tensor<1x8x28x128xf32>) outs(%1210 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb122(%1212: f32, %1213: f32):
      linalg.yield %1212 : f32
    } -> tensor<1x8x28x128xf32>
    %1214 = tensor.collapse_shape %1211 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1215 = tensor.expand_shape %1214 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %1216 = tensor.empty() : tensor<1x8x128x28xf32>
    %1217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1209 : tensor<1x8x128x28xf32>) outs(%1216 : tensor<1x8x128x28xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb123(%1218: f32, %1219: f32):
      linalg.yield %1218 : f32
    } -> tensor<1x8x128x28xf32>
    %1220 = tensor.collapse_shape %1217 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x128x28xf32> into tensor<28672xf32>
    %1221 = tensor.expand_shape %1220 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 128, 28] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<8x128x28xf32>
    %1222 = arith.constant {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1223 = tensor.splat %1222 {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<8x28x28xf32>
    %1224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1215, %1221 : tensor<8x28x128xf32>, tensor<8x128x28xf32>) outs(%1223 : tensor<8x28x28xf32>) attrs =  {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb124(%1225: f32, %1226: f32, %1227: f32):
      %1228 = arith.mulf %1225, %1226 : f32
      %1229 = arith.addf %1227, %1228 : f32
      linalg.yield %1229 : f32
    } -> tensor<8x28x28xf32>
    %1230 = tensor.collapse_shape %1224 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<8x28x28xf32> into tensor<6272xf32>
    %1231 = tensor.expand_shape %1230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 28] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<6272xf32> into tensor<1x8x28x28xf32>
    %1232 = arith.constant {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.0883883461 : f32
    %1233 = tensor.splat %1232 {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28x28xf32>
    %1234 = tensor.empty() : tensor<1x8x28x28xf32>
    %1235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1231, %1233 : tensor<1x8x28x28xf32>, tensor<1x8x28x28xf32>) outs(%1234 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb125(%1236: f32, %1237: f32, %1238: f32):
      %1239 = arith.mulf %1236, %1237 : f32
      linalg.yield %1239 : f32
    } -> tensor<1x8x28x28xf32>
    %1240 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0xff800000 : f32
    %1241 = tensor.splat %1240 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28xf32>
    %1242 = linalg.reduce ins(%1235:tensor<1x8x28x28xf32>) outs(%1241:tensor<1x8x28xf32>) dimensions = [3]
    (%1243: f32, %1244: f32) {
      %1245 = arith.maximumf %1243, %1244 : f32
      linalg.yield %1245 : f32
    }
    %1246 = tensor.collapse_shape %1242 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %1247 = tensor.expand_shape %1246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %1248 = tensor.empty() : tensor<1x8x28x28xf32>
    %1249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1235, %1247 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%1248 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb126(%1250: f32, %1251: f32, %1252: f32):
      %1253 = arith.subf %1250, %1251 : f32
      linalg.yield %1253 : f32
    } -> tensor<1x8x28x28xf32>
    %1254 = tensor.empty() : tensor<1x8x28x28xf32>
    %1255 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1249 : tensor<1x8x28x28xf32>) outs(%1254 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb127(%1256: f32, %1257: f32):
      %1258 = math.exp %1256 : f32
      linalg.yield %1258 : f32
    } -> tensor<1x8x28x28xf32>
    %1259 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1260 = tensor.splat %1259 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28xf32>
    %1261 = linalg.reduce ins(%1255:tensor<1x8x28x28xf32>) outs(%1260:tensor<1x8x28xf32>) dimensions = [3]
    (%1262: f32, %1263: f32) {
      %1264 = arith.addf %1262, %1263 : f32
      linalg.yield %1264 : f32
    }
    %1265 = tensor.collapse_shape %1261 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %1266 = tensor.expand_shape %1265 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %1267 = tensor.empty() : tensor<1x8x28x28xf32>
    %1268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1255, %1266 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%1267 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb128(%1269: f32, %1270: f32, %1271: f32):
      %1272 = arith.divf %1269, %1270 : f32
      linalg.yield %1272 : f32
    } -> tensor<1x8x28x28xf32>
    %1273 = tensor.empty() : tensor<1x8x28x28xf32>
    %1274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1268 : tensor<1x8x28x28xf32>) outs(%1273 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb129(%1275: f32, %1276: f32):
      linalg.yield %1275 : f32
    } -> tensor<1x8x28x28xf32>
    %1277 = tensor.collapse_shape %1274 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28x28xf32> into tensor<6272xf32>
    %1278 = tensor.expand_shape %1277 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 28] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<6272xf32> into tensor<8x28x28xf32>
    %1279 = tensor.empty() : tensor<1x8x28x128xf32>
    %1280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1207 : tensor<1x8x28x128xf32>) outs(%1279 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb130(%1281: f32, %1282: f32):
      linalg.yield %1281 : f32
    } -> tensor<1x8x28x128xf32>
    %1283 = tensor.collapse_shape %1280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1284 = tensor.expand_shape %1283 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %1285 = arith.constant {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} 0.000000e+00 : f32
    %1286 = tensor.splat %1285 {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<8x28x128xf32>
    %1287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1278, %1284 : tensor<8x28x28xf32>, tensor<8x28x128xf32>) outs(%1286 : tensor<8x28x128xf32>) attrs =  {prov.region_id = "matmul_19", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} {
    ^bb131(%1288: f32, %1289: f32, %1290: f32):
      %1291 = arith.mulf %1288, %1289 : f32
      %1292 = arith.addf %1290, %1291 : f32
      linalg.yield %1292 : f32
    } -> tensor<8x28x128xf32>
    %1293 = tensor.collapse_shape %1287 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<8x28x128xf32> into tensor<28672xf32>
    %1294 = tensor.expand_shape %1293 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1295 = tensor.empty() : tensor<1x28x8x128xf32>
    %1296 = linalg.transpose ins(%1294:tensor<1x8x28x128xf32>) outs(%1295:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1297 = tensor.collapse_shape %1296 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1299 = tensor.empty() : tensor<1024x1024xf32>
    %1300 = linalg.transpose ins(%29:tensor<1024x1024xf32>) outs(%1299:tensor<1024x1024xf32>) permutation = [1, 0]
    %1301 = tensor.collapse_shape %1298 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wo"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1302 = tensor.expand_shape %1301 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wo"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1303 = tensor.empty() : tensor<28x1024xf32>
    %1304 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1305 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1304 : f32) outs(%1303 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1306 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wo", prov.transposed_b = "true"} ins(%1302, %1300 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%1305 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1307 = tensor.collapse_shape %1306 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wo"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1308 = tensor.expand_shape %1307 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.attn.wo"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1309 = tensor.empty() : tensor<1x28x1024xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%990, %1308 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1309 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb132(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.mulf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x28x1024xf32>
    %1315 = tensor.empty() : tensor<1x28x1024xf32>
    %1316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944, %1310 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1315 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb133(%1317: f32, %1318: f32, %1319: f32):
      %1320 = arith.addf %1317, %1318 : f32
      linalg.yield %1320 : f32
    } -> tensor<1x28x1024xf32>
    %1321 = tensor.collapse_shape %985 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1322 = tensor.expand_shape %1321 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1323 = tensor.empty() : tensor<1x28x1024xf32>
    %1324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316 : tensor<1x28x1024xf32>) outs(%1323 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb134(%1325: f32, %1326: f32):
      %1327 = arith.constant 2.000000e+00 : f32
      %1328 = math.powf %1325, %1327 : f32
      linalg.yield %1328 : f32
    } -> tensor<1x28x1024xf32>
    %1329 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} 0.000000e+00 : f32
    %1330 = tensor.splat %1329 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} : tensor<1x28xf32>
    %1331 = linalg.reduce ins(%1324:tensor<1x28x1024xf32>) outs(%1330:tensor<1x28xf32>) dimensions = [2]
    (%1332: f32, %1333: f32) {
      %1334 = arith.addf %1332, %1333 : f32
      linalg.yield %1334 : f32
    }
    %1335 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} 1.024000e+03 : f32
    %1336 = tensor.splat %1335 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} : tensor<1x28xf32>
    %1337 = tensor.empty() : tensor<1x28xf32>
    %1338 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1331, %1336 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1337 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb135(%1339: f32, %1340: f32, %1341: f32):
      %1342 = arith.divf %1339, %1340 : f32
      linalg.yield %1342 : f32
    } -> tensor<1x28xf32>
    %1343 = tensor.collapse_shape %1338 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %1344 = tensor.expand_shape %1343 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %1345 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} 1.000000e-05 : f32
    %1346 = tensor.splat %1345 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} : tensor<1x28x1xf32>
    %1347 = tensor.empty() : tensor<1x28x1xf32>
    %1348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1344, %1346 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1347 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb136(%1349: f32, %1350: f32, %1351: f32):
      %1352 = arith.addf %1349, %1350 : f32
      linalg.yield %1352 : f32
    } -> tensor<1x28x1xf32>
    %1353 = tensor.empty() : tensor<1x28x1xf32>
    %1354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1348 : tensor<1x28x1xf32>) outs(%1353 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_9", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb137(%1355: f32, %1356: f32):
      %1357 = math.rsqrt %1355 : f32
      linalg.yield %1357 : f32
    } -> tensor<1x28x1xf32>
    %1358 = tensor.empty() : tensor<1x28x1024xf32>
    %1359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316, %1354 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1358 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb138(%1360: f32, %1361: f32, %1362: f32):
      %1363 = arith.mulf %1360, %1361 : f32
      linalg.yield %1363 : f32
    } -> tensor<1x28x1024xf32>
    %1364 = tensor.empty() : tensor<1x28x1024xf32>
    %1365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1359, %32 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1364 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_norm"} {
    ^bb139(%1366: f32, %1367: f32, %1368: f32):
      %1369 = arith.mulf %1366, %1367 : f32
      linalg.yield %1369 : f32
    } -> tensor<1x28x1024xf32>
    %1370 = tensor.collapse_shape %984 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1371 = tensor.expand_shape %1370 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1372 = arith.constant {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} 1.000000e+00 : f32
    %1373 = tensor.splat %1372 {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1x1024xf32>
    %1374 = tensor.empty() : tensor<1x1x1024xf32>
    %1375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1371, %1373 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1374 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb140(%1376: f32, %1377: f32, %1378: f32):
      %1379 = arith.addf %1376, %1377 : f32
      linalg.yield %1379 : f32
    } -> tensor<1x1x1024xf32>
    %1380 = tensor.empty() : tensor<1x28x1024xf32>
    %1381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1365, %1375 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1380 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb141(%1382: f32, %1383: f32, %1384: f32):
      %1385 = arith.mulf %1382, %1383 : f32
      linalg.yield %1385 : f32
    } -> tensor<1x28x1024xf32>
    %1386 = tensor.collapse_shape %983 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1387 = tensor.expand_shape %1386 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1388 = tensor.empty() : tensor<1x28x1024xf32>
    %1389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1381, %1387 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1388 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb142(%1390: f32, %1391: f32, %1392: f32):
      %1393 = arith.addf %1390, %1391 : f32
      linalg.yield %1393 : f32
    } -> tensor<1x28x1024xf32>
    %1394 = tensor.empty() : tensor<1024x1024xf32>
    %1395 = linalg.transpose ins(%34:tensor<1024x1024xf32>) outs(%1394:tensor<1024x1024xf32>) permutation = [1, 0]
    %1396 = tensor.collapse_shape %1389 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wq"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1397 = tensor.expand_shape %1396 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wq"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1398 = tensor.empty() : tensor<28x1024xf32>
    %1399 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1400 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1399 : f32) outs(%1398 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1401 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wq", prov.transposed_b = "true"} ins(%1397, %1395 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%1400 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1402 = tensor.collapse_shape %1401 [[0 : i64, 1 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wq"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1403 = tensor.expand_shape %1402 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wq"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1404 = tensor.collapse_shape %1403 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1405 = tensor.expand_shape %1404 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1406 = tensor.empty() : tensor<1x28x8x128xf32>
    %1407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1405 : tensor<1x28x8x128xf32>) outs(%1406 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb143(%1408: f32, %1409: f32):
      %1410 = arith.constant 2.000000e+00 : f32
      %1411 = math.powf %1408, %1410 : f32
      linalg.yield %1411 : f32
    } -> tensor<1x28x8x128xf32>
    %1412 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} 0.000000e+00 : f32
    %1413 = tensor.splat %1412 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} : tensor<1x28x8xf32>
    %1414 = linalg.reduce ins(%1407:tensor<1x28x8x128xf32>) outs(%1413:tensor<1x28x8xf32>) dimensions = [3]
    (%1415: f32, %1416: f32) {
      %1417 = arith.addf %1415, %1416 : f32
      linalg.yield %1417 : f32
    }
    %1418 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} 1.280000e+02 : f32
    %1419 = tensor.splat %1418 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} : tensor<1x28x8xf32>
    %1420 = tensor.empty() : tensor<1x28x8xf32>
    %1421 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1414, %1419 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%1420 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb144(%1422: f32, %1423: f32, %1424: f32):
      %1425 = arith.divf %1422, %1423 : f32
      linalg.yield %1425 : f32
    } -> tensor<1x28x8xf32>
    %1426 = tensor.collapse_shape %1421 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1427 = tensor.expand_shape %1426 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1428 = arith.constant {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} 1.000000e-05 : f32
    %1429 = tensor.splat %1428 {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} : tensor<1x28x8x1xf32>
    %1430 = tensor.empty() : tensor<1x28x8x1xf32>
    %1431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1427, %1429 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%1430 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb145(%1432: f32, %1433: f32, %1434: f32):
      %1435 = arith.addf %1432, %1433 : f32
      linalg.yield %1435 : f32
    } -> tensor<1x28x8x1xf32>
    %1436 = tensor.empty() : tensor<1x28x8x1xf32>
    %1437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1431 : tensor<1x28x8x1xf32>) outs(%1436 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_10", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb146(%1438: f32, %1439: f32):
      %1440 = math.rsqrt %1438 : f32
      linalg.yield %1440 : f32
    } -> tensor<1x28x8x1xf32>
    %1441 = tensor.empty() : tensor<1x28x8x128xf32>
    %1442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1405, %1437 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%1441 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb147(%1443: f32, %1444: f32, %1445: f32):
      %1446 = arith.mulf %1443, %1444 : f32
      linalg.yield %1446 : f32
    } -> tensor<1x28x8x128xf32>
    %1447 = tensor.empty() : tensor<1x28x8x128xf32>
    %1448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1442, %37 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%1447 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.norm_q"} {
    ^bb148(%1449: f32, %1450: f32, %1451: f32):
      %1452 = arith.mulf %1449, %1450 : f32
      linalg.yield %1452 : f32
    } -> tensor<1x28x8x128xf32>
    %1453 = tensor.collapse_shape %950 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
    %1454 = tensor.expand_shape %1453 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
    %1455 = tensor.empty() : tensor<1x64x4x2x128xf32>
    %1456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1454 : tensor<1x64x4x1x128xf32>) outs(%1455 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb149(%1457: f32, %1458: f32):
      linalg.yield %1457 : f32
    } -> tensor<1x64x4x2x128xf32>
    %1459 = tensor.collapse_shape %1456 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
    %1460 = tensor.expand_shape %1459 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
    %1461 = tensor.collapse_shape %952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
    %1462 = tensor.expand_shape %1461 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
    %1463 = tensor.empty() : tensor<1x64x4x2x128xf32>
    %1464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1462 : tensor<1x64x4x1x128xf32>) outs(%1463 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb150(%1465: f32, %1466: f32):
      linalg.yield %1465 : f32
    } -> tensor<1x64x4x2x128xf32>
    %1467 = tensor.collapse_shape %1464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
    %1468 = tensor.expand_shape %1467 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
    %1469 = tensor.empty() : tensor<1x8x28x128xf32>
    %1470 = linalg.transpose ins(%1448:tensor<1x28x8x128xf32>) outs(%1469:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1471 = tensor.empty() : tensor<1x8x64x128xf32>
    %1472 = linalg.transpose ins(%1460:tensor<1x64x8x128xf32>) outs(%1471:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
    %1473 = tensor.empty() : tensor<1x8x64x128xf32>
    %1474 = linalg.transpose ins(%1468:tensor<1x64x8x128xf32>) outs(%1473:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
    %1475 = tensor.empty() : tensor<1x8x128x64xf32>
    %1476 = linalg.transpose ins(%1472:tensor<1x8x64x128xf32>) outs(%1475:tensor<1x8x128x64xf32>) permutation = [0, 1, 3, 2]
    %1477 = tensor.empty() : tensor<1x8x28x128xf32>
    %1478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1470 : tensor<1x8x28x128xf32>) outs(%1477 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "expand_22", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb151(%1479: f32, %1480: f32):
      linalg.yield %1479 : f32
    } -> tensor<1x8x28x128xf32>
    %1481 = tensor.collapse_shape %1478 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1482 = tensor.expand_shape %1481 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 128] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<28672xf32> into tensor<8x28x128xf32>
    %1483 = tensor.empty() : tensor<1x8x128x64xf32>
    %1484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x8x128x64xf32>) outs(%1483 : tensor<1x8x128x64xf32>) attrs =  {prov.region_id = "expand_23", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb152(%1485: f32, %1486: f32):
      linalg.yield %1485 : f32
    } -> tensor<1x8x128x64xf32>
    %1487 = tensor.collapse_shape %1484 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x128x64xf32> into tensor<65536xf32>
    %1488 = tensor.expand_shape %1487 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 128, 64] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<65536xf32> into tensor<8x128x64xf32>
    %1489 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1490 = tensor.splat %1489 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8x28x64xf32>
    %1491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1482, %1488 : tensor<8x28x128xf32>, tensor<8x128x64xf32>) outs(%1490 : tensor<8x28x64xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb153(%1492: f32, %1493: f32, %1494: f32):
      %1495 = arith.mulf %1492, %1493 : f32
      %1496 = arith.addf %1494, %1495 : f32
      linalg.yield %1496 : f32
    } -> tensor<8x28x64xf32>
    %1497 = tensor.collapse_shape %1491 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8x28x64xf32> into tensor<14336xf32>
    %1498 = tensor.expand_shape %1497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 64] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<14336xf32> into tensor<1x8x28x64xf32>
    %1499 = arith.constant {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.0883883461 : f32
    %1500 = tensor.splat %1499 {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28x64xf32>
    %1501 = tensor.empty() : tensor<1x8x28x64xf32>
    %1502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1498, %1500 : tensor<1x8x28x64xf32>, tensor<1x8x28x64xf32>) outs(%1501 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb154(%1503: f32, %1504: f32, %1505: f32):
      %1506 = arith.mulf %1503, %1504 : f32
      linalg.yield %1506 : f32
    } -> tensor<1x8x28x64xf32>
    %1507 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0xff800000 : f32
    %1508 = tensor.splat %1507 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28xf32>
    %1509 = linalg.reduce ins(%1502:tensor<1x8x28x64xf32>) outs(%1508:tensor<1x8x28xf32>) dimensions = [3]
    (%1510: f32, %1511: f32) {
      %1512 = arith.maximumf %1510, %1511 : f32
      linalg.yield %1512 : f32
    }
    %1513 = tensor.collapse_shape %1509 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %1514 = tensor.expand_shape %1513 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %1515 = tensor.empty() : tensor<1x8x28x64xf32>
    %1516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1502, %1514 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%1515 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb155(%1517: f32, %1518: f32, %1519: f32):
      %1520 = arith.subf %1517, %1518 : f32
      linalg.yield %1520 : f32
    } -> tensor<1x8x28x64xf32>
    %1521 = tensor.empty() : tensor<1x8x28x64xf32>
    %1522 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1516 : tensor<1x8x28x64xf32>) outs(%1521 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb156(%1523: f32, %1524: f32):
      %1525 = math.exp %1523 : f32
      linalg.yield %1525 : f32
    } -> tensor<1x8x28x64xf32>
    %1526 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1527 = tensor.splat %1526 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28xf32>
    %1528 = linalg.reduce ins(%1522:tensor<1x8x28x64xf32>) outs(%1527:tensor<1x8x28xf32>) dimensions = [3]
    (%1529: f32, %1530: f32) {
      %1531 = arith.addf %1529, %1530 : f32
      linalg.yield %1531 : f32
    }
    %1532 = tensor.collapse_shape %1528 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28xf32> into tensor<224xf32>
    %1533 = tensor.expand_shape %1532 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<224xf32> into tensor<1x8x28x1xf32>
    %1534 = tensor.empty() : tensor<1x8x28x64xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1522, %1533 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%1534 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb157(%1536: f32, %1537: f32, %1538: f32):
      %1539 = arith.divf %1536, %1537 : f32
      linalg.yield %1539 : f32
    } -> tensor<1x8x28x64xf32>
    %1540 = tensor.empty() : tensor<1x8x28x64xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1535 : tensor<1x8x28x64xf32>) outs(%1540 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "expand_24", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb158(%1542: f32, %1543: f32):
      linalg.yield %1542 : f32
    } -> tensor<1x8x28x64xf32>
    %1544 = tensor.collapse_shape %1541 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x28x64xf32> into tensor<14336xf32>
    %1545 = tensor.expand_shape %1544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 28, 64] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<14336xf32> into tensor<8x28x64xf32>
    %1546 = tensor.empty() : tensor<1x8x64x128xf32>
    %1547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1474 : tensor<1x8x64x128xf32>) outs(%1546 : tensor<1x8x64x128xf32>) attrs =  {prov.region_id = "expand_25", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb159(%1548: f32, %1549: f32):
      linalg.yield %1548 : f32
    } -> tensor<1x8x64x128xf32>
    %1550 = tensor.collapse_shape %1547 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x8x64x128xf32> into tensor<65536xf32>
    %1551 = tensor.expand_shape %1550 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 64, 128] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<65536xf32> into tensor<8x64x128xf32>
    %1552 = arith.constant {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} 0.000000e+00 : f32
    %1553 = tensor.splat %1552 {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8x28x128xf32>
    %1554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1545, %1551 : tensor<8x28x64xf32>, tensor<8x64x128xf32>) outs(%1553 : tensor<8x28x128xf32>) attrs =  {prov.region_id = "matmul_23", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} {
    ^bb160(%1555: f32, %1556: f32, %1557: f32):
      %1558 = arith.mulf %1555, %1556 : f32
      %1559 = arith.addf %1557, %1558 : f32
      linalg.yield %1559 : f32
    } -> tensor<8x28x128xf32>
    %1560 = tensor.collapse_shape %1554 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<8x28x128xf32> into tensor<28672xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1562 = tensor.empty() : tensor<1x28x8x128xf32>
    %1563 = linalg.transpose ins(%1561:tensor<1x8x28x128xf32>) outs(%1562:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1564 = tensor.collapse_shape %1563 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1565 = tensor.expand_shape %1564 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1566 = tensor.empty() : tensor<1024x1024xf32>
    %1567 = linalg.transpose ins(%36:tensor<1024x1024xf32>) outs(%1566:tensor<1024x1024xf32>) permutation = [1, 0]
    %1568 = tensor.collapse_shape %1565 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wo"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1569 = tensor.expand_shape %1568 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wo"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1570 = tensor.empty() : tensor<28x1024xf32>
    %1571 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1572 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1571 : f32) outs(%1570 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1573 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wo", prov.transposed_b = "true"} ins(%1569, %1567 : tensor<28x1024xf32>, tensor<1024x1024xf32>) outs(%1572 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1574 = tensor.collapse_shape %1573 [[0 : i64, 1 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wo"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1575 = tensor.expand_shape %1574 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.cross_attn.wo"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1576 = tensor.empty() : tensor<1x28x1024xf32>
    %1577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1322, %1575 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1576 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb161(%1578: f32, %1579: f32, %1580: f32):
      %1581 = arith.mulf %1578, %1579 : f32
      linalg.yield %1581 : f32
    } -> tensor<1x28x1024xf32>
    %1582 = tensor.empty() : tensor<1x28x1024xf32>
    %1583 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1316, %1577 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1582 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb162(%1584: f32, %1585: f32, %1586: f32):
      %1587 = arith.addf %1584, %1585 : f32
      linalg.yield %1587 : f32
    } -> tensor<1x28x1024xf32>
    %1588 = tensor.collapse_shape %988 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1589 = tensor.expand_shape %1588 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1590 = tensor.empty() : tensor<1x28x1024xf32>
    %1591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1583 : tensor<1x28x1024xf32>) outs(%1590 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_11", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb163(%1592: f32, %1593: f32):
      %1594 = arith.constant 2.000000e+00 : f32
      %1595 = math.powf %1592, %1594 : f32
      linalg.yield %1595 : f32
    } -> tensor<1x28x1024xf32>
    %1596 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} 0.000000e+00 : f32
    %1597 = tensor.splat %1596 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} : tensor<1x28xf32>
    %1598 = linalg.reduce ins(%1591:tensor<1x28x1024xf32>) outs(%1597:tensor<1x28xf32>) dimensions = [2]
    (%1599: f32, %1600: f32) {
      %1601 = arith.addf %1599, %1600 : f32
      linalg.yield %1601 : f32
    }
    %1602 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} 1.024000e+03 : f32
    %1603 = tensor.splat %1602 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} : tensor<1x28xf32>
    %1604 = tensor.empty() : tensor<1x28xf32>
    %1605 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1598, %1603 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1604 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb164(%1606: f32, %1607: f32, %1608: f32):
      %1609 = arith.divf %1606, %1607 : f32
      linalg.yield %1609 : f32
    } -> tensor<1x28xf32>
    %1610 = tensor.collapse_shape %1605 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %1611 = tensor.expand_shape %1610 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %1612 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} 1.000000e-05 : f32
    %1613 = tensor.splat %1612 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} : tensor<1x28x1xf32>
    %1614 = tensor.empty() : tensor<1x28x1xf32>
    %1615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1611, %1613 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1614 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb165(%1616: f32, %1617: f32, %1618: f32):
      %1619 = arith.addf %1616, %1617 : f32
      linalg.yield %1619 : f32
    } -> tensor<1x28x1xf32>
    %1620 = tensor.empty() : tensor<1x28x1xf32>
    %1621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1615 : tensor<1x28x1xf32>) outs(%1620 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_11", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb166(%1622: f32, %1623: f32):
      %1624 = math.rsqrt %1622 : f32
      linalg.yield %1624 : f32
    } -> tensor<1x28x1xf32>
    %1625 = tensor.empty() : tensor<1x28x1024xf32>
    %1626 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1583, %1621 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1625 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb167(%1627: f32, %1628: f32, %1629: f32):
      %1630 = arith.mulf %1627, %1628 : f32
      linalg.yield %1630 : f32
    } -> tensor<1x28x1024xf32>
    %1631 = tensor.empty() : tensor<1x28x1024xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1626, %39 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1631 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn_norm"} {
    ^bb168(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.mulf %1633, %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<1x28x1024xf32>
    %1637 = tensor.collapse_shape %987 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1638 = tensor.expand_shape %1637 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1639 = arith.constant {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} 1.000000e+00 : f32
    %1640 = tensor.splat %1639 {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1x1024xf32>
    %1641 = tensor.empty() : tensor<1x1x1024xf32>
    %1642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1638, %1640 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1641 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb169(%1643: f32, %1644: f32, %1645: f32):
      %1646 = arith.addf %1643, %1644 : f32
      linalg.yield %1646 : f32
    } -> tensor<1x1x1024xf32>
    %1647 = tensor.empty() : tensor<1x28x1024xf32>
    %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1632, %1642 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1647 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb170(%1649: f32, %1650: f32, %1651: f32):
      %1652 = arith.mulf %1649, %1650 : f32
      linalg.yield %1652 : f32
    } -> tensor<1x28x1024xf32>
    %1653 = tensor.collapse_shape %986 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1654 = tensor.expand_shape %1653 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1655 = tensor.empty() : tensor<1x28x1024xf32>
    %1656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1648, %1654 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1655 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb171(%1657: f32, %1658: f32, %1659: f32):
      %1660 = arith.addf %1657, %1658 : f32
      linalg.yield %1660 : f32
    } -> tensor<1x28x1024xf32>
    %1661 = tensor.empty() : tensor<1024x2816xf32>
    %1662 = linalg.transpose ins(%40:tensor<2816x1024xf32>) outs(%1661:tensor<1024x2816xf32>) permutation = [1, 0]
    %1663 = tensor.collapse_shape %1656 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w1"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1664 = tensor.expand_shape %1663 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w1"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1665 = tensor.empty() : tensor<28x2816xf32>
    %1666 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1667 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1666 : f32) outs(%1665 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %1668 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w1", prov.transposed_b = "true"} ins(%1664, %1662 : tensor<28x1024xf32>, tensor<1024x2816xf32>) outs(%1667 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %1669 = tensor.collapse_shape %1668 [[0 : i64, 1 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w1"} : tensor<28x2816xf32> into tensor<78848xf32>
    %1670 = tensor.expand_shape %1669 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 2816] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w1"} : tensor<78848xf32> into tensor<1x28x2816xf32>
    %1671 = tensor.empty() : tensor<1x28x2816xf32>
    %1672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1670 : tensor<1x28x2816xf32>) outs(%1671 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "sigmoid_4", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn"} {
    ^bb172(%1673: f32, %1674: f32):
      %1675 = arith.constant 1.000000e+00 : f32
      %1676 = arith.negf %1673 : f32
      %1677 = math.exp %1676 : f32
      %1678 = arith.addf %1675, %1677 : f32
      %1679 = arith.divf %1675, %1678 : f32
      linalg.yield %1679 : f32
    } -> tensor<1x28x2816xf32>
    %1680 = tensor.empty() : tensor<1x28x2816xf32>
    %1681 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1670, %1672 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%1680 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn"} {
    ^bb173(%1682: f32, %1683: f32, %1684: f32):
      %1685 = arith.mulf %1682, %1683 : f32
      linalg.yield %1685 : f32
    } -> tensor<1x28x2816xf32>
    %1686 = tensor.empty() : tensor<1024x2816xf32>
    %1687 = linalg.transpose ins(%42:tensor<2816x1024xf32>) outs(%1686:tensor<1024x2816xf32>) permutation = [1, 0]
    %1688 = tensor.collapse_shape %1656 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w3"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1689 = tensor.expand_shape %1688 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w3"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1690 = tensor.empty() : tensor<28x2816xf32>
    %1691 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1692 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1691 : f32) outs(%1690 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %1693 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w3", prov.transposed_b = "true"} ins(%1689, %1687 : tensor<28x1024xf32>, tensor<1024x2816xf32>) outs(%1692 : tensor<28x2816xf32>) -> tensor<28x2816xf32>
    %1694 = tensor.collapse_shape %1693 [[0 : i64, 1 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w3"} : tensor<28x2816xf32> into tensor<78848xf32>
    %1695 = tensor.expand_shape %1694 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 2816] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w3"} : tensor<78848xf32> into tensor<1x28x2816xf32>
    %1696 = tensor.empty() : tensor<1x28x2816xf32>
    %1697 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1681, %1695 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%1696 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn"} {
    ^bb174(%1698: f32, %1699: f32, %1700: f32):
      %1701 = arith.mulf %1698, %1699 : f32
      linalg.yield %1701 : f32
    } -> tensor<1x28x2816xf32>
    %1702 = tensor.empty() : tensor<2816x1024xf32>
    %1703 = linalg.transpose ins(%41:tensor<1024x2816xf32>) outs(%1702:tensor<2816x1024xf32>) permutation = [1, 0]
    %1704 = tensor.collapse_shape %1697 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w2"} : tensor<1x28x2816xf32> into tensor<78848xf32>
    %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64]] output_shape [28, 2816] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w2"} : tensor<78848xf32> into tensor<28x2816xf32>
    %1706 = tensor.empty() : tensor<28x1024xf32>
    %1707 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1708 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1707 : f32) outs(%1706 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1709 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w2", prov.transposed_b = "true"} ins(%1705, %1703 : tensor<28x2816xf32>, tensor<2816x1024xf32>) outs(%1708 : tensor<28x1024xf32>) -> tensor<28x1024xf32>
    %1710 = tensor.collapse_shape %1709 [[0 : i64, 1 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w2"} : tensor<28x1024xf32> into tensor<28672xf32>
    %1711 = tensor.expand_shape %1710 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1.ffn.w2"} : tensor<28672xf32> into tensor<1x28x1024xf32>
    %1712 = tensor.empty() : tensor<1x28x1024xf32>
    %1713 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1589, %1711 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1712 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb175(%1714: f32, %1715: f32, %1716: f32):
      %1717 = arith.mulf %1714, %1715 : f32
      linalg.yield %1717 : f32
    } -> tensor<1x28x1024xf32>
    %1718 = tensor.empty() : tensor<1x28x1024xf32>
    %1719 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1583, %1713 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1718 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.blocks.1"} {
    ^bb176(%1720: f32, %1721: f32, %1722: f32):
      %1723 = arith.addf %1720, %1721 : f32
      linalg.yield %1723 : f32
    } -> tensor<1x28x1024xf32>
    %1724 = tensor.empty() : tensor<1x2048xf32>
    %1725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162 : tensor<1x2048xf32>) outs(%1724 : tensor<1x2048xf32>) attrs =  {prov.region_id = "sigmoid_5", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.adaLN_modulation.0"} {
    ^bb177(%1726: f32, %1727: f32):
      %1728 = arith.constant 1.000000e+00 : f32
      %1729 = arith.negf %1726 : f32
      %1730 = math.exp %1729 : f32
      %1731 = arith.addf %1728, %1730 : f32
      %1732 = arith.divf %1728, %1731 : f32
      linalg.yield %1732 : f32
    } -> tensor<1x2048xf32>
    %1733 = tensor.empty() : tensor<1x2048xf32>
    %1734 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%162, %1725 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%1733 : tensor<1x2048xf32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.adaLN_modulation.0"} {
    ^bb178(%1735: f32, %1736: f32, %1737: f32):
      %1738 = arith.mulf %1735, %1736 : f32
      linalg.yield %1738 : f32
    } -> tensor<1x2048xf32>
    %1739 = tensor.empty() : tensor<2048x2048xf32>
    %1740 = linalg.transpose ins(%50:tensor<2048x2048xf32>) outs(%1739:tensor<2048x2048xf32>) permutation = [1, 0]
    %1741 = tensor.empty() : tensor<1x2048xf32>
    %1742 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1743 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1742 : f32) outs(%1741 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %1744 = linalg.matmul {prov.region_id = "matmul_28", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.adaLN_modulation.1", prov.transposed_b = "true"} ins(%1734, %1740 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1743 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %1745 = tensor.empty() : tensor<1x2048xf32>
    %1746 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1744, %51 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%1745 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.adaLN_modulation.1"} {
    ^bb179(%1747: f32, %1748: f32, %1749: f32):
      %1750 = arith.addf %1747, %1748 : f32
      linalg.yield %1750 : f32
    } -> tensor<1x2048xf32>
    %1751 = "tensor.extract_slice"(%1746) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : (tensor<1x2048xf32>) -> tensor<1x1024xf32>
    %1752 = "tensor.extract_slice"(%1746) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : (tensor<1x2048xf32>) -> tensor<1x1024xf32>
    %1753 = tensor.empty() : tensor<1x28x1024xf32>
    %1754 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1719 : tensor<1x28x1024xf32>) outs(%1753 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_12", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb180(%1755: f32, %1756: f32):
      %1757 = arith.constant 2.000000e+00 : f32
      %1758 = math.powf %1755, %1757 : f32
      linalg.yield %1758 : f32
    } -> tensor<1x28x1024xf32>
    %1759 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} 0.000000e+00 : f32
    %1760 = tensor.splat %1759 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} : tensor<1x28xf32>
    %1761 = linalg.reduce ins(%1754:tensor<1x28x1024xf32>) outs(%1760:tensor<1x28xf32>) dimensions = [2]
    (%1762: f32, %1763: f32) {
      %1764 = arith.addf %1762, %1763 : f32
      linalg.yield %1764 : f32
    }
    %1765 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} 1.024000e+03 : f32
    %1766 = tensor.splat %1765 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} : tensor<1x28xf32>
    %1767 = tensor.empty() : tensor<1x28xf32>
    %1768 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1761, %1766 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1767 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb181(%1769: f32, %1770: f32, %1771: f32):
      %1772 = arith.divf %1769, %1770 : f32
      linalg.yield %1772 : f32
    } -> tensor<1x28xf32>
    %1773 = tensor.collapse_shape %1768 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} : tensor<1x28xf32> into tensor<28xf32>
    %1774 = tensor.expand_shape %1773 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} : tensor<28xf32> into tensor<1x28x1xf32>
    %1775 = arith.constant {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} 1.000000e-05 : f32
    %1776 = tensor.splat %1775 {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} : tensor<1x28x1xf32>
    %1777 = tensor.empty() : tensor<1x28x1xf32>
    %1778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1774, %1776 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1777 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb182(%1779: f32, %1780: f32, %1781: f32):
      %1782 = arith.addf %1779, %1780 : f32
      linalg.yield %1782 : f32
    } -> tensor<1x28x1xf32>
    %1783 = tensor.empty() : tensor<1x28x1xf32>
    %1784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1778 : tensor<1x28x1xf32>) outs(%1783 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_12", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb183(%1785: f32, %1786: f32):
      %1787 = math.rsqrt %1785 : f32
      linalg.yield %1787 : f32
    } -> tensor<1x28x1xf32>
    %1788 = tensor.empty() : tensor<1x28x1024xf32>
    %1789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1719, %1784 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1788 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_50", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb184(%1790: f32, %1791: f32, %1792: f32):
      %1793 = arith.mulf %1790, %1791 : f32
      linalg.yield %1793 : f32
    } -> tensor<1x28x1024xf32>
    %1794 = tensor.empty() : tensor<1x28x1024xf32>
    %1795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1789, %45 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1794 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn_norm"} {
    ^bb185(%1796: f32, %1797: f32, %1798: f32):
      %1799 = arith.mulf %1796, %1797 : f32
      linalg.yield %1799 : f32
    } -> tensor<1x28x1024xf32>
    %1800 = tensor.collapse_shape %1752 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1801 = tensor.expand_shape %1800 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1802 = arith.constant {prov.region_id = "add_38", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} 1.000000e+00 : f32
    %1803 = tensor.splat %1802 {prov.region_id = "add_38", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : tensor<1x1x1024xf32>
    %1804 = tensor.empty() : tensor<1x1x1024xf32>
    %1805 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1801, %1803 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1804 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} {
    ^bb186(%1806: f32, %1807: f32, %1808: f32):
      %1809 = arith.addf %1806, %1807 : f32
      linalg.yield %1809 : f32
    } -> tensor<1x1x1024xf32>
    %1810 = tensor.empty() : tensor<1x28x1024xf32>
    %1811 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1795, %1805 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1810 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} {
    ^bb187(%1812: f32, %1813: f32, %1814: f32):
      %1815 = arith.mulf %1812, %1813 : f32
      linalg.yield %1815 : f32
    } -> tensor<1x28x1024xf32>
    %1816 = tensor.collapse_shape %1751 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : tensor<1x1024xf32> into tensor<1024xf32>
    %1817 = tensor.expand_shape %1816 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %1818 = tensor.empty() : tensor<1x28x1024xf32>
    %1819 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1811, %1817 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1818 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_39", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer"} {
    ^bb188(%1820: f32, %1821: f32, %1822: f32):
      %1823 = arith.addf %1820, %1821 : f32
      linalg.yield %1823 : f32
    } -> tensor<1x28x1024xf32>
    %1824 = tensor.collapse_shape %1819 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1"} : tensor<1x28x1024xf32> into tensor<28672xf32>
    %1825 = tensor.expand_shape %1824 [[0 : i64, 1 : i64]] output_shape [28, 1024] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1"} : tensor<28672xf32> into tensor<28x1024xf32>
    %1826 = tensor.empty() : tensor<1024x4096xf32>
    %1827 = linalg.transpose ins(%46:tensor<4096x1024xf32>) outs(%1826:tensor<1024x4096xf32>) permutation = [1, 0]
    %1828 = tensor.empty() : tensor<28x4096xf32>
    %1829 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1830 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1829 : f32) outs(%1828 : tensor<28x4096xf32>) -> tensor<28x4096xf32>
    %1831 = linalg.matmul {prov.region_id = "matmul_29", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1", prov.transposed_b = "true"} ins(%1825, %1827 : tensor<28x1024xf32>, tensor<1024x4096xf32>) outs(%1830 : tensor<28x4096xf32>) -> tensor<28x4096xf32>
    %1832 = tensor.empty() : tensor<28x4096xf32>
    %1833 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1831, %47 : tensor<28x4096xf32>, tensor<4096xf32>) outs(%1832 : tensor<28x4096xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1"} {
    ^bb189(%1834: f32, %1835: f32, %1836: f32):
      %1837 = arith.addf %1834, %1835 : f32
      linalg.yield %1837 : f32
    } -> tensor<28x4096xf32>
    %1838 = tensor.collapse_shape %1833 [[0 : i64, 1 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1"} : tensor<28x4096xf32> into tensor<114688xf32>
    %1839 = tensor.expand_shape %1838 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 4096] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc1"} : tensor<114688xf32> into tensor<1x28x4096xf32>
    %1840 = tensor.empty() : tensor<1x28x4096xf32>
    %1841 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1839 : tensor<1x28x4096xf32>) outs(%1840 : tensor<1x28x4096xf32>) attrs =  {prov.region_id = "sigmoid_6", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.act"} {
    ^bb190(%1842: f32, %1843: f32):
      %1844 = arith.constant 1.000000e+00 : f32
      %1845 = arith.negf %1842 : f32
      %1846 = math.exp %1845 : f32
      %1847 = arith.addf %1844, %1846 : f32
      %1848 = arith.divf %1844, %1847 : f32
      linalg.yield %1848 : f32
    } -> tensor<1x28x4096xf32>
    %1849 = tensor.empty() : tensor<1x28x4096xf32>
    %1850 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1839, %1841 : tensor<1x28x4096xf32>, tensor<1x28x4096xf32>) outs(%1849 : tensor<1x28x4096xf32>) attrs =  {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.act"} {
    ^bb191(%1851: f32, %1852: f32, %1853: f32):
      %1854 = arith.mulf %1851, %1852 : f32
      linalg.yield %1854 : f32
    } -> tensor<1x28x4096xf32>
    %1855 = tensor.collapse_shape %1850 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2"} : tensor<1x28x4096xf32> into tensor<114688xf32>
    %1856 = tensor.expand_shape %1855 [[0 : i64, 1 : i64]] output_shape [28, 4096] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2"} : tensor<114688xf32> into tensor<28x4096xf32>
    %1857 = tensor.empty() : tensor<4096x20xf32>
    %1858 = linalg.transpose ins(%48:tensor<20x4096xf32>) outs(%1857:tensor<4096x20xf32>) permutation = [1, 0]
    %1859 = tensor.empty() : tensor<28x20xf32>
    %1860 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %1861 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%1860 : f32) outs(%1859 : tensor<28x20xf32>) -> tensor<28x20xf32>
    %1862 = linalg.matmul {prov.region_id = "matmul_30", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2", prov.transposed_b = "true"} ins(%1856, %1858 : tensor<28x4096xf32>, tensor<4096x20xf32>) outs(%1861 : tensor<28x20xf32>) -> tensor<28x20xf32>
    %1863 = tensor.empty() : tensor<28x20xf32>
    %1864 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1862, %49 : tensor<28x20xf32>, tensor<20xf32>) outs(%1863 : tensor<28x20xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2"} {
    ^bb192(%1865: f32, %1866: f32, %1867: f32):
      %1868 = arith.addf %1865, %1866 : f32
      linalg.yield %1868 : f32
    } -> tensor<28x20xf32>
    %1869 = tensor.collapse_shape %1864 [[0 : i64, 1 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2"} : tensor<28x20xf32> into tensor<560xf32>
    %1870 = tensor.expand_shape %1869 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 20] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.final_layer.ffn.fc2"} : tensor<560xf32> into tensor<1x28x20xf32>
    %1871 = "tensor.extract_slice"(%1870) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 24, 20>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : (tensor<1x28x20xf32>) -> tensor<1x24x20xf32>
    func.return %1871 : tensor<1x24x20xf32>
  }
}
